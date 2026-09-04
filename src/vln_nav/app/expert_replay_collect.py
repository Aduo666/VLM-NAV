from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, fields
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from vln_nav.navigation.grounding_dino_detector import GroundingDinoConfig, GroundingDinoDetector
from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.occupancy_grid import GlobalOccupancyMap, OccupancyGridConfig
from vln_nav.navigation.rgbd_geometry_map import RGBDGeometryMapConfig, RGBDVirtualRayProjector
from vln_nav.navigation.vlm_client import VLMClient
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from vln_nav.training.dataset_writer import ExpertReplayDatasetWriter
from vln_nav.training.expert_replay import ExpertReplayConfig, build_expert_replay
from vln_nav.training.frontier import FrontierExtractor, FrontierExtractorConfig, FrontierTracker
from vln_nav.training.frontier_oracle import (
    FrontierOracleConfig,
    label_frontiers_from_reference_route,
)
from vln_nav.training.interiornav import load_canonical_episode
from vln_nav.training.offline_perception import (
    LowConfidenceVLMVerifier,
    OfflinePerceptionConfig,
    OfflineSemanticTracker,
)
from vln_nav.training.perception_scheduler import PerceptionScheduler, PerceptionSchedulerConfig
from vln_nav.training.query_compiler import QueryCompiler
from vln_nav.training.route_alignment import align_instruction_to_route, progress_alignment_at_frame
from vln_nav.training.schemas import (
    CameraCalibration,
    DebugOracle,
    EXPERT_REPLAY_SCHEMA_VERSION,
    ExpertReplayFrame,
    MotionHistoryToken,
    ProgressObservation,
    ProgressSupervision,
    ProgressTrainingView,
    SpatialSubgoalObservation,
    SpatialSubgoalSupervision,
    SpatialSubgoalTrainingView,
    TRAINING_SAMPLE_SCHEMA_VERSION,
    TrainingSampleV3,
)
from vln_nav.training.timeline import fixed_timeline_bounds, render_timeline_frame


@dataclass(frozen=True)
class CollectionLightingConfig:
    enabled: bool = True
    intensity: float = 3000.0
    exposure: float = 2.0
    shadows_enabled: bool = False


@dataclass(frozen=True)
class CollectionSettings:
    data_root: Path
    output_root: Path
    render_warmup_frames: int
    render_frames_per_pose: int
    pose_position_tolerance_m: float
    pose_orientation_tolerance_deg: float
    rgb_history_frames: int
    motion_history_frames: int
    local_map_size_m: float
    local_map_cells: int
    global_map_cells: int
    lighting: CollectionLightingConfig
    replay: ExpertReplayConfig
    occupancy: OccupancyGridConfig
    rgbd: RGBDGeometryMapConfig
    frontier: FrontierExtractorConfig
    frontier_oracle: FrontierOracleConfig
    scheduler: PerceptionSchedulerConfig
    perception: OfflinePerceptionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect causal CAPTAINav samples by replaying an InteriorNav expert path"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--training-config",
        default=str(PROJECT_ROOT / "training" / "configs" / "expert_replay.yaml"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-headless to inspect replay rendering in the Isaac Sim window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.split).strip() != "train":
        raise ValueError(
            f"Expert replay collection accepts only split=train; got {args.split!r}"
        )
    app_config = load_config(args.config)
    if app_config.robot.name != "agx" or app_config.robot.is_g1:
        raise RuntimeError("Expert replay V1 is intentionally restricted to robot.profile=agx")
    settings = _load_collection_settings(
        path=Path(args.training_config).expanduser().resolve(),
        output_override=str(args.output_root),
    )
    episode = load_canonical_episode(
        data_root=settings.data_root,
        split=str(args.split),
        episode_index=int(args.episode_index),
        episode_id=str(args.episode_id),
    )
    if not Path(episode.scene_usd_path).is_file():
        raise FileNotFoundError(f"InteriorNav scene USD does not exist: {episode.scene_usd_path}")
    replay = build_expert_replay(episode, config=settings.replay)
    queries = QueryCompiler().compile(episode.instruction)
    alignments = align_instruction_to_route(episode=episode, replay=replay)
    logger = _setup_logger()
    logger.info(
        "[ExpertReplay] episode=%s split=%s scene=%s poses=%s queries=%s alignments=%s",
        episode.episode_id,
        episode.split,
        episode.scene_id,
        len(replay),
        [query.text for query in queries],
        len(alignments),
    )
    app_config.simulation.headless = bool(args.headless)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(_simulation_app_settings(app_config))
    try:
        return _collect(
            simulation_app=simulation_app,
            app_config=app_config,
            settings=settings,
            episode=episode,
            replay=replay,
            queries=queries,
            alignments=alignments,
            overwrite=bool(args.overwrite),
            logger=logger,
        )
    except BaseException:
        logger.exception("[ExpertReplayFatal] collection failed")
        raise
    finally:
        simulation_app.close()


def _collect(
    *,
    simulation_app,
    app_config,
    settings: CollectionSettings,
    episode,
    replay,
    queries,
    alignments,
    overwrite: bool,
    logger,
) -> int:
    try:
        from isaacsim.core.prims import SingleArticulation as RobotArticulation
    except ImportError:
        from isaacsim.core.prims import Articulation as RobotArticulation
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera
    from pxr import UsdLux

    from vln_nav.runtime.isaac_runtime_utils import (
        resolve_existing_prim_path,
        resolve_robot_relative_prim_path,
        wait_for_stage_assets,
    )

    world = World(
        physics_dt=float(app_config.robot.simulation.physics_dt),
        rendering_dt=float(app_config.robot.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    add_reference_to_stage(str(episode.scene_usd_path), app_config.scene.prim_path)
    wait_for_stage_assets(
        simulation_app=simulation_app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(app_config.scene.stage_load_timeout_sec),
        logger=logger,
        label="InteriorNav expert replay scene",
    )
    add_reference_to_stage(str(app_config.robot.asset.usd_path), app_config.robot.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=simulation_app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(app_config.scene.stage_load_timeout_sec),
        logger=logger,
        label="AGX expert replay robot",
    )
    if settings.lighting.enabled:
        dome = UsdLux.DomeLight.Define(stage, "/World/ExpertReplayDomeLight")
        dome.CreateIntensityAttr().Set(float(settings.lighting.intensity))
        dome.CreateExposureAttr().Set(float(settings.lighting.exposure))
        shadow_api = UsdLux.ShadowAPI.Apply(dome.GetPrim())
        shadow_api.CreateShadowEnableAttr().Set(bool(settings.lighting.shadows_enabled))
        logger.info(
            "[ExpertReplayLight] intensity=%.1f exposure=%.1f shadows=%s",
            settings.lighting.intensity,
            settings.lighting.exposure,
            settings.lighting.shadows_enabled,
        )
    articulation_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=app_config.robot.asset.prim_path,
            prim_path=app_config.robot.asset.articulation_path,
        ),
        kind="expert replay AGX articulation",
    )
    robot = world.scene.add(
        RobotArticulation(
            prim_path=articulation_path,
            name="expert_replay_robot",
            reset_xform_properties=True,
        )
    )
    pose_writer = _ReplayPoseWriter(
        robot=robot,
        spawn_z_m=float(app_config.robot.spawn_z_m),
    )
    pose_writer.set_default(pose=replay[0])
    front_camera_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=app_config.robot.asset.prim_path,
            prim_path=app_config.robot.sensors.camera.front_prim_path,
        ),
        kind="expert replay front camera",
    )
    camera = Camera(
        prim_path=front_camera_path,
        frequency=max(1, int(app_config.robot.sensors.camera.frequency_hz)),
        resolution=(
            int(app_config.robot.sensors.camera.width),
            int(app_config.robot.sensors.camera.height),
        ),
    )
    world.reset()
    robot.initialize()
    pose_writer.set(pose=replay[0])
    camera.initialize(attach_rgb_annotator=True)
    _attach_depth_annotator(camera=camera)
    # Isaac render products need one played simulation sample before paused
    # render-only updates can produce valid RGB-D buffers.
    world.step(render=True)
    for _ in range(max(0, int(settings.render_warmup_frames) - 1)):
        _render_without_physics(simulation_app=simulation_app)
    camera_pose_writer = _ReplayCameraPoseWriter(camera=camera, robot=robot)
    logger.info(
        "[ExpertReplayCamera] calibrated relative_position_m=%s relative_orientation_wxyz=%s",
        np.round(camera_pose_writer.relative_position_m, 4).tolist(),
        np.round(camera_pose_writer.relative_orientation_wxyz, 5).tolist(),
    )
    pose_writer.set(pose=replay[0])
    _render_without_physics(simulation_app=simulation_app)
    pose_writer.verify(
        pose=replay[0],
        position_tolerance_m=settings.pose_position_tolerance_m,
        orientation_tolerance_deg=settings.pose_orientation_tolerance_deg,
    )
    camera_pose_writer.set(robot_state=_normalized_pose(robot.get_world_pose()))
    _render_without_physics(simulation_app=simulation_app)
    camera_pose_writer.verify(
        robot_state=_normalized_pose(robot.get_world_pose()),
        frame_index=0,
        position_tolerance_m=settings.pose_position_tolerance_m,
        orientation_tolerance_deg=settings.pose_orientation_tolerance_deg,
    )
    _capture_rgbd(
        camera=camera,
        world=world,
        simulation_app=simulation_app,
        max_attempts=max(8, int(settings.render_warmup_frames)),
    )
    world.pause()

    detector_config = app_config.perception.grounding.detector
    detector = GroundingDinoDetector(
        config=GroundingDinoConfig(
            repo_path=str(detector_config.repo_path),
            config_path=str(detector_config.config_path),
            checkpoint_path=str(detector_config.checkpoint_path),
            device=str(detector_config.device),
            box_threshold=float(detector_config.box_threshold),
            text_threshold=float(detector_config.text_threshold),
            text_encoder_path=str(detector_config.text_encoder_path),
            max_detections=int(detector_config.max_detections),
            nms_iou_threshold=float(detector_config.nms_iou_threshold),
        ),
        logger=logger,
    )
    detector.require_ready()
    vlm_client = VLMClient(
        api_key=app_config.llm.api_key,
        model_name=app_config.llm.model_name,
        base_url=app_config.llm.api_base_url,
        timeout_sec=app_config.llm.timeout_sec,
        retry_cooldown_sec=app_config.llm.retry_cooldown_sec,
        upload_max_edge_px=app_config.perception.grounding.semantic_verification.max_edge_px,
        image_detail=app_config.llm.camera_image_detail,
        logger=logger,
    )
    tracker = OfflineSemanticTracker(
        detector=detector,
        config=settings.perception,
        verifier=LowConfidenceVLMVerifier(client=vlm_client),
    )
    scheduler = PerceptionScheduler(settings.scheduler)
    mapper = GlobalOccupancyMap(config=settings.occupancy, logger=logger)
    projector = RGBDVirtualRayProjector(
        config=settings.rgbd,
        floor_z_m=float(episode.reference_path_xyz[0][2]),
    )
    frontier_extractor = FrontierExtractor(settings.frontier)
    frontier_tracker = FrontierTracker(match_radius_m=settings.frontier.persistent_match_radius_m)
    writer = ExpertReplayDatasetWriter(
        output_root=settings.output_root,
        episode=episode,
        overwrite=overwrite,
    )
    writer.write_json(
        writer.episode_dir / "route_alignment.json",
        {"alignments": [item.to_dict() for item in alignments]},
    )
    history_paths: deque[str] = deque(maxlen=max(1, int(settings.rgb_history_frames)))
    motion_history: deque[MotionHistoryToken] = deque(
        maxlen=max(1, int(settings.motion_history_frames))
    )
    trajectory: list[tuple[float, float]] = []
    reference_path_xy = np.asarray(episode.reference_path_xyz, dtype=np.float64)[:, :2]
    timeline_bounds = fixed_timeline_bounds(reference_path_xy=reference_path_xy)
    previous_rgb: np.ndarray | None = None
    previous_depth: np.ndarray | None = None
    previous_view_matrix: np.ndarray | None = None

    for pose in replay:
        pose_changed = (
            float(pose.delta_distance_m) > 1.0e-4
            or abs(float(pose.delta_yaw_rad)) > 1.0e-4
        )
        pose_writer.set(pose=pose)
        # Let the articulation hierarchy consume the root teleport first, then
        # place the capture camera explicitly from its calibrated root offset.
        _render_without_physics(simulation_app=simulation_app)
        pose_writer.verify(
            pose=pose,
            position_tolerance_m=settings.pose_position_tolerance_m,
            orientation_tolerance_deg=settings.pose_orientation_tolerance_deg,
        )
        camera_pose_writer.set(robot_state=_normalized_pose(robot.get_world_pose()))
        for _ in range(max(1, int(settings.render_frames_per_pose))):
            _render_without_physics(simulation_app=simulation_app)
        pose_writer.verify(
            pose=pose,
            position_tolerance_m=settings.pose_position_tolerance_m,
            orientation_tolerance_deg=settings.pose_orientation_tolerance_deg,
        )
        camera_pose_writer.verify(
            robot_state=_normalized_pose(robot.get_world_pose()),
            frame_index=pose.frame_index,
            position_tolerance_m=settings.pose_position_tolerance_m,
            orientation_tolerance_deg=settings.pose_orientation_tolerance_deg,
        )
        rgb, depth = _capture_rgbd(
            camera=camera,
            world=world,
            simulation_app=simulation_app,
            max_attempts=max(8, int(settings.render_frames_per_pose) + 4),
            previous_rgb=previous_rgb,
            previous_depth=previous_depth,
            require_changed=pose_changed,
        )
        previous_rgb = rgb
        previous_depth = depth
        intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64).reshape(3, 3)
        view_matrix = np.asarray(camera.get_view_matrix_ros(), dtype=np.float64).reshape(4, 4)
        if (
            pose_changed
            and previous_view_matrix is not None
            and np.allclose(view_matrix, previous_view_matrix, rtol=0.0, atol=1.0e-7)
        ):
            raise RuntimeError(
                "Expert replay camera view matrix remained fixed after a path motion: "
                f"frame={pose.frame_index} delta_distance_m={pose.delta_distance_m:.6f} "
                f"delta_yaw_rad={pose.delta_yaw_rad:.6f}"
            )
        previous_view_matrix = view_matrix.copy()
        robot_pose = Pose2D(x=pose.position_xyz[0], y=pose.position_xyz[1], yaw_rad=pose.yaw_rad)
        ray_frame = projector.project(
            depth_image=depth,
            camera_intrinsics=intrinsics,
            camera_view_matrix_ros=view_matrix,
        )
        mapper.update_from_world_rays(
            robot_pose=robot_pose,
            sensor_world_xy=ray_frame.sensor_world_xy,
            beam_endpoints_world_xy=ray_frame.endpoints_world_xy,
            beam_hit_mask=ray_frame.hit_mask,
            timestamp_sec=pose.timestamp_sec,
        )
        pose_xyyaw = (pose.position_xyz[0], pose.position_xyz[1], pose.yaw_rad)
        schedule = scheduler.decide(
            timestamp_sec=pose.timestamp_sec,
            pose_xyyaw=pose_xyyaw,
            query_labels=tuple(query.text for query in queries),
        )
        perception_reason = "scheduled_reuse"
        if schedule.should_run:
            landmarks, perception_reason = tracker.observe(
                rgb=rgb,
                depth=depth,
                camera_intrinsics=intrinsics,
                camera_view_matrix_ros=view_matrix,
                queries=queries,
                frame_index=pose.frame_index,
                timestamp_sec=pose.timestamp_sec,
            )
            scheduler.record_request(
                timestamp_sec=pose.timestamp_sec,
                pose_xyyaw=pose_xyyaw,
                query_labels=tuple(query.text for query in queries),
                had_detection=any(item.last_seen_frame_index == pose.frame_index for item in landmarks),
            )
        else:
            landmarks = tracker.snapshot(timestamp_sec=pose.timestamp_sec)
        trajectory.append((pose.position_xyz[0], pose.position_xyz[1]))
        snapshot = mapper.snapshot()
        frontiers = frontier_tracker.update(
            frontier_extractor.extract(
                occupied_mask=snapshot.occupied_mask,
                free_mask=snapshot.free_mask,
                inflated_occupied_mask=snapshot.inflated_occupied,
                resolution_m=snapshot.resolution_m,
                origin_x=snapshot.origin_x,
                origin_y=snapshot.origin_y,
                robot_pose=pose_xyyaw,
            )
        )
        oracle = label_frontiers_from_reference_route(
            frontiers=frontiers,
            occupied_mask=snapshot.occupied_mask,
            free_mask=snapshot.free_mask,
            resolution_m=snapshot.resolution_m,
            origin_x=snapshot.origin_x,
            origin_y=snapshot.origin_y,
            robot_xy=pose_xyyaw[:2],
            reference_path_xy=reference_path_xy,
            config=settings.frontier_oracle,
        )
        completed_prefix_weights, active_weights, active_alignment, completion = (
            progress_alignment_at_frame(
                alignments,
                frame_index=pose.frame_index,
                token_count=len(episode.instruction_words),
            )
        )
        future_route = tuple(
            (float(point[0]), float(point[1]))
            for point in reference_path_xy[oracle.nearest_reference_path_index :]
        )
        mode = _navigation_mode(
            default_mode=oracle.navigation_mode,
            active_alignment=active_alignment,
            landmarks=landmarks,
            frame_index=pose.frame_index,
            final_frame_index=len(replay) - 1,
        )
        relative_frame_dir = Path("frames") / f"{pose.frame_index:05d}"
        rgb_relative = str(relative_frame_dir / "rgb.png")
        depth_relative = str(relative_frame_dir / "depth.npz")
        calibration = CameraCalibration(
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            intrinsics=_matrix_tuple(intrinsics),
            camera_view_matrix_ros=_matrix_tuple(view_matrix),
        )
        replay_frame = ExpertReplayFrame(
            schema_version=EXPERT_REPLAY_SCHEMA_VERSION,
            episode_id=episode.episode_id,
            pose=pose,
            rgb_path=rgb_relative,
            depth_path=depth_relative,
            calibration=calibration,
        )
        timeline = render_timeline_frame(
            rgb=rgb,
            snapshot=snapshot,
            robot_pose_xyyaw=pose_xyyaw,
            trajectory_world_xy=tuple(trajectory),
            landmarks=landmarks,
            frontiers=oracle.frontier_tokens,
            instruction=episode.instruction,
            active_alignment=active_alignment,
            selected_frontier_id=oracle.selected_frontier_id,
            future_reference_path_xy=future_route,
            oracle_crossing_world_xy=oracle.first_unknown_world_xy,
            oracle_reason=oracle.reason,
            frame_index=pose.frame_index,
            display_bounds=timeline_bounds,
        )
        paths = writer.write_frame_assets(
            frame=replay_frame,
            rgb=rgb,
            depth=depth,
            snapshot=snapshot,
            robot_pose_xyyaw=pose_xyyaw,
            timeline_image=timeline,
            local_size_m=settings.local_map_size_m,
            local_cells=settings.local_map_cells,
            global_cells=settings.global_map_cells,
            occupied_threshold=settings.occupancy.occupied_threshold,
            free_threshold=settings.occupancy.free_threshold,
        )
        history_paths.append(paths["rgb"])
        motion_history.append(
            MotionHistoryToken(
                frame_index=int(pose.frame_index),
                motion_type=str(pose.motion_type),
                delta_distance_m=float(pose.delta_distance_m),
                delta_yaw_rad=float(pose.delta_yaw_rad),
            )
        )
        quality_flags = _quality_flags(
            active_alignment=active_alignment,
            oracle_reason=oracle.reason,
            perception_reason=perception_reason,
            ray_count=ray_frame.ray_count,
        )
        sample_id = (
            f"{episode.split}:{episode.scene_id}:{episode.episode_id}:{pose.frame_index:05d}"
        )
        goal_distance_m, goal_bearing_rad = _relative_goal_metrics(
            pose_xyyaw=pose_xyyaw,
            goal_xy=episode.goal_xyz[:2],
        )
        sample = TrainingSampleV3(
            schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
            sample_id=sample_id,
            progress=ProgressTrainingView(
                observation=ProgressObservation(
                    episode_id=episode.episode_id,
                    frame_index=pose.frame_index,
                    timestamp_sec=pose.timestamp_sec,
                    instruction=episode.instruction,
                    instruction_words=episode.instruction_words,
                    current_rgb_path=paths["rgb"],
                    rgb_history_paths=tuple(history_paths),
                    motion_history=tuple(motion_history),
                ),
                supervision=ProgressSupervision(
                    completed_prefix_weights=completed_prefix_weights,
                    active_span_weights=active_weights,
                    active_span_text="" if active_alignment is None else active_alignment.text,
                    active_event_type=(
                        "CONTINUE" if active_alignment is None else active_alignment.event_type
                    ),
                    active_span_completion_probability=float(completion),
                    path_progress=float(pose.path_progress),
                    alignment_confidence=(
                        0.0 if active_alignment is None else float(active_alignment.confidence)
                    ),
                    supervision_valid=bool(
                        active_alignment is not None
                        and float(active_alignment.confidence) >= 0.50
                    ),
                ),
            ),
            spatial_subgoal=SpatialSubgoalTrainingView(
                observation=SpatialSubgoalObservation(
                    episode_id=episode.episode_id,
                    frame_index=pose.frame_index,
                    timestamp_sec=pose.timestamp_sec,
                    instruction=episode.instruction,
                    instruction_words=episode.instruction_words,
                    current_rgb_path=paths["rgb"],
                    robot_pose_xyyaw=pose_xyyaw,
                    local_map_path=paths["local_map"],
                    global_map_path=paths["global_map"],
                    trajectory_world_xy=tuple(trajectory),
                    observed_landmarks=landmarks,
                    frontiers=oracle.frontier_tokens,
                    query_phrases=queries,
                ),
                supervision=SpatialSubgoalSupervision(
                    navigation_mode=mode,
                    frontier_ids=tuple(item.frontier_id for item in oracle.frontier_tokens),
                    frontier_value_targets=oracle.frontier_value_targets,
                    selected_frontier_id=oracle.selected_frontier_id,
                    goal_relative_distance_m=goal_distance_m,
                    goal_relative_bearing_rad=goal_bearing_rad,
                    remaining_route_distance_m=_remaining_route_distance(
                        reference_path_xy=reference_path_xy,
                        start_index=oracle.nearest_reference_path_index,
                    ),
                    oracle_confidence=float(oracle.confidence),
                    supervision_valid=bool(
                        oracle.selected_frontier_id is not None
                        and oracle.reason == "reliable_route_frontier_crossing"
                    ),
                ),
            ),
            debug_oracle=DebugOracle(
                future_reference_path_xy=future_route,
                nearest_reference_path_index=oracle.nearest_reference_path_index,
                route_alignment_path="route_alignment.json",
                active_span_id=None if active_alignment is None else active_alignment.span_id,
                oracle_reason=oracle.reason,
                oracle_crossing_world_xy=oracle.first_unknown_world_xy,
                oracle_confidence=oracle.confidence,
            ),
            quality_flags=quality_flags,
        )
        writer.write_sample(sample=sample, paths=paths)
        logger.info(
            "[ExpertReplayFrame] frame=%s/%s motion=%s perception=%s frontiers=%s selected=%s oracle=%s confidence=%.3f mode=%s flags=%s",
            pose.frame_index,
            len(replay) - 1,
            pose.motion_type,
            schedule.reason if schedule.should_run else "reuse",
            len(oracle.frontier_tokens),
            oracle.selected_frontier_id,
            oracle.reason,
            oracle.confidence,
            mode,
            list(quality_flags),
        )

    writer.write_final_observed_map(
        snapshot=mapper.snapshot(),
        output_cells=settings.global_map_cells,
    )
    manifest = writer.finalize()
    logger.info("[ExpertReplayComplete] manifest=%s timeline=%s", manifest, writer.episode_dir / "timeline.html")
    return 0


def _navigation_mode(
    *,
    default_mode: str,
    active_alignment,
    landmarks,
    frame_index: int,
    final_frame_index: int,
) -> str:
    if frame_index >= final_frame_index:
        return "ARRIVED"
    if active_alignment is not None:
        text = active_alignment.text.lower()
        if any(
            landmark.last_seen_frame_index == frame_index and landmark.label in text
            for landmark in landmarks
        ):
            return "APPROACH_OBSERVED_TARGET"
    return str(default_mode)


def _relative_goal_metrics(
    *,
    pose_xyyaw: tuple[float, float, float],
    goal_xy: tuple[float, float],
) -> tuple[float, float]:
    delta_x = float(goal_xy[0]) - float(pose_xyyaw[0])
    delta_y = float(goal_xy[1]) - float(pose_xyyaw[1])
    cosine = float(np.cos(float(pose_xyyaw[2])))
    sine = float(np.sin(float(pose_xyyaw[2])))
    forward = cosine * delta_x + sine * delta_y
    left = -sine * delta_x + cosine * delta_y
    return float(np.hypot(delta_x, delta_y)), float(np.arctan2(left, forward))


def _remaining_route_distance(
    *,
    reference_path_xy: np.ndarray,
    start_index: int,
) -> float:
    route = np.asarray(reference_path_xy, dtype=np.float64)
    start = max(0, min(int(start_index), max(0, route.shape[0] - 1)))
    remaining = route[start:, :2]
    if remaining.shape[0] <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(remaining, axis=0), axis=1)))


def _quality_flags(*, active_alignment, oracle_reason: str, perception_reason: str, ray_count: int) -> tuple[str, ...]:
    flags: list[str] = []
    if active_alignment is None or float(active_alignment.confidence) < 0.50:
        flags.append("weak_route_alignment")
    if oracle_reason in {
        "no_reachable_frontier",
        "no_route_frontier_crossing",
        "no_aligned_route_frontier",
        "ambiguous_route_frontier",
    }:
        flags.append(f"frontier_oracle:{oracle_reason}")
    if perception_reason not in {"ok", "scheduled_reuse", "grounding_dino_no_detection"}:
        flags.append(f"perception:{perception_reason}")
    if int(ray_count) <= 0:
        flags.append("no_valid_rgbd_rays")
    return tuple(flags)


class _ReplayPoseWriter:
    """Teleport the initialized articulation along a privileged replay path."""

    def __init__(self, *, robot, spawn_z_m: float) -> None:
        self._robot = robot
        self._spawn_z_m = float(spawn_z_m)

    def set_default(self, *, pose) -> None:
        position, orientation = self.state(pose=pose)
        self._robot.set_default_state(position=position, orientation=orientation)

    def set(self, *, pose) -> None:
        position, orientation = self.state(pose=pose)
        self._robot.set_world_pose(position=position, orientation=orientation)

    def verify(
        self,
        *,
        pose,
        position_tolerance_m: float,
        orientation_tolerance_deg: float,
    ) -> None:
        expected_position, expected_orientation = self.state(pose=pose)
        actual_position, actual_orientation = self._robot.get_world_pose()
        actual_position = np.asarray(actual_position, dtype=np.float64).reshape(3)
        actual_orientation = np.asarray(actual_orientation, dtype=np.float64).reshape(4)
        position_error = float(np.linalg.norm(actual_position - expected_position))
        orientation_error_deg = _quaternion_error_deg(
            actual_orientation,
            expected_orientation,
        )
        if (
            position_error > float(position_tolerance_m)
            or orientation_error_deg > float(orientation_tolerance_deg)
        ):
            raise RuntimeError(
                "Expert replay articulation pose did not match the requested path pose: "
                f"frame={pose.frame_index} position_error_m={position_error:.6f} "
                f"orientation_error_deg={orientation_error_deg:.4f} "
                f"position_tolerance_m={position_tolerance_m:.4f} "
                f"orientation_tolerance_deg={orientation_tolerance_deg:.2f}"
            )

    def state(self, *, pose) -> tuple[np.ndarray, np.ndarray]:
        yaw_half = 0.5 * float(pose.yaw_rad)
        return (
            np.asarray(
                [
                    float(pose.position_xyz[0]),
                    float(pose.position_xyz[1]),
                    float(pose.position_xyz[2]) + self._spawn_z_m,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)],
                dtype=np.float32,
            ),
        )


class _ReplayCameraPoseWriter:
    """Keep the capture camera rigidly attached to a teleported replay root."""

    def __init__(self, *, camera, robot) -> None:
        self._camera = camera
        robot_position, robot_orientation = _normalized_pose(robot.get_world_pose())
        camera_position, camera_orientation = _normalized_pose(
            camera.get_world_pose(camera_axes="world")
        )
        inverse_robot_orientation = _quaternion_conjugate_wxyz(robot_orientation)
        self.relative_position_m = _quaternion_rotate_wxyz(
            inverse_robot_orientation,
            camera_position - robot_position,
        )
        self.relative_orientation_wxyz = _quaternion_normalize_wxyz(
            _quaternion_multiply_wxyz(inverse_robot_orientation, camera_orientation)
        )

    def set(self, *, robot_state: tuple[np.ndarray, np.ndarray]) -> None:
        position, orientation = self._world_camera_pose(robot_state=robot_state)
        self._camera.set_world_pose(
            position=position,
            orientation=orientation,
            camera_axes="world",
        )

    def verify(
        self,
        *,
        robot_state: tuple[np.ndarray, np.ndarray],
        frame_index: int,
        position_tolerance_m: float,
        orientation_tolerance_deg: float,
    ) -> None:
        expected_position, expected_orientation = self._world_camera_pose(
            robot_state=robot_state
        )
        actual_position, actual_orientation = _normalized_pose(
            self._camera.get_world_pose(camera_axes="world")
        )
        position_error = float(np.linalg.norm(actual_position - expected_position))
        orientation_error_deg = _quaternion_error_deg(
            actual_orientation,
            expected_orientation,
        )
        if (
            position_error > float(position_tolerance_m)
            or orientation_error_deg > float(orientation_tolerance_deg)
        ):
            raise RuntimeError(
                "Expert replay camera pose did not follow the requested path pose: "
                f"frame={frame_index} position_error_m={position_error:.6f} "
                f"orientation_error_deg={orientation_error_deg:.4f} "
                f"position_tolerance_m={position_tolerance_m:.4f} "
                f"orientation_tolerance_deg={orientation_tolerance_deg:.2f}"
            )

    def _world_camera_pose(
        self,
        *,
        robot_state: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        robot_position, robot_orientation = _normalized_pose(robot_state)
        camera_position = robot_position + _quaternion_rotate_wxyz(
            robot_orientation,
            self.relative_position_m,
        )
        camera_orientation = _quaternion_normalize_wxyz(
            _quaternion_multiply_wxyz(
                robot_orientation,
                self.relative_orientation_wxyz,
            )
        )
        return camera_position.astype(np.float32), camera_orientation.astype(np.float32)


def _normalized_pose(pose) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = pose
    return (
        np.asarray(position, dtype=np.float64).reshape(3),
        _quaternion_normalize_wxyz(
            np.asarray(orientation, dtype=np.float64).reshape(4)
        ),
    )


def _quaternion_normalize_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("Replay pose contains a zero-length quaternion")
    return value / norm


def _quaternion_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _quaternion_normalize_wxyz(quaternion)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def _quaternion_error_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    alignment = abs(
        float(
            np.dot(
                _quaternion_normalize_wxyz(lhs),
                _quaternion_normalize_wxyz(rhs),
            )
        )
    )
    return float(np.degrees(2.0 * np.arccos(np.clip(alignment, 0.0, 1.0))))


def _quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(lhs, dtype=np.float64).reshape(4)
    rw, rx, ry, rz = np.asarray(rhs, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quaternion_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    orientation = _quaternion_normalize_wxyz(quaternion)
    vector_quaternion = np.asarray(
        [0.0, *np.asarray(vector, dtype=np.float64).reshape(3)],
        dtype=np.float64,
    )
    rotated = _quaternion_multiply_wxyz(
        _quaternion_multiply_wxyz(orientation, vector_quaternion),
        _quaternion_conjugate_wxyz(orientation),
    )
    return rotated[1:]


def _render_without_physics(*, simulation_app) -> None:
    # The timeline is paused, so this updates Kit and render products without
    # integrating another physics step. world.render() alone can leave camera
    # annotators stale when the robot is teleported between replay poses.
    simulation_app.update()


def _attach_depth_annotator(*, camera) -> None:
    method = getattr(camera, "add_distance_to_image_plane_to_frame", None)
    if not callable(method):
        raise RuntimeError("Isaac Sim Camera has no distance-to-image-plane annotator")
    method()


def _capture_rgbd(
    *,
    camera,
    world,
    simulation_app,
    max_attempts: int,
    previous_rgb: np.ndarray | None = None,
    previous_depth: np.ndarray | None = None,
    require_changed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    last_rgb_shape: tuple[int, ...] | None = None
    last_depth_shape: tuple[int, ...] | None = None
    available_keys: tuple[str, ...] = ()
    for _ in range(max(1, int(max_attempts))):
        frame = camera.get_current_frame()
        if isinstance(frame, dict):
            available_keys = tuple(sorted(str(key) for key in frame))
        else:
            frame = {}
        rgb, last_rgb_shape = _get_rgb(camera=camera, frame=frame)
        depth, last_depth_shape = _get_depth(camera=camera, frame=frame)
        if rgb is not None and depth is not None:
            changed = _rgbd_changed(
                rgb=rgb,
                depth=depth,
                previous_rgb=previous_rgb,
                previous_depth=previous_depth,
            )
            if not require_changed or changed:
                return rgb, depth
        _render_without_physics(simulation_app=simulation_app)
    raise RuntimeError(
        "Expert replay RGB-D stream remained invalid or stale after rendered retries: "
        f"rgb_shape={last_rgb_shape} depth_shape={last_depth_shape} "
        f"require_changed={bool(require_changed)} "
        f"available_frame_keys={list(available_keys)}"
    )


def _rgbd_changed(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    previous_rgb: np.ndarray | None,
    previous_depth: np.ndarray | None,
) -> bool:
    if previous_rgb is None or previous_depth is None:
        return True
    rgb_changed = rgb.shape != previous_rgb.shape or not np.array_equal(rgb, previous_rgb)
    depth_changed = depth.shape != previous_depth.shape or not np.array_equal(
        depth,
        previous_depth,
        equal_nan=True,
    )
    return bool(rgb_changed or depth_changed)


def _get_rgb(*, camera, frame: dict) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
    # get_current_frame() is simulation-time gated and remains cached while the
    # replay timeline is paused. The annotator itself contains the latest render.
    values = []
    try:
        values.append(camera.get_rgba())
    except Exception:
        values.append(None)
    values.append(frame.get("rgb"))
    last_shape: tuple[int, ...] | None = None
    for value in values:
        if value is None:
            continue
        image = _to_numpy(value)
        last_shape = tuple(int(size) for size in image.shape)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] < 3 or image.size == 0:
            continue
        image = image[..., :3]
        if np.issubdtype(image.dtype, np.floating):
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if image.size and float(np.max(image)) <= 1.0:
                image *= 255.0
        return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8)), last_shape
    return None, last_shape


def _get_depth(*, camera, frame: dict) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
    values = []
    try:
        values.append(camera.get_depth())
    except Exception:
        values.append(None)
    values.append(frame.get("distance_to_image_plane"))
    last_shape: tuple[int, ...] | None = None
    for value in values:
        if value is None:
            continue
        depth = np.array(_to_numpy(value), dtype=np.float32, copy=True)
        last_shape = tuple(int(size) for size in depth.shape)
        if depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or depth.size == 0:
            continue
        depth[~np.isfinite(depth)] = np.nan
        return depth, last_shape
    return None, last_shape


def _to_numpy(value) -> np.ndarray:
    if value is None:
        raise RuntimeError("Isaac camera returned no data")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _matrix_tuple(matrix: np.ndarray):
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=np.float64))


def _simulation_app_settings(config) -> dict:
    settings = {
        "headless": bool(config.simulation.headless),
        "width": int(config.simulation.width),
        "height": int(config.simulation.height),
        "multi_gpu": bool(config.simulation.multi_gpu),
        "extra_args": [
            "--/app/hydraEngine/waitIdle=1",
            "--/app/updateOrder/checkForHydraRenderComplete=1000",
        ],
    }
    if config.simulation.active_gpu is not None:
        settings["active_gpu"] = int(config.simulation.active_gpu)
    if config.simulation.physics_gpu is not None:
        settings["physics_gpu"] = int(config.simulation.physics_gpu)
    return settings


def _load_collection_settings(*, path: Path, output_override: str) -> CollectionSettings:
    if not path.is_file():
        raise FileNotFoundError(f"Expert replay config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expert replay config must be a YAML object: {path}")
    data = _mapping(payload, "data")
    capture = _mapping(payload, "capture")
    lighting = _construct(CollectionLightingConfig, _mapping(payload, "lighting"), "lighting")
    replay = _construct(ExpertReplayConfig, _mapping(payload, "replay"), "replay")
    occupancy = _construct(OccupancyGridConfig, _mapping(payload, "occupancy"), "occupancy")
    rgbd = _construct(RGBDGeometryMapConfig, _mapping(payload, "rgbd"), "rgbd")
    frontier = _construct(FrontierExtractorConfig, _mapping(payload, "frontier"), "frontier")
    frontier_oracle = _construct(
        FrontierOracleConfig,
        _mapping(payload, "frontier_oracle"),
        "frontier_oracle",
    )
    scheduler = _construct(PerceptionSchedulerConfig, _mapping(payload, "scheduler"), "scheduler")
    perception = _construct(OfflinePerceptionConfig, _mapping(payload, "perception"), "perception")
    output_value = str(output_override).strip() or str(
        data.get("output_root", "/Data/qiang/data/CAPTAINav/expert_replay")
    )
    return CollectionSettings(
        data_root=_project_path(str(data.get("interiornav_root", "/Data/qiang/interiornav_data"))),
        output_root=_project_path(output_value),
        render_warmup_frames=int(capture.get("render_warmup_frames", 8)),
        render_frames_per_pose=int(capture.get("render_frames_per_pose", 2)),
        pose_position_tolerance_m=float(capture.get("pose_position_tolerance_m", 0.02)),
        pose_orientation_tolerance_deg=float(
            capture.get("pose_orientation_tolerance_deg", 2.0)
        ),
        rgb_history_frames=max(1, int(capture.get("rgb_history_frames", 8))),
        motion_history_frames=max(1, int(capture.get("motion_history_frames", 32))),
        local_map_size_m=float(capture.get("local_map_size_m", 12.0)),
        local_map_cells=int(capture.get("local_map_cells", 128)),
        global_map_cells=int(capture.get("global_map_cells", 320)),
        lighting=lighting,
        replay=replay,
        occupancy=occupancy,
        rgbd=rgbd,
        frontier=frontier,
        frontier_oracle=frontier_oracle,
        scheduler=scheduler,
        perception=perception,
    )


def _construct(cls, values: dict, section: str):
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown expert replay fields under {section}: {unknown}")
    return cls(**values)


def _mapping(payload: dict, name: str) -> dict:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expert replay config section {name} must be a mapping")
    return value


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("captainav.expert_replay")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
