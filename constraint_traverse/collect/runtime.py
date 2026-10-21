from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time

import numpy as np
import yaml

from vln_nav.active_perception.bev_goal_map import render_bev_goal_map
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap, LocalVoxelMapConfig
from vln_nav.control.fine_path_tracker import fit_agx_command
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import AgxVelocityAdapter, VelocityCommand
from vln_nav.runtime.agx_geometry import validate_agx_geometry
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.isaac_runtime_utils import (
    bind_viewport_to_camera,
    get_joint_names,
    get_world_pose,
    resolve_existing_prim_path,
    resolve_robot_relative_prim_path,
    set_camera_horizontal_fov,
    set_local_xform,
    wait_for_stage_assets,
)
from vln_nav.runtime.scene_lighting import add_indoor_lighting

from .bev import BevConfig, local_voxel_config_from_mapping, snapshot_to_robot_bev
from .episode import EpisodeMode, EpisodeSession
from .recorder import ControlRecord, EpisodeRecorder, SampleRecord
from .sampling import PhysicsHistory, PhysicsState, capture_aligned_frame, pair_aligned_frames
from .teleop import ContinuousTeleop, TeleopClock, TeleopConfig
from .transforms import (
    BodyMotionEstimator,
    pose_xyyaw,
    quaternion_wxyz_from_yaw,
    relative_goal_xy,
)
from .ui import CollectorPanel, PanelStatus


@dataclass(frozen=True)
class RuntimeOverrides:
    usd_path: Path
    output_root: Path


def _load_navigation_mapping(
    project_root: Path,
) -> tuple[dict, dict, dict, LocalVoxelMapConfig]:
    config_path = Path(project_root) / "configs/navigation.yaml"
    with config_path.open(encoding="utf-8") as handle:
        navigation = yaml.safe_load(handle) or {}
    mapping = navigation.get("mapping")
    map_selector = navigation.get("map_selector")
    navigation_camera = navigation.get("camera")
    if not isinstance(mapping, dict) or set(mapping) != {
        "camera",
        "map",
        "max_frame_skew_sec",
    }:
        raise ValueError("主导航 mapping 配置字段不完整")
    if not isinstance(map_selector, dict) or set(map_selector) != {
        "sources",
        "image_size_px",
        "view_size_m",
        "rear_view_m",
        "render_period_sec",
        "obstacle_min_relative_z_m",
        "obstacle_max_relative_z_m",
    }:
        raise ValueError("主导航 map_selector 配置字段不完整")
    camera = mapping["camera"]
    if not isinstance(camera, dict) or set(camera) != {
        "width",
        "height",
        "frequency_hz",
    }:
        raise ValueError("主导航 mapping.camera 配置字段不完整")
    if any(int(camera[name]) <= 0 for name in camera):
        raise ValueError("主导航 mapping.camera 配置必须为正整数")
    if (
        not isinstance(navigation_camera, dict)
        or "horizontal_fov_deg" not in navigation_camera
        or not 0.0 < float(navigation_camera["horizontal_fov_deg"]) < 180.0
    ):
        raise ValueError("主导航前视相机水平视角配置无效")
    map_values = dict(mapping["map"])
    integration_period = float(map_values.pop("integration_period_sec"))
    if integration_period <= 0 or float(mapping["max_frame_skew_sec"]) <= 0:
        raise ValueError("主导航地图周期和帧同步阈值必须为正数")
    map_config = local_voxel_config_from_mapping(map_values)
    return mapping, map_selector, navigation_camera, map_config


class ConstraintTraverseCollector:
    def __init__(
        self,
        *,
        app,
        project_root: Path,
        config: dict,
        overrides: RuntimeOverrides,
        logger: logging.Logger,
    ) -> None:
        self.app = app
        self.project_root = Path(project_root)
        self.config = config
        self.overrides = overrides
        self.logger = logger

    def run(self) -> int:
        import omni.ui as ui
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
            is_stage_loading,
        )
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        profile = load_agx_profile(project_root=self.project_root)
        scene = self.config["scene"]
        cameras_cfg = self.config["cameras"]
        mapping_cfg, map_selector_cfg, navigation_camera_cfg, map_cfg = (
            _load_navigation_mapping(self.project_root)
        )
        map_camera_cfg = mapping_cfg["camera"]
        map_runtime_cfg = mapping_cfg["map"]
        bev_cfg = BevConfig.from_mapping(self.config["bev"])
        teleop_cfg = TeleopConfig.from_mapping(self.config["teleop"])
        collection = self.config["collection"]

        dt = float(profile.simulation.physics_dt)
        world = World(
            physics_dt=dt,
            rendering_dt=float(profile.simulation.rendering_dt),
            stage_units_in_meters=1.0,
        )
        dt = float(world.get_physics_dt())
        stage = get_current_stage()
        scene_path = self.overrides.usd_path.expanduser().resolve()
        if not scene_path.is_file():
            raise FileNotFoundError(f"Scene USD does not exist: {scene_path}")

        add_reference_to_stage(str(scene_path), str(scene["prim_path"]))
        if bool(scene["add_default_ground_plane"]):
            world.scene.add_default_ground_plane()
        add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
        wait_for_stage_assets(
            simulation_app=self.app,
            is_stage_loading=is_stage_loading,
            timeout_sec=float(scene["stage_load_timeout_sec"]),
            logger=self.logger,
            label="constraint traversal collection scene",
        )
        if bool(scene["add_indoor_lighting"]):
            try:
                add_indoor_lighting(
                    stage=stage,
                    environment_prim_path=str(scene["prim_path"]),
                    logger=self.logger,
                )
            except Exception:
                self.logger.exception("[CollectorLighting] failed; continuing")

        spawn = self.config["spawn"]
        scene_bounds = _scene_world_bounds(stage, str(scene["prim_path"]))
        holding_xy = 0.5 * (scene_bounds[0][:2] + scene_bounds[1][:2])
        holding_z = float(profile.spawn_z_m)
        set_local_xform(
            stage=stage,
            prim_path=profile.asset.prim_path,
            translation=[float(holding_xy[0]), float(holding_xy[1]), holding_z],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        articulation_path = resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=profile.asset.articulation_path,
        )
        validate_agx_geometry(stage, profile, articulation_path)
        robot = world.scene.add(
            SingleArticulation(
                prim_path=articulation_path,
                name="constraint_traverse_agx",
                reset_xform_properties=False,
            )
        )

        front_path = resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=profile.sensors.camera.front_prim_path,
            ),
            kind="constraint-traverse front camera",
        )
        down_path = resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=profile.sensors.camera.mapping_down_prim_path,
            ),
            kind="constraint-traverse down camera",
        )
        observer_path = resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=profile.sensors.observer_camera_prim_path,
            ),
            kind="constraint-traverse observer camera",
        )
        for name, path in (("front", front_path), ("down", down_path), ("observer", observer_path)):
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
                raise RuntimeError(f"{name} prim is not a Camera: {path}")

        focal_mm = set_camera_horizontal_fov(
            stage=stage,
            prim_path=front_path,
            fov_deg=float(navigation_camera_cfg["horizontal_fov_deg"]),
        )
        self.logger.info(
            "[FrontCamera] horizontal_fov=%.1fdeg focal_length=%.3fmm path=%s",
            float(navigation_camera_cfg["horizontal_fov_deg"]),
            focal_mm,
            front_path,
        )

        resolution = (
            int(map_camera_cfg["width"]),
            int(map_camera_cfg["height"]),
        )
        frequency = int(map_camera_cfg["frequency_hz"])
        front_camera = Camera(
            prim_path=front_path,
            name="constraint_traverse_front",
            frequency=frequency,
            resolution=resolution,
        )
        down_camera = Camera(
            prim_path=down_path,
            name="constraint_traverse_down",
            frequency=frequency,
            resolution=resolution,
        )

        world.reset()
        robot.initialize()
        world.pause()
        for camera in (front_camera, down_camera):
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()

        holding_position = np.asarray(
            [float(holding_xy[0]), float(holding_xy[1]), holding_z],
            dtype=np.float32,
        )
        holding_orientation = quaternion_wxyz_from_yaw(0.0)
        self.logger.info(
            "[CollectorPlacement] 初始车体中心位置=%s",
            holding_position.tolist(),
        )
        for _ in range(int(spawn["query_warmup_frames"])):
            robot.set_world_pose(
                position=holding_position,
                orientation=holding_orientation,
            )
            if hasattr(robot, "set_linear_velocity"):
                robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            if hasattr(robot, "set_angular_velocity"):
                robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            world.step(render=True)

        bind_viewport_to_camera(
            camera_prim_path=observer_path,
            logger=self.logger,
            enabled=False,
        )

        drive = create_locomotion_controller(
            profile=profile,
            robot=robot,
            stage=stage,
            articulation_path=articulation_path,
            available_joint_names=get_joint_names(robot),
            logger=self.logger,
            steering_centering_max_effort_nm=float(collection["steering_centering_max_effort_nm"]),
        )
        adapter = AgxVelocityAdapter(drive, profile.agx_controller.continuous)
        teleop = ContinuousTeleop(teleop_cfg)
        teleop.bind_keyboard()
        teleop.set_enabled(False)

        command_queue: list[object] = []
        panel = CollectorPanel(
            ui,
            image_size_px=int(map_selector_cfg["image_size_px"]),
            command_sink=command_queue.append,
        )

        for _ in range(int(cameras_cfg["warmup_render_frames"])):
            robot.set_world_pose(
                position=holding_position,
                orientation=holding_orientation,
            )
            _zero_robot_velocity(robot)
            adapter.stop()
            world.step(render=True)

        position, orientation = get_world_pose(robot)
        episode_map = LocalVoxelMap(map_cfg)
        episode_map.reset(np.asarray(position[:3], dtype=np.float64))
        latest_snapshot = episode_map.snapshot()
        motion = BodyMotionEstimator()
        session = EpisodeSession()
        recorder = EpisodeRecorder(
            output_root=self.overrides.output_root,
            scene_path=scene_path,
            jpeg_quality=int(collection["jpeg_quality"]),
            png_compression=int(collection["png_compression"]),
        )

        wheel_names = tuple(wheel.drive_joint for wheel in profile.agx_controller.continuous.wheels)
        steering_names = tuple(
            wheel.steering_joint for wheel in profile.agx_controller.continuous.wheels
        )
        last_front_timestamp = -float("inf")
        last_down_timestamp = -float("inf")
        last_bev = np.zeros((3, bev_cfg.height, bev_cfg.width), dtype=np.uint8)
        last_bev[2] = 255
        last_ui_update = -float("inf")
        current_episode_id = ""
        render_stride = max(
            1,
            round(float(profile.simulation.rendering_dt) / dt),
        )
        frame_index = 0
        mapping_state = {"last_integration_time": -float("inf")}
        last_map_render_time = -float("inf")
        last_control_limit_log_time = -float("inf")
        history = PhysicsHistory()
        desired = VelocityCommand()
        applied = VelocityCommand()
        control_clock = TeleopClock(1.0 / frequency, float(world.current_time))

        self.logger.info(
            "[Collector] scene=%s output=%s placement=manual map_config=%s",
            scene_path,
            self.overrides.output_root,
            self.project_root / "configs/navigation.yaml",
        )
        try:
            while self.app.is_running():
                now = float(world.current_time)
                if world.is_playing():
                    position, orientation = get_world_pose(robot)
                else:
                    position, orientation = _world_pose_from_prim(
                        stage,
                        articulation_path,
                    )
                pose = pose_xyyaw(position, orientation)
                if world.is_playing():
                    measured_v, measured_w = motion.update(pose, now)
                else:
                    measured_v, measured_w = 0.0, 0.0

                while command_queue:
                    command = command_queue.pop(0)
                    effect, current_episode_id = self._handle_ui_command(
                        command=command,
                        session=session,
                        recorder=recorder,
                        episode_map=episode_map,
                        teleop=teleop,
                        adapter=adapter,
                        robot=robot,
                        pose=pose,
                        position=position,
                        now=now,
                        measured_v=measured_v,
                        measured_w=measured_w,
                        motion=motion,
                        current_episode_id=current_episode_id,
                        scene_path=scene_path,
                        profile=profile,
                        world=world,
                        stage=stage,
                        articulation_path=articulation_path,
                        mapping_state=mapping_state,
                        mapping_cfg=mapping_cfg,
                        front_path=front_path,
                        down_path=down_path,
                        last_bev=last_bev,
                    )
                    if effect:
                        desired = VelocityCommand()
                        applied = VelocityCommand()
                        control_clock.reset(float(world.current_time))
                    if effect == "RESET_HISTORY":
                        history.clear()
                        measured_v, measured_w = 0.0, 0.0
                    if world.is_playing():
                        position, orientation = get_world_pose(robot)
                    else:
                        position, orientation = _world_pose_from_prim(
                            stage,
                            articulation_path,
                        )
                    pose = pose_xyyaw(position, orientation)
                    now = float(world.current_time)
                    if effect == "RESET_HISTORY":
                        motion.update(pose, now)
                    latest_snapshot = episode_map.snapshot()

                teleop.set_enabled(world.is_playing() and session.robot_placed and not session.placing)
                ui_now = time.monotonic()
                if not world.is_playing():
                    adapter.stop()
                    teleop.reset()
                    if ui_now - last_ui_update >= float(
                        self.config["ui"]["update_period_sec"]
                    ):
                        panel.update_status(
                            PanelStatus(
                                mode=session.mode.value,
                                episode_id=current_episode_id,
                                sample_count=session.sample_count,
                                elapsed_sec=session.elapsed(now),
                                robot_placed=session.robot_placed,
                                goal_selected=session.goal_world_xy is not None,
                                can_reset_start=(
                                    session.last_start_pose_world is not None
                                ),
                            )
                        )
                        last_ui_update = ui_now
                    self.app.update()
                    continue

                readback = drive.readback()
                current_state = PhysicsState(
                    timestamp_sec=now,
                    physics_step=int(world.current_time_step_index),
                    position=np.asarray(position).copy(),
                    orientation=np.asarray(orientation).copy(),
                    pose=pose.copy(),
                    velocity=np.asarray([measured_v, measured_w], dtype=np.float64),
                    steering_position=_ordered_readback(readback["steering_position"], steering_names),
                    steering_velocity=_ordered_readback(readback["steering_velocity"], steering_names),
                    wheel_velocity=_ordered_readback(readback["wheel_velocity"], wheel_names),
                    front_world_from_camera=np.linalg.inv(np.asarray(front_camera.get_view_matrix_ros(), dtype=np.float64)),
                    down_world_from_camera=np.linalg.inv(np.asarray(down_camera.get_view_matrix_ros(), dtype=np.float64)),
                )
                history.append(current_state)
                front_match = capture_aligned_frame(
                    source="front",
                    camera=front_camera,
                    history=history,
                    after_sec=last_front_timestamp,
                )
                down_match = capture_aligned_frame(
                    source="down",
                    camera=down_camera,
                    history=history,
                    after_sec=last_down_timestamp,
                )
                pair = pair_aligned_frames(
                    front_match, down_match,
                    max_skew_sec=float(mapping_cfg["max_frame_skew_sec"]),
                    not_before_sec=(session.started_at_sec if session.recording else -float("inf")),
                )
                new_pair = pair is not None
                front = None if pair is None else pair.front
                down = None if pair is None else pair.down
                desired = control_clock.step(
                    teleop, now, stopped=session.success_requested_at_sec is not None,
                )
                try:
                    applied, command_limited = _apply_velocity_command(adapter, desired)
                    if command_limited and now - last_control_limit_log_time >= 1.0:
                        self.logger.info(
                            "[CollectorControl] desired=[%.3f,%.3f] applied=[%.3f,%.3f]",
                            desired.linear_mps, desired.angular_radps,
                            applied.linear_mps, applied.angular_radps,
                        )
                        last_control_limit_log_time = now
                except Exception as exc:
                    self.logger.error("[CollectorControl] command rejected: %s", exc)
                    teleop.reset()
                    adapter.stop()
                    desired = VelocityCommand()
                    applied = VelocityCommand()
                if session.recording:
                    recorder.append_control(ControlRecord(
                        timestamp_sec=now,
                        physics_step=current_state.physics_step,
                        pose_world_xyyaw=current_state.pose,
                        world_position_xyz=current_state.position,
                        world_orientation_wxyz=current_state.orientation,
                        measured_velocity_vw=current_state.velocity,
                        human_action_vw=np.asarray([desired.linear_mps, desired.angular_radps]),
                        applied_action_vw=np.asarray([applied.linear_mps, applied.angular_radps]),
                        steering_position=current_state.steering_position,
                        steering_velocity=current_state.steering_velocity,
                        wheel_velocity=current_state.wheel_velocity,
                    ))
                map_active = session.robot_placed and session.mode in {
                    EpisodeMode.IDLE,
                    EpisodeMode.RECORDING,
                }
                if new_pair:
                    observation_state = pair.state
                    observation_time = max(front.timestamp_sec, down.timestamp_sec)
                    skew = abs(float(front.timestamp_sec - down.timestamp_sec))
                    if skew <= float(mapping_cfg["max_frame_skew_sec"]):
                        last_front_timestamp = float(front.timestamp_sec)
                        last_down_timestamp = float(down.timestamp_sec)
                        if map_active:
                            map_integrated = False
                            integration_period = float(
                                map_runtime_cfg["integration_period_sec"]
                            )
                            if (
                                observation_time - mapping_state["last_integration_time"]
                                >= integration_period
                            ):
                                episode_map.integrate_batch(
                                    [
                                        TimedDepthObservation(
                                            source="front",
                                            timestamp_sec=front.timestamp_sec,
                                            depth_m=front.depth_m,
                                            intrinsics=front.intrinsics,
                                            camera_to_world=front.world_from_camera,
                                        ),
                                        TimedDepthObservation(
                                            source="down",
                                            timestamp_sec=down.timestamp_sec,
                                            depth_m=down.depth_m,
                                            intrinsics=down.intrinsics,
                                            camera_to_world=down.world_from_camera,
                                        ),
                                    ],
                                    center_world_xyz=np.asarray(
                                        observation_state.position[:3], dtype=np.float64
                                    ),
                                )
                                mapping_state["last_integration_time"] = observation_time
                                map_integrated = True
                            snapshot = episode_map.snapshot()
                            latest_snapshot = snapshot
                            last_bev = snapshot_to_robot_bev(
                                snapshot,
                                robot_world_xyyaw=observation_state.pose,
                                floor_z_m=float(observation_state.position[2]) - float(profile.spawn_z_m),
                                config=bev_cfg,
                            )
                            if session.recording:
                                sample = SampleRecord(
                                    sample_time_sec=max(
                                        float(front.timestamp_sec),
                                        float(down.timestamp_sec),
                                    ),
                                    front_timestamp_sec=float(front.timestamp_sec),
                                    down_timestamp_sec=float(down.timestamp_sec),
                                    front_rgb=front.rgb,
                                    front_depth_m=front.depth_m,
                                    down_depth_m=down.depth_m,
                                    bev=last_bev,
                                    pose_world_xyyaw=observation_state.pose,
                                    world_position_xyz=observation_state.position,
                                    world_orientation_wxyz=observation_state.orientation,
                                    measured_velocity_vw=observation_state.velocity,
                                    human_action_vw=np.asarray(
                                        [desired.linear_mps, desired.angular_radps],
                                        dtype=np.float32,
                                    ),
                                    applied_action_vw=np.asarray(
                                        [applied.linear_mps, applied.angular_radps],
                                        dtype=np.float32,
                                    ),
                                    steering_position=observation_state.steering_position,
                                    steering_velocity=observation_state.steering_velocity,
                                    wheel_velocity=observation_state.wheel_velocity,
                                    front_intrinsics=front.intrinsics,
                                    down_intrinsics=down.intrinsics,
                                    front_base_from_camera=front.base_from_camera,
                                    down_base_from_camera=down.base_from_camera,
                                    front_world_from_camera=front.world_from_camera,
                                    down_world_from_camera=down.world_from_camera,
                                    map_version=int(snapshot.version),
                                    state_timestamp_sec=observation_state.timestamp_sec,
                                    action_timestamp_sec=now,
                                    physics_step=observation_state.physics_step,
                                    action_physics_step=current_state.physics_step,
                                    front_physics_step=front.frame_index,
                                    down_physics_step=down.frame_index,
                                    map_integrated=map_integrated,
                                    floor_z_m=float(observation_state.position[2]) - float(profile.spawn_z_m),
                                )
                                recorder.append(sample)
                                session.note_sample(observation_state.pose)
                                path = _complete_requested_episode(session, recorder, sample, collection)
                                if path is not None:
                                    current_episode_id = ""
                                    self.logger.info("[Collector] human success saved path=%s", path)
                    elif map_active:
                        self.logger.warning(
                            "[CollectorMap] skipped RGB-D pair skew=%.3fs limit=%.3fs",
                            skew,
                            float(mapping_cfg["max_frame_skew_sec"]),
                        )

                if ui_now - last_ui_update >= float(self.config["ui"]["update_period_sec"]):
                    panel.update_status(
                        PanelStatus(
                            mode=session.mode.value,
                            episode_id=current_episode_id,
                            sample_count=session.sample_count,
                            elapsed_sec=session.elapsed(now),
                            robot_placed=session.robot_placed,
                            goal_selected=session.goal_world_xy is not None,
                            can_reset_start=session.last_start_pose_world is not None,
                        )
                    )
                    last_ui_update = ui_now

                if (
                    session.robot_placed
                    and now - last_map_render_time
                    >= float(map_selector_cfg["render_period_sec"])
                ):
                    goal_local_xy = None
                    if session.goal_world_xy is not None:
                        goal_local_xy = relative_goal_xy(
                            pose,
                            session.goal_world_xy,
                        )
                    panel.update_map(render_bev_goal_map(
                        latest_snapshot,
                        robot_world_xyz=position[:3],
                        robot_yaw_rad=float(pose[2]),
                        view_size_m=float(map_selector_cfg["view_size_m"]),
                        rear_view_m=float(map_selector_cfg["rear_view_m"]),
                        image_size_px=int(map_selector_cfg["image_size_px"]),
                        obstacle_min_relative_z_m=float(
                            map_selector_cfg["obstacle_min_relative_z_m"]
                        ),
                        obstacle_max_relative_z_m=float(
                            map_selector_cfg["obstacle_max_relative_z_m"]
                        ),
                        goal_local_xy=goal_local_xy,
                    ), robot_world_xyyaw=pose)
                    last_map_render_time = now

                max_duration = float(collection["max_episode_duration_sec"])
                if session.recording and session.elapsed(now) > max_duration:
                    adapter.stop()
                    teleop.reset()
                    recorder.interrupt("timeout")
                    session.abort()
                    self.logger.info("[Collector] timed-out episode preserved as incomplete")
                    current_episode_id = ""

                _step_physics(world, render=(frame_index % render_stride == 0))
                frame_index += 1
        finally:
            try:
                adapter.stop()
            except Exception:
                pass
            try:
                recorder.close()
            finally:
                teleop.shutdown()
                panel.shutdown()
        return 0

    def _handle_ui_command(
        self,
        *,
        command: object,
        session: EpisodeSession,
        recorder: EpisodeRecorder,
        episode_map: LocalVoxelMap,
        teleop: ContinuousTeleop,
        adapter: AgxVelocityAdapter,
        robot,
        pose: np.ndarray,
        position: np.ndarray,
        now: float,
        measured_v: float,
        measured_w: float,
        motion: BodyMotionEstimator,
        current_episode_id: str,
        scene_path: Path,
        profile,
        world,
        stage,
        articulation_path: str,
        mapping_state: dict[str, float],
        mapping_cfg: dict,
        front_path: str,
        down_path: str,
        last_bev: np.ndarray,
    ) -> tuple[str, str]:
        collection = self.config["collection"]

        if command == "BORN":
            if session.mode is EpisodeMode.IDLE:
                teleop.reset()
                adapter.stop()
                session.begin_born_placement()
                _select_robot_for_placement(articulation_path)
                world.pause()
                self.logger.info("[CollectorPlacement] robot placement started")
                return "RESET_CONTROL", current_episode_id
            if session.mode is EpisodeMode.PLACING_BORN:
                placed_position, placed_orientation = _world_pose_from_prim(
                    stage,
                    articulation_path,
                )
                robot.set_world_pose(
                    position=placed_position,
                    orientation=placed_orientation,
                )
                _zero_robot_velocity(robot)
                session.confirm_born_placement()
                _clear_selection()
                motion.reset()
                world.play()
                for _ in range(int(self.config["spawn"]["settle_frames"])):
                    adapter.stop()
                    world.step(render=True)
                settled_position, _ = get_world_pose(robot)
                _zero_robot_velocity(robot)
                episode_map.reset(
                    np.asarray(settled_position[:3], dtype=np.float64)
                )
                _clear_bev(last_bev)
                mapping_state["last_integration_time"] = -float("inf")
                self.logger.info(
                    "[CollectorPlacement] robot placement confirmed position=%s",
                    np.asarray(settled_position, dtype=np.float64).tolist(),
                )
                return "RESET_HISTORY", current_episode_id
            return "", current_episode_id

        if (
            isinstance(command, tuple)
            and len(command) == 3
            and command[0] == "SET_HINT_WORLD"
        ):
            if session.mode is not EpisodeMode.IDLE or not session.robot_placed:
                return "", current_episode_id
            goal_world_xy = np.asarray(command[1:3], dtype=np.float64)
            session.set_goal(goal_world_xy)
            self.logger.info(
                "[CollectorHint] map hint selected xy=%s",
                goal_world_xy.tolist(),
            )
            return "", current_episode_id

        if command == "RESET_MAP":
            if session.mode is not EpisodeMode.IDLE or not session.robot_placed:
                return "", current_episode_id
            episode_map.reset(np.asarray(position[:3], dtype=np.float64))
            _clear_bev(last_bev)
            mapping_state["last_integration_time"] = -float("inf")
            self.logger.info("[CollectorMap] map reset")
            return "", current_episode_id

        if command == "START":
            if session.mode is not EpisodeMode.IDLE:
                return "", current_episode_id
            if not session.robot_placed:
                return "", current_episode_id
            if (
                abs(float(measured_v)) > float(collection["start_max_linear_mps"])
                or abs(float(measured_w)) > float(collection["start_max_angular_radps"])
            ):
                return "", current_episode_id
            teleop.reset()
            adapter.stop()
            session.begin(
                start_pose_world=pose,
                start_position_world_xyz=position,
                timestamp_sec=now,
            )
            temp_dir = recorder.begin(
                start_pose_world=pose,
                start_position_world_xyz=position,
                start_time_sec=now,
                initial_map=episode_map.snapshot(),
                extra_meta={
                    "physics_dt_sec": float(world.get_physics_dt()),
                    "policy_period_sec": 1.0 / int(mapping_cfg["camera"]["frequency_hz"]),
                    "controller": asdict(profile.agx_controller.continuous),
                    "teleop": dict(self.config["teleop"]),
                    "cameras": {
                        "front_prim_path": front_path,
                        "down_prim_path": down_path,
                        "width": int(mapping_cfg["camera"]["width"]),
                        "height": int(mapping_cfg["camera"]["height"]),
                        "frequency_hz": int(mapping_cfg["camera"]["frequency_hz"]),
                        "max_frame_skew_sec": float(
                            mapping_cfg["max_frame_skew_sec"]
                        ),
                    },
                    "bev": dict(self.config["bev"]),
                    "voxel_map": {
                        name: value
                        for name, value in mapping_cfg["map"].items()
                        if name != "integration_period_sec"
                    },
                    "map_integration_period_sec": float(
                        mapping_cfg["map"]["integration_period_sec"]
                    ),
                    "collection": dict(self.config["collection"]),
                    "applied_action_limits": {
                        "max_linear_mps": float(
                            profile.agx_controller.continuous.max_linear_mps
                        ),
                        "max_angular_radps": float(
                            profile.agx_controller.continuous.max_angular_radps
                        ),
                    },
                    "wheel_joint_order": [
                        wheel.drive_joint
                        for wheel in profile.agx_controller.continuous.wheels
                    ],
                    "steering_joint_order": [
                        wheel.steering_joint
                        for wheel in profile.agx_controller.continuous.wheels
                    ],
                },
            )
            episode_id = temp_dir.name.removeprefix(".episode_").removesuffix(".tmp")
            return "RESET_CONTROL", episode_id

        if command == "SUCCESS":
            if not session.recording:
                return "", current_episode_id
            if session.sample_count < int(collection["min_success_samples"]):
                return "", current_episode_id
            if (
                abs(float(measured_v)) > float(collection["start_max_linear_mps"])
                or abs(float(measured_w))
                > float(collection["start_max_angular_radps"])
            ):
                return "", current_episode_id
            teleop.reset()
            adapter.stop()
            session.success_requested_at_sec = now
            self.logger.info("[Collector] human success confirmed; waiting for terminal RGB-D")
            return "RESET_CONTROL", current_episode_id

        if command == "DISCARD":
            if not session.recording:
                return "", current_episode_id
            teleop.reset()
            adapter.stop()
            recorder.discard()
            session.abort()
            self.logger.info("[Collector] episode discarded")
            return "RESET_CONTROL", ""

        if command == "RESET_TO_START":
            if session.recording:
                recorder.discard()
                session.abort()
            if (
                session.last_start_pose_world is None
                or session.last_start_position_world_xyz is None
            ):
                return "", current_episode_id
            session.abort()
            teleop.reset()
            adapter.stop()
            _teleport_robot(
                robot,
                session.last_start_pose_world,
                start_position_world_xyz=session.last_start_position_world_xyz,
            )
            motion.reset()
            episode_map.reset(
                np.asarray(
                    [
                        session.last_start_position_world_xyz[0],
                        session.last_start_position_world_xyz[1],
                        session.last_start_position_world_xyz[2],
                    ],
                    dtype=np.float64,
                )
            )
            _clear_bev(last_bev)
            mapping_state["last_integration_time"] = -float("inf")
            self.logger.info("[Collector] reset to recorded start")
            return "RESET_HISTORY", ""

        self.logger.warning("[CollectorUI] unknown command=%s", command)
        return "", current_episode_id


def _teleport_robot(
    robot,
    pose_xyyaw_world: np.ndarray,
    *,
    start_position_world_xyz: np.ndarray,
) -> None:
    pose = np.asarray(pose_xyyaw_world, dtype=np.float64).reshape(3)
    position = np.asarray(start_position_world_xyz, dtype=np.float32).reshape(3)
    position[:2] = pose[:2]
    orientation = quaternion_wxyz_from_yaw(float(pose[2]))
    robot.set_world_pose(position=position, orientation=orientation)
    if hasattr(robot, "set_linear_velocity"):
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
    if hasattr(robot, "set_angular_velocity"):
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))


def _step_physics(world, *, render: bool) -> None:
    world.step(render=False, update_fabric=True)
    if render:
        world.render()


def _complete_requested_episode(
    session: EpisodeSession, recorder: EpisodeRecorder, sample: SampleRecord, collection: dict,
) -> Path | None:
    requested = session.success_requested_at_sec
    if requested is None or min(sample.front_timestamp_sec, sample.down_timestamp_sec) <= requested + 1.0e-9:
        return None
    if session.sample_count < int(collection["min_success_samples"]):
        return None
    if (
        abs(sample.measured_velocity_vw[0]) > float(collection["start_max_linear_mps"])
        or abs(sample.measured_velocity_vw[1]) > float(collection["start_max_angular_radps"])
    ):
        return None
    path = recorder.finish(
        success=True,
        end_pose_world=sample.pose_world_xyyaw,
        end_time_sec=sample.sample_time_sec,
        success_requested_time_sec=requested,
    )
    session.complete()
    return path


def _zero_robot_velocity(robot) -> None:
    if hasattr(robot, "set_linear_velocity"):
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
    if hasattr(robot, "set_angular_velocity"):
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))


def _clear_bev(bev: np.ndarray) -> None:
    bev.fill(0)
    bev[2] = 255


def _world_pose_from_prim(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid pose prim: {prim_path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    quaternion = transform.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    position = np.asarray(
        [float(translation[0]), float(translation[1]), float(translation[2])],
        dtype=np.float32,
    )
    orientation = np.asarray(
        [
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
        dtype=np.float32,
    )
    return position, orientation


def _select_robot_for_placement(prim_path: str) -> None:
    import carb.settings
    import omni.usd

    carb.settings.get_settings().set_bool("/app/viewport/snapEnabled", False)
    omni.usd.get_context().get_selection().set_selected_prim_paths(
        [str(prim_path)],
        True,
    )


def _clear_selection() -> None:
    import omni.usd

    omni.usd.get_context().get_selection().set_selected_prim_paths([], True)


def _scene_world_bounds(stage, scene_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(str(scene_prim_path))
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Scene prim is invalid: {scene_prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bbox = cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedRange()
    minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
    maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or not (
        np.isfinite(minimum).all() and np.isfinite(maximum).all()
    ):
        raise RuntimeError("Unable to determine finite scene bounds")
    if np.any(maximum <= minimum):
        raise RuntimeError(f"Invalid scene bounds: min={minimum}, max={maximum}")
    return minimum, maximum


def _ordered_readback(values: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in names if name not in values]
    if missing:
        raise RuntimeError(f"Joint readback is missing: {missing}")
    return np.asarray([float(values[name]) for name in names], dtype=np.float32)


def _apply_velocity_command(
    adapter: AgxVelocityAdapter,
    desired: VelocityCommand,
) -> tuple[VelocityCommand, bool]:
    command = fit_agx_command(desired, adapter.config)
    return adapter.apply(command), command != desired
