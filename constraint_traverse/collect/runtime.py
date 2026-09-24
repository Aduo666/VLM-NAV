from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time

import numpy as np
import yaml

from vln_nav.active_perception.camera_geometry import (
    capture_camera_frame,
    world_from_pose_wxyz,
)
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap, LocalVoxelMapConfig
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
    set_local_xform,
    wait_for_stage_assets,
)
from vln_nav.runtime.scene_lighting import add_indoor_lighting

from .bev import BevConfig, local_voxel_config_from_mapping, snapshot_to_robot_bev
from .episode import EpisodeMode, EpisodeSession
from .recorder import EpisodeRecorder, SampleRecord
from .teleop import ContinuousTeleop, TeleopConfig
from .transforms import (
    BodyMotionEstimator,
    pose_xyyaw,
    quaternion_wxyz_from_yaw,
    relative_goal_xyyaw,
    world_xy_to_body,
)
from .ui import CollectorPanel, PanelStatus


@dataclass(frozen=True)
class RuntimeOverrides:
    usd_path: Path
    output_root: Path


def _load_navigation_mapping(
    project_root: Path,
) -> tuple[dict, LocalVoxelMapConfig]:
    config_path = Path(project_root) / "configs/navigation.yaml"
    with config_path.open(encoding="utf-8") as handle:
        navigation = yaml.safe_load(handle) or {}
    mapping = navigation.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != {
        "camera",
        "map",
        "max_frame_skew_sec",
    }:
        raise ValueError("主导航 mapping 配置字段不完整")
    camera = mapping["camera"]
    if not isinstance(camera, dict) or set(camera) != {
        "width",
        "height",
        "frequency_hz",
    }:
        raise ValueError("主导航 mapping.camera 配置字段不完整")
    if any(int(camera[name]) <= 0 for name in camera):
        raise ValueError("主导航 mapping.camera 配置必须为正整数")
    map_values = dict(mapping["map"])
    integration_period = float(map_values.pop("integration_period_sec"))
    if integration_period <= 0 or float(mapping["max_frame_skew_sec"]) <= 0:
        raise ValueError("主导航地图周期和帧同步阈值必须为正数")
    map_config = local_voxel_config_from_mapping(map_values)
    return mapping, map_config


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
        mapping_cfg, map_cfg = _load_navigation_mapping(self.project_root)
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
        holding_z = float(scene_bounds[1][2]) + float(spawn["holding_height_m"])
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
        for camera in (front_camera, down_camera):
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()

        holding_position = np.asarray(
            [float(holding_xy[0]), float(holding_xy[1]), holding_z],
            dtype=np.float32,
        )
        holding_orientation = quaternion_wxyz_from_yaw(0.0)
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
        )
        adapter = AgxVelocityAdapter(drive, profile.agx_controller.continuous)
        teleop = ContinuousTeleop(teleop_cfg)
        teleop.bind_keyboard()

        command_queue: list[object] = []
        panel = CollectorPanel(
            ui,
            image_size_px=int(self.config["ui"]["bev_image_size_px"]),
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

        self.logger.info(
            "[Collector] scene=%s output=%s placement=manual map_config=%s",
            scene_path,
            self.overrides.output_root,
            self.project_root / "configs/navigation.yaml",
        )
        world.pause()

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
                    _, current_episode_id = self._handle_ui_command(
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
                    if world.is_playing():
                        position, orientation = get_world_pose(robot)
                    else:
                        position, orientation = _world_pose_from_prim(
                            stage,
                            articulation_path,
                        )
                    pose = pose_xyyaw(position, orientation)

                ui_now = time.monotonic()
                if not world.is_playing():
                    adapter.stop()
                    teleop.reset()
                    if ui_now - last_ui_update >= float(
                        self.config["ui"]["update_period_sec"]
                    ):
                        start_body, goal_body, trajectory_body = _body_overlay(
                            session=session,
                            pose=pose,
                        )
                        panel.update_bev(
                            last_bev,
                            config=bev_cfg,
                            start_body_xy=start_body,
                            goal_body_xy=goal_body,
                            trajectory_body_xy=trajectory_body,
                        )
                        panel.update_status(
                            PanelStatus(
                                mode=session.mode.value,
                                episode_id=current_episode_id,
                                sample_count=session.sample_count,
                                elapsed_sec=session.elapsed(now),
                                robot_placed=session.robot_placed,
                                goal_selected=session.goal_pose_world is not None,
                                can_reset_start=(
                                    session.last_start_pose_world is not None
                                ),
                            )
                        )
                        last_ui_update = ui_now
                    self.app.update()
                    continue

                desired = teleop.step(dt)
                if session.recording and session.warmup_pairs_remaining > 0:
                    teleop.reset()
                    desired = VelocityCommand()
                try:
                    applied = adapter.apply(desired)
                except Exception as exc:
                    self.logger.error("[CollectorControl] command rejected: %s", exc)
                    teleop.reset()
                    adapter.stop()
                    desired = VelocityCommand()
                    applied = VelocityCommand()

                world_from_base = world_from_pose_wxyz(position, orientation)
                front = capture_camera_frame(
                    source="front",
                    camera=front_camera,
                    frame_index=frame_index,
                    world_from_base=world_from_base,
                )
                down = capture_camera_frame(
                    source="down",
                    camera=down_camera,
                    frame_index=frame_index,
                    world_from_base=world_from_base,
                )
                new_pair = (
                    front is not None
                    and down is not None
                    and front.timestamp_sec > last_front_timestamp + 1.0e-9
                    and down.timestamp_sec > last_down_timestamp + 1.0e-9
                )
                map_active = session.robot_placed and session.mode in {
                    EpisodeMode.IDLE,
                    EpisodeMode.RECORDING,
                }
                if new_pair:
                    skew = abs(float(front.timestamp_sec - down.timestamp_sec))
                    if skew <= float(mapping_cfg["max_frame_skew_sec"]):
                        last_front_timestamp = float(front.timestamp_sec)
                        last_down_timestamp = float(down.timestamp_sec)
                        if map_active:
                            integration_period = float(
                                map_runtime_cfg["integration_period_sec"]
                            )
                            if (
                                now - mapping_state["last_integration_time"]
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
                                        position[:3], dtype=np.float64
                                    ),
                                )
                                mapping_state["last_integration_time"] = now
                            snapshot = episode_map.snapshot()
                            last_bev = snapshot_to_robot_bev(
                                snapshot,
                                robot_world_xyyaw=pose,
                                floor_z_m=float(position[2]) - float(profile.spawn_z_m),
                                config=bev_cfg,
                            )
                            if session.recording:
                                readback = drive.readback()
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
                                    pose_world_xyyaw=pose,
                                    world_position_xyz=position,
                                    world_orientation_wxyz=orientation,
                                    measured_velocity_vw=np.asarray(
                                        [measured_v, measured_w],
                                        dtype=np.float32,
                                    ),
                                    human_action_vw=np.asarray(
                                        [desired.linear_mps, desired.angular_radps],
                                        dtype=np.float32,
                                    ),
                                    applied_action_vw=np.asarray(
                                        [applied.linear_mps, applied.angular_radps],
                                        dtype=np.float32,
                                    ),
                                    steering_position=_ordered_readback(
                                        readback["steering_position"], steering_names
                                    ),
                                    steering_velocity=_ordered_readback(
                                        readback["steering_velocity"], steering_names
                                    ),
                                    wheel_velocity=_ordered_readback(
                                        readback["wheel_velocity"], wheel_names
                                    ),
                                    front_intrinsics=front.intrinsics,
                                    down_intrinsics=down.intrinsics,
                                    front_base_from_camera=front.base_from_camera,
                                    down_base_from_camera=down.base_from_camera,
                                    map_version=int(snapshot.version),
                                    goal_body_xyyaw=relative_goal_xyyaw(
                                        pose,
                                        session.goal_pose_world,
                                    ),
                                )
                                if not session.consume_warmup_pair():
                                    recorder.append(sample)
                                    session.note_sample(pose)
                    elif map_active:
                        self.logger.warning(
                            "[CollectorMap] skipped RGB-D pair skew=%.3fs limit=%.3fs",
                            skew,
                            float(mapping_cfg["max_frame_skew_sec"]),
                        )

                if ui_now - last_ui_update >= float(self.config["ui"]["update_period_sec"]):
                    start_body, goal_body, trajectory_body = _body_overlay(
                        session=session,
                        pose=pose,
                    )
                    panel.update_bev(
                        last_bev,
                        config=bev_cfg,
                        start_body_xy=start_body,
                        goal_body_xy=goal_body,
                        trajectory_body_xy=trajectory_body,
                    )
                    panel.update_status(
                        PanelStatus(
                            mode=session.mode.value,
                            episode_id=current_episode_id,
                            sample_count=session.sample_count,
                            elapsed_sec=session.elapsed(now),
                            robot_placed=session.robot_placed,
                            goal_selected=session.goal_pose_world is not None,
                            can_reset_start=session.last_start_pose_world is not None,
                        )
                    )
                    last_ui_update = ui_now

                max_duration = float(collection["max_episode_duration_sec"])
                if session.recording and session.elapsed(now) > max_duration:
                    adapter.stop()
                    teleop.reset()
                    recorder.discard()
                    session.abort()
                    self.logger.info("[Collector] timed-out episode discarded")
                    current_episode_id = ""

                world.step(render=(frame_index % render_stride == 0))
                frame_index += 1
        finally:
            try:
                adapter.stop()
            except Exception:
                pass
            if recorder.active:
                recorder.discard()
            recorder.close()
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
                _select_prim(articulation_path)
                world.pause()
                self.logger.info("[CollectorPlacement] robot placement started")
                return "", current_episode_id
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
                return "", current_episode_id
            return "", current_episode_id

        if (
            isinstance(command, tuple)
            and len(command) == 3
            and command[0] == "SET_GOAL_BODY"
        ):
            if session.mode is not EpisodeMode.IDLE or not session.robot_placed:
                return "", current_episode_id
            forward = float(command[1])
            left = float(command[2])
            cosine = math.cos(float(pose[2]))
            sine = math.sin(float(pose[2]))
            goal_x = float(pose[0]) + cosine * forward - sine * left
            goal_y = float(pose[1]) + sine * forward + cosine * left
            if math.hypot(forward, left) > 1.0e-6:
                goal_yaw = float(pose[2]) + math.atan2(left, forward)
            else:
                goal_yaw = float(pose[2])
            goal_yaw = math.atan2(math.sin(goal_yaw), math.cos(goal_yaw))
            goal_pose = np.asarray([goal_x, goal_y, goal_yaw], dtype=np.float64)
            session.set_goal(goal_pose)
            self.logger.info("[CollectorGoal] map goal selected pose=%s", goal_pose.tolist())
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
            if not session.robot_placed or session.goal_pose_world is None:
                return "", current_episode_id
            if (
                abs(float(measured_v)) > float(collection["start_max_linear_mps"])
                or abs(float(measured_w)) > float(collection["start_max_angular_radps"])
            ):
                return "", current_episode_id
            teleop.reset()
            adapter.stop()
            motion.reset()
            episode_map.reset(np.asarray(position[:3], dtype=np.float64))
            _clear_bev(last_bev)
            mapping_state["last_integration_time"] = -float("inf")
            session.begin(
                start_pose_world=pose,
                start_position_world_xyz=position,
                timestamp_sec=now,
                warmup_pairs=int(collection["episode_map_warmup_pairs"]),
            )
            temp_dir = recorder.begin(
                start_pose_world=pose,
                start_position_world_xyz=position,
                goal_pose_world=session.goal_pose_world,
                start_time_sec=now,
                extra_meta={
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
            return "", episode_id

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
            end_pose = pose.copy()
            path = recorder.finish(
                success=True,
                end_pose_world=end_pose,
                end_time_sec=now,
            )
            session.complete()
            self.logger.info("[Collector] success saved path=%s", path)
            return "", ""

        if command == "DISCARD":
            if not session.recording:
                return "", current_episode_id
            teleop.reset()
            adapter.stop()
            recorder.discard()
            session.abort()
            self.logger.info("[Collector] episode discarded")
            return "", ""

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
            return "", ""

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


def _select_prim(prim_path: str) -> None:
    import omni.usd

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


def _body_overlay(
    *,
    session: EpisodeSession,
    pose: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    current_xy = (float(pose[0]), float(pose[1]))
    yaw = float(pose[2])
    start = None
    goal = None
    trajectory = None
    if session.last_start_pose_world is not None:
        start = world_xy_to_body(
            session.last_start_pose_world[:2],
            robot_world_xy=current_xy,
            robot_yaw_rad=yaw,
        )[0]
    if session.goal_pose_world is not None:
        goal = world_xy_to_body(
            session.goal_pose_world[:2],
            robot_world_xy=current_xy,
            robot_yaw_rad=yaw,
        )[0]
    if session.trajectory_world_xy:
        trajectory = world_xy_to_body(
            np.asarray(session.trajectory_world_xy, dtype=np.float64),
            robot_world_xy=current_xy,
            robot_yaw_rad=yaw,
        )
    return start, goal, trajectory
