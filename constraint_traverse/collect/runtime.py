from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import math
from pathlib import Path
import random
import time

import numpy as np

from vln_nav.active_perception.camera_geometry import (
    capture_camera_frame,
    world_from_pose_wxyz,
)
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap
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
from .episode import EpisodeSession
from .recorder import EpisodeRecorder, SampleRecord
from .teleop import ContinuousTeleop, TeleopConfig
from .transforms import (
    BodyMotionEstimator,
    pose_xyyaw,
    quaternion_wxyz_from_yaw,
    world_xy_to_body,
)
from .ui import CollectorPanel, PanelStatus


@dataclass(frozen=True)
class RuntimeOverrides:
    usd_path: Path
    output_root: Path
    spawn_x_m: float | None = None
    spawn_y_m: float | None = None
    spawn_yaw_deg: float | None = None


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
        simulation = self.config["simulation"]
        scene = self.config["scene"]
        cameras_cfg = self.config["cameras"]
        map_cfg = local_voxel_config_from_mapping(self.config["voxel_map"])
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
                prim_path=str(cameras_cfg["front_prim_path"]),
            ),
            kind="constraint-traverse front camera",
        )
        down_path = resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=str(cameras_cfg["down_prim_path"]),
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
            int(cameras_cfg["width"]),
            int(cameras_cfg["height"]),
        )
        frequency = int(cameras_cfg["frequency_hz"])
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

        spawn_x, spawn_y, floor_z, spawn_yaw_deg = _resolve_initial_spawn(
            stage=stage,
            scene_prim_path=str(scene["prim_path"]),
            robot_prim_path=profile.asset.prim_path,
            scene_bounds=scene_bounds,
            config=spawn,
            override_x_m=self.overrides.spawn_x_m,
            override_y_m=self.overrides.spawn_y_m,
            override_yaw_deg=self.overrides.spawn_yaw_deg,
            logger=self.logger,
        )
        _place_robot_on_floor(
            robot,
            x_m=spawn_x,
            y_m=spawn_y,
            floor_z_m=floor_z,
            base_height_m=float(profile.spawn_z_m),
            yaw_deg=spawn_yaw_deg,
        )
        for _ in range(int(spawn["settle_frames"])):
            world.step(render=True)

        bind_viewport_to_camera(
            camera_prim_path=observer_path,
            logger=self.logger,
            enabled=True,
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

        command_queue: list[str] = []
        panel = CollectorPanel(
            ui,
            image_size_px=int(self.config["ui"]["bev_image_size_px"]),
            command_sink=command_queue.append,
        )

        for _ in range(int(cameras_cfg["warmup_render_frames"])):
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
        last_message = "IDLE: drive to a start pose, then press START DEMO"
        current_episode_id = ""
        render_stride = max(
            1,
            round(float(profile.simulation.rendering_dt) / dt),
        )
        frame_index = 0

        self.logger.info(
            "[Collector] scene=%s output=%s spawn=[%.2f,%.2f,z_floor=%.2f,%.1fdeg]",
            scene_path,
            self.overrides.output_root,
            spawn_x,
            spawn_y,
            floor_z,
            spawn_yaw_deg,
        )

        try:
            while self.app.is_running():
                if not world.is_playing():
                    adapter.stop()
                    teleop.reset()
                    self.app.update()
                    continue

                now = float(world.current_time)
                position, orientation = get_world_pose(robot)
                pose = pose_xyyaw(position, orientation)
                measured_v, measured_w = motion.update(pose, now)

                while command_queue:
                    command = command_queue.pop(0)
                    last_message, current_episode_id = self._handle_ui_command(
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
                    )
                    position, orientation = get_world_pose(robot)
                    pose = pose_xyyaw(position, orientation)

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
                    last_message = f"Control rejected: {exc}"

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
                if new_pair:
                    skew = abs(float(front.timestamp_sec - down.timestamp_sec))
                    if skew <= float(cameras_cfg["max_frame_skew_sec"]):
                        last_front_timestamp = float(front.timestamp_sec)
                        last_down_timestamp = float(down.timestamp_sec)
                        if session.recording:
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
                                center_world_xyz=np.asarray(position[:3], dtype=np.float64),
                            )
                            snapshot = episode_map.snapshot()
                            last_bev = snapshot_to_robot_bev(
                                snapshot,
                                robot_world_xyyaw=pose,
                                floor_z_m=float(position[2]) - float(profile.spawn_z_m),
                                config=bev_cfg,
                            )
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
                            )
                            if session.consume_warmup_pair():
                                last_message = (
                                    "RECORDING: initializing episode map "
                                    f"({session.warmup_pairs_remaining} pairs remaining)"
                                )
                            else:
                                recorder.append(sample)
                                session.note_sample(pose)
                    elif session.recording:
                        last_message = (
                            f"Skipped RGB-D pair: frame skew {skew:.3f}s exceeds limit"
                        )

                if now - last_ui_update >= float(self.config["ui"]["update_period_sec"]):
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
                            message=last_message,
                        )
                    )
                    last_ui_update = now

                max_duration = float(collection["max_episode_duration_sec"])
                if session.recording and session.elapsed(now) > max_duration:
                    adapter.stop()
                    teleop.reset()
                    end_pose = pose_xyyaw(*get_world_pose(robot))
                    path = recorder.finish(
                        success=False,
                        end_pose_world=end_pose,
                        end_time_sec=now,
                        failure_reason="timeout",
                    )
                    session.complete(goal_pose_world=end_pose)
                    session.last_goal_pose_world = None
                    last_message = f"FAILURE(timeout) saved: {path.name}"
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
        command: str,
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
    ) -> tuple[str, str]:
        collection = self.config["collection"]

        if command == "START":
            if session.recording:
                return "Already recording", current_episode_id
            if (
                abs(float(measured_v)) > float(collection["start_max_linear_mps"])
                or abs(float(measured_w)) > float(collection["start_max_angular_radps"])
            ):
                return "Stop the robot before START DEMO", current_episode_id
            teleop.reset()
            adapter.stop()
            motion.reset()
            episode_map.reset(np.asarray(position[:3], dtype=np.float64))
            session.begin(
                start_pose_world=pose,
                start_position_world_xyz=position,
                timestamp_sec=now,
                warmup_pairs=int(collection["episode_map_warmup_pairs"]),
            )
            temp_dir = recorder.begin(
                start_pose_world=pose,
                start_position_world_xyz=position,
                start_time_sec=now,
                extra_meta={
                    "teleop": dict(self.config["teleop"]),
                    "cameras": dict(self.config["cameras"]),
                    "bev": dict(self.config["bev"]),
                    "voxel_map": dict(self.config["voxel_map"]),
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
            return "RECORDING: manually traverse the constrained space", episode_id

        if command == "SUCCESS":
            if not session.recording:
                return "No active episode", current_episode_id
            if session.sample_count < int(collection["min_success_samples"]):
                return (
                    f"Need at least {int(collection['min_success_samples'])} samples",
                    current_episode_id,
                )
            if (
                abs(float(measured_v)) > float(collection["start_max_linear_mps"])
                or abs(float(measured_w))
                > float(collection["start_max_angular_radps"])
            ):
                return "Stop the robot before SUCCESS", current_episode_id
            teleop.reset()
            adapter.stop()
            end_pose = pose.copy()
            path = recorder.finish(
                success=True,
                end_pose_world=end_pose,
                end_time_sec=now,
            )
            session.complete(goal_pose_world=end_pose)
            return f"SUCCESS saved: {path.name}", ""

        if command == "FAILURE":
            if not session.recording:
                return "No active episode", current_episode_id
            teleop.reset()
            adapter.stop()
            end_pose = pose.copy()
            path = recorder.finish(
                success=False,
                end_pose_world=end_pose,
                end_time_sec=now,
                failure_reason="manual_failure",
            )
            session.complete(goal_pose_world=end_pose)
            session.last_goal_pose_world = None
            return f"FAILURE saved: {path.name}", ""

        if command == "DISCARD":
            if not session.recording:
                return "No active episode", current_episode_id
            teleop.reset()
            adapter.stop()
            recorder.discard()
            session.abort()
            return "Episode discarded", ""

        if command == "RESET_TO_START":
            if session.recording:
                recorder.discard()
                session.abort()
            if (
                session.last_start_pose_world is None
                or session.last_start_position_world_xyz is None
            ):
                return "No recorded START pose available", current_episode_id
            session.abort()
            session.last_goal_pose_world = None
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
            return "Reset to previous START; press START DEMO when ready", ""

        return f"Unknown UI command: {command}", current_episode_id


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


def _place_robot_on_floor(
    robot,
    *,
    x_m: float,
    y_m: float,
    floor_z_m: float,
    base_height_m: float,
    yaw_deg: float,
) -> None:
    position = np.asarray(
        [float(x_m), float(y_m), float(floor_z_m) + float(base_height_m)],
        dtype=np.float32,
    )
    orientation = quaternion_wxyz_from_yaw(math.radians(float(yaw_deg)))
    robot.set_world_pose(position=position, orientation=orientation)
    if hasattr(robot, "set_linear_velocity"):
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
    if hasattr(robot, "set_angular_velocity"):
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))


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
        raise RuntimeError("Unable to determine finite scene bounds for auto spawn")
    if np.any(maximum <= minimum):
        raise RuntimeError(f"Invalid scene bounds: min={minimum}, max={maximum}")
    return minimum, maximum


def _resolve_initial_spawn(
    *,
    stage,
    scene_prim_path: str,
    robot_prim_path: str,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    config: dict,
    override_x_m: float | None,
    override_y_m: float | None,
    override_yaw_deg: float | None,
    logger: logging.Logger,
) -> tuple[float, float, float, float]:
    if (override_x_m is None) != (override_y_m is None):
        raise ValueError("--spawn-x and --spawn-y must be provided together")

    query = _physx_scene_query_interface()
    if override_x_m is not None and override_y_m is not None:
        yaw_deg = 0.0 if override_yaw_deg is None else float(override_yaw_deg)
        hit = _downward_floor_hit(
            query=query,
            x_m=float(override_x_m),
            y_m=float(override_y_m),
            scene_bounds=scene_bounds,
            config=config,
            robot_prim_path=robot_prim_path,
        )
        if hit is None or not _spawn_volume_is_clear(
            query=query,
            x_m=float(override_x_m),
            y_m=float(override_y_m),
            floor_z_m=float(hit[0]) if hit is not None else 0.0,
            yaw_deg=yaw_deg,
            config=config,
            robot_prim_path=robot_prim_path,
        ):
            raise RuntimeError(
                "Manual spawn XY is not a collision-free floor location; "
                "omit --spawn-x/--spawn-y to use automatic placement"
            )
        return (
            float(override_x_m),
            float(override_y_m),
            float(hit[0]),
            yaw_deg,
        )

    floor_level = _estimate_floor_level(
        query=query,
        scene_bounds=scene_bounds,
        config=config,
        robot_prim_path=robot_prim_path,
        logger=logger,
    )
    logger.info("[AutoSpawn] estimated floor_z=%.3fm", floor_level)

    minimum, maximum = scene_bounds
    margin = float(config["bounds_margin_m"])
    x_min, x_max = float(minimum[0] + margin), float(maximum[0] - margin)
    y_min, y_max = float(minimum[1] + margin), float(maximum[1] - margin)
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Scene XY extent is too small for configured auto-spawn margin")

    step = float(config["sample_step_m"])
    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=np.float64)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=np.float64)
    candidates = [(float(x), float(y)) for x in xs for y in ys]
    rng = random.Random(int(config["random_seed"]))
    rng.shuffle(candidates)
    candidates = candidates[: int(config["max_attempts"])]

    for attempt, (x_m, y_m) in enumerate(candidates, start=1):
        yaw_deg = (
            float(override_yaw_deg)
            if override_yaw_deg is not None
            else rng.uniform(-180.0, 180.0)
        )
        result = _evaluate_spawn_candidate(
            query=query,
            x_m=x_m,
            y_m=y_m,
            yaw_deg=yaw_deg,
            floor_level_m=floor_level,
            scene_bounds=scene_bounds,
            config=config,
            robot_prim_path=robot_prim_path,
        )
        if result is not None:
            logger.info(
                "[AutoSpawn] selected attempt=%d/%d xy=[%.2f,%.2f] floor_z=%.2f yaw=%.1fdeg",
                attempt,
                len(candidates),
                result[0],
                result[1],
                result[2],
                result[3],
            )
            return result

    raise RuntimeError(
        "Auto spawn could not find a collision-free floor location. "
        "Increase spawn.max_attempts, reduce clearance, or pass --spawn-x/--spawn-y."
    )


def _physx_scene_query_interface():
    try:
        from omni.physx import get_physx_scene_query_interface
    except ImportError as exc:
        raise RuntimeError("PhysX scene-query API is unavailable") from exc
    return get_physx_scene_query_interface()


def _estimate_floor_level(
    *,
    query,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    config: dict,
    robot_prim_path: str,
    logger: logging.Logger,
) -> float:
    minimum, maximum = scene_bounds
    margin = float(config["bounds_margin_m"])
    x_min, x_max = float(minimum[0] + margin), float(maximum[0] - margin)
    y_min, y_max = float(minimum[1] + margin), float(maximum[1] - margin)
    rng = random.Random(int(config["random_seed"]) + 1777)
    samples = []
    count = int(config["floor_estimation_samples"])
    for _ in range(count):
        x_m = rng.uniform(x_min, x_max)
        y_m = rng.uniform(y_min, y_max)
        hit = _downward_floor_hit(
            query=query,
            x_m=x_m,
            y_m=y_m,
            scene_bounds=scene_bounds,
            config=config,
            robot_prim_path=robot_prim_path,
        )
        if hit is not None:
            samples.append(float(hit[0]))
    if not samples:
        grid_side = max(6, int(math.ceil(math.sqrt(count))))
        xs = np.linspace(x_min, x_max, grid_side, dtype=np.float64)
        ys = np.linspace(y_min, y_max, grid_side, dtype=np.float64)
        for x_m in xs:
            for y_m in ys:
                hit = _downward_floor_hit(
                    query=query,
                    x_m=float(x_m),
                    y_m=float(y_m),
                    scene_bounds=scene_bounds,
                    config=config,
                    robot_prim_path=robot_prim_path,
                )
                if hit is not None:
                    samples.append(float(hit[0]))
        if samples:
            logger.warning(
                "[AutoSpawn] random probes found no floor; grid probes found %d hits",
                len(samples),
            )

    if not samples:
        raise RuntimeError(
            "Auto spawn found no horizontal collidable floor. "
            "Check scene collision geometry or provide --spawn-x and --spawn-y."
        )

    logger.info("[AutoSpawn] accepted floor probe hits=%d", len(samples))
    bin_size = float(config["floor_height_bin_m"])
    bins: dict[int, list[float]] = {}
    for value in samples:
        key = int(round(value / bin_size))
        bins.setdefault(key, []).append(value)
    best_key, values = max(
        bins.items(),
        key=lambda item: (len(item[1]), -float(np.mean(item[1]))),
    )
    floor_level = float(np.median(np.asarray(values, dtype=np.float64)))
    logger.info(
        "[AutoSpawn] floor bin=%d hits=%d floor_z=%.3f",
        best_key,
        len(values),
        floor_level,
    )
    return floor_level


def _evaluate_spawn_candidate(
    *,
    query,
    x_m: float,
    y_m: float,
    yaw_deg: float,
    floor_level_m: float,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    config: dict,
    robot_prim_path: str,
) -> tuple[float, float, float, float] | None:
    hit = _downward_floor_hit(
        query=query,
        x_m=x_m,
        y_m=y_m,
        scene_bounds=scene_bounds,
        config=config,
        robot_prim_path=robot_prim_path,
    )
    if hit is None:
        return None
    floor_z, _normal_z = hit
    if abs(float(floor_z) - float(floor_level_m)) > float(
        config["floor_height_tolerance_m"]
    ):
        return None
    if not _spawn_volume_is_clear(
        query=query,
        x_m=x_m,
        y_m=y_m,
        floor_z_m=floor_z,
        yaw_deg=yaw_deg,
        config=config,
        robot_prim_path=robot_prim_path,
    ):
        return None
    return float(x_m), float(y_m), float(floor_z), float(yaw_deg)


def _downward_floor_hit(
    *,
    query,
    x_m: float,
    y_m: float,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    config: dict,
    robot_prim_path: str,
) -> tuple[float, float] | None:
    import carb

    minimum, maximum = scene_bounds
    min_z = float(minimum[2])
    max_z = float(maximum[2])
    configured_height = float(config["floor_probe_height_m"])
    probe_depth = float(config["floor_probe_depth_m"])
    probe_heights = sorted({
        float(np.clip(value, min_z + 0.05, max_z + 0.25))
        for value in (
            min_z + 0.75,
            min_z + 1.25,
            min_z + configured_height,
            min_z + configured_height + 1.5,
            max_z + 0.25,
        )
    })
    minimum_normal_z = math.cos(math.radians(float(config["max_slope_deg"])))
    for probe_z in probe_heights:
        distance = max(0.10, probe_z - min_z + probe_depth)
        hit = query.raycast_closest(
            carb.Float3(float(x_m), float(y_m), float(probe_z)),
            carb.Float3(0.0, 0.0, -1.0),
            float(distance),
        )
        if not hit or not bool(hit.get("hit", False)):
            continue
        hit_path = str(hit.get("rigidBody", ""))
        if hit_path.startswith(str(robot_prim_path)):
            continue
        position = hit.get("position")
        normal = hit.get("normal")
        if position is None or normal is None:
            continue
        floor_z = float(position[2])
        normal_z = float(normal[2])
        if abs(normal_z) < minimum_normal_z:
            continue
        return floor_z, normal_z
    return None


def _spawn_volume_is_clear(
    *,
    query,
    x_m: float,
    y_m: float,
    floor_z_m: float,
    yaw_deg: float,
    config: dict,
    robot_prim_path: str,
) -> bool:
    import carb

    half_length = float(config["robot_half_length_m"]) + float(config["clearance_m"])
    half_width = float(config["robot_half_width_m"]) + float(config["clearance_m"])
    height = float(config["collision_check_height_m"])
    bottom = float(floor_z_m) + float(config["collision_bottom_offset_m"])
    origin = carb.Float3(
        float(x_m),
        float(y_m),
        bottom + 0.5 * height,
    )
    yaw = math.radians(float(yaw_deg))
    rotation = carb.Float4(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
    extent = carb.Float3(half_length, half_width, 0.5 * height)
    blocked = False

    def report_hit(hit) -> bool:
        nonlocal blocked
        rigid_body = str(getattr(hit, "rigid_body", ""))
        collision = str(getattr(hit, "collision", ""))
        if rigid_body.startswith(str(robot_prim_path)) or collision.startswith(
            str(robot_prim_path)
        ):
            return True
        blocked = True
        return False

    query.overlap_box(extent, origin, rotation, report_hit, False)
    return not blocked


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
    if session.last_goal_pose_world is not None:
        goal = world_xy_to_body(
            session.last_goal_pose_world[:2],
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
