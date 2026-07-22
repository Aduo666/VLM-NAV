from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from vln_nav.app.nav import (
    _get_joint_names,
    _get_world_pose,
    _resolve_robot_relative_prim_path,
    _set_local_xform,
    _wait_for_stage_assets,
)
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.navigation.odometry import SimOdometrySensor
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the robot selected by robot.profile")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Project YAML config")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.headless:
        config.headless = True
    logger = setup_logging(config.log_level)
    if config.robot_profile is None:
        raise RuntimeError("Robot profile was not loaded")

    # Kit also parses process arguments; remove this app's --config/--headless flags.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(config.headless),
            "width": int(config.sim_width),
            "height": int(config.sim_height),
            "active_gpu": config.active_gpu,
            "physics_gpu": config.physics_gpu,
            "multi_gpu": bool(config.multi_gpu),
        }
    )
    try:
        try:
            return _run(simulation_app=simulation_app, config=config, logger=logger)
        except BaseException as exc:
            print(f"[SmokeTest] FAILED error={type(exc).__name__}:{exc}", flush=True)
            raise
    finally:
        simulation_app.close()


def _run(*, simulation_app, config, logger) -> int:
    print("[SmokeTest] stage=IMPORT_ISAAC_API", flush=True)
    try:
        from isaacsim.core.prims import SingleArticulation as RobotArticulation
    except ImportError:
        from isaacsim.core.prims import Articulation as RobotArticulation
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from pxr import Usd, UsdLux, UsdPhysics

    profile = config.robot_profile
    print(f"[SmokeTest] stage=CREATE_WORLD profile={profile.name}", flush=True)
    world = World(
        physics_dt=float(profile.simulation.physics_dt),
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    dome = UsdLux.DomeLight.Define(stage, "/World/SmokeTestLight")
    dome.CreateIntensityAttr().Set(1500.0)

    asset_stage = Usd.Stage.Open(str(profile.asset.usd_path))
    if asset_stage is None:
        raise RuntimeError(f"Smoke-test could not open robot USD: {profile.asset.usd_path}")
    default_prim = asset_stage.GetDefaultPrim()
    articulation_roots = [
        str(prim.GetPath())
        for prim in asset_stage.Traverse()
        if UsdPhysics.ArticulationRootAPI(prim)
    ]
    print(
        f"[SmokeTest] stage=USD_COMPOSED default_prim={default_prim.GetPath()} "
        f"prim_count={sum(1 for _ in asset_stage.Traverse())} "
        f"articulation_roots={articulation_roots}",
        flush=True,
    )

    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    print(f"[SmokeTest] stage=LOAD_USD path={profile.asset.usd_path}", flush=True)
    if not profile.is_g1:
        _wait_for_stage_assets(
            simulation_app=simulation_app,
            is_stage_loading=is_stage_loading,
            timeout_sec=float(config.stage_load_timeout_sec),
            logger=logger,
            label="smoke-test robot",
        )
    _set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[0.0, 0.0, float(profile.spawn_z_m)],
        rotation_deg=[0.0, 0.0, 0.0],
    )
    print("[SmokeTest] stage=SPAWN_POSE_SET", flush=True)
    articulation_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    print(f"[SmokeTest] stage=CREATE_ARTICULATION path={articulation_path}", flush=True)
    try:
        robot = world.scene.add(
            RobotArticulation(
                prim_path=articulation_path,
                name="smoke_test_robot",
                reset_xform_properties=not profile.is_g1,
            )
        )
    except BaseException as exc:
        print(f"[SmokeTest] stage=CREATE_ARTICULATION_FAILED error={type(exc).__name__}:{exc}", flush=True)
        raise
    print(f"[SmokeTest] stage=RESET articulation={articulation_path}", flush=True)
    world.reset()
    robot.initialize()

    print("[SmokeTest] stage=VALIDATE_SENSORS", flush=True)
    _validate_profile_sensors(stage=stage, profile=profile)
    joint_names = _get_joint_names(robot)
    controller = create_locomotion_controller(
        profile=profile,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=joint_names,
        logger=logger,
    )
    print("[SmokeTest] stage=CONTROLLER_READY", flush=True)
    _validate_camera_frames(
        world=world,
        controller=controller,
        profile=profile,
    )
    print("[SmokeTest] stage=CAMERAS_READY", flush=True)
    odom = SimOdometrySensor(
        logger=logger,
        source="smoke_test_world_pose",
        pose_filter_alpha=(
            1.0 if profile.g1_controller is None else profile.g1_controller.odom_filter_alpha
        ),
    )

    sim_time_sec = 0.0
    standup_sec = 0.5 if profile.g1_controller is None else profile.g1_controller.standup_duration_sec
    logger.info("[SmokeTest] profile=%s stage=STAND duration_sec=%.2f", profile.name, standup_sec)
    sim_time_sec, _ = _run_timed_action(
        world=world,
        robot=robot,
        controller=controller,
        odom=odom,
        action="STOP",
        duration_sec=float(standup_sec),
        sim_time_sec=sim_time_sec,
        physics_dt=float(profile.simulation.physics_dt),
    )

    sequence = ("FORWARD", "STOP", "TURN_LEFT", "STOP", "TURN_RIGHT", "STOP")
    for action in sequence:
        if action == "STOP":
            timeout_sec = max(0.5, float(profile.primitives.settle_max_wait_sec))
        elif action == "FORWARD":
            timeout_sec = float(profile.primitives.forward_timeout_sec)
        else:
            timeout_sec = float(profile.primitives.turn_timeout_sec)
        logger.info("[SmokeTest] action=%s timeout_sec=%.2f", action, timeout_sec)
        sim_time_sec, result = _run_primitive(
            world=world,
            robot=robot,
            controller=controller,
            odom=odom,
            action=action,
            timeout_sec=timeout_sec,
            sim_time_sec=sim_time_sec,
            physics_dt=float(profile.simulation.physics_dt),
            forward_target_m=float(profile.primitives.forward_distance_m),
            turn_target_rad=float(np.deg2rad(profile.primitives.turn_angle_deg)),
        )
        logger.info("[SmokeTest] action=%s result=%s", action, result)
        print(f"[SmokeTest] action={action} result={result}", flush=True)
        if result.startswith("timeout") and action != "STOP":
            raise RuntimeError(f"Smoke-test primitive timed out: {action}")

    logger.info("[SmokeTest] PASS profile=%s", profile.name)
    print(f"[SmokeTest] PASS profile={profile.name}", flush=True)
    return 0


def _run_primitive(
    *,
    world,
    robot,
    controller,
    odom,
    action: str,
    timeout_sec: float,
    sim_time_sec: float,
    physics_dt: float,
    forward_target_m: float,
    turn_target_rad: float,
) -> tuple[float, str]:
    start_pose = None
    started_at = float(sim_time_sec)
    while sim_time_sec - started_at < max(float(physics_dt), float(timeout_sec)):
        controller.apply(action, speed_scale=0.0 if action == "STOP" else 1.0)
        advance_physics = getattr(controller, "advance_physics", None)
        if callable(advance_physics):
            advance_physics()
        world.step(render=False)
        sim_time_sec += float(physics_dt)
        position, orientation = _get_world_pose(robot)
        state = odom.update(
            world_position_xyz=position,
            world_orientation_wxyz=orientation,
            timestamp_sec=sim_time_sec,
        )
        if start_pose is None:
            start_pose = state.pose
        if action == "FORWARD":
            traveled = float(np.hypot(state.pose.x - start_pose.x, state.pose.y - start_pose.y))
            if traveled >= float(forward_target_m):
                controller.apply("STOP", speed_scale=0.0)
                return sim_time_sec, f"distance_reached:{traveled:.3f}m"
        elif action in {"TURN_LEFT", "TURN_RIGHT"}:
            yaw_delta = abs(_wrap_angle(state.pose.yaw_rad - start_pose.yaw_rad))
            if yaw_delta >= float(turn_target_rad):
                controller.apply("STOP", speed_scale=0.0)
                return sim_time_sec, f"yaw_reached:{np.degrees(yaw_delta):.2f}deg"
    controller.apply("STOP", speed_scale=0.0)
    if action == "STOP":
        return sim_time_sec, "complete"
    if start_pose is None:
        return sim_time_sec, "timeout:no_pose"
    position, orientation = _get_world_pose(robot)
    state = odom.update(
        world_position_xyz=position,
        world_orientation_wxyz=orientation,
        timestamp_sec=sim_time_sec,
    )
    traveled = float(np.hypot(state.pose.x - start_pose.x, state.pose.y - start_pose.y))
    yaw_delta = abs(_wrap_angle(state.pose.yaw_rad - start_pose.yaw_rad))
    return sim_time_sec, f"timeout:distance={traveled:.3f}m,yaw={np.degrees(yaw_delta):.2f}deg"


def _run_timed_action(**kwargs) -> tuple[float, str]:
    return _run_primitive(
        forward_target_m=float("inf"),
        turn_target_rad=float("inf"),
        timeout_sec=float(kwargs.pop("duration_sec")),
        **kwargs,
    )


def _validate_profile_sensors(*, stage, profile) -> None:
    expected = (
        ("front camera", profile.sensors.front_camera_prim_path, "Camera"),
        ("observer camera", profile.sensors.observer_camera_prim_path, "Camera"),
        ("LiDAR", profile.sensors.lidar_prim_path, "Lidar"),
    )
    for label, relative_path, expected_type in expected:
        if label == "observer camera" and not profile.sensors.observer_camera_enabled:
            continue
        path = _resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=relative_path,
        )
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Smoke-test required {label} is missing: {path}")
        if prim.GetTypeName() != expected_type:
            raise RuntimeError(
                f"Smoke-test {label} type mismatch at {path}: "
                f"expected {expected_type}, got {prim.GetTypeName()}"
            )


def _validate_camera_frames(*, world, controller, profile) -> None:
    from isaacsim.sensors.camera import Camera

    cameras = []
    camera_specs = [
        ("front", profile.sensors.front_camera_prim_path),
    ]
    if profile.sensors.observer_camera_enabled:
        camera_specs.append(("observer", profile.sensors.observer_camera_prim_path))
    for name, relative_path in camera_specs:
        camera = Camera(
            prim_path=_resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=relative_path,
            ),
            name=f"smoke_test_{name}_camera",
            frequency=int(profile.sensors.camera_frequency_hz),
            resolution=(160, 120),
        )
        camera.initialize()
        cameras.append((name, camera))

    render_stride = max(
        1,
        int(round(float(profile.simulation.rendering_dt) / float(profile.simulation.physics_dt))),
    )
    for step_index in range(12 * render_stride):
        controller.apply("STOP", speed_scale=0.0)
        advance_physics = getattr(controller, "advance_physics", None)
        if callable(advance_physics):
            advance_physics()
        world.step(render=False)
        if (step_index + 1) % render_stride == 0:
            world.render()
    for name, camera in cameras:
        rgba = camera.get_rgba()
        if rgba is None or np.asarray(rgba).size == 0:
            raise RuntimeError(f"Smoke-test {name} camera did not produce an image")


def _wrap_angle(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


if __name__ == "__main__":
    raise SystemExit(main())
