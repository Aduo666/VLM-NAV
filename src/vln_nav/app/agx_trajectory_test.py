"""AGX continuous-control and narrow-passage acceptance.

This entry deliberately excludes VLN, NavDP, RGB-D mapping, and safety
intervention.  It answers two lower-level questions only: can the chassis
track a smooth path without a steering deadlock, and can it repeatably follow
known feasible paths through a narrow passage?
"""
from __future__ import annotations

import logging
import math
import sys

import numpy as np
import yaml

from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import AgxVelocityAdapter, PathTracker
from vln_nav.navigation.odometry import SimOdometrySensor
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from vln_nav.runtime.isaac_runtime_utils import (
    bind_viewport_to_camera,
    get_joint_names,
    get_world_pose,
    resolve_existing_prim_path,
    resolve_robot_relative_prim_path,
    set_local_xform,
    wait_for_stage_assets,
)
from vln_nav.runtime.robot_profile import load_robot_profile


SETTINGS_PATH = PROJECT_ROOT / "configs/agx_trajectory_test.yaml"
DOOR_SCENARIOS = {"DOOR STRAIGHT", "DOOR OFFSET", "DOOR ANGLED"}


def _load_settings() -> dict:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if set(data) != {"passage", "test"}:
        raise ValueError("AGX trajectory test requires exactly passage and test")
    passage_fields = {
        "gap_m",
        "wall_thickness_m",
        "wall_segment_length_m",
        "wall_height_m",
        "robot_half_width_m",
        "door_forward_m",
        "path_exit_distance_m",
        "offset_left_m",
        "angled_yaw_deg",
    }
    if set(data["passage"]) != passage_fields:
        raise ValueError(f"passage requires exactly: {sorted(passage_fields)}")
    test_fields = {
        "settle_after_reset_sec",
        "steering_log_period_sec",
        "path_visualization_z_m",
    }
    if set(data["test"]) != test_fields:
        raise ValueError(f"test requires exactly: {sorted(test_fields)}")
    numeric = [*data["passage"].values(), *data["test"].values()]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("AGX trajectory test settings must be finite")
    positive = (
        data["passage"]["gap_m"],
        data["passage"]["wall_thickness_m"],
        data["passage"]["wall_segment_length_m"],
        data["passage"]["wall_height_m"],
        data["passage"]["robot_half_width_m"],
        data["passage"]["door_forward_m"],
        data["passage"]["path_exit_distance_m"],
        data["test"]["settle_after_reset_sec"],
        data["test"]["steering_log_period_sec"],
    )
    if not all(float(value) > 0 for value in positive):
        raise ValueError("AGX trajectory test dimensions and periods must be positive")
    if not 0 < abs(float(data["passage"]["angled_yaw_deg"])) < 45:
        raise ValueError("passage.angled_yaw_deg must be within (0, 45) degrees")
    if float(data["passage"]["gap_m"]) <= 2.0 * float(
        data["passage"]["robot_half_width_m"]
    ):
        raise ValueError("passage gap must be wider than the configured robot body")
    return data


def _validate_geometry(stage, profile, articulation_path):
    """Check configured wheel geometry against the loaded articulation."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(profile.asset.prim_path)
    by_name = {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            if prim.GetName() in by_name:
                raise ValueError(f"Ambiguous joint name: {prim.GetName()}")
            by_name[prim.GetName()] = prim
    chassis_inverse = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(articulation_path)
    ).GetInverse()
    for wheel in profile.agx_controller.continuous.wheels:
        for name, expected_axis in (
            (wheel.steering_joint, (0, 0, wheel.steering_sign)),
            (wheel.drive_joint, (0, wheel.drive_sign, 0)),
        ):
            if name not in by_name:
                raise ValueError(f"Configured AGX joint is missing from USD: {name}")
            joint = UsdPhysics.RevoluteJoint(by_name[name])
            body = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            body_to_world = cache.GetLocalToWorldTransform(body)
            basis = {
                "X": (1, 0, 0),
                "Y": (0, 1, 0),
                "Z": (0, 0, 1),
            }[str(joint.GetAxisAttr().Get())]
            local_rotation = joint.GetLocalRot0Attr().Get()
            rotation = Gf.Rotation(
                Gf.Quatd(
                    float(local_rotation.GetReal()),
                    Gf.Vec3d(*local_rotation.GetImaginary()),
                )
            )
            axis = chassis_inverse.TransformDir(
                body_to_world.TransformDir(rotation.TransformDir(Gf.Vec3d(*basis)))
            )
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            if np.linalg.norm(axis - expected_axis) > 0.05:
                raise ValueError(
                    f"USD joint {name} axis={axis}, expected={expected_axis}"
                )
            if name == wheel.steering_joint:
                position = chassis_inverse.Transform(
                    body_to_world.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get()))
                )
                if np.linalg.norm(
                    np.asarray(position)[:2] - [wheel.x_m, wheel.y_m]
                ) > 0.04:
                    raise ValueError(
                        f"USD joint {name} XY={tuple(position)[:2]} disagrees "
                        "with continuous wheel geometry"
                    )
                lower = float(joint.GetLowerLimitAttr().Get())
                upper = float(joint.GetUpperLimitAttr().Get())
                required = math.degrees(
                    float(profile.agx_controller.continuous.max_steering_rad)
                )
                if lower > -required or upper < required:
                    raise ValueError(
                        f"USD steering limits for {name} do not support "
                        f"+/-{required:.1f} degrees"
                    )


def _smoothstep(values: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def _path_and_door(kind: str, settings: dict) -> tuple[np.ndarray, tuple[np.ndarray, float] | None]:
    passage = settings["passage"]
    if kind == "STRAIGHT":
        return np.column_stack((np.linspace(0.0, 3.0, 121), np.zeros(121))), None
    if kind in {"LEFT ARC", "RIGHT ARC"}:
        angle = np.linspace(0.0, math.pi / 2.0, 121)
        side = 1.0 if kind == "LEFT ARC" else -1.0
        path = np.column_stack(
            (1.5 * np.sin(angle), side * 1.5 * (1.0 - np.cos(angle)))
        )
        return path, None
    if kind == "S CURVE":
        x = np.linspace(0.0, 5.0, 201)
        left = 0.55 * np.sin(2.0 * math.pi * x / 5.0) * np.sin(math.pi * x / 5.0)
        return np.column_stack((x, left)), None
    if kind not in DOOR_SCENARIOS:
        raise ValueError(f"Unknown trajectory scenario: {kind}")

    door_forward = float(passage["door_forward_m"])
    exit_distance = float(passage["path_exit_distance_m"])
    total_length = door_forward + exit_distance
    samples = 181
    if kind == "DOOR STRAIGHT":
        x = np.linspace(0.0, total_length, samples)
        path = np.column_stack((x, np.zeros(samples)))
        door = (np.asarray([door_forward, 0.0]), 0.0)
    elif kind == "DOOR OFFSET":
        x = np.linspace(0.0, total_length, samples)
        offset = float(passage["offset_left_m"])
        left = offset * _smoothstep((x - 0.20) / max(door_forward - 0.65, 0.30))
        path = np.column_stack((x, left))
        door = (np.asarray([door_forward, offset]), 0.0)
    else:
        yaw = math.radians(float(passage["angled_yaw_deg"]))
        arc = np.linspace(0.0, total_length, samples)
        alignment_length = max(0.50, door_forward - 0.65)
        heading = yaw * _smoothstep(arc / alignment_length)
        ds = np.diff(arc)
        x = np.r_[0.0, np.cumsum(np.cos(heading[1:]) * ds)]
        y = np.r_[0.0, np.cumsum(np.sin(heading[1:]) * ds)]
        path = np.column_stack((x, y))
        door_index = int(np.argmin(np.abs(arc - door_forward)))
        door = (path[door_index].copy(), yaw)
    return np.ascontiguousarray(path), door


def _transform_xy(points_xy: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return points @ np.asarray([[cosine, sine], [-sine, cosine]]) + [x, y]


def _author_passage(stage, door, settings: dict) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    root = "/World/NarrowPassage"
    stage.RemovePrim(root)
    if door is None:
        return
    UsdGeom.Xform.Define(stage, root)
    center, yaw = door
    passage = settings["passage"]
    gap = float(passage["gap_m"])
    thickness = float(passage["wall_thickness_m"])
    segment_length = float(passage["wall_segment_length_m"])
    height = float(passage["wall_height_m"])
    tangent = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    for index, side in enumerate((-1.0, 1.0)):
        block_center = center + tangent * side * (0.5 * gap + 0.5 * segment_length)
        cube = UsdGeom.Cube.Define(stage, f"{root}/Wall_{index}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(0.62, 0.30, 0.16)])
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(
            Gf.Vec3d(float(block_center[0]), float(block_center[1]), 0.5 * height)
        )
        xform.AddRotateZOp().Set(math.degrees(yaw))
        xform.AddScaleOp().Set(Gf.Vec3f(thickness, segment_length, height))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _author_path(stage, world_path: np.ndarray, z_m: float) -> None:
    from pxr import UsdGeom

    path = "/World/AcceptancePath"
    stage.RemovePrim(path)
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(world_path)])
    curve.CreatePointsAttr(
        [(float(x), float(y), float(z_m)) for x, y in world_path]
    )
    curve.CreateWidthsAttr([0.025])
    curve.CreateDisplayColorAttr([(0.10, 0.82, 0.22)])


def _degree_map(values: dict[str, float]) -> dict[str, float]:
    return {name: round(math.degrees(float(value)), 1) for name, value in values.items()}


def main() -> int:
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    settings = _load_settings()
    logger = logging.getLogger("agx_trajectory")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/agx_trajectory_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "active_gpu": config.simulation.active_gpu,
            "physics_gpu": config.simulation.physics_gpu,
            "multi_gpu": bool(config.simulation.multi_gpu),
        }
    )
    try:
        return _run(app, config, profile, settings, logger)
    finally:
        app.close()


def _run(app, config, profile, settings, logger) -> int:
    import omni.ui as ui
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import (
        add_reference_to_stage,
        get_current_stage,
        is_stage_loading,
    )
    from pxr import UsdGeom, UsdLux

    dt = float(profile.simulation.physics_dt)
    world = World(
        physics_dt=dt,
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/TestLight").CreateIntensityAttr(1500.0)
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=config.scene.stage_load_timeout_sec,
        logger=logger,
        label="AGX continuous and narrow-passage test",
    )
    spawn_position = np.asarray([0.0, 0.0, float(profile.spawn_z_m)], dtype=np.float32)
    spawn_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=spawn_position,
        rotation_deg=[0.0, 0.0, 0.0],
    )
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    _validate_geometry(stage, profile, articulation_path)
    robot = world.scene.add(
        SingleArticulation(
            prim_path=articulation_path,
            name="agx_trajectory",
            reset_xform_properties=False,
        )
    )
    observer_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=profile.sensors.observer_camera_prim_path,
        ),
        kind="AGX trajectory observer camera",
    )
    if not stage.GetPrimAtPath(observer_path).IsA(UsdGeom.Camera):
        raise ValueError(f"Observer prim is not a Camera: {observer_path}")

    world.reset()
    robot.initialize()
    initial_positions_value = robot.get_joint_positions()
    if initial_positions_value is None:
        raise RuntimeError("Unable to read initial AGX articulation joint positions")
    initial_joint_positions = np.asarray(
        initial_positions_value, dtype=np.float32
    ).reshape(-1).copy()
    joint_names = get_joint_names(robot)
    drive = create_locomotion_controller(
        profile=profile,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=joint_names,
        logger=logger,
    )
    adapter = AgxVelocityAdapter(drive, profile.agx_controller.continuous)
    side_clearance = 0.5 * (
        float(settings["passage"]["gap_m"])
        - 2.0 * float(settings["passage"]["robot_half_width_m"])
    )
    logger.info(
        "[PassageConfig] gap=%.3fm body_width=%.3fm nominal_side_clearance=%.3fm",
        float(settings["passage"]["gap_m"]),
        2.0 * float(settings["passage"]["robot_half_width_m"]),
        side_clearance,
    )
    bind_viewport_to_camera(camera_prim_path=observer_path, logger=logger, enabled=True)

    event: list[str | None] = [None]
    window = ui.Window("AGX Continuous / Narrow Passage", width=570, height=330)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Continuous tracking first, then known-path narrow-passage acceptance")
            with ui.HStack(height=40):
                for name in ("STRAIGHT", "LEFT ARC", "RIGHT ARC", "S CURVE"):
                    ui.Button(name, clicked_fn=lambda value=name: event.__setitem__(0, value))
            with ui.HStack(height=44):
                for name in ("DOOR STRAIGHT", "DOOR OFFSET", "DOOR ANGLED"):
                    ui.Button(name, clicked_fn=lambda value=name: event.__setitem__(0, value))
            with ui.HStack(height=40):
                ui.Button("RESET", clicked_fn=lambda: event.__setitem__(0, "RESET"))
                ui.Button("STOP", clicked_fn=lambda: event.__setitem__(0, "STOP"))
            status_label = ui.Label("Ready: choose a trajectory")
            steering_label = ui.Label("Steering: stopped")

    odom = SimOdometrySensor(logger=logger)
    tracker: PathTracker | None = None
    pending_scenario: str | None = None
    pending_start_time = float("inf")
    active_scenario: str | None = None
    status_text = "IDLE"
    previous_tracking_status = ""
    previous_adapter_status = ""
    last_steering_log_time = -float("inf")
    frame_index = 0
    render_stride = max(1, round(float(profile.simulation.rendering_dt) / dt))

    def reset_robot() -> None:
        nonlocal odom, tracker, active_scenario, previous_tracking_status
        adapter.stop()
        tracker = None
        active_scenario = None
        previous_tracking_status = ""
        robot.set_world_pose(position=spawn_position, orientation=spawn_orientation)
        robot.set_joint_positions(initial_joint_positions)
        robot.set_joint_velocities(np.zeros_like(initial_joint_positions))
        stage.RemovePrim("/World/AcceptancePath")
        stage.RemovePrim("/World/NarrowPassage")
        odom = SimOdometrySensor(logger=logger)

    try:
        while app.is_running():
            if not world.is_playing():
                adapter.stop()
                tracker = None
                pending_scenario = None
                event[0] = None
                status_label.text = "PAUSED: resume, then choose a trajectory"
                app.update()
                continue

            now = float(world.current_time)
            position, orientation = get_world_pose(robot)
            state = odom.update(
                world_position_xyz=position,
                world_orientation_wxyz=orientation,
                timestamp_sec=now,
            )

            if event[0] is not None:
                action, event[0] = event[0], None
                if action == "STOP":
                    adapter.stop()
                    tracker = None
                    pending_scenario = None
                    active_scenario = None
                    status_text = "STOPPED"
                    logger.info("[TrajectoryControl] STOP")
                else:
                    reset_robot()
                    if action == "RESET":
                        pending_scenario = None
                        status_text = "RESET COMPLETE"
                        logger.info("[TrajectoryControl] RESET")
                    else:
                        pending_scenario = action
                        pending_start_time = now + float(
                            settings["test"]["settle_after_reset_sec"]
                        )
                        status_text = f"RESETTING FOR {action}"
                        logger.info("[TrajectoryControl] reset_for=%s", action)

            if pending_scenario is not None and now >= pending_start_time:
                position, orientation = get_world_pose(robot)
                state = odom.update(
                    world_position_xyz=position,
                    world_orientation_wxyz=orientation,
                    timestamp_sec=now,
                )
                local_path, door = _path_and_door(pending_scenario, settings)
                path = _transform_xy(
                    local_path,
                    state.pose.x,
                    state.pose.y,
                    state.pose.yaw_rad,
                )
                world_path = _transform_xy(
                    path,
                    state.origin_world_xy[0],
                    state.origin_world_xy[1],
                    state.origin_world_yaw_rad,
                )
                world_door = None
                if door is not None:
                    door_odom = _transform_xy(
                        door[0][None, :],
                        state.pose.x,
                        state.pose.y,
                        state.pose.yaw_rad,
                    )
                    door_center = _transform_xy(
                        door_odom,
                        state.origin_world_xy[0],
                        state.origin_world_xy[1],
                        state.origin_world_yaw_rad,
                    )[0]
                    world_door = (
                        door_center,
                        door[1] + state.pose.yaw_rad + state.origin_world_yaw_rad,
                    )
                _author_passage(stage, world_door, settings)
                _author_path(
                    stage,
                    world_path,
                    float(settings["test"]["path_visualization_z_m"]),
                )
                tracker = PathTracker(path, profile.agx_controller.continuous)
                active_scenario = pending_scenario
                pending_scenario = None
                previous_tracking_status = ""
                status_text = f"TRACKING {active_scenario}"
                logger.info(
                    "[TrajectoryStart] scenario=%s length=%.3fm door=%s",
                    active_scenario,
                    float(tracker.s[-1]),
                    None if world_door is None else np.round(world_door[0], 3).tolist(),
                )

            if tracker is not None:
                result = tracker.step(
                    (state.pose.x, state.pose.y, state.pose.yaw_rad),
                    dt,
                )
                applied = adapter.apply(result.command)
                tracker.feedback(applied)
                status_text = (
                    f"{active_scenario}: {result.status}/{adapter.status}\n"
                    f"remaining={result.remaining_m:.2f}m  error={result.cross_track_m:.2f}m  "
                    f"v={applied.linear_mps:.3f}m/s  w={applied.angular_radps:.3f}rad/s"
                )
                if result.status != previous_tracking_status:
                    logger.info("[TrajectoryTracking] %s", status_text.replace("\n", " "))
                    previous_tracking_status = result.status
                diagnostics = adapter.steering_diagnostics()
                adapter_changed = adapter.status != previous_adapter_status
                periodic_transient = (
                    adapter.status == "STEERING_TRANSIENT"
                    and now - last_steering_log_time
                    >= float(settings["test"]["steering_log_period_sec"])
                )
                if adapter_changed or periodic_transient:
                    logger.info(
                        "[Steering] status=%s max_error_deg=%.2f target_deg=%s measured_deg=%s",
                        adapter.status,
                        math.degrees(float(diagnostics["max_error_rad"])),
                        _degree_map(diagnostics["target_rad"]),
                        _degree_map(diagnostics["measured_rad"]),
                    )
                    previous_adapter_status = adapter.status
                    last_steering_log_time = now
                steering_label.text = (
                    f"Steering: {adapter.status} | max error "
                    f"{math.degrees(float(diagnostics['max_error_rad'])):.1f} deg"
                )
                if result.status != "TRACKING":
                    adapter.stop()
                    tracker = None
                    logger.info(
                        "[TrajectoryEnd] scenario=%s result=%s path_error=%.3fm",
                        active_scenario,
                        result.status,
                        result.cross_track_m,
                    )
            else:
                adapter.stop()
                steering_label.text = "Steering: stopped / centering"

            if frame_index % render_stride == 0:
                status_label.text = status_text
            world.step(render=frame_index % render_stride == 0)
            frame_index += 1
    finally:
        adapter.stop()
        window.visible = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
