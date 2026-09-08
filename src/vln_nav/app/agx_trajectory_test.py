"""AGX continuous-control acceptance on an empty floor; no VLM or navigation."""
from __future__ import annotations

import logging
import math
import sys
import time

import numpy as np

from vln_nav.app.nav import (
    _bind_viewport_to_camera, _get_joint_names, _get_world_pose,
    _resolve_robot_relative_prim_path, _set_local_xform, _wait_for_stage_assets,
)
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import AgxVelocityAdapter, PathTracker, VelocityCommand
from vln_nav.app.agx_safety_test_support import (
    SafetyTestSensors, configure_obstacles, load_test_settings, save_safety_debug,
)
from vln_nav.navigation.odometry import SimOdometrySensor
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from vln_nav.runtime.robot_profile import load_robot_profile


def _validate_geometry(stage, profile, articulation_path):
    """Check nominal URDF geometry against the loaded USD before allowing motion."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(profile.asset.prim_path)
    by_name = {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            if prim.GetName() in by_name:
                raise ValueError(f"Ambiguous joint name: {prim.GetName()}")
            by_name[prim.GetName()] = prim
    chassis_inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(articulation_path)).GetInverse()
    for wheel in profile.agx_controller.continuous.wheels:
        for name, expected_axis in (
            (wheel.steering_joint, (0, 0, wheel.steering_sign)),
            (wheel.drive_joint, (0, wheel.drive_sign, 0)),
        ):
            joint = UsdPhysics.RevoluteJoint(by_name[name])
            body = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            body_to_world = cache.GetLocalToWorldTransform(body)
            basis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[str(joint.GetAxisAttr().Get())]
            local_rotation = joint.GetLocalRot0Attr().Get()
            rotation = Gf.Rotation(Gf.Quatd(float(local_rotation.GetReal()),
                                           Gf.Vec3d(*local_rotation.GetImaginary())))
            axis = chassis_inverse.TransformDir(body_to_world.TransformDir(rotation.TransformDir(Gf.Vec3d(*basis))))
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            if np.linalg.norm(axis - expected_axis) > 0.05:
                raise ValueError(f"USD joint {name} axis={axis}, expected={expected_axis}; check AGX continuous signs")
            if name == wheel.steering_joint:
                position = chassis_inverse.Transform(body_to_world.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get())))
                if np.linalg.norm(np.asarray(position)[:2] - [wheel.x_m, wheel.y_m]) > 0.04:
                    raise ValueError(f"USD joint {name} XY={tuple(position)[:2]} disagrees with continuous wheel geometry")
                lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
                limit = math.degrees(profile.agx_controller.continuous.max_steering_rad)
                if lower > -limit or upper < limit:
                    raise ValueError(f"USD steering limits for {name} do not support +/-{limit:.1f} degrees")


def _make_path(pose, kind):
    kind = {"BLOCKED STRAIGHT": "STRAIGHT", "INNER CURVE": "LEFT ARC",
            "NARROW DOOR": "STRAIGHT"}.get(kind, kind)
    if kind == "STRAIGHT":
        local = np.column_stack((np.linspace(0, 3.0, 100), np.zeros(100)))
    elif kind == "IRREGULAR":
        # Unequal lane changes with continuous slope and curvature, not sharp
        # polyline corners. End straight sections test entry and final stopping.
        anchors = np.array([[0, 0], [.8, 0], [2.6, .7], [4.8, -.6],
                            [6.8, .4], [8.0, .4]])
        segments = []
        for start, end in zip(anchors[:-1], anchors[1:]):
            count = max(2, int(math.ceil((end[0] - start[0]) / .025)))
            t = np.linspace(0, 1, count, endpoint=False)
            blend = t ** 3 * (10 - 15 * t + 6 * t ** 2)
            segments.append(np.column_stack((start[0] + t * (end[0] - start[0]),
                                             start[1] + blend * (end[1] - start[1]))))
        local = np.vstack([*segments, anchors[-1:]])
    elif kind in {"LEFT ARC", "RIGHT ARC"}:
        angle = np.linspace(0, math.pi / 2, 100)
        side = 1 if kind == "LEFT ARC" else -1
        local = np.column_stack((1.5 * np.sin(angle), side * 1.5 * (1 - np.cos(angle))))
    else:
        raise ValueError(f"Unknown acceptance path: {kind}")
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    return local @ np.array([[c, s], [-s, c]]) + [pose.x, pose.y]


def main() -> int:
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    logger = logging.getLogger("agx_trajectory")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    log_path = PROJECT_ROOT / "logs/agx_trajectory_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(handler.formatter)
    logger.addHandler(file_handler)
    logger.info("[Trajectory] log=%s", log_path)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False, "active_gpu": config.simulation.active_gpu,
                         "physics_gpu": config.simulation.physics_gpu,
                         "multi_gpu": bool(config.simulation.multi_gpu)})
    try:
        return _run(app, config, profile, logger)
    finally:
        app.close()


def _run(app, config, profile, logger):
    import omni.ui as ui
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux

    dt = profile.simulation.physics_dt
    test_settings = load_test_settings(PROJECT_ROOT)
    world = World(physics_dt=dt, rendering_dt=profile.simulation.rendering_dt, stage_units_in_meters=1.0)
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/TestLight").CreateIntensityAttr(1500.0)
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    _wait_for_stage_assets(simulation_app=app, is_stage_loading=is_stage_loading,
                          timeout_sec=config.scene.stage_load_timeout_sec, logger=logger, label="AGX trajectory test")
    _set_local_xform(stage=stage, prim_path=profile.asset.prim_path,
                     translation=[0, 0, profile.spawn_z_m], rotation_deg=[0, 0, 0])
    articulation_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path, prim_path=profile.asset.articulation_path)
    _validate_geometry(stage, profile, articulation_path)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name="agx_trajectory",
                                             reset_xform_properties=False))
    camera_path = _resolve_robot_relative_prim_path(robot_root_path=profile.asset.prim_path,
                                                   prim_path=profile.sensors.camera.front_prim_path)
    if not stage.GetPrimAtPath(camera_path).IsA(UsdGeom.Camera):
        raise ValueError(f"Front RGB-D camera missing: {camera_path}")
    camera_config = test_settings["camera"]
    camera = Camera(prim_path=camera_path, name="agx_safety_rgbd",
                    frequency=int(camera_config["frequency_hz"]),
                    resolution=(int(camera_config["width"]), int(camera_config["height"])))
    near_config = test_settings["near_camera"]
    near_path = _resolve_robot_relative_prim_path(robot_root_path=profile.asset.prim_path,
                                                 prim_path=near_config["prim_path"])
    near_prim = stage.GetPrimAtPath(near_path)
    authored_near_camera = near_prim.IsValid()
    if authored_near_camera and not near_prim.IsA(UsdGeom.Camera):
        raise ValueError(f"Near-field prim is not a Camera: {near_path}")
    if not stage.GetPrimAtPath(near_path.rsplit("/", 1)[0]).IsValid():
        raise ValueError(f"Near-field camera parent missing: {near_path}")
    if near_path == camera_path:
        raise ValueError("Near-field camera must not replace the front camera")
    near_camera = Camera(prim_path=near_path, name="agx_safety_near_rgbd",
                         frequency=int(camera_config["frequency_hz"]),
                         resolution=(int(camera_config["width"]), int(camera_config["height"])))
    if not authored_near_camera:
        pitch = math.radians(float(near_config["pitch_down_deg"]))
        near_camera.set_local_pose(translation=np.array(near_config["position_m"], dtype=float),
            orientation=np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]), camera_axes="world")
    world.reset()
    robot.initialize()
    for sensor_camera in (camera, near_camera):
        sensor_camera.initialize()
        sensor_camera.add_distance_to_image_plane_to_frame()
        sensor_camera.add_instance_id_segmentation_to_frame()
    if not authored_near_camera:
        aperture = 2 * near_camera.get_focal_length() * math.tan(math.radians(near_config["horizontal_fov_deg"]) / 2)
        near_camera.set_horizontal_aperture(aperture)
    logger.info("[SafetyTest] near_camera=%s source=%s; front camera unchanged", near_path,
                "existing USD camera" if authored_near_camera else "temporary test Stage camera")
    drive = create_locomotion_controller(profile=profile, robot=robot, stage=stage,
        articulation_path=articulation_path, available_joint_names=_get_joint_names(robot), logger=logger)
    adapter = AgxVelocityAdapter(drive, profile.agx_controller.continuous)
    observer = _resolve_robot_relative_prim_path(robot_root_path=profile.asset.prim_path,
                                                prim_path=profile.sensors.observer_camera_prim_path)
    if not stage.GetPrimAtPath(observer).IsA(UsdGeom.Camera):
        raise ValueError(f"Observer camera missing: {observer}")
    _bind_viewport_to_camera(camera_prim_path=observer, logger=logger, enabled=True)
    odom = SimOdometrySensor(logger=logger)
    sensors = SafetyTestSensors(test_settings, logger, dt, required_sources=("front", "near"),
                                robot_root=profile.asset.prim_path)
    debug_path = PROJECT_ROOT / test_settings["debug"]["output_path"]
    logger.info("[Trajectory] RGB-D safety enabled; debug=%s", debug_path)
    request = [None]
    window = ui.Window("AGX Continuous Trajectory Test", width=560, height=400)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("RGB-D safety acceptance. HOLD brakes; no automatic detour.")
            with ui.HStack(height=40):
                for kind in ("STRAIGHT", "LEFT ARC", "RIGHT ARC"):
                    ui.Button(kind, clicked_fn=lambda k=kind: request.__setitem__(0, k))
            ui.Button("IRREGULAR - mixed left/right bends", height=40,
                      clicked_fn=lambda: request.__setitem__(0, "IRREGULAR"))
            with ui.HStack(height=40):
                for kind in ("BLOCKED STRAIGHT", "INNER CURVE", "NARROW DOOR"):
                    ui.Button(kind, clicked_fn=lambda k=kind: request.__setitem__(0, k))
            def toggle_depth():
                sensors.frozen = not sensors.frozen
                freeze_button.text = "RESUME RGB-D" if sensors.frozen else "FREEZE RGB-D"
                logger.info("[SafetyTest] RGB-D freeze=%s", sensors.frozen)
            freeze_button = ui.Button("FREEZE RGB-D", height=32, clicked_fn=toggle_depth)
            ui.Button("STOP / CANCEL", height=40, clicked_fn=lambda: request.__setitem__(0, "STOP"))
            status = ui.Label("Ready: select a path after settling")
            telemetry = ui.Label("Actual speed: 0 m/s; yaw: 0 rad/s")
            safety_label = ui.Label("Safety: waiting for RGB-D")
    tracker = None
    sim_time, frame = 0.0, 0
    render_stride = max(1, round(profile.simulation.rendering_dt / dt))
    previous_status = ""
    idle_label = "IDLE"
    reference_world_path = None
    last_debug_time = -float("inf")
    last_safety_key, last_safety_log = None, -float("inf")
    decision = None
    pause_invalidated = False
    # Start idle, never start moving just because the simulator was opened.
    try:
        while app.is_running():
            if not world.is_playing():
                adapter.stop()
                tracker = None
                request[0] = None
                idle_label = "PAUSED: select a new path after resuming"
                status.text = idle_label
                if not pause_invalidated:
                    sensors.reset((0, 0, 0), float(world.current_time))
                    pause_invalidated = True
                app.update()
                continue
            # Use the simulator's clock, also used by camera rendering_time.
            sim_time = float(world.current_time)
            position, orientation = _get_world_pose(robot)
            state = odom.update(world_position_xyz=position, world_orientation_wxyz=orientation,
                                timestamp_sec=sim_time)
            world_pose = (float(position[0]), float(position[1]),
                          state.pose.yaw_rad + state.origin_world_yaw_rad)
            if pause_invalidated:
                sensors.reset(world_pose, sim_time)
                pause_invalidated = False
            if request[0] is not None:
                kind, request[0] = request[0], None
                was_tracking = tracker is not None
                adapter.stop()
                tracker = None
                idle_label = "CANCELLED" if kind == "STOP" else "WAIT: select again after the robot stops"
                stopped = state.linear_velocity_mps < 0.03 and abs(state.angular_velocity_radps) < 0.05
                if kind != "STOP" and not was_tracking and stopped and sim_time >= 1.0:
                    path = _make_path(state.pose, kind)
                    tracker = PathTracker(path, profile.agx_controller.continuous)
                    # Display in world coordinates, not in the odom-relative frame.
                    c, s = math.cos(state.origin_world_yaw_rad), math.sin(state.origin_world_yaw_rad)
                    world_path = path @ np.array([[c, s], [-s, c]]) + state.origin_world_xy
                    configure_obstacles(stage, kind, world_pose)
                    sensors.reset(world_pose, sim_time)
                    reference_world_path = world_path
                    curve = UsdGeom.BasisCurves.Define(stage, "/World/AcceptancePath")
                    curve.CreateTypeAttr("linear")
                    curve.CreateCurveVertexCountsAttr([len(path)])
                    curve.CreatePointsAttr([(float(x), float(y), 0.025) for x, y in world_path])
                    curve.CreateWidthsAttr([0.025])
                    curve.CreateDisplayColorAttr([(0.1, 0.8, 0.2)])
                    logger.info("[Trajectory] start=%s length=%.2fm", kind, tracker.s[-1])
            label = idle_label
            if tracker is not None:
                result = tracker.step((state.pose.x, state.pose.y, state.pose.yaw_rad), dt)
                decision = sensors.supervisor.evaluate(result.command, pose=world_pose,
                    measured=VelocityCommand(state.linear_velocity_mps, state.angular_velocity_radps),
                    now=sim_time, wall_now=time.monotonic(), probe_command=result.nominal_command)
                applied = adapter.apply(decision.command)
                tracker.feedback(applied)
                if decision.status == "HOLD" and result.status == "TRACKING":
                    # Safety waiting is not movement execution; keep the same
                    # reference path, without silently timing it out while held.
                    tracker.elapsed = max(0.0, tracker.elapsed - dt)
                label = (f"{result.status}/{adapter.status} remaining={result.remaining_m:.2f}m "
                         f"error={result.cross_track_m:.2f}m\n"
                         f"Command: v={applied.linear_mps:.3f}m/s w={applied.angular_radps:.3f}rad/s")
                if result.status != "TRACKING":
                    logger.info("[Trajectory] %s", label)
                    idle_label = label
                    tracker = None
            else:
                adapter.stop()
                decision = sensors.supervisor.evaluate(VelocityCommand(), pose=world_pose,
                    measured=VelocityCommand(state.linear_velocity_mps, state.angular_velocity_radps),
                    now=sim_time, wall_now=time.monotonic())
            safety_key = (decision.status, decision.reason)
            if safety_key != last_safety_key or (
                decision.status == "HOLD" and sim_time - last_safety_log >= sensors.config.hold_log_period_sec
            ):
                logger.info("[TrajectorySafety] %s reason=%s cap_scale=%.2f age=%.3fs sensor=%s blocker_xy=%s",
                            decision.status, decision.reason, decision.scale, decision.sensor_age_sec, sensors.last_reason,
                            decision.blocker_world_xy)
                last_safety_key, last_safety_log = safety_key, sim_time
            if label.split()[0] != previous_status:
                logger.info("[Trajectory] %s", label)
                previous_status = label.split()[0]
            if frame % render_stride == 0:
                status.text = label
                telemetry.text = f"Actual speed: {state.linear_velocity_mps:.3f} m/s; yaw: {state.angular_velocity_radps:.3f} rad/s"
                safety_label.text = (f"Safety: {decision.status} / {decision.reason}\n"
                                     f"RGB-D age={decision.sensor_age_sec:.2f}s | scale={decision.scale:.2f} | {sensors.last_reason}")
            world.step(render=frame % render_stride == 0)
            if frame % render_stride == 0:
                new_position, new_orientation = _get_world_pose(robot)
                qw, qx, qy, qz = map(float, new_orientation)
                new_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                sensors.consume(camera, now=float(world.current_time),
                                pose=(float(new_position[0]), float(new_position[1]), new_yaw))
                sensors.consume(near_camera, now=float(world.current_time), source="near",
                                pose=(float(new_position[0]), float(new_position[1]), new_yaw))
            if sim_time - last_debug_time >= float(test_settings["debug"]["period_sec"]):
                save_safety_debug(debug_path, sensors, pose=world_pose, reference_path=reference_world_path,
                                  decision=decision, status=f"{decision.status}: {decision.reason}")
                last_debug_time = sim_time
            frame += 1
    finally:
        adapter.stop()
        window.visible = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
