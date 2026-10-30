from __future__ import annotations

import math
import time

import numpy as np
import torch

from constraint_traverse.benchmark.geometry import DoorScene, PassageGeometry
from constraint_traverse.reference_path import angle_difference

from ..recording import inspect_record
from . import snapshot


def place(runtime, x, y, yaw):
    runtime.stop_motion()
    runtime.robot.set_world_pose(position=np.asarray([x, y, runtime.profile.spawn_z_m]),
        orientation=np.asarray([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    runtime.robot.set_linear_velocity(np.zeros(3))
    runtime.robot.set_angular_velocity(np.zeros(3))
    runtime.world.physics_sim_view.update_articulations_kinematic()
    for _ in range(12):
        runtime.step([0, 0])


def verify_motion(runtime):
    results = {}
    runtime.goal = np.asarray([1.5, 0, 0])
    saved = runtime.capture_snapshot()
    sampling = runtime.navigator.sampling
    for name, command in (("forward", [0.3, 0]), ("reverse", [-0.3, 0]),
                          ("turn", [0.3, 0.25]), ("spin", [0, 0.3]), ("brake", [0, 0])):
        snapshot.restore(runtime, saved)
        runtime.world.play()
        if name == "brake":
            for _ in range(90):
                runtime.step([0.3, 0])
        initial = sampling.read_state(runtime.robot, runtime.drive)
        predicted = torch.as_tensor(initial)
        checks = []
        dt = runtime.world.get_physics_dt()
        for index in range(round(4 / dt)):
            predicted = sampling.execution_model.step(predicted, command)
            runtime.step(command)
            if index + 1 in {round(0.2 / dt), round(1 / dt), round(4 / dt)}:
                actual = sampling.read_state(runtime.robot, runtime.drive)
                checks.append(dict(duration_sec=(index + 1) * dt,
                    position_error_m=float(np.linalg.norm(actual[:2] - predicted[:2].numpy())),
                    yaw_error_rad=abs(float(angle_difference(actual[2], float(predicted[2])))),
                    actual_speed_mps=float(np.linalg.norm(actual[3:5])), actual_angular_radps=float(actual[5])))
        results[name] = checks
        if checks[0]["position_error_m"] > 0.02 or checks[0]["yaw_error_rad"] > math.radians(3):
            raise AssertionError(f"一个控制周期的运动预测误差超出验收范围：{name}，{checks}")
        displacement = actual[:2] - initial[:2]
        if name in {"forward", "reverse"} and displacement[0] * command[0] <= 0:
            raise AssertionError(f"实际行驶方向与控制方向不一致：{name}")
        if name in {"turn", "spin"} and angle_difference(actual[2], initial[2]) < 0.1:
            raise AssertionError(f"实际转向没有产生预期朝向变化：{name}")
        if name == "brake" and checks[-1]["actual_speed_mps"] > 0.025:
            raise AssertionError("底盘制动后没有停稳")
        runtime.logger.info("[MotionAcceptance] %s %s", name, checks)
    snapshot.restore(runtime, saved)
    runtime.world.play()
    return results


def observe_environment(runtime):
    # 在已知测试场景中手动放置并采集真实 RGB-D，规划器只接收这些观测形成的地图。
    runtime.reset_map()
    observations = 0
    for x in (-2.8, -1.6, 1.6, 2.8):
        for y in (-1.0, 0.0, 1.0):
            for yaw in (0.0, math.pi):
                place(runtime, x, y, yaw)
                for _ in range(3):
                    runtime.observations = runtime.sensors.capture()
                    runtime.update_map(runtime.observations)
                    observations += len(runtime.observations)
    place(runtime, -1.6, 0, 0)
    runtime.observations = runtime.sensors.capture()
    runtime.update_map(runtime.observations)
    runtime.logger.info("[MapAcceptance] actual_rgbd_frames=%d map_version=%d", observations, runtime.local_map.version)
    return observations


def run_trials(runtime, label, geometry=None):
    runtime.goal = np.asarray([1.6, 0, 0])
    initial = runtime.experiment.save_start()
    results = []
    for index in range(runtime.args.repeats):
        runtime.experiment.saved = initial
        runtime.experiment.reset()
        runtime.start_navigation()
        started = time.monotonic()
        minimum_clearance = math.inf
        while runtime.experiment.recorder is not None:
            runtime.step()
            if geometry is not None:
                pose = runtime.navigator.sampling.read_state(runtime.robot, runtime.drive)[:3]
                minimum_clearance = min(minimum_clearance, geometry.clearance(pose))
            if runtime.panel is not None:
                runtime.panel.update(runtime)
            if time.monotonic() - started > 240:
                runtime.experiment.stop("WALL_TIMEOUT")
        record = inspect_record(runtime.experiment.last_record)
        pose = runtime.navigator.sampling.read_state(runtime.robot, runtime.drive)[:3]
        passed = geometry is None or geometry.completely_passed(pose)
        result = dict(status=record["status"], goal_distance_m=float(np.linalg.norm(pose[:2] - runtime.goal[:2])),
            completely_passed=passed, minimum_clearance_m=None if geometry is None else minimum_clearance,
            records=record["verified_counts"], wall_duration_sec=time.monotonic() - started)
        results.append(result)
        runtime.logger.info("[NavigationAcceptance] scene=%s repeat=%d result=%s", label, index + 1, result)
        if record["status"] != "SUCCESS" or not passed:
            raise AssertionError(f"自动通行验收没有通过：{label}，{result}")
    return results


def verify_collision_record(runtime):
    place(runtime, -1.0, 1.5, 0)
    runtime.goal = np.asarray([1.6, 1.5, 0])
    path = runtime.experiment.start()
    for _ in range(240):
        runtime.step([0.3, 0])
        if runtime.experiment.recorder is None:
            break
    if runtime.experiment.recorder is not None:
        runtime.experiment.stop("CONTACT_TEST_FAILED")
    record = inspect_record(path)
    if record["status"] != "COLLISION":
        raise AssertionError(f"物理墙体接触没有保存碰撞终止：{record['status']}")
    return dict(status=record["status"], records=record["verified_counts"])


def verify_navigation(runtime):
    from pxr import Gf, UsdGeom, UsdPhysics

    runtime.confirm_placement()
    for _ in range(60):
        runtime.step([0, 0])
    motion = verify_motion(runtime) if runtime.args.checks in {"all", "motion"} else None
    if runtime.args.checks == "motion":
        return dict(motion_prediction=motion)
    open_frames = opened = None
    if runtime.args.checks != "contacts":
        runtime.experiment.metadata["test_scene"] = dict(name="open_floor", map_initialization="REAL_RGBD_MULTIVIEW_MANUAL_PLACEMENT")
        open_frames = observe_environment(runtime)
        opened = run_trials(runtime, "open_floor")
    footprint = runtime.navigator.sampling.footprint
    parts = np.asarray([[part.min_x_m, part.max_x_m, part.min_y_m, part.max_y_m] for part in footprint.parts])
    width = float(parts[:, 3].max() - parts[:, 2].min())
    scene = DoorScene(width + runtime.args.door_clearance)
    for index, bounds in enumerate(scene.obstacles):
        cube = UsdGeom.Cube.Define(runtime.stage, f"/World/Environment/AcceptanceDoor/Wall{index}")
        cube.CreateSizeAttr(1)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.65, 0.7)])
        transform = UsdGeom.Xformable(cube)
        transform.AddTranslateOp().Set(Gf.Vec3d((bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2, scene.height_m / 2))
        transform.AddScaleOp().Set(Gf.Vec3d(bounds[1] - bounds[0], bounds[3] - bounds[2], scene.height_m))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    runtime.world.reset()
    runtime.robot.initialize()
    runtime.create_drive()
    runtime.navigator.reset()
    runtime.world.play()
    geometry = PassageGeometry(parts, scene, margin_m=footprint.margin_m, sweep_step_m=0.01)
    runtime.experiment.metadata["test_scene"] = dict(name="wide_door", door_width_m=scene.width_m,
        door_thickness_m=scene.thickness_m, wall_bounds=scene.obstacles.tolist(),
        map_initialization="REAL_RGBD_MULTIVIEW_MANUAL_PLACEMENT")
    if runtime.args.checks == "contacts":
        return dict(physical_collision_record=verify_collision_record(runtime))
    door_frames = observe_environment(runtime)
    door = run_trials(runtime, "wide_door", geometry)
    contact = verify_collision_record(runtime)
    return dict(motion_prediction=motion, open_floor=opened, wide_door=door,
        body_width_m=width, door_width_m=scene.width_m, total_clearance_m=runtime.args.door_clearance,
        map_initialization="REAL_RGBD_MULTIVIEW_MANUAL_PLACEMENT",
        preobserved_rgbd_frames=dict(open_floor=open_frames, wide_door=door_frames), unknown_policy="blocked",
        physical_collision_record=contact)
