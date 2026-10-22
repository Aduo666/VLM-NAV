from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np
import shapely

from constraint_traverse.benchmark.run import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description="三层导航窄门配对验收")
    parser.add_argument("--paths", nargs="+", choices=("COARSE", "FINE"), default=["FINE"])
    parser.add_argument("--timeout", type=float, default=35)
    parser.add_argument("--offset", type=float, nargs="+", default=[0, 0.6])
    args = parser.parse_args()
    config = load_config(ROOT / "constraint_traverse/configs/benchmark.yaml")
    config["scene"]["door_clearance_m"] = 0.20
    logging.basicConfig(level=logging.INFO)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, **config["simulation"]})
    try:
        with tempfile.TemporaryDirectory(prefix="constraint_navigation_") as directory:
            return verify(app, config, Path(directory), args)
    finally:
        app.close()


def verify(app, config, output, args):
    from constraint_traverse.benchmark.runtime import PassageRuntime
    from constraint_traverse.navigation import SamplingNavigationController
    from constraint_traverse.reference_path import ReferencePath
    from vln_nav.app import navigation
    from vln_nav.navigation.fine_local_planner import FineLocalPlanner
    from vln_nav.navigation.odometry import SimOdometrySensor
    from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig, PlanningGrid, FootprintCollisionChecker
    from vln_nav.navigation.trajectory_refiner import TrajectoryRefiner
    from vln_nav.navigation.navdp_client import NavDPClient
    from vln_nav.runtime.isaac_runtime_utils import resolve_robot_relative_prim_path, set_camera_horizontal_fov
    from isaacsim.sensors.camera import Camera
    import omni.ui as ui

    loaded = navigation._load_settings()
    runtime = PassageRuntime(app, ROOT, config, output, True, [1.6, 0])
    runtime._initialize()
    resolution, size = 0.05, 120
    edges = -3 + np.arange(size + 1) * resolution
    rows, columns = np.indices((size, size))
    cells = shapely.box(edges[columns], edges[rows], edges[columns + 1], edges[rows + 1])
    occupied = shapely.intersects(cells, runtime.geometry.obstacle_geometry)
    grid = PlanningGrid(occupied, ~occupied, np.zeros_like(occupied), np.zeros_like(occupied, dtype=np.float32), -3, -3, resolution, 1)
    footprint = FootprintConfig(tuple(FootprintPartConfig(f"part_{i}", *part) for i, part in enumerate(runtime.geometry.parts)), 0.03)
    checker = FootprintCollisionChecker(footprint, yaw_resolution_deg=loaded[6].yaw_resolution_deg)
    camera = None
    if "COARSE" in args.paths:
        camera_config = loaded[0]["camera"]
        camera_path = resolve_robot_relative_prim_path(robot_root_path=runtime.profile.asset.prim_path,
            prim_path=runtime.profile.sensors.camera.front_prim_path)
        set_camera_horizontal_fov(stage=runtime.stage, prim_path=camera_path, fov_deg=camera_config["horizontal_fov_deg"])
        camera = Camera(prim_path=camera_path, name="validation_front", frequency=camera_config["frequency_hz"],
                        resolution=(camera_config["width"], camera_config["height"]))
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        for _ in range(30):
            runtime.world.step(render=True)
    results = []
    for offset in args.offset:
        for path_source in args.paths:
            for method in ("MPPI", "KMPPI"):
                if runtime.snapshot is not None:
                    runtime._reset()
                position, _ = runtime.robot.get_world_pose()
                position[:2] = [-1.6, offset]
                runtime.robot.set_world_pose(position=position, orientation=np.asarray([1, 0, 0, 0]))
                for _ in range(60):
                    runtime._apply([0, 0])
                    runtime.world.step(render=camera is not None)
                runtime.goal = np.asarray([1.6, offset])
                if not runtime._start(method, "goal"):
                    raise RuntimeError(runtime.result_text)
                controller = SamplingNavigationController(method=method, hardware=runtime.profile.agx_controller.continuous,
                    footprint=footprint, physics_dt=runtime.dt, config_path=ROOT / "constraint_traverse/configs/benchmark.yaml", logger=runtime.logger)
                worker = (navigation._FinePlanningWorker(planner=FineLocalPlanner(loaded[6], checker),
                          restricted_analyzer=None, trajectory_refiner=TrajectoryRefiner(loaded[10], checker))
                          if path_source == "FINE" else navigation._NavDPWorker(NavDPClient(loaded[1])))
                reference = None
                sensor = SimOdometrySensor(logger=runtime.logger)
                start_time = runtime.world.current_time
                last_request = -float("inf")
                request = cycles = 0
                previous = None
                reason = "TIMEOUT"
                while app.is_running() and runtime.world.current_time - start_time < args.timeout:
                    wall_start = time.perf_counter()
                    now = float(runtime.world.current_time)
                    state = controller.read_state(runtime.robot, runtime.drive)
                    pos, quat = runtime.robot.get_world_pose()
                    odom = sensor.update(world_position_xyz=pos, world_orientation_wxyz=quat, timestamp_sec=now)
                    completed = worker.poll()
                    if completed is not None:
                        if path_source == "FINE":
                            job, planned_grid, outcome, error = completed
                            if error is None and outcome.plan is not None:
                                plan = outcome.plan
                                reference = ReferencePath(plan.path_world_xy, plan.path_world_yaw_rad, plan.path_motion_direction,
                                    runtime.goal, "FINE", job.request_id, job.submitted_sim_time_sec, planned_grid.map_version, plan.partial)
                            elif error is not None:
                                runtime.logger.warning("[ValidationPath] %s", error)
                        else:
                            job, result, error = completed
                            if error is None:
                                chosen = navigation._freeze_candidates(result, job)[result.model_choice_index]
                                points = navigation._odom_path_to_world(chosen.odom_path_xy, odom)
                                if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 1.0e-4:
                                    reference = ReferencePath.from_positions(points, initial_yaw_rad=float(state[2]),
                                        goal_world_xy=runtime.goal, source="COARSE", request_id=job.request_id,
                                        created_sim_time_sec=job.odom_state.timestamp_sec, map_version=1, partial=True)
                            elif error is not None:
                                runtime.logger.warning("[ValidationPath] %s", error)
                    needs_path = (path_source == "COARSE" or reference is None
                                  or reference.needs_extension(state[:3], lookahead_m=1.2))
                    if needs_path and not worker.busy and now - last_request >= 0.5:
                        request += 1
                        if path_source == "FINE":
                            job = navigation._FinePlanningJob(request, grid, tuple(state[:3]), tuple(runtime.goal), None, False,
                                1.2, now, 0., initial_speed_mps=float(state[3]))
                        else:
                            rgb, depth = navigation._capture_rgbd(camera)
                            forward, left = navigation._odom_goal_to_local(navigation._world_path_to_odom(runtime.goal[None], odom)[0], odom)
                            job = navigation._InferenceJob(request, np.asarray(camera.get_intrinsics_matrix()), rgb, depth, forward, left, odom)
                        if worker.submit(job):
                            last_request = now
                    if reference is None:
                        runtime._apply([0, 0])
                    else:
                        controller.step(robot=runtime.robot, drive=runtime.drive, grid=grid, goal_world_xy=runtime.goal,
                                        now=now, reference_path=reference)
                        if controller.last_plan is not None and controller.last_plan is not previous:
                            previous = controller.last_plan
                            cycles += 1
                    if runtime.contacts:
                        reason = "COLLISION"
                        break
                    distance = float(np.linalg.norm(state[:2] - runtime.goal))
                    if (distance <= 0.30 and runtime.geometry.completely_passed(state[:3])
                            and np.linalg.norm(state[3:5]) < 0.025 and abs(state[5]) < 0.04):
                        reason = "SUCCESS"
                        break
                    runtime.world.step(render=camera is not None and runtime.world.current_time_step_index % 2 == 0)
                    time.sleep(max(0., runtime.dt - (time.perf_counter() - wall_start)))
                final = controller.read_state(runtime.robot, runtime.drive)
                result = dict(path_source=path_source, controller=method, offset_m=offset, result=reason,
                              duration_sec=float(runtime.world.current_time - start_time),
                              goal_distance_m=float(np.linalg.norm(final[:2] - runtime.goal)),
                              complete_body_passed=runtime.geometry.completely_passed(final[:3]),
                              collisions=len(runtime.contacts), planning_cycles=cycles)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                results.append(result)
                controller.close(wait=True)
                runtime._finish(reason)
    events = []
    panel = navigation._GoalMapPanel(ui, image_size_px=128, command_sink=events.append, bind_keyboard=False)
    for combo in (panel._mode_combo, panel._controller_combo):
        combo.model.get_item_value_model().set_value(1)
        combo.model.get_item_value_model().set_value(0)
    before = len(events)
    panel.update_controls("FINE", "TEST", "KMPPI")
    panel.shutdown()
    panel_passed = (all(event in events for event in (("SET_MODE", "COARSE"), ("SET_MODE", "FINE"),
                    ("SET_CONTROLLER", "MPPI"), ("SET_CONTROLLER", "KMPPI"))) and len(events) == before)
    print(json.dumps(dict(panel_passed=panel_passed, results=results), ensure_ascii=False), flush=True)
    runtime.subscription = None
    return 0 if panel_passed and all(row["result"] == "SUCCESS" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
