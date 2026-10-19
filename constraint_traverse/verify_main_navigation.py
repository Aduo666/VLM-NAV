from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class NavigationMeasurements(logging.Handler):
    def __init__(self):
        super().__init__()
        self.reached = 0
        self.s2_requests = 0
        self.s2_after_reached = False

    def emit(self, record):
        message = record.getMessage()
        if "[NavigationGoalReached]" in message:
            self.reached += 1
        if "[S2Request]" in message:
            self.s2_requests += 1
            self.s2_after_reached |= self.reached > 0


def main():
    parser = argparse.ArgumentParser(description="主导航室内 RGB-D 与局部目标闭环验收")
    parser.add_argument("--method", choices=("MPPI", "KMPPI", "COARSE"), default="MPPI")
    parser.add_argument("--episode", default="kujiale_0067_bedroom_001")
    parser.add_argument("--distance", type=float, help="单独测试前方指定距离的局部目标；省略时使用真实 S2")
    parser.add_argument("--duration", type=float, default=45.0)
    args = parser.parse_args()
    from vln_nav.app import navigation
    from vln_nav.runtime.episode import _parse_episode
    profile = navigation.load_agx_profile(project_root=ROOT)
    loaded = navigation._load_settings()
    settings, client, s2client, goal_adapter, maps, map_config, fine, footprint, corridor, graph, refiner = loaded
    settings["execution"]["default_mode"] = args.method
    if args.distance is not None:
        if not math.isfinite(args.distance) or args.distance <= 0:
            parser.error("测试目标距离必须为正数")
        settings["internnav_s2"]["enabled"] = False
        settings["goal"]["min_distance_m"] = 0.01
        settings["execution"]["goal_reached_tolerance_m"] = settings["internnav_s2"]["goal_reached_tolerance_m"]
    with navigation.EPISODES_PATH.open(encoding="utf-8") as handle:
        row = next(item for item in json.load(handle)["episodes"] if item["id"] == args.episode)
    episode = _parse_episode(row, project_root=ROOT)
    output = ROOT / "constraint_traverse/data/main_navigation_validation"
    output.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.method.lower()
    logger = logging.getLogger("constraint_traverse.main_validation")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(output / f"{stem}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    measurements = NavigationMeasurements()
    logger.addHandler(measurements)
    frames, plans, latencies, violations = [], [], [], []
    last_plan = None
    sent_goal = False
    initial_position = final_position = None
    target_world = None
    start_sim = None

    def observe(*, world, robot, state, command, grid, controller, navigation_active):
        nonlocal start_sim, sent_goal, initial_position, final_position, target_world, last_plan
        now = float(world.current_time)
        start_sim = now if start_sim is None else start_sim
        frames.append((now, time.perf_counter()))
        position, _ = robot.get_world_pose()
        final_position = np.asarray(position[:2]).copy()
        if initial_position is None:
            initial_position = final_position.copy()
        if args.distance is not None and not sent_goal and grid is not None and grid.map_version >= 2:
            yaw = state.pose.yaw_rad
            target_world = final_position + args.distance * np.asarray([math.cos(yaw), math.sin(yaw)])
            command[0] = ("SELECT_GOAL", args.distance, 0.0)
            sent_goal = True
        if controller is not None and controller.last_plan is not None and controller.last_plan is not last_plan:
            last_plan = controller.last_plan
            plans.append(last_plan.diagnostics)
            latencies.append(last_plan.planning_sec)
            violations.append(last_plan.diagnostics["candidate_motion_violation_fraction"])
        return (measurements.reached > 0 if args.distance is not None else measurements.s2_after_reached) or now - start_sim >= args.duration

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, **settings["simulation"]})
    try:
        navigation._run(app, profile, episode, settings, client, s2client, goal_adapter, maps,
                        map_config, fine, footprint, corridor, graph, refiner, logger, step_observer=observe)
    finally:
        app.close()
    frame_gaps = np.diff(np.asarray(frames)[:, 1]) if len(frames) > 1 else np.asarray([])
    report = {"method": args.method, "episode": args.episode,
              "condition": "live_s2" if args.distance is None else "local_goal_actual_rgbd",
              "requested_distance_m": args.distance, "goals_reached": measurements.reached,
              "s2_requests": measurements.s2_requests, "s2_after_reached": measurements.s2_after_reached,
              "frames": len(frames), "planning_cycles": len(plans),
              "planning_p95_sec": float(np.percentile(latencies, 95)) if latencies else None,
              "frame_gap_p95_sec": float(np.percentile(frame_gaps, 95)) if len(frame_gaps) else None,
              "frame_gap_max_sec": float(frame_gaps.max()) if len(frame_gaps) else None,
              "candidate_motion_violation_mean": float(np.mean(violations)) if violations else None,
              "displacement_m": float(np.linalg.norm(final_position - initial_position)),
              "goal_distance_m": float(np.linalg.norm(target_world - final_position)) if target_world is not None else None,
              "plans": plans}
    with (output / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "plans"}, ensure_ascii=False), flush=True)
    print(output / f"{stem}.json", flush=True)
    return 0 if measurements.reached > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
