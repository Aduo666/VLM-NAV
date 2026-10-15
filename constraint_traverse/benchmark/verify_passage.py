from __future__ import annotations

import copy
from datetime import datetime
import json
import logging
import math

import numpy as np

from .run import ROOT, load_config


def main():
    config = load_config(ROOT / "constraint_traverse/configs/benchmark.yaml")
    config["scene"]["door_clearance_m"] = 0.20
    config["trial"]["timeout_sec"] = 25.0
    logging.basicConfig(level=logging.INFO)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, **config["simulation"]})
    try:
        from .runtime import PassageRuntime
        output = ROOT / "constraint_traverse/data/passage_validation"
        runtime = PassageRuntime(app, ROOT, config, output, True, [1.6, 0])
        runtime._initialize()
        if not runtime._start("MPPI", "goal"):
            raise RuntimeError(runtime.result_text)
        runtime._finish("VALIDATION_SETUP")
        baseline = copy.deepcopy((runtime.snapshot, runtime.saved_link_states, runtime.saved_controller))
        reports = []
        cases = (
            ("straight", [-1.6, 0, 0], [1.6, 0], 25.0),
            ("angled", [-1.6, 0, 15], [1.6, 0], 25.0),
            ("detour", [-1.6, 0.8, 0], [1.6, 0.8], 25.0),
            ("wall_stop", [-0.6, 0.9, 0], [1.6, 0.9], 3.0),
        )
        for name, pose, goal, timeout in cases:
            starts = []
            for method in ("MPPI", "KMPPI"):
                runtime.snapshot, runtime.saved_link_states, runtime.saved_controller = copy.deepcopy(baseline)
                runtime._reset()
                position, _ = runtime.robot.get_world_pose()
                position[:2] = pose[:2]
                yaw = math.radians(pose[2])
                runtime.robot.set_world_pose(position=position, orientation=np.asarray([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
                runtime.robot.set_linear_velocity(np.zeros(3))
                runtime.robot.set_angular_velocity(np.zeros(3))
                runtime.robot.set_joint_velocities(np.zeros_like(runtime.robot.get_joint_velocities()))
                runtime.last_command[:] = 0
                for _ in range(60):
                    runtime._apply([0, 0])
                    runtime.world.step(render=False)
                runtime.goal = np.asarray(goal, dtype=float)
                runtime.config["trial"]["timeout_sec"] = timeout
                if not runtime._start(method, "goal"):
                    raise RuntimeError(f"{name}/{method}: {runtime.result_text}")
                initial = runtime._state().copy()
                starts.append(initial)
                weights, jumps, bounds, gaps = [], [], [], []
                last = None
                while runtime.mode == "RUNNING" and app.is_running():
                    runtime._running_step()
                    if runtime.last_plan is not None and runtime.last_plan is not last:
                        last = runtime.last_plan
                        weights.append(last.diagnostics["ess"])
                        jumps.append(last.diagnostics["command_change"])
                        bounds.append(last.diagnostics["control_bound_excess"])
                        gaps.append(last.min_clearance_m)
                    if runtime.mode == "RUNNING":
                        runtime.world.step(render=False)
                final = runtime._state()
                report = {"case": name, "method": method, **runtime.summary[-1],
                          "initial_pose": initial[:3].tolist(),
                          "displacement_m": float(np.linalg.norm(final[:2] - initial[:2])),
                          "ess_median": float(np.median(weights)) if weights else None,
                          "command_change_max": max(jumps, default=0),
                          "control_bound_excess_max": max(bounds, default=0),
                          "prediction_min_clearance_m": min(gaps, default=None)}
                reports.append(report)
                print(json.dumps(report, ensure_ascii=False), flush=True)
            difference = float(np.max(np.abs(starts[0] - starts[1])))
            for report in reports[-2:]:
                report["paired_initial_state_max_difference"] = difference
        path = output / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_report.json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(reports, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"report": str(path)}, ensure_ascii=False), flush=True)
        runtime.subscription = None
    finally:
        app.close()


if __name__ == "__main__":
    main()
