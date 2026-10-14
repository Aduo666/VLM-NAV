from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np

from .run import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description="验证 Isaac Sim 四轮执行、刚体恢复、碰撞回调和面板接口")
    parser.add_argument("--output", type=Path, default=ROOT / "constraint_traverse/data/benchmark_validation")
    args = parser.parse_args()
    config = load_config(ROOT / "constraint_traverse/configs/benchmark.yaml")
    logging.basicConfig(level=logging.INFO)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, **config["simulation"]})
    try:
        from .runtime import PassageRuntime
        from .ui import BenchmarkPanel
        import omni.ui as ui
        runtime = PassageRuntime(app, ROOT, config, args.output, True, [1.6, 0])
        runtime._initialize()
        if not runtime._start("MPPI", "goal"):
            raise RuntimeError(runtime.result_text)
        runtime._finish("VALIDATION_SETUP")
        measurements = {}
        first_trace = None
        for repetition in range(2):
            runtime._reset()
            initial = runtime._state()
            trace, predicted = [], []
            action = np.asarray([0.25, 0.15])
            for step in range(60):
                command = action if step < 30 else np.zeros(2)
                estimate = runtime.execution_model.step(runtime._state(), command).numpy()
                runtime._apply(command)
                runtime.world.step(render=False)
                trace.append(runtime._state().copy())
                predicted.append(estimate)
            trace, predicted = np.asarray(trace), np.asarray(predicted)
            measurements[f"run_{repetition}"] = {
                "initial_state": initial.tolist(), "reset_errors": copy.deepcopy(runtime.reset_errors),
                "one_step_position_rmse_m": float(np.sqrt(np.mean((trace[:, :2] - predicted[:, :2]) ** 2))),
                "one_step_yaw_rmse_rad": float(np.sqrt(np.mean((trace[:, 2] - predicted[:, 2]) ** 2))),
                "final_speed_mps": float(np.linalg.norm(trace[-1, 3:5])),
            }
            if first_trace is None:
                first_trace = trace
            else:
                measurements["repeated_motion_max_position_difference_m"] = float(np.max(np.linalg.norm(trace[:, :2] - first_trace[:, :2], axis=1)))
        motion_checks = {}
        for name, action in (("forward", [0.3, 0]), ("reverse", [-0.3, 0]),
                             ("curve", [0.25, 0.15]), ("spin", [0, 0.25])):
            runtime._reset()
            initial = runtime._state()
            for _ in range(60):
                runtime._apply(action)
                runtime.world.step(render=False)
            final = runtime._state()
            motion_checks[name] = {"delta_x_m": float(final[0] - initial[0]),
                                   "delta_y_m": float(final[1] - initial[1]),
                                   "delta_yaw_rad": float(final[2] - initial[2])}
        measurements["motion_checks"] = motion_checks
        runtime._reset()
        for _ in range(60):
            runtime._apply([0.3, 0])
            runtime.world.step(render=False)
        brake_initial = runtime._state()
        predicted_braking = runtime.model.braking_rollout(brake_initial, config["safety"]["reaction_time_sec"]).cpu().numpy()
        brake_poses = [brake_initial[:3]]
        for _ in range(300):
            runtime._apply([0, 0])
            runtime.world.step(render=False)
            state = runtime._state()
            brake_poses.append(state[:3].copy())
            if np.linalg.norm(state[3:5]) < config["trial"]["stopped_linear_mps"] and abs(state[5]) < config["trial"]["stopped_angular_radps"]:
                break
        brake_poses = np.asarray(brake_poses)
        measurements["braking"] = {
            "initial_speed_mps": float(np.linalg.norm(brake_initial[3:5])),
            "stopping_time_sec": (len(brake_poses) - 1) * runtime.dt,
            "actual_distance_m": float(np.linalg.norm(np.diff(brake_poses[:, :2], axis=0), axis=1).sum()),
            "predicted_distance_m": float(np.linalg.norm(np.diff(predicted_braking[:, :2], axis=0), axis=1).sum()),
            "final_speed_mps": float(np.linalg.norm(state[3:5])),
        }
        runtime._reset()
        commands = []
        panel = BenchmarkPanel(ui, runtime.renderer, commands.append)
        panel.update(runtime.renderer.render(runtime._state()[:3], runtime.goal), "READY", "VALIDATION")
        app.update()
        panel._start()
        measurements["panel_start_command"] = commands[-1]
        panel.close()
        runtime.mode, runtime.start_time = "RUNNING", float(runtime.world.current_time)
        position, orientation = runtime.robot.get_world_pose()
        position[:2] = [0, runtime.scene.width_m / 2 + 0.1]
        runtime.robot.set_world_pose(position=position, orientation=orientation)
        for _ in range(10):
            runtime.world.step(render=False)
        measurements["known_obstacle_contact_detected"] = bool(runtime.contacts)
        measurements["contact_events"] = runtime.contacts
        measurements["passed"] = (
            measurements["known_obstacle_contact_detected"]
            and measurements["repeated_motion_max_position_difference_m"] < 0.01
            and all(measurements[f"run_{i}"]["one_step_position_rmse_m"] < 0.01 for i in range(2))
            and measurements["braking"]["final_speed_mps"] < config["trial"]["stopped_linear_mps"]
            and measurements["braking"]["actual_distance_m"] <= measurements["braking"]["predicted_distance_m"] + config["safety"]["margin_m"]
            and motion_checks["forward"]["delta_x_m"] > 0.03
            and motion_checks["reverse"]["delta_x_m"] < -0.03
            and motion_checks["curve"]["delta_x_m"] > 0.03
            and motion_checks["curve"]["delta_yaw_rad"] > 0.03
            and motion_checks["spin"]["delta_yaw_rad"] > 0.03
        )
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_validation.json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(measurements, handle, indent=2, ensure_ascii=False)
        print(json.dumps({"passed": measurements["passed"], "report": str(path)}, ensure_ascii=False))
        runtime.subscription = None
        return 0 if measurements["passed"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
