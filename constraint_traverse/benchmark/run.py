from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = {"simulation", "scene", "planner", "motion", "cost", "safety", "trial"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"实验配置需要包含 {sorted(required)}")
    from .model import MotionLimits
    MotionLimits(**cfg["motion"])
    for section, fields in {
        "planner": ("dt_sec", "horizon_steps", "samples", "iterations", "temperature", "noise_linear_std", "noise_angular_std", "kernel_support_points", "kernel_sigma", "reference_lookahead_m"),
        "trial": ("goal_tolerance_m", "timeout_sec", "stopped_hold_sec", "settle_frames", "map_size_px"),
        "safety": ("sweep_step_m", "reaction_time_sec"),
        "cost": ("reference_dt_sec", "goal_distance_scale_m", "penetration_scale_m", "obstacle_soft_distance_m"),
    }.items():
        if any(float(cfg[section][field]) <= 0 for field in fields):
            raise ValueError(f"{section} 的数值配置必须为正数")
    if int(cfg["planner"]["kernel_support_points"]) > int(cfg["planner"]["horizon_steps"]):
        raise ValueError("KMPPI 支持点数量超过预测步数")
    if not 0 <= float(cfg["planner"].get("noise_correlation", 0.85)) < 1:
        raise ValueError("控制采样相关系数需要位于 [0, 1)")
    if not 0 < float(cfg["planner"].get("ess_fraction", 0.12)) <= 1:
        raise ValueError("有效样本比例需要位于 (0, 1]")
    for name in ("connection_position_m", "connection_yaw_rad", "connection_speed_mps", "connection_steering_rad"):
        value = float(cfg["planner"].get(name, 0.05))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"控制连接误差阈值必须为有限正数：{name}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="AGX 窄门 MPPI/KMPPI 交互实验")
    parser.add_argument("--config", type=Path, default=ROOT / "constraint_traverse/configs/benchmark.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "constraint_traverse/data/benchmark")
    parser.add_argument("--method", choices=("MPPI", "KMPPI"))
    parser.add_argument("--door-clearance", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-yaw-deg", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--goal", type=float, nargs=2)
    parser.add_argument("--navigation-condition", choices=("goal", "reference"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("重复次数必须为正数")
    if args.headless and args.goal is None:
        parser.error("无界面模式需要 --goal 世界坐标")
    cfg = copy.deepcopy(load_config(args.config))
    for argument, section, field in (
        (args.method, "planner", "method"), (args.door_clearance, "scene", "door_clearance_m"),
        (args.start_y, "scene", "start_y_m"), (args.start_yaw_deg, "scene", "start_yaw_deg"),
        (args.seed, "planner", "seed"), (args.navigation_condition, "trial", "navigation_condition"),
        (args.timeout, "trial", "timeout_sec"),
    ):
        if argument is not None:
            cfg[section][field] = argument
    if cfg["scene"]["door_clearance_m"] <= 0 or cfg["trial"]["timeout_sec"] <= 0:
        parser.error("门洞余量和超时时间必须为正数")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, **cfg["simulation"]})
    try:
        from .runtime import PassageRuntime
        return PassageRuntime(app, ROOT, cfg, args.output, args.headless, args.goal, args.repeat).run()
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
