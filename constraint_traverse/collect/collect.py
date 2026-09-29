from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys

import yaml

from .runtime import ConstraintTraverseCollector, RuntimeOverrides


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "constraint_traverse/configs/collect.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "constraint_traverse/data"
DEFAULT_LOG_PATH = PROJECT_ROOT / "constraint_traverse/logs/collect_latest.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect human AGX demonstrations in constrained spaces"
    )
    parser.add_argument("--usd", required=True, type=Path, help="Scene USD path")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Collector YAML configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset output root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = _make_logger(DEFAULT_LOG_PATH)
    try:
        config = _load_config(args.config)
        overrides = RuntimeOverrides(
            usd_path=args.usd,
            output_root=args.output.expanduser().resolve(),
        )
    except Exception:
        logger.exception("[CollectorConfig] invalid collector arguments or configuration")
        return 1

    from isaacsim import SimulationApp

    simulation = config["simulation"]
    app = SimulationApp(
        {
            "headless": False,
            "active_gpu": simulation["active_gpu"],
            "physics_gpu": simulation["physics_gpu"],
            "multi_gpu": simulation["multi_gpu"],
        }
    )
    try:
        collector = ConstraintTraverseCollector(
            app=app,
            project_root=PROJECT_ROOT,
            config=config,
            overrides=overrides,
            logger=logger,
        )
        return collector.run()
    except Exception:
        logger.exception("[CollectorFatal] collector crashed")
        return 1
    finally:
        app.close()


def _load_config(path: Path) -> dict:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Collector config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    expected = {
        "simulation",
        "scene",
        "spawn",
        "cameras",
        "bev",
        "teleop",
        "collection",
        "ui",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"collect.yaml 必须包含这些配置段：{sorted(expected)}")
    _validate_top_level_config(data)
    return data


def _validate_top_level_config(data: dict) -> None:
    simulation = data["simulation"]
    if set(simulation) != {"active_gpu", "physics_gpu", "multi_gpu"}:
        raise ValueError("simulation 配置字段不完整")
    if not isinstance(simulation["multi_gpu"], bool):
        raise ValueError("simulation.multi_gpu 必须为布尔值")
    for name in ("active_gpu", "physics_gpu"):
        if simulation[name] is not None and int(simulation[name]) < 0:
            raise ValueError(f"simulation.{name} 必须为空或非负整数")

    scene = data["scene"]
    if set(scene) != {
        "prim_path",
        "add_default_ground_plane",
        "stage_load_timeout_sec",
        "add_indoor_lighting",
    }:
        raise ValueError("scene 配置字段不完整")
    if not str(scene["prim_path"]).startswith("/"):
        raise ValueError("scene.prim_path 必须是绝对 USD 路径")
    if float(scene["stage_load_timeout_sec"]) <= 0:
        raise ValueError("scene.stage_load_timeout_sec 必须为正数")

    spawn = data["spawn"]
    required_spawn = {
        "holding_height_m",
        "query_warmup_frames",
        "settle_frames",
    }
    if set(spawn) != required_spawn:
        raise ValueError(f"spawn 必须包含这些字段：{sorted(required_spawn)}")
    if any(float(spawn[name]) <= 0 for name in required_spawn):
        raise ValueError("spawn 数值配置必须为正数")

    cameras = data["cameras"]
    if set(cameras) != {"warmup_render_frames"}:
        raise ValueError("cameras 配置字段不完整")
    if int(cameras["warmup_render_frames"]) <= 0:
        raise ValueError("cameras.warmup_render_frames 必须为正整数")

    collection = data["collection"]
    required_collection = {
        "jpeg_quality",
        "png_compression",
        "max_episode_duration_sec",
        "min_success_samples",
        "start_max_linear_mps",
        "start_max_angular_radps",
        "min_depth_valid_ratio",
        "endpoint_position_tolerance_m",
        "endpoint_yaw_tolerance_deg",
        "success_goal_position_tolerance_m",
    }
    if set(collection) != required_collection:
        raise ValueError(
            f"collection 必须包含这些字段：{sorted(required_collection)}"
        )
    if not 1 <= int(collection["jpeg_quality"]) <= 100:
        raise ValueError("collection.jpeg_quality 必须位于 1 至 100")
    if not 0 <= int(collection["png_compression"]) <= 9:
        raise ValueError("collection.png_compression 必须位于 0 至 9")
    for name in required_collection - {
        "jpeg_quality",
        "png_compression",
    }:
        value = float(collection[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"collection.{name} 必须为有限正数")
    if not 0 < float(collection["min_depth_valid_ratio"]) <= 1:
        raise ValueError("collection.min_depth_valid_ratio 必须位于 0 至 1")

    ui = data["ui"]
    if set(ui) != {"update_period_sec"}:
        raise ValueError("ui 配置字段不完整")
    if float(ui["update_period_sec"]) <= 0:
        raise ValueError("ui 配置值无效")


def _make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("constraint_traverse.collect")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("[Collector] log=%s", path)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
