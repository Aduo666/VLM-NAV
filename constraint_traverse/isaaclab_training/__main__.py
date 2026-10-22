from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab 窄门示范训练与评估")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="检查原始数据并构建训练缓存")
    prepare.add_argument("--dataset", type=Path, default=Path("/Data/qiang/data/constraint_traverse"))
    prepare.add_argument("--cache", type=Path, default=Path("/Data/qiang/constraint_traverse_training/cache"))
    train = sub.add_parser("train", help="启动 BC 训练")
    train.add_argument("--cache", type=Path, default=Path("/Data/qiang/constraint_traverse_training/cache"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", choices=("lstm", "mlp"), default="lstm")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, help="每批轨迹数量；新训练默认 8，恢复训练使用检查点配置")
    train.add_argument("--learning-rate", type=float, help="学习率；新训练默认 0.0003，恢复训练使用检查点配置")
    train.add_argument("--seed", type=int, help="随机种子；新训练默认 0，恢复训练使用检查点配置")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", type=Path)
    sub.add_parser("check", help="运行数据接口、模型与 PPO 自动测试")
    for name in ("check-sim", "evaluate"):
        p = sub.add_parser(name, help="检查真实仿真任务" if name == "check-sim" else "运行闭环评估")
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--cache", type=Path, default=Path("/Data/qiang/constraint_traverse_training/cache"))
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--gui", action="store_true")
        if name == "evaluate":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--perturbations", type=int, default=3)
            p.add_argument("--output", type=Path)
            p.add_argument("--doors", type=Path)
    args = parser.parse_args()
    if args.command != "prepare":
        expected = {"torch": "2.7.0", "rsl-rl-lib": "3.1.2"}
        if args.command in ("check-sim", "evaluate"):
            expected.update({"isaaclab": "0.54.2", "isaaclab-rl": "0.4.7"})
        for package, version in expected.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(f"训练环境缺少 {package}，请激活专用训练环境") from exc
            if actual.split("+")[0] != version:
                raise RuntimeError(f"{package} 版本应为 {version}，当前为 {actual}，请激活专用训练环境")
    if args.command == "prepare":
        from .data import prepare as build
        manifest = build(args.dataset, args.cache)
        print(json.dumps({key: manifest[key] for key in ("frames", "decisions")}, ensure_ascii=False))
    elif args.command == "train":
        from .bc import train as run
        run(args.cache, args.output, model=args.model, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, device=args.device, resume=args.resume)
    elif args.command == "check":
        import unittest
        from . import test_training, test_geometry, test_temporal
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                                   for module in (test_training, test_geometry, test_temporal))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    else:
        from .simulation import run
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
