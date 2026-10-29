from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import signal
import tempfile

from .cameras import CAMERA_PATHS, DEFAULT_MAP_SOURCES, normalize_map_sources


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="机械臂主动感知实验：手动运行、状态恢复与数据记录")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "check-sim"):
        sub = commands.add_parser(command, help="交互实验" if command == "run" else "物理仿真验收")
        sub.add_argument("--usd", type=Path, help="场景 USD；省略时使用空白地面")
        sub.add_argument("--start", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW_DEG"),
                         help="机器人起始世界坐标及朝向角度")
        sub.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
        sub.add_argument("--output", type=Path)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--gpu", type=int, default=0)
        sub.add_argument("--timeout", type=float, default=120.0)
        sub.add_argument("--map-sources", nargs="*", choices=tuple(CAMERA_PATHS), default=DEFAULT_MAP_SOURCES,
                         help="参与建图的相机，可组合 front、arm、down；省略名称时关闭全部融合")
        if command == "check-sim":
            sub.add_argument("--gui", action="store_true", help="显示验收过程")
            sub.add_argument("--resets", type=int, default=10)
    inspect = commands.add_parser("inspect", help="检查已经保存的试验记录")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return args
    try:
        args.map_sources = normalize_map_sources(args.map_sources)
    except ValueError as error:
        parser.error(str(error))
    import math
    values = [args.timeout, *(args.start or []), *(args.goal or [])]
    if not all(math.isfinite(value) for value in values) or args.timeout <= 0 or args.gpu < 0:
        parser.error("坐标与时间参数需要使用有限数值，超时必须为正数，GPU 编号不能为负数")
    args.headless = args.command == "check-sim" and not args.gui
    if args.command == "check-sim" and args.resets < 1:
        parser.error("恢复检查次数必须为正整数")
    if args.usd is not None:
        args.usd = args.usd.expanduser().resolve()
        if not args.usd.is_file():
            parser.error(f"场景文件不存在：{args.usd}")
        if args.headless and args.start is None:
            parser.error("室内场景的无界面验收需要明确提供 --start X Y Z YAW_DEG")
    return args


def run_application(args, *, cleanup_output=None):
    from isaacsim import SimulationApp
    from constraint_traverse.lifecycle import BackgroundTasks, bind_shutdown

    app = SimulationApp(dict(headless=args.headless, active_gpu=args.gpu,
                             physics_gpu=args.gpu, multi_gpu=False))
    logger = logging.getLogger("constraint_traverse.active_passage")
    tasks = BackgroundTasks(logger)
    runtime = None
    subscription = None
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        import omni.kit.app
        from .simulation.runtime import ExperimentRuntime

        runtime = tasks.add(ExperimentRuntime(app, args, ROOT))
        subscription = bind_shutdown(omni.kit.app.get_app(), 1, tasks)
        runtime.initialize()
        if args.command == "check-sim":
            from .simulation.verify import verify
            print(json.dumps(verify(runtime, args.resets), ensure_ascii=False, indent=2), flush=True)
            return 0
        return runtime.run()
    except KeyboardInterrupt:
        if runtime is not None and runtime.experiment is not None:
            runtime.experiment.close("INTERRUPTED")
        import omni.kit.app
        omni.kit.app.get_app().post_quit(130)
        return 130
    except Exception as error:
        if runtime is not None and runtime.experiment is not None:
            runtime.experiment.close("ERROR", details=dict(error_type=type(error).__name__, message=str(error)))
        logger.exception("实验程序执行失败")
        import omni.kit.app
        omni.kit.app.get_app().post_quit(1)
        return 1
    finally:
        tasks.close()
        subscription = None
        signal.signal(signal.SIGTERM, previous_term)
        try:
            if cleanup_output is not None:
                cleanup_output()
        finally:
            app.close(wait_for_replicator=False)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.command == "inspect":
        from .recording import inspect_record
        print(json.dumps(inspect_record(args.path), ensure_ascii=False, indent=2))
        return 0
    if args.command == "check-sim" and args.output is None:
        temporary = tempfile.TemporaryDirectory(prefix="active-passage-check-")
        with temporary as directory:
            args.output = Path(directory)
            return run_application(args, cleanup_output=temporary.cleanup)
    args.output = ((ROOT / "constraint_traverse/data/active_passage") if args.output is None
                   else args.output.expanduser().resolve())
    return run_application(args)


if __name__ == "__main__":
    raise SystemExit(main())
