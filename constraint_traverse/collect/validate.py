from __future__ import annotations

import argparse
import json
from pathlib import Path

from .map_state import verify_episode_bevs
from .recorder import recover_episode_directory, validate_episode_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate collected traversal episodes")
    parser.add_argument("path", type=Path, help="Episode directory or dataset root")
    parser.add_argument("--recover", action="store_true", help="恢复指定未完成记录中已提交的状态批次")
    parser.add_argument("--rebuild-bev", action="store_true", help="从初始地图与 RGB-D 重建并逐帧核对 BEV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.recover:
        try:
            path = recover_episode_directory(args.path.expanduser().resolve())
        except Exception as exc:
            print(f"[RECOVERY FAILED] {exc}")
            return 1
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        print(f"[RECOVERED {meta['status'].upper()}] {path}")
        return 0
    targets = _find_episode_dirs(args.path)
    if not targets:
        print(f"No episode directories found under: {args.path}")
        return 1

    failed = 0
    for episode in targets:
        errors = validate_episode_directory(episode)
        if args.rebuild_bev and not errors:
            errors.extend(verify_episode_bevs(episode))
        if errors:
            failed += 1
            print(f"[INVALID] {episode}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {episode}")

    print(f"Validated {len(targets)} episode(s), invalid={failed}")
    return 1 if failed else 0


def _find_episode_dirs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if (path / "meta.json").is_file() or path.name.startswith(("episode_", ".episode_")):
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_dir()
        and candidate.name.startswith(("episode_", ".episode_"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
