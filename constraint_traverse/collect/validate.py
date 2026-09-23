from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recorder import validate_episode_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate collected traversal episodes")
    parser.add_argument("path", type=Path, help="Episode directory or dataset root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = _find_episode_dirs(args.path)
    if not targets:
        print(f"No episode directories found under: {args.path}")
        return 1

    failed = 0
    for episode in targets:
        errors = validate_episode_directory(episode)
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
    if (path / "meta.json").is_file() and (path / "state.npz").is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("episode_*")
        if candidate.is_dir()
        and (candidate / "meta.json").is_file()
        and (candidate / "state.npz").is_file()
    )


if __name__ == "__main__":
    raise SystemExit(main())
