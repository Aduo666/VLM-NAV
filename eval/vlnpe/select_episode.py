#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

from adapter import normalize_episode


DATA_ROOT = Path(os.getenv("VLNPE_DATA_ROOT", "/Data/qiang/InternData-N1/vln_pe"))
SPLIT = os.getenv("VLNPE_SPLIT", "val_seen")
EPISODE_INDEX = int(os.getenv("VLNPE_EPISODE_INDEX", "0"))
EPISODE_ID = os.getenv("VLNPE_EPISODE_ID", "")
OUTPUT_PATH = Path(os.getenv("VLNPE_OUTPUT_PATH", "eval/current_episode.json"))


def main() -> int:
    split_path = DATA_ROOT / "raw_data" / "r2r" / SPLIT / f"{SPLIT}.json.gz"
    if not split_path.exists():
        raise FileNotFoundError(f"VLNPE split file not found: {split_path}")

    with gzip.open(split_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    episodes = data.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"No episodes found in {split_path}")

    selected = None
    if str(EPISODE_ID).strip():
        for episode in episodes:
            if str(episode.get("episode_id", "")) == str(EPISODE_ID).strip():
                selected = episode
                break
        if selected is None:
            raise RuntimeError(f"episode_id={EPISODE_ID!r} not found in split={SPLIT}")
    else:
        index = int(EPISODE_INDEX)
        if index < 0 or index >= len(episodes):
            raise IndexError(f"EPISODE_INDEX={index} out of range for split={SPLIT}, count={len(episodes)}")
        selected = episodes[index]

    # Convert R2R format to nav.py-compatible format
    normalized = normalize_episode(
        selected,
        tasks_path=split_path,
        source_root=str(DATA_ROOT.parent / ".." / "mp3d_pe"),
    )

    payload = {
        "schema_version": "vlnpe-r2r-v1",
        "source_dataset": "vlnpe",
        "source_root": str(DATA_ROOT),
        "split": SPLIT,
        "episodes": [normalized],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scene_id = selected.get("scene_id", "")
    episode_id = selected.get("episode_id", "")
    instruction = selected.get("instruction", {})
    if isinstance(instruction, dict):
        instruction_text = str(instruction.get("instruction_text", ""))
    else:
        instruction_text = str(instruction)
    print(f"selected split={SPLIT} index={episodes.index(selected)} scene={scene_id} episode={episode_id}")
    print(f"scene_usd={normalized.get('scene_usd_path', 'NOT_FOUND')}")
    print(f"wrote {OUTPUT_PATH}")
    print(f"instruction={instruction_text[:240]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VLNPE episode selection failed: {exc}", file=sys.stderr)
        raise
