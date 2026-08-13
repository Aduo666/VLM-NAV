#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from vln_nav.training.expert_replay import ExpertReplayConfig, build_expert_replay
from vln_nav.training.interiornav import load_canonical_episode
from vln_nav.training.query_compiler import QueryCompiler
from vln_nav.training.route_alignment import align_instruction_to_route
from vln_nav.training.schemas import TRAINING_SAMPLE_SCHEMA_VERSION


FORBIDDEN_STUDENT_KEYS = {
    "future_reference_path_xy",
    "nearest_reference_path_index",
    "route_alignment_path",
    "active_span_id",
    "oracle_reason",
    "oracle_crossing_world_xy",
    "oracle_confidence",
    "active_goal_world_xy",
    "selected_frontier_id",
    "frontier_value_targets",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate expert replay inputs and causal outputs")
    parser.add_argument("--data-root", default="/Data/qiang/interiornav_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode = load_canonical_episode(
        data_root=args.data_root,
        split=args.split,
        episode_index=args.episode_index,
    )
    replay = build_expert_replay(episode, config=ExpertReplayConfig())
    queries = QueryCompiler().compile(episode.instruction)
    alignments = align_instruction_to_route(episode=episode, replay=replay)
    peak_frames = [item.peak_frame_index for item in alignments]
    if peak_frames != sorted(peak_frames):
        raise RuntimeError(f"Route alignment is not monotonic: {peak_frames}")
    max_turn_deg = max(abs(float(np.degrees(item.delta_yaw_rad))) for item in replay)
    if max_turn_deg > 15.0 + 1.0e-5:
        raise RuntimeError(f"Replay turn step exceeded 15 deg: {max_turn_deg:.3f}")
    print(
        json.dumps(
            {
                "episode": {
                    "split": episode.split,
                    "scene_id": episode.scene_id,
                    "episode_id": episode.episode_id,
                    "instruction": episode.instruction,
                    "reference_path_points": len(episode.reference_path_xyz),
                    "room_regions": len(episode.rooms),
                },
                "replay": {
                    "frames": len(replay),
                    "duration_sec": replay[-1].timestamp_sec,
                    "max_turn_step_deg": max_turn_deg,
                    "motion_counts": {
                        motion: sum(item.motion_type == motion for item in replay)
                        for motion in sorted({item.motion_type for item in replay})
                    },
                },
                "queries": [item.to_dict() for item in queries],
                "route_alignment": [item.to_dict() for item in alignments],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if str(args.episode_dir).strip():
        _validate_collected_episode(Path(args.episode_dir).expanduser().resolve())
    return 0


def _validate_collected_episode(episode_dir: Path) -> None:
    manifest_path = episode_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Collected manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    if not frames:
        raise RuntimeError(f"Collected manifest has no frames: {manifest_path}")
    for relative_path in manifest.get("episode_artifacts", {}).values():
        if not (episode_dir / str(relative_path)).is_file():
            raise FileNotFoundError(f"Missing episode artifact: {relative_path}")
    previous_rgb: np.ndarray | None = None
    previous_depth: np.ndarray | None = None
    for frame in frames:
        sample_path = episode_dir / str(frame["sample_path"])
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        student = sample.get("student", {})
        if sample.get("schema_version") != TRAINING_SAMPLE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Stale training schema in {sample_path}: "
                f"expected={TRAINING_SAMPLE_SCHEMA_VERSION!r} "
                f"actual={sample.get('schema_version')!r}"
            )
        leaked = FORBIDDEN_STUDENT_KEYS.intersection(_recursive_keys(student))
        if leaked:
            raise RuntimeError(f"Privileged fields leaked into {sample_path}: {sorted(leaked)}")
        targets = sample.get("targets", {})
        frontiers = student.get("observation", {}).get("frontiers", [])
        frontier_ids = [int(item["frontier_id"]) for item in frontiers]
        if frontier_ids != [int(value) for value in targets.get("frontier_ids", [])]:
            raise RuntimeError(f"Student/target frontier IDs differ in {sample_path}")
        for frontier in frontiers:
            if not bool(frontier.get("reachable", False)):
                raise RuntimeError(f"Unreachable frontier leaked into student candidates: {sample_path}")
            if "boundary_world_xy" not in frontier or "navigation_goal_world_xy" not in frontier:
                raise RuntimeError(f"Frontier lacks boundary/goal separation: {sample_path}")
        values = [float(value) for value in targets.get("frontier_value_targets", [])]
        if len(values) != len(frontier_ids):
            raise RuntimeError(f"Frontier target length mismatch in {sample_path}")
        selected = targets.get("selected_frontier_id")
        oracle_reason = str(sample.get("debug_oracle", {}).get("oracle_reason", ""))
        if selected is not None and int(selected) not in frontier_ids:
            raise RuntimeError(f"Selected frontier is absent from candidates in {sample_path}")
        if selected is None and any(abs(value) > 1.0e-8 for value in values):
            raise RuntimeError(f"Unreliable frame still carries soft frontier labels: {sample_path}")
        if selected is not None and oracle_reason != "reliable_route_frontier_crossing":
            raise RuntimeError(f"Hard frontier label has unreliable reason in {sample_path}")
        for relative_path in frame.get("paths", {}).values():
            if not (episode_dir / str(relative_path)).is_file():
                raise FileNotFoundError(f"Missing collected artifact: {relative_path}")
        paths = frame.get("paths", {})
        frame_record = json.loads(
            (episode_dir / str(paths["rgb"])).with_name("frame.json").read_text(encoding="utf-8")
        )
        pose = frame_record.get("pose", {})
        require_changed = (
            float(pose.get("delta_distance_m", 0.0)) > 1.0e-4
            or abs(float(pose.get("delta_yaw_rad", 0.0))) > 1.0e-4
        )
        rgb = np.asarray(Image.open(episode_dir / str(paths["rgb"])).convert("RGB"))
        with np.load(episode_dir / str(paths["depth"])) as depth_payload:
            depth = np.asarray(depth_payload["depth_m"], dtype=np.float32)
        if require_changed and previous_rgb is not None and previous_depth is not None:
            rgb_same = rgb.shape == previous_rgb.shape and np.array_equal(rgb, previous_rgb)
            depth_same = depth.shape == previous_depth.shape and np.array_equal(
                depth,
                previous_depth,
                equal_nan=True,
            )
            if rgb_same and depth_same:
                raise RuntimeError(
                    f"Stale RGB-D on moving replay frame in {sample_path}: "
                    f"delta_distance_m={pose.get('delta_distance_m')} "
                    f"delta_yaw_rad={pose.get('delta_yaw_rad')}"
                )
        previous_rgb = rgb
        previous_depth = depth
    print(f"validated collected episode: frames={len(frames)} path={episode_dir}")


def _recursive_keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(_recursive_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_recursive_keys(item))
        return result
    return set()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"expert replay validation failed: {exc}", file=sys.stderr)
        raise
