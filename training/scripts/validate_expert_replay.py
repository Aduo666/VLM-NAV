#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
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
    parser.add_argument(
        "--report-path",
        default="",
        help="Quality report path; defaults to <episode-dir>/quality_report.json",
    )
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
        episode_dir = Path(args.episode_dir).expanduser().resolve()
        report = _validate_collected_episode(episode_dir)
        report_path = (
            Path(args.report_path).expanduser().resolve()
            if str(args.report_path).strip()
            else episode_dir / "quality_report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"quality report: {report_path}")
    return 0


def _validate_collected_episode(episode_dir: Path) -> dict:
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
    previous_view_matrix: np.ndarray | None = None
    previous_completed_prefix: np.ndarray | None = None
    previous_path_progress: float | None = None
    previous_known_ratio: float | None = None
    moving_frames = 0
    rgb_changed_frames = 0
    depth_changed_frames = 0
    view_changed_frames = 0
    depth_valid_ratios: list[float] = []
    known_ratios: list[float] = []
    known_ratio_decrease_frames = 0
    frontier_counts: list[int] = []
    reliable_frontier_labels = 0
    progress_valid_frames = 0
    progress_monotonic_violations = 0
    path_progress_violations = 0
    prefix_shape_violations = 0
    oracle_reasons: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    local_map_shapes: Counter[str] = Counter()
    global_map_shapes: Counter[str] = Counter()
    for frame in frames:
        sample_path = episode_dir / str(frame["sample_path"])
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if sample.get("schema_version") != TRAINING_SAMPLE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Stale training schema in {sample_path}: "
                f"expected={TRAINING_SAMPLE_SCHEMA_VERSION!r} "
                f"actual={sample.get('schema_version')!r}"
            )
        progress = sample.get("progress", {})
        progress_observation = progress.get("observation", {})
        progress_supervision = progress.get("supervision", {})
        spatial = sample.get("spatial_subgoal", {})
        spatial_observation = spatial.get("observation", {})
        spatial_supervision = spatial.get("supervision", {})
        if not progress_observation or not spatial_observation:
            raise RuntimeError(f"Missing progress/spatial observation contract in {sample_path}")
        leaked = FORBIDDEN_STUDENT_KEYS.intersection(
            _recursive_keys(
                {
                    "progress_observation": progress_observation,
                    "spatial_observation": spatial_observation,
                }
            )
        )
        if leaked:
            raise RuntimeError(f"Privileged fields leaked into {sample_path}: {sorted(leaked)}")
        if int(progress_observation.get("frame_index", -1)) != int(
            spatial_observation.get("frame_index", -2)
        ):
            raise RuntimeError(f"Progress/spatial frame mismatch in {sample_path}")
        if str(progress_observation.get("instruction", "")) != str(
            spatial_observation.get("instruction", "")
        ):
            raise RuntimeError(f"Progress/spatial instruction mismatch in {sample_path}")

        instruction_words = progress_observation.get("instruction_words", [])
        completed_prefix = np.asarray(
            progress_supervision.get("completed_prefix_weights", []),
            dtype=np.float64,
        )
        active_span = np.asarray(
            progress_supervision.get("active_span_weights", []),
            dtype=np.float64,
        )
        if completed_prefix.shape != (len(instruction_words),):
            raise RuntimeError(f"Completed-prefix length mismatch in {sample_path}")
        if active_span.shape != (len(instruction_words),):
            raise RuntimeError(f"Active-span length mismatch in {sample_path}")
        if np.any((completed_prefix < -1.0e-6) | (completed_prefix > 1.0 + 1.0e-6)):
            raise RuntimeError(f"Completed-prefix values outside [0,1] in {sample_path}")
        if previous_completed_prefix is not None:
            decreases = completed_prefix + 1.0e-6 < previous_completed_prefix
            if np.any(decreases):
                progress_monotonic_violations += 1
        if completed_prefix.size > 1 and np.any(
            completed_prefix[1:] > completed_prefix[:-1] + 1.0e-6
        ):
            prefix_shape_violations += 1
        previous_completed_prefix = completed_prefix
        path_progress = float(progress_supervision.get("path_progress", 0.0))
        if previous_path_progress is not None and path_progress + 1.0e-6 < previous_path_progress:
            path_progress_violations += 1
        previous_path_progress = path_progress
        progress_valid_frames += int(bool(progress_supervision.get("supervision_valid", False)))

        frontiers = spatial_observation.get("frontiers", [])
        frontier_ids = [int(item["frontier_id"]) for item in frontiers]
        frontier_counts.append(len(frontiers))
        if frontier_ids != [
            int(value) for value in spatial_supervision.get("frontier_ids", [])
        ]:
            raise RuntimeError(f"Student/target frontier IDs differ in {sample_path}")
        for frontier in frontiers:
            if not bool(frontier.get("reachable", False)):
                raise RuntimeError(f"Unreachable frontier leaked into student candidates: {sample_path}")
            if "boundary_world_xy" not in frontier or "navigation_goal_world_xy" not in frontier:
                raise RuntimeError(f"Frontier lacks boundary/goal separation: {sample_path}")
        values = [
            float(value) for value in spatial_supervision.get("frontier_value_targets", [])
        ]
        if len(values) != len(frontier_ids):
            raise RuntimeError(f"Frontier target length mismatch in {sample_path}")
        selected = spatial_supervision.get("selected_frontier_id")
        oracle_reason = str(sample.get("debug_oracle", {}).get("oracle_reason", ""))
        oracle_reasons[oracle_reason or "missing"] += 1
        if selected is not None and int(selected) not in frontier_ids:
            raise RuntimeError(f"Selected frontier is absent from candidates in {sample_path}")
        if selected is None and any(abs(value) > 1.0e-8 for value in values):
            raise RuntimeError(f"Unreliable frame still carries soft frontier labels: {sample_path}")
        if selected is not None and oracle_reason != "reliable_route_frontier_crossing":
            raise RuntimeError(f"Hard frontier label has unreliable reason in {sample_path}")
        spatial_valid = bool(spatial_supervision.get("supervision_valid", False))
        if spatial_valid != (selected is not None):
            raise RuntimeError(f"Spatial supervision validity mismatch in {sample_path}")
        reliable_frontier_labels += int(spatial_valid)
        quality_flags.update(str(value) for value in sample.get("quality_flags", []))
        for relative_path in frame.get("paths", {}).values():
            if not (episode_dir / str(relative_path)).is_file():
                raise FileNotFoundError(f"Missing collected artifact: {relative_path}")
        paths = frame.get("paths", {})
        if progress_observation.get("current_rgb_path") != paths.get("rgb"):
            raise RuntimeError(f"Progress RGB path mismatch in {sample_path}")
        if spatial_observation.get("current_rgb_path") != paths.get("rgb"):
            raise RuntimeError(f"Spatial RGB path mismatch in {sample_path}")
        if spatial_observation.get("local_map_path") != paths.get("local_map"):
            raise RuntimeError(f"Spatial local-map path mismatch in {sample_path}")
        if spatial_observation.get("global_map_path") != paths.get("global_map"):
            raise RuntimeError(f"Spatial global-map path mismatch in {sample_path}")
        frame_record = json.loads(
            (episode_dir / str(paths["rgb"])).with_name("frame.json").read_text(encoding="utf-8")
        )
        pose = frame_record.get("pose", {})
        view_matrix = np.asarray(
            frame_record.get("calibration", {}).get("camera_view_matrix_ros", []),
            dtype=np.float64,
        )
        if view_matrix.shape != (4, 4):
            raise RuntimeError(f"Invalid camera view matrix in {sample_path}: {view_matrix.shape}")
        require_changed = (
            float(pose.get("delta_distance_m", 0.0)) > 1.0e-4
            or abs(float(pose.get("delta_yaw_rad", 0.0))) > 1.0e-4
        )
        rgb = np.asarray(Image.open(episode_dir / str(paths["rgb"])).convert("RGB"))
        with np.load(episode_dir / str(paths["depth"])) as depth_payload:
            depth = np.asarray(depth_payload["depth_m"], dtype=np.float32)
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        depth_valid_ratios.append(float(np.mean(valid_depth)))

        with np.load(episode_dir / str(paths["local_map"])) as local_payload:
            local_occupancy = np.asarray(local_payload["occupancy"], dtype=np.uint8)
        with np.load(episode_dir / str(paths["global_map"])) as global_payload:
            global_occupancy = np.asarray(global_payload["occupancy"], dtype=np.uint8)
        _validate_occupancy_channels(local_occupancy, path=sample_path, name="local_map")
        _validate_occupancy_channels(global_occupancy, path=sample_path, name="global_map")
        local_map_shapes[str(tuple(local_occupancy.shape))] += 1
        global_map_shapes[str(tuple(global_occupancy.shape))] += 1
        known_ratio = float(np.mean((global_occupancy[0] > 0) | (global_occupancy[1] > 0)))
        known_ratios.append(known_ratio)
        if previous_known_ratio is not None and known_ratio + 1.0e-4 < previous_known_ratio:
            known_ratio_decrease_frames += 1
        previous_known_ratio = known_ratio

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
            moving_frames += 1
            rgb_changed_frames += int(not rgb_same)
            depth_changed_frames += int(not depth_same)
        if (
            require_changed
            and previous_view_matrix is not None
            and np.allclose(view_matrix, previous_view_matrix, rtol=0.0, atol=1.0e-7)
        ):
            raise RuntimeError(
                f"Fixed camera view matrix on moving replay frame in {sample_path}: "
                f"delta_distance_m={pose.get('delta_distance_m')} "
                f"delta_yaw_rad={pose.get('delta_yaw_rad')}"
            )
        if require_changed and previous_view_matrix is not None:
            view_changed_frames += 1
        previous_rgb = rgb
        previous_depth = depth
        previous_view_matrix = view_matrix
    if progress_monotonic_violations:
        raise RuntimeError(
            f"Semantic progress decreased in {progress_monotonic_violations} frame(s): {episode_dir}"
        )
    if path_progress_violations:
        raise RuntimeError(
            f"Path progress decreased in {path_progress_violations} frame(s): {episode_dir}"
        )
    if prefix_shape_violations:
        raise RuntimeError(
            f"Completed-prefix target is not prefix-shaped in {prefix_shape_violations} frame(s)"
        )

    frame_count = len(frames)
    report = {
        "status": "pass",
        "schema_version": TRAINING_SAMPLE_SCHEMA_VERSION,
        "episode_dir": str(episode_dir),
        "frame_count": frame_count,
        "rgbd": {
            "moving_frames": moving_frames,
            "rgb_changed_moving_rate": _safe_ratio(rgb_changed_frames, moving_frames),
            "depth_changed_moving_rate": _safe_ratio(depth_changed_frames, moving_frames),
            "view_matrix_changed_moving_rate": _safe_ratio(view_changed_frames, moving_frames),
            "depth_valid_ratio_mean": _mean(depth_valid_ratios),
            "depth_valid_ratio_min": min(depth_valid_ratios, default=0.0),
        },
        "maps": {
            "local_shapes": dict(local_map_shapes),
            "global_shapes": dict(global_map_shapes),
            "known_ratio_first": known_ratios[0] if known_ratios else 0.0,
            "known_ratio_final": known_ratios[-1] if known_ratios else 0.0,
            "known_ratio_growth": (
                known_ratios[-1] - known_ratios[0] if len(known_ratios) >= 2 else 0.0
            ),
            "known_ratio_decrease_frames": known_ratio_decrease_frames,
        },
        "progress": {
            "valid_supervision_frames": progress_valid_frames,
            "valid_supervision_rate": _safe_ratio(progress_valid_frames, frame_count),
            "monotonic_violations": progress_monotonic_violations,
            "prefix_shape_violations": prefix_shape_violations,
            "path_progress_violations": path_progress_violations,
        },
        "spatial_subgoal": {
            "frontier_count_mean": _mean(frontier_counts),
            "frontier_count_min": min(frontier_counts, default=0),
            "frontier_count_max": max(frontier_counts, default=0),
            "reliable_label_frames": reliable_frontier_labels,
            "reliable_label_rate": _safe_ratio(reliable_frontier_labels, frame_count),
            "unlabeled_frame_rate": 1.0 - _safe_ratio(reliable_frontier_labels, frame_count),
            "oracle_reasons": dict(oracle_reasons),
        },
        "quality_flags": dict(quality_flags),
    }
    print(f"validated collected episode: frames={frame_count} path={episode_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _validate_occupancy_channels(
    occupancy: np.ndarray,
    *,
    path: Path,
    name: str,
) -> None:
    values = np.asarray(occupancy)
    if values.ndim != 3 or values.shape[0] != 3:
        raise RuntimeError(f"{name} must have shape [3,H,W] in {path}; got {values.shape}")
    active = values > 0
    if np.any(np.sum(active.astype(np.uint8), axis=0) != 1):
        raise RuntimeError(f"{name} occupied/free/unknown channels overlap or contain gaps in {path}")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else float(numerator) / float(denominator)


def _mean(values) -> float:
    return 0.0 if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


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
