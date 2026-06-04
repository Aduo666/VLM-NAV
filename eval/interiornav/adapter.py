"""Convert InteriorNav episode format to nav.py-compatible task format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def normalize_episode(episode: dict, *, tasks_path: Path, source_root: str | None = None) -> dict:
    """Convert a single InteriorNav episode to the format nav.py expects."""
    normalized = dict(episode)
    scene_id = str(episode.get("scene_id", episode.get("scan", ""))).strip()
    split = _infer_split_from_task_path(tasks_path)
    episode_id = episode.get("episode_id", episode.get("trajectory_id", episode.get("id", "")))

    if not str(normalized.get("id", "")).strip():
        if scene_id and episode_id:
            normalized["id"] = f"{split}_{scene_id}_{episode_id}" if split else f"{scene_id}_{episode_id}"
        else:
            normalized["id"] = f"task_{abs(hash(json.dumps(episode, sort_keys=True, default=str))) % 1000000}"

    if "instruction" not in normalized:
        normalized["instruction"] = ""

    # Map InteriorNav field names to nav.py field names
    if "robot_start_position" not in normalized and isinstance(episode.get("start_position"), list):
        normalized["robot_start_position"] = episode["start_position"]

    if "robot_start_rotation_wxyz" not in normalized and isinstance(episode.get("start_rotation"), list):
        normalized["robot_start_rotation_wxyz"] = episode["start_rotation"]

    if "robot_start_orientation_deg" not in normalized and isinstance(normalized.get("robot_start_rotation_wxyz"), list):
        try:
            normalized["robot_start_orientation_deg"] = _quaternion_wxyz_to_euler_xyz_deg(
                np.asarray(normalized["robot_start_rotation_wxyz"][:4], dtype=np.float64)
            )
        except Exception:
            pass

    if "goal_radius_m" not in normalized and isinstance(episode.get("goals"), dict):
        radius = episode["goals"].get("radius")
        if radius is not None:
            normalized["goal_radius_m"] = float(radius)
    if "success_distance_m" not in normalized and normalized.get("goal_radius_m") is not None:
        normalized["success_distance_m"] = float(normalized["goal_radius_m"])

    if "arrival_distance_m" not in normalized:
        normalized["arrival_distance_m"] = 1.4

    # Infer scene USD path from scene_id
    if not str(normalized.get("scene_usd_path", "")).strip() and scene_id:
        scene_path = _infer_scene_usd_path(tasks_path=tasks_path, scene_id=scene_id, source_root=source_root)
        if scene_path is not None:
            normalized["scene_usd_path"] = str(scene_path)

    # Flatten nested instruction dict
    normalized["instruction"] = _task_instruction_text(normalized)
    return normalized


def _task_instruction_text(task: dict) -> str:
    instruction = task.get("instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", "")).strip()
    return str(instruction).strip()


def _infer_split_from_task_path(tasks_path: Path) -> str:
    stem = tasks_path.name
    if stem.endswith(".json.gz"):
        stem = stem[: -len(".json.gz")]
    elif stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return stem


def _infer_scene_usd_path(*, tasks_path: Path, scene_id: str, source_root: str | None = None) -> Path | None:
    if source_root:
        data_root = Path(str(source_root)).expanduser()
        candidate = data_root / "scene_data" / scene_id / f"{scene_id}.usda"
        if candidate.exists():
            return candidate
    for parent in [tasks_path.parent, *tasks_path.parents]:
        if parent.name == "raw_data":
            data_root = parent.parent
            candidate = data_root / "scene_data" / scene_id / f"{scene_id}.usda"
            if candidate.exists():
                return candidate
    if tasks_path.parent.parent.name == "raw_data":
        data_root = tasks_path.parent.parent.parent
        candidate = data_root / "scene_data" / scene_id / f"{scene_id}.usda"
        return candidate
    return None


def _quaternion_wxyz_to_euler_xyz_deg(quaternion_wxyz: np.ndarray) -> list[float]:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]
