"""Convert VLN-PE (R2R) episode format to nav.py-compatible task format."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def normalize_episode(
    episode: dict,
    *,
    tasks_path: Path,
    source_root: str | None = None,
) -> dict:
    """Convert a VLN-PE (R2R) episode to nav.py format.

    R2R format differs from InteriorNav:
      - start_rotation: [x, y, z, w] (not wxyz)
      - scene_id: "mp3d/XXXX/XXXX.glb"
      - goals: list of {"position": [...], "radius": float}
      - instruction: nested dict {"instruction_text": "..."}
    """
    normalized = dict(episode)
    scene_key = str(episode.get("scene_id", "")).strip()
    # Extract scene hash from "mp3d/<hash>/<hash>.glb"
    scene_hash = ""
    if scene_key:
        parts = scene_key.replace("/", " ").split()
        for p in parts:
            p = p.replace(".glb", "").strip()
            if len(p) >= 10:
                scene_hash = p
                break
    split = _infer_split_from_task_path(tasks_path)
    episode_id = episode.get("episode_id", episode.get("trajectory_id", ""))

    if not str(normalized.get("id", "")).strip():
        if scene_hash and episode_id:
            normalized["id"] = f"{split}_{scene_hash}_{episode_id}" if split else f"{scene_hash}_{episode_id}"
        else:
            normalized["id"] = f"task_{abs(hash(json.dumps(episode, sort_keys=True, default=str))) % 1000000}"

    if "instruction" not in normalized:
        normalized["instruction"] = ""

    # Map R2R start_position
    if "robot_start_position" not in normalized and isinstance(episode.get("start_position"), list):
        normalized["robot_start_position"] = episode["start_position"]

    # R2R start_rotation is [x, y, z, w] - convert to [w, x, y, z] first
    rot = episode.get("start_rotation", [])
    if isinstance(rot, list) and len(rot) >= 4:
        # R2R format: quaternion [x, y, z, w]
        # Our converter expects [w, x, y, z]
        if "robot_start_rotation_wxyz" not in normalized and "robot_start_orientation_deg" not in normalized:
            wxyz = [float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])]
            normalized["robot_start_rotation_wxyz"] = wxyz
            try:
                normalized["robot_start_orientation_deg"] = _quaternion_wxyz_to_euler_xyz_deg(
                    np.asarray(wxyz, dtype=np.float64)
                )
            except Exception:
                pass

    # Goals -> success_distance_m
    goals = episode.get("goals", [])
    if isinstance(goals, list) and goals:
        radius = goals[0].get("radius")
        if radius is not None:
            normalized["goal_radius_m"] = float(radius)
    if "success_distance_m" not in normalized and normalized.get("goal_radius_m") is not None:
        normalized["success_distance_m"] = float(normalized["goal_radius_m"])
    if "arrival_distance_m" not in normalized:
        normalized["arrival_distance_m"] = 1.4

    # Infer scene USD path
    if not str(normalized.get("scene_usd_path", "")).strip() and scene_hash:
        scene_path = _infer_scene_usd_path(tasks_path=tasks_path, scene_hash=scene_hash, source_root=source_root)
        if scene_path is not None:
            normalized["scene_usd_path"] = str(scene_path)

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


def _infer_scene_usd_path(*, tasks_path: Path, scene_hash: str, source_root: str | None = None) -> Path | None:
    """Find a USD file for the given scene hash under the MP3D data root."""
    if source_root:
        scans_dir = Path(str(source_root)).expanduser() / "scans"
        if scans_dir.exists():
            for candidate_dir in scans_dir.iterdir():
                if candidate_dir.is_dir() and candidate_dir.name == scene_hash:
                    mesh_dir = candidate_dir / "matterport_mesh"
                    if mesh_dir.exists():
                        # Prefer fixed.usd/fixed.usda over non_metric files
                        candidates: list[Path] = []
                        for subdir in mesh_dir.iterdir():
                            for f in subdir.iterdir():
                                if f.suffix in (".usd", ".usda"):
                                    candidates.append(f)
                        fixed = [p for p in candidates if p.name.startswith("fixed")]
                        # Prefer exact "fixed.usd/.usda" over "fixed_docker.usd"
                        exact = [p for p in fixed if p.stem == "fixed"]
                        if exact:
                            return exact[0]
                        if fixed:
                            return fixed[0]
    # Try from tasks_path parent chain
    for parent in [tasks_path.parent, *tasks_path.parents]:
        if parent.name == "raw_data":
            data_root = parent.parent.parent.parent  # .../InternData-N1/vln_pe
            scans_dir = data_root.parent / "mp3d_pe" / "scans"
            if not scans_dir.exists():
                scans_dir = data_root.parent.parent / "mp3d_pe" / "scans"  # try /Data/qiang/mp3d_pe/scans
            if scans_dir.exists():
                for candidate_dir in scans_dir.iterdir():
                    if candidate_dir.name == scene_hash:
                        return _find_first_usd(candidate_dir)
    return None


def _find_first_usd(scene_dir: Path) -> Path | None:
    """Recursively find first .usd/.usda file in scene directory, preferring fixed.usd."""
    candidates: list[Path] = []
    for subdir in scene_dir.rglob("*"):
        if subdir.is_file() and subdir.suffix in (".usd", ".usda"):
            candidates.append(subdir)
    fixed = [p for p in candidates if p.stem == "fixed"]
    if fixed:
        return fixed[0]
    return candidates[0] if candidates else None


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


