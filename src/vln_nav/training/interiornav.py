from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .schemas import CANONICAL_EPISODE_SCHEMA_VERSION, CanonicalEpisode, RoomRegion


DEFAULT_INTERIORNAV_ROOT = Path("/Data/qiang/interiornav_data")


def split_path(data_root: str | Path, split: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    split_name = str(split).strip()
    return root / "raw_data" / split_name / f"{split_name}.json.gz"


def load_canonical_episodes(
    path: str | Path,
    *,
    split: str | None = None,
    source_root: str | Path | None = None,
) -> list[CanonicalEpisode]:
    dataset_path = Path(path).expanduser().resolve()
    payload = _read_json(dataset_path)
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError(f"InteriorNav file has no episodes list: {dataset_path}")
    root = _resolve_source_root(dataset_path=dataset_path, source_root=source_root)
    split_name = str(split or payload.get("split", "") or _infer_split(dataset_path)).strip()
    records = [
        _normalize_episode(episode, index=index, split=split_name, source_root=root)
        for index, episode in enumerate(episodes)
        if isinstance(episode, dict)
    ]
    if not records:
        raise ValueError(f"InteriorNav file contains no usable episodes: {dataset_path}")
    return records


def load_canonical_episode(
    *,
    data_root: str | Path,
    split: str,
    episode_index: int | None = None,
    episode_id: str | None = None,
) -> CanonicalEpisode:
    records = load_canonical_episodes(
        split_path(data_root, split),
        split=split,
        source_root=data_root,
    )
    requested_id = str(episode_id or "").strip()
    if requested_id:
        for record in records:
            if record.episode_id == requested_id:
                return record
        raise KeyError(f"InteriorNav episode_id={requested_id!r} not found in split={split}")
    index = 0 if episode_index is None else int(episode_index)
    if not 0 <= index < len(records):
        raise IndexError(f"episode_index={index} out of range for split={split}, count={len(records)}")
    return records[index]


def nearest_path_index(path_xy: np.ndarray, robot_xy: tuple[float, float]) -> int:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        raise ValueError("path_xy must have shape [N, 2]")
    delta = path[:, :2] - np.asarray(robot_xy, dtype=np.float64)[None, :]
    return int(np.argmin(np.einsum("ij,ij->i", delta, delta)))


def _normalize_episode(
    episode: dict[str, Any],
    *,
    index: int,
    split: str,
    source_root: Path,
) -> CanonicalEpisode:
    episode_id = str(
        episode.get("episode_id", episode.get("trajectory_id", episode.get("id", index)))
    ).strip()
    scene_id = str(episode.get("scene_id", episode.get("scan", ""))).strip()
    if not scene_id:
        raise ValueError(f"Episode {episode_id} has no scene_id")

    instruction_value = episode.get("instruction", "")
    instruction = (
        str(instruction_value.get("instruction_text", "")).strip()
        if isinstance(instruction_value, dict)
        else str(instruction_value).strip()
    )
    raw_tokens = (
        instruction_value.get("instruction_tokens", [])
        if isinstance(instruction_value, dict)
        else []
    )
    tokens = tuple(int(value) for value in raw_tokens if int(value) != 0)

    reference_path = _xyz_array(episode.get("reference_path", []), field_name="reference_path")
    if reference_path.shape[0] < 2:
        raise ValueError(f"Episode {episode_id} reference_path needs at least two points")
    start = _xyz_vector(
        episode.get("start_position", episode.get("robot_start_position", reference_path[0])),
        default_z=float(reference_path[0, 2]),
    )
    rotation = _wxyz_quaternion(
        episode.get("start_rotation", episode.get("robot_start_rotation_wxyz")),
        reference_path=reference_path,
    )
    goals = episode.get("goals", {})
    goal_value = goals.get("position") if isinstance(goals, dict) else None
    goal = _xyz_vector(reference_path[-1] if goal_value is None else goal_value)
    goal_radius = float(
        goals.get("radius", 1.0) if isinstance(goals, dict) else 1.0
    )
    info = episode.get("info", {})
    geodesic = info.get("geodesic_distance") if isinstance(info, dict) else None
    scene_usd_path = source_root / "scene_data" / scene_id / f"{scene_id}.usda"
    rooms = _load_rooms(source_root=source_root, scene_id=scene_id)
    return CanonicalEpisode(
        schema_version=CANONICAL_EPISODE_SCHEMA_VERSION,
        source_dataset="interiornav",
        split=str(split),
        episode_id=episode_id,
        scene_id=scene_id,
        scene_usd_path=str(scene_usd_path),
        instruction=instruction,
        instruction_words=tuple(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", instruction)),
        source_instruction_token_ids=tokens,
        start_xyz=tuple(float(value) for value in start),
        start_rotation_wxyz=tuple(float(value) for value in rotation),
        reference_path_xyz=tuple(tuple(float(value) for value in point) for point in reference_path),
        goal_xyz=tuple(float(value) for value in goal),
        goal_radius_m=max(0.05, goal_radius),
        geodesic_distance_m=None if geodesic is None else float(geodesic),
        rooms=rooms,
    )


def _load_rooms(*, source_root: Path, scene_id: str) -> tuple[RoomRegion, ...]:
    rooms_path = source_root / "scene_data" / scene_id / "rooms.json"
    if not rooms_path.is_file():
        return ()
    payload = json.loads(rooms_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return ()
    result: list[RoomRegion] = []
    for room in payload:
        if not isinstance(room, dict):
            continue
        polygon = np.asarray(room.get("polygon", []), dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
            continue
        result.append(
            RoomRegion(
                label=str(room.get("room_type", "unknown")).strip().lower() or "unknown",
                polygon_xy=tuple((float(point[0]), float(point[1])) for point in polygon),
            )
        )
    return tuple(result)


def _read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_source_root(*, dataset_path: Path, source_root: str | Path | None) -> Path:
    if source_root is not None:
        return Path(source_root).expanduser().resolve()
    for parent in dataset_path.parents:
        if parent.name == "raw_data":
            return parent.parent
    return DEFAULT_INTERIORNAV_ROOT.resolve()


def _xyz_array(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{field_name} must have shape [N,2+]" )
    if array.shape[1] == 2:
        array = np.concatenate([array, np.zeros((array.shape[0], 1), dtype=np.float64)], axis=1)
    return array[:, :3]


def _xyz_vector(value: Any, *, default_z: float = 0.0) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 2:
        raise ValueError("position must contain at least x and y")
    if vector.size < 3:
        vector = np.asarray([vector[0], vector[1], default_z], dtype=np.float64)
    return vector[:3]


def _wxyz_quaternion(value: Any, *, reference_path: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    if quaternion.size >= 4 and np.linalg.norm(quaternion[:4]) > 1.0e-8:
        quaternion = quaternion[:4] / np.linalg.norm(quaternion[:4])
        return quaternion
    direction = reference_path[1, :2] - reference_path[0, :2]
    yaw = float(np.arctan2(direction[1], direction[0]))
    return np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


def _infer_split(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem
