from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

from vln_nav.active_perception.local_geometry import LocalVoxelSnapshot

from .map_state import snapshot_arrays

DEPTH_INVALID_MM = np.uint16(0)
DEPTH_NO_RETURN_MM = np.uint16(65535)
DEPTH_MAX_FINITE_MM = 65534
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SampleRecord:
    sample_time_sec: float
    front_timestamp_sec: float
    down_timestamp_sec: float
    front_rgb: np.ndarray
    front_depth_m: np.ndarray
    down_depth_m: np.ndarray
    bev: np.ndarray
    pose_world_xyyaw: np.ndarray
    world_position_xyz: np.ndarray
    world_orientation_wxyz: np.ndarray
    measured_velocity_vw: np.ndarray
    human_action_vw: np.ndarray
    applied_action_vw: np.ndarray
    steering_position: np.ndarray
    steering_velocity: np.ndarray
    wheel_velocity: np.ndarray
    front_intrinsics: np.ndarray
    down_intrinsics: np.ndarray
    front_base_from_camera: np.ndarray
    down_base_from_camera: np.ndarray
    front_world_from_camera: np.ndarray
    down_world_from_camera: np.ndarray
    map_version: int
    state_timestamp_sec: float
    action_timestamp_sec: float
    physics_step: int
    action_physics_step: int
    front_physics_step: int
    down_physics_step: int
    map_integrated: bool
    floor_z_m: float


@dataclass(frozen=True)
class ControlRecord:
    timestamp_sec: float
    physics_step: int
    pose_world_xyyaw: np.ndarray
    world_position_xyz: np.ndarray
    world_orientation_wxyz: np.ndarray
    measured_velocity_vw: np.ndarray
    human_action_vw: np.ndarray
    applied_action_vw: np.ndarray
    steering_position: np.ndarray
    steering_velocity: np.ndarray
    wheel_velocity: np.ndarray


class EpisodeRecorder:

    def __init__(
        self,
        *,
        output_root: Path,
        scene_path: Path,
        jpeg_quality: int = 92,
        png_compression: int = 1,
        max_pending_writes: int = 12,
    ) -> None:
        self.output_root = Path(output_root)
        self.scene_path = Path(scene_path)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.png_compression = int(np.clip(png_compression, 0, 9))
        self.max_pending_writes = max(1, int(max_pending_writes))
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="constraint-traverse-writer",
        )
        self._pending: deque[Future] = deque()
        self._active = False
        self._episode_dir: Path | None = None
        self._final_dir: Path | None = None
        self._meta_seed: dict = {}
        self._state: dict[str, list] = {}
        self._front_intrinsics: np.ndarray | None = None
        self._down_intrinsics: np.ndarray | None = None
        self._controls: dict[str, list] = {}
        self._checkpoint_samples = 0
        self._checkpoint_controls = 0
        self._checkpoint_index = 0
        self._last_checkpoint_time = -float("inf")

    @property
    def active(self) -> bool:
        return self._active

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    def begin(
        self,
        *,
        start_pose_world: np.ndarray,
        start_position_world_xyz: np.ndarray,
        start_time_sec: float,
        initial_map: LocalVoxelSnapshot,
        extra_meta: dict | None = None,
    ) -> Path:
        if self._active:
            raise RuntimeError("Recorder already has an active episode")
        scene_dir = self.output_root / self.scene_path.stem
        scene_dir.mkdir(parents=True, exist_ok=True)
        index = _next_episode_index(scene_dir)
        final_dir = scene_dir / f"episode_{index:06d}"
        temp_dir = scene_dir / f".episode_{index:06d}.tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        for name in ("front_rgb", "front_depth", "down_depth", "bev", "chunks"):
            (temp_dir / name).mkdir(parents=True, exist_ok=True)

        self._episode_dir = temp_dir
        self._final_dir = final_dir
        self._active = True
        self._front_intrinsics = None
        self._down_intrinsics = None
        self._state = {
            "sample_time_sec": [],
            "front_timestamp_sec": [],
            "down_timestamp_sec": [],
            "pose_world_xyyaw": [],
            "world_position_xyz": [],
            "world_orientation_wxyz": [],
            "measured_velocity_vw": [],
            "human_action_vw": [],
            "applied_action_vw": [],
            "steering_position": [],
            "steering_velocity": [],
            "wheel_velocity": [],
            "front_base_from_camera": [],
            "down_base_from_camera": [],
            "front_world_from_camera": [],
            "down_world_from_camera": [],
            "map_version": [],
            "state_timestamp_sec": [],
            "action_timestamp_sec": [],
            "physics_step": [],
            "action_physics_step": [],
            "front_physics_step": [],
            "down_physics_step": [],
            "map_integrated": [],
            "floor_z_m": [],
        }
        self._controls = {name: [] for name in ControlRecord.__dataclass_fields__}
        self._checkpoint_samples = self._checkpoint_controls = self._checkpoint_index = 0
        self._last_checkpoint_time = float(start_time_sec)
        self._meta_seed = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": f"{index:06d}",
            "scene_name": self.scene_path.stem,
            "usd_path": str(self.scene_path),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_revision": _repository_revision(),
            "start_pose_world_xyyaw": _json_vector(start_pose_world),
            "start_position_world_xyz": _json_vector(start_position_world_xyz),
            "goal_world_xy": None,
            "start_time_sec": float(start_time_sec),
            "status": "recording",
            "success": False,
            "success_source": "human",
            "sample_count": 0,
            "control_count": 0,
            "valid": None,
            "validation_errors": [],
            "depth_encoding": {
                "format": "uint16_png_millimeters",
                "invalid_value": int(DEPTH_INVALID_MM),
                "no_return_value": int(DEPTH_NO_RETURN_MM),
            },
            "bev_channel_order": ["occupied", "free", "unknown"],
        }
        if extra_meta:
            self._meta_seed.update(extra_meta)
        _atomic_savez(temp_dir / "initial_map.npz", snapshot_arrays(initial_map))
        _atomic_json(temp_dir / "meta.json", self._meta_seed)
        return temp_dir

    def append_control(self, control: ControlRecord) -> None:
        if not self._active:
            raise RuntimeError("Recorder is not active")
        for name in self._controls:
            self._controls[name].append(np.asarray(getattr(control, name)).copy())
        if control.timestamp_sec - self._last_checkpoint_time >= 1.0:
            self._queue_checkpoint()

    def append(self, sample: SampleRecord) -> int:
        if not self._active or self._episode_dir is None:
            raise RuntimeError("Recorder is not active")
        index = len(self._state["sample_time_sec"])
        stem = f"{index:06d}"

        self._submit(
            _write_sample_files,
            episode_dir=self._episode_dir,
            stem=stem,
            front_rgb=np.asarray(sample.front_rgb, dtype=np.uint8).copy(),
            front_depth_m=np.asarray(sample.front_depth_m, dtype=np.float32).copy(),
            down_depth_m=np.asarray(sample.down_depth_m, dtype=np.float32).copy(),
            bev=np.asarray(sample.bev, dtype=np.uint8).copy(),
            jpeg_quality=self.jpeg_quality,
            png_compression=self.png_compression,
        )
        if self._front_intrinsics is None:
            self._front_intrinsics = np.asarray(sample.front_intrinsics, dtype=np.float64).copy()
            self._down_intrinsics = np.asarray(sample.down_intrinsics, dtype=np.float64).copy()
        else:
            if not np.allclose(self._front_intrinsics, sample.front_intrinsics, atol=1.0e-6):
                raise RuntimeError("Front camera intrinsics changed during an episode")
            if not np.allclose(self._down_intrinsics, sample.down_intrinsics, atol=1.0e-6):
                raise RuntimeError("Down camera intrinsics changed during an episode")

        values = {
            "sample_time_sec": float(sample.sample_time_sec),
            "front_timestamp_sec": float(sample.front_timestamp_sec),
            "down_timestamp_sec": float(sample.down_timestamp_sec),
            "pose_world_xyyaw": np.asarray(sample.pose_world_xyyaw, dtype=np.float64),
            "world_position_xyz": np.asarray(sample.world_position_xyz, dtype=np.float64),
            "world_orientation_wxyz": np.asarray(sample.world_orientation_wxyz, dtype=np.float64),
            "measured_velocity_vw": np.asarray(sample.measured_velocity_vw, dtype=np.float32),
            "human_action_vw": np.asarray(sample.human_action_vw, dtype=np.float32),
            "applied_action_vw": np.asarray(sample.applied_action_vw, dtype=np.float32),
            "steering_position": np.asarray(sample.steering_position, dtype=np.float32),
            "steering_velocity": np.asarray(sample.steering_velocity, dtype=np.float32),
            "wheel_velocity": np.asarray(sample.wheel_velocity, dtype=np.float32),
            "front_base_from_camera": np.asarray(sample.front_base_from_camera, dtype=np.float32),
            "down_base_from_camera": np.asarray(sample.down_base_from_camera, dtype=np.float32),
            "front_world_from_camera": np.asarray(sample.front_world_from_camera, dtype=np.float64),
            "down_world_from_camera": np.asarray(sample.down_world_from_camera, dtype=np.float64),
            "map_version": int(sample.map_version),
            "state_timestamp_sec": float(sample.state_timestamp_sec),
            "action_timestamp_sec": float(sample.action_timestamp_sec),
            "physics_step": int(sample.physics_step),
            "action_physics_step": int(sample.action_physics_step),
            "front_physics_step": int(sample.front_physics_step),
            "down_physics_step": int(sample.down_physics_step),
            "map_integrated": bool(sample.map_integrated),
            "floor_z_m": float(sample.floor_z_m),
        }
        for key, value in values.items():
            self._state[key].append(np.asarray(value).copy())
        self._queue_checkpoint()
        return index

    def _submit(self, function, *args, **kwargs) -> None:
        self._pending.append(self._executor.submit(function, *args, **kwargs))
        if len(self._pending) >= self.max_pending_writes:
            self._pending.popleft().result()

    def _queue_checkpoint(self) -> None:
        if self._episode_dir is None:
            return
        samples = len(self._state["sample_time_sec"])
        controls = len(self._controls["timestamp_sec"])
        if (samples, controls) == (self._checkpoint_samples, self._checkpoint_controls):
            return
        arrays = {
            "sample_start": np.asarray(self._checkpoint_samples),
            "control_start": np.asarray(self._checkpoint_controls),
            **{
                f"state_{key}": np.asarray(values[self._checkpoint_samples:])
                for key, values in self._state.items()
            },
            **{
                f"control_{key}": np.asarray(values[self._checkpoint_controls:])
                for key, values in self._controls.items()
            },
        }
        if self._front_intrinsics is not None:
            arrays["front_intrinsics"] = self._front_intrinsics.copy()
            arrays["down_intrinsics"] = self._down_intrinsics.copy()
        # 同一写入线程先完成传感器文件，再原子提交这一批状态。
        self._submit(
            _atomic_savez,
            self._episode_dir / "chunks" / f"{self._checkpoint_index:06d}.npz",
            arrays,
        )
        self._checkpoint_samples, self._checkpoint_controls = samples, controls
        self._checkpoint_index += 1
        if controls:
            self._last_checkpoint_time = float(self._controls["timestamp_sec"][-1])

    def finish(
        self,
        *,
        success: bool,
        end_pose_world: np.ndarray,
        end_time_sec: float,
        failure_reason: str | None = None,
        success_requested_time_sec: float | None = None,
    ) -> Path:
        if not self._active or self._episode_dir is None or self._final_dir is None:
            raise RuntimeError("Recorder is not active")
        self._queue_checkpoint()
        self._flush_pending()
        count = len(self._state["sample_time_sec"])
        if count == 0:
            raise ValueError("示范没有同步样本")
        end_pose = np.asarray(self._state["pose_world_xyyaw"][-1], dtype=np.float64)
        if not np.allclose(end_pose, end_pose_world, atol=1.0e-6):
            raise ValueError("结束位姿必须对应最后一个同步样本")
        end_time = float(self._state["sample_time_sec"][-1])
        if abs(end_time - end_time_sec) > 1.0e-6:
            raise ValueError("结束时间必须对应最后一个同步样本")

        arrays = {
            key: np.asarray(values)
            for key, values in self._state.items()
        }
        arrays["front_intrinsics"] = (
            np.asarray(self._front_intrinsics, dtype=np.float64)
            if self._front_intrinsics is not None
            else np.empty((0, 0), dtype=np.float64)
        )
        arrays["down_intrinsics"] = (
            np.asarray(self._down_intrinsics, dtype=np.float64)
            if self._down_intrinsics is not None
            else np.empty((0, 0), dtype=np.float64)
        )
        _atomic_savez(self._episode_dir / "state.npz", arrays)
        _atomic_savez(self._episode_dir / "controls.npz", {
            key: np.asarray(values) for key, values in self._controls.items()
        })

        start_time = float(self._meta_seed["start_time_sec"])
        meta = {
            **self._meta_seed,
            "status": "complete",
            "success": bool(success),
            "goal_world_xy": _json_vector(end_pose[:2]) if success else None,
            "success_requested_time_sec": success_requested_time_sec,
            "failure_reason": None if success else str(failure_reason or "unspecified"),
            "end_pose_world_xyyaw": _json_vector(end_pose),
            "end_time_sec": end_time,
            "duration_sec": max(0.0, end_time - start_time),
            "sample_count": int(count),
            "control_count": len(self._controls["timestamp_sec"]),
            "valid": None,
            "validation_errors": [],
        }
        _atomic_json(self._episode_dir / "meta.json", meta)

        errors = validate_episode_directory(self._episode_dir)
        meta["valid"] = not errors
        meta["validation_errors"] = errors
        _atomic_json(self._episode_dir / "meta.json", meta)

        if self._final_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {self._final_dir}")
        self._episode_dir.rename(self._final_dir)
        result = self._final_dir
        if errors:
            logging.getLogger(__name__).error("采集数据校验失败：%s；%s", result, errors)
        self._clear()
        return result

    def discard(self) -> None:
        self._flush_pending(ignore_errors=True)
        if self._episode_dir is not None and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir)
        self._clear()

    def close(self) -> None:
        try:
            if self._active:
                self.interrupt("collector_closed")
            else:
                self._flush_pending()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def interrupt(self, reason: str) -> Path | None:
        if self._episode_dir is None:
            return None
        path = self._episode_dir
        if not (path / "meta.json").is_file():
            self._flush_pending(ignore_errors=True)
            self._clear()
            return path
        self._queue_checkpoint()
        self._flush_pending(ignore_errors=True)
        result = recover_episode_directory(path, reason=reason)
        self._clear()
        return result

    def _flush_pending(self, *, ignore_errors: bool = False) -> None:
        while self._pending:
            future = self._pending.popleft()
            try:
                future.result()
            except Exception:
                if not ignore_errors:
                    raise

    def _clear(self) -> None:
        self._active = False
        self._episode_dir = None
        self._final_dir = None
        self._meta_seed = {}
        self._state = {}
        self._controls = {}
        self._front_intrinsics = None
        self._down_intrinsics = None


def validate_episode_directory(path: Path) -> list[str]:
    path = Path(path)
    errors: list[str] = []
    meta_path = path / "meta.json"
    state_path = path / "state.npz"
    if not meta_path.is_file():
        return ["missing meta.json"]
    if not state_path.is_file():
        return ["missing state.npz"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = int(meta.get("schema_version", 1))
        if version not in (1, SCHEMA_VERSION):
            errors.append(f"unsupported schema_version: {version}")
        if meta.get("valid") is False:
            errors.extend(meta.get("validation_errors") or ["metadata marks episode invalid"])
        if version == SCHEMA_VERSION and meta.get("status") != "complete":
            errors.append("episode is incomplete")
        with np.load(state_path, allow_pickle=False) as data:
            required = {
                "sample_time_sec",
                "front_timestamp_sec",
                "down_timestamp_sec",
                "pose_world_xyyaw",
                "world_position_xyz",
                "world_orientation_wxyz",
                "measured_velocity_vw",
                "human_action_vw",
                "applied_action_vw",
                "steering_position",
                "steering_velocity",
                "wheel_velocity",
                "front_base_from_camera",
                "down_base_from_camera",
                "map_version",
                "front_intrinsics",
                "down_intrinsics",
            }
            missing = sorted(required - set(data.files))
            if missing:
                errors.append(f"state.npz missing arrays: {missing}")
                return errors
            count = int(len(data["sample_time_sec"]))
            if count != int(meta.get("sample_count", -1)):
                errors.append("meta sample_count does not match state.npz")
            sequence_keys = required - {"front_intrinsics", "down_intrinsics"}
            for key in sequence_keys:
                if len(data[key]) != count:
                    errors.append(f"{key} length does not match sample count")
            if count == 0:
                errors.append("episode has no samples")
            else:
                expected_shapes = {
                    "pose_world_xyyaw": (count, 3),
                    "world_position_xyz": (count, 3),
                    "world_orientation_wxyz": (count, 4),
                    "measured_velocity_vw": (count, 2),
                    "human_action_vw": (count, 2),
                    "applied_action_vw": (count, 2),
                    "steering_position": (count, 4),
                    "steering_velocity": (count, 4),
                    "wheel_velocity": (count, 4),
                    "front_base_from_camera": (count, 4, 4),
                    "down_base_from_camera": (count, 4, 4),
                    "map_version": (count,),
                    "front_intrinsics": (3, 3),
                    "down_intrinsics": (3, 3),
                }
                for key, expected_shape in expected_shapes.items():
                    if tuple(data[key].shape) != expected_shape:
                        errors.append(
                            f"{key} shape {data[key].shape} != {expected_shape}"
                        )
                ts = np.asarray(data["sample_time_sec"], dtype=np.float64)
                if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                    errors.append("sample timestamps are not finite and strictly increasing")
                finite_keys = sequence_keys | {"front_intrinsics", "down_intrinsics"}
                for key in sorted(finite_keys):
                    if not np.isfinite(np.asarray(data[key], dtype=np.float64)).all():
                        errors.append(f"{key} contains non-finite values")
                if bool(meta.get("success")):
                    goal_value = meta.get("goal_world_xy")
                    if goal_value is None:
                        legacy_goal = meta.get("goal_pose_world_xyyaw")
                        goal_value = (
                            None if legacy_goal is None else legacy_goal[:2]
                        )
                    goal_world_xy = np.asarray(goal_value, dtype=np.float64)
                    if (
                        goal_world_xy.shape != (2,)
                        or not np.isfinite(goal_world_xy).all()
                    ):
                        errors.append("successful episode world goal XY is invalid")
                    else:
                        poses = np.asarray(
                            data["pose_world_xyyaw"], dtype=np.float64
                        )
                        tolerance = float(meta.get("collection", {}).get(
                            "success_goal_position_tolerance_m", 0.30
                        )) if version == 1 else 1.0e-6
                        if np.linalg.norm(
                            poses[-1, :2] - goal_world_xy
                        ) > tolerance:
                            errors.append("last world position exceeds goal tolerance")

                teleop = meta.get("teleop", {})
                if isinstance(teleop, dict):
                    action = np.asarray(data["human_action_vw"], dtype=np.float64)
                    max_v = float(teleop.get("max_linear_mps", np.inf))
                    max_w = float(teleop.get("max_angular_radps", np.inf))
                    if np.any(np.abs(action[:, 0]) > max_v + 1.0e-5):
                        errors.append("human linear action exceeds configured limit")
                    if np.any(np.abs(action[:, 1]) > max_w + 1.0e-5):
                        errors.append("human angular action exceeds configured limit")

                applied_limits = meta.get("applied_action_limits", {})
                if isinstance(applied_limits, dict):
                    action = np.asarray(data["applied_action_vw"], dtype=np.float64)
                    max_v = float(applied_limits.get("max_linear_mps", np.inf))
                    max_w = float(applied_limits.get("max_angular_radps", np.inf))
                    if np.any(np.abs(action[:, 0]) > max_v + 1.0e-5):
                        errors.append("applied linear action exceeds controller limit")
                    if np.any(np.abs(action[:, 1]) > max_w + 1.0e-5):
                        errors.append("applied angular action exceeds controller limit")

                cameras = meta.get("cameras", {})
                if isinstance(cameras, dict):
                    max_skew = float(cameras.get("max_frame_skew_sec", np.inf))
                    front_ts = np.asarray(data["front_timestamp_sec"], dtype=np.float64)
                    down_ts = np.asarray(data["down_timestamp_sec"], dtype=np.float64)
                    if np.any(np.abs(front_ts - down_ts) > max_skew + 1.0e-6):
                        errors.append("front/down timestamp skew exceeds configured limit")

                collection = meta.get("collection", {})
                if isinstance(collection, dict):
                    poses = np.asarray(data["pose_world_xyyaw"], dtype=np.float64)
                    start = np.asarray(
                        meta.get("start_pose_world_xyyaw", []), dtype=np.float64
                    )
                    end = np.asarray(
                        meta.get("end_pose_world_xyyaw", []), dtype=np.float64
                    )
                    position_tolerance = float(
                        collection.get("endpoint_position_tolerance_m", 0.15)
                    )
                    yaw_tolerance = np.radians(float(
                        collection.get("endpoint_yaw_tolerance_deg", 10.0)
                    ))
                    if start.shape != (3,) or end.shape != (3,):
                        errors.append("episode metadata start/end pose is invalid")
                    else:
                        if np.linalg.norm(poses[0, :2] - start[:2]) > position_tolerance:
                            errors.append("first sample position is too far from START pose")
                        if abs(_wrap_angle(float(poses[0, 2] - start[2]))) > yaw_tolerance:
                            errors.append("first sample yaw is too far from START pose")
                        if np.linalg.norm(poses[-1, :2] - end[:2]) > position_tolerance:
                            errors.append("last sample position is too far from end pose")
                        if abs(_wrap_angle(float(poses[-1, 2] - end[2]))) > yaw_tolerance:
                            errors.append("last sample yaw is too far from end pose")

                map_version = np.asarray(data["map_version"], dtype=np.int64)
                if np.any(np.diff(map_version) < 0):
                    errors.append("map_version is not monotonic")
                if version == SCHEMA_VERSION:
                    _validate_v2(path, meta, data, count, errors)

            expected_dirs = ("front_rgb", "front_depth", "down_depth", "bev")
            for dirname in expected_dirs:
                folder = path / dirname
                files = sorted(folder.glob("*")) if folder.is_dir() else []
                if len(files) != count:
                    errors.append(f"{dirname} file count {len(files)} != {count}")
            if count > 0 and all(
                len(list((path / dirname).glob("*"))) == count
                for dirname in expected_dirs
            ):
                _validate_sensor_files(path, meta, count, errors)
    except Exception as exc:
        errors.append(f"validation exception: {exc}")
    return errors


def _validate_v2(path: Path, meta: dict, data, count: int, errors: list[str]) -> None:
    from .bev import local_voxel_config_from_mapping
    from .map_state import load_snapshot
    from vln_nav.active_perception.local_geometry import LocalVoxelMap

    additional = {
        "state_timestamp_sec", "action_timestamp_sec", "physics_step",
        "action_physics_step", "front_physics_step", "down_physics_step",
        "map_integrated", "floor_z_m",
    }
    camera_transforms = {"front_world_from_camera", "down_world_from_camera"}
    missing = (additional | camera_transforms) - set(data.files)
    if missing:
        errors.append(f"state.npz missing arrays: {sorted(missing)}")
        return
    for name in additional:
        if data[name].shape != (count,) or not np.isfinite(data[name]).all():
            errors.append(f"{name} has invalid shape or values")
    if any(data[name].shape != (count,) for name in additional):
        return
    for name in camera_transforms:
        if data[name].shape != (count, 4, 4) or not np.isfinite(data[name]).all():
            errors.append(f"{name} has invalid shape or values")
    times = data["sample_time_sec"]
    if not np.allclose(data["state_timestamp_sec"], times, atol=1.0e-6, rtol=0):
        errors.append("state_timestamp_sec does not match observation time")
    if not np.allclose(np.maximum(data["front_timestamp_sec"], data["down_timestamp_sec"]), times, atol=1.0e-6, rtol=0):
        errors.append("sample time does not match latest camera observation")
    steps = data["physics_step"]
    if np.any(np.diff(steps) <= 0):
        errors.append("observation physics steps are not strictly increasing")
    if not np.array_equal(steps, np.maximum(data["front_physics_step"], data["down_physics_step"])):
        errors.append("observation physics step does not match latest camera observation")
    if np.any(data["action_timestamp_sec"] < times - 1.0e-6):
        errors.append("action precedes its observation")
    if times[0] < float(meta["start_time_sec"]) - 1.0e-6:
        errors.append("first observation precedes START")
    if abs(float(times[-1]) - float(meta["end_time_sec"])) > 1.0e-6:
        errors.append("end time does not match terminal observation")
    requested = meta.get("success_requested_time_sec")
    if bool(meta.get("success")) and (
        requested is None
        or min(data["front_timestamp_sec"][-1], data["down_timestamp_sec"][-1]) <= float(requested) + 1.0e-9
    ):
        errors.append("terminal observation precedes SUCCESS")
    if not np.allclose(data["pose_world_xyyaw"][-1], meta["end_pose_world_xyyaw"], atol=1e-6, rtol=0):
        errors.append("end pose does not match terminal observation")
    if bool(meta.get("success")):
        collection = meta["collection"]
        if count < int(collection["min_success_samples"]):
            errors.append("successful episode has too few samples")
        if (
            abs(data["measured_velocity_vw"][-1, 0]) > float(collection["start_max_linear_mps"])
            or abs(data["measured_velocity_vw"][-1, 1]) > float(collection["start_max_angular_radps"])
        ):
            errors.append("terminal observation is not stationary")
    snapshot = load_snapshot(path / "initial_map.npz")
    local_map = LocalVoxelMap(local_voxel_config_from_mapping(meta["voxel_map"]))
    local_map.restore(snapshot)
    if not np.array_equal(local_map.snapshot().state, snapshot.state):
        errors.append("initial map state does not match its probabilities")
    if snapshot.version > int(data["map_version"][0]):
        errors.append("map version precedes initial map")
    with np.load(path / "controls.npz", allow_pickle=False) as controls:
        required = set(ControlRecord.__dataclass_fields__)
        if not required.issubset(controls.files):
            errors.append("controls.npz is missing physical control arrays")
            return
        total = len(controls["timestamp_sec"])
        if total != int(meta.get("control_count", -1)) or total == 0:
            errors.append("control count does not match metadata")
            return
        shapes = {
            "timestamp_sec": (total,), "physics_step": (total,),
            "pose_world_xyyaw": (total, 3), "world_position_xyz": (total, 3),
            "world_orientation_wxyz": (total, 4), "measured_velocity_vw": (total, 2),
            "human_action_vw": (total, 2), "applied_action_vw": (total, 2),
            "steering_position": (total, 4), "steering_velocity": (total, 4),
            "wheel_velocity": (total, 4),
        }
        for name, shape in shapes.items():
            if controls[name].shape != shape or not np.isfinite(controls[name]).all():
                errors.append(f"control {name} has invalid shape or values")
        if any(controls[name].shape != shape for name, shape in shapes.items()):
            return
        control_steps = controls["physics_step"]
        control_times = controls["timestamp_sec"]
        if np.any(np.diff(control_steps) != 1) or np.any(np.diff(control_times) <= 0):
            errors.append("physical control sequence has missing or unordered steps")
        dt = float(meta["physics_dt_sec"])
        if not np.allclose(np.diff(control_times), dt, atol=1.0e-6, rtol=1.0e-5):
            errors.append("physical control timestamps do not follow physics_dt_sec")
        poses = controls["pose_world_xyyaw"]
        delta = np.diff(poses[:, :2], axis=0)
        forward_delta = np.cos(poses[:-1, 2]) * delta[:, 0] + np.sin(poses[:-1, 2]) * delta[:, 1]
        yaw_delta = np.diff(poses[:, 2])
        yaw_delta = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
        integrated = controls["measured_velocity_vw"][1:] * np.diff(control_times)[:, None]
        if not np.allclose(integrated, np.column_stack((forward_delta, yaw_delta)), atol=1e-5, rtol=1e-4):
            errors.append("physical velocity integrals do not match pose increments")
        for step_name, time_name, value_names in (
            ("physics_step", "state_timestamp_sec", (
                "pose_world_xyyaw", "world_position_xyz", "world_orientation_wxyz",
                "measured_velocity_vw", "steering_position", "steering_velocity", "wheel_velocity",
            )),
            ("action_physics_step", "action_timestamp_sec", ("human_action_vw", "applied_action_vw")),
            ("front_physics_step", "front_timestamp_sec", ()),
            ("down_physics_step", "down_timestamp_sec", ()),
        ):
            indices = np.searchsorted(control_steps, data[step_name])
            if np.any(indices >= total) or not np.array_equal(control_steps[indices], data[step_name]):
                errors.append(f"{step_name} is missing from physical control sequence")
                continue
            if not np.allclose(control_times[indices], data[time_name], atol=1e-6, rtol=0):
                errors.append(f"{time_name} does not match physical control sequence")
            for name in value_names:
                if not np.allclose(controls[name][indices], data[name], atol=1e-6, rtol=0):
                    errors.append(f"sample {name} differs from physical control sequence")
        for name, limits in (("human_action_vw", meta["teleop"]), ("applied_action_vw", meta["applied_action_limits"])):
            maximum = np.asarray([limits["max_linear_mps"], limits["max_angular_radps"]])
            if np.any(np.abs(controls[name]) > maximum + 1.0e-5):
                errors.append(f"control {name} exceeds configured limits")


def _validate_sensor_files(
    path: Path,
    meta: dict,
    count: int,
    errors: list[str],
) -> None:
    cameras = meta.get("cameras", {})
    bev_config = meta.get("bev", {})
    collection = meta.get("collection", {})
    camera_height = int(cameras.get("height", 0))
    camera_width = int(cameras.get("width", 0))
    bev_resolution = float(bev_config.get("resolution_m", 0.0))
    if camera_height <= 0 or camera_width <= 0 or bev_resolution <= 0:
        errors.append("camera or BEV dimensions are missing from metadata")
        return
    bev_height = int(round(
        (float(bev_config["forward_m"]) + float(bev_config["backward_m"]))
        / bev_resolution
    ))
    bev_width = int(round(
        (float(bev_config["left_m"]) + float(bev_config["right_m"]))
        / bev_resolution
    ))
    minimum_depth_ratio = float(collection.get("min_depth_valid_ratio", 0.02))
    for index in range(count):
        stem = f"{index:06d}"
        try:
            with Image.open(path / "front_rgb" / f"{stem}.jpg") as image:
                image.load()
                if image.size != (camera_width, camera_height):
                    errors.append(
                        f"front_rgb/{stem}.jpg size {image.size} is invalid"
                    )
            front_depth = decode_depth_png(path / "front_depth" / f"{stem}.png")
            down_depth = decode_depth_png(path / "down_depth" / f"{stem}.png")
            for name, depth in (
                ("front_depth", front_depth),
                ("down_depth", down_depth),
            ):
                if depth.shape != (camera_height, camera_width):
                    errors.append(f"{name}/{stem}.png shape {depth.shape} is invalid")
                    continue
                valid_ratio = float(np.mean(np.isfinite(depth) & (depth > 0.0)))
                if valid_ratio < minimum_depth_ratio:
                    errors.append(
                        f"{name}/{stem}.png valid ratio {valid_ratio:.4f} is too low"
                    )
            bev = np.load(path / "bev" / f"{stem}.npy", allow_pickle=False)
            if bev.shape != (3, bev_height, bev_width) or bev.dtype != np.uint8:
                errors.append(
                    f"bev/{stem}.npy format {bev.shape}/{bev.dtype} is invalid"
                )
            elif np.any(np.sum(bev > 0, axis=0) != 1):
                errors.append(f"bev/{stem}.npy channels are not mutually exclusive")
        except Exception as exc:
            errors.append(f"sample {stem} sensor validation failed: {exc}")
        if len(errors) >= 20:
            errors.append("sensor validation stopped after 20 errors")
            return


def decode_depth_png(path: Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.dtype != np.uint16:
        raise ValueError(f"Invalid uint16 depth PNG: {path}")
    return _decode_depth_mm(encoded)


def _decode_depth_mm(encoded: np.ndarray) -> np.ndarray:
    depth = encoded.astype(np.float32) / 1000.0
    depth[encoded == DEPTH_INVALID_MM] = np.nan
    depth[encoded == DEPTH_NO_RETURN_MM] = np.inf
    return depth


def quantize_depth_m(depth_m: np.ndarray) -> np.ndarray:
    return _decode_depth_mm(_encode_depth_mm(depth_m))


def _encode_depth_mm(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    finite = np.isfinite(depth) & (depth > 0.0)
    mm = np.rint(depth[finite] * 1000.0)
    encoded[finite] = np.clip(mm, 1, DEPTH_MAX_FINITE_MM).astype(np.uint16)
    encoded[np.isposinf(depth)] = DEPTH_NO_RETURN_MM
    return encoded


def _write_sample_files(
    *,
    episode_dir: Path,
    stem: str,
    front_rgb: np.ndarray,
    front_depth_m: np.ndarray,
    down_depth_m: np.ndarray,
    bev: np.ndarray,
    jpeg_quality: int,
    png_compression: int,
) -> None:
    _save_rgb_jpeg(
        episode_dir / "front_rgb" / f"{stem}.jpg",
        front_rgb,
        quality=jpeg_quality,
    )
    _save_depth_png(
        episode_dir / "front_depth" / f"{stem}.png",
        front_depth_m,
        compression=png_compression,
    )
    _save_depth_png(
        episode_dir / "down_depth" / f"{stem}.png",
        down_depth_m,
        compression=png_compression,
    )
    np.save(
        episode_dir / "bev" / f"{stem}.npy",
        bev,
        allow_pickle=False,
    )


def _save_rgb_jpeg(path: Path, rgb: np.ndarray, *, quality: int) -> None:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"RGB must be HxWx3+, got {array.shape}")
    Image.fromarray(array[..., :3], mode="RGB").save(
        path,
        format="JPEG",
        quality=int(quality),
        subsampling=0,
    )


def _save_depth_png(path: Path, depth_m: np.ndarray, *, compression: int) -> None:
    encoded = _encode_depth_mm(depth_m)
    ok = cv2.imwrite(
        str(path),
        encoded,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    )
    if not ok:
        raise IOError(f"Failed to write depth PNG: {path}")


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def recover_episode_directory(path: Path, *, reason: str = "interrupted") -> Path:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("只能恢复当前格式的未完成记录")
    if meta.get("status") == "complete":
        if path.name != f".episode_{meta['episode_id']}.tmp":
            raise ValueError("记录已经完成保存")
        errors = validate_episode_directory(path)
        meta["valid"] = not errors
        meta["validation_errors"] = list(dict.fromkeys(errors))
        _atomic_json(path / "meta.json", meta)
        final = path.with_name(f"episode_{meta['episode_id']}")
        if final.exists():
            raise FileExistsError(final)
        path.rename(final)
        return final
    samples: dict[str, list[np.ndarray]] = {}
    controls: dict[str, list[np.ndarray]] = {}
    sample_count = control_count = 0
    intrinsics: dict[str, np.ndarray] = {}
    recovery_errors = []
    for serial, chunk in enumerate(sorted((path / "chunks").glob("*.npz"))):
        try:
            if chunk.stem != f"{serial:06d}":
                raise ValueError("状态批次编号不连续")
            with np.load(chunk, allow_pickle=False) as data:
                if int(data["sample_start"]) != sample_count or int(data["control_start"]) != control_count:
                    raise ValueError("状态批次的起始索引不连续")
                state_values = {key[6:]: data[key].copy() for key in data.files if key.startswith("state_")}
                control_values = {
                    key[8:]: data[key].copy()
                    for key in data.files if key.startswith("control_") and key != "control_start"
                }
                size = len(state_values["sample_time_sec"])
                control_size = len(control_values["timestamp_sec"])
                if any(len(value) != size for value in state_values.values()):
                    raise ValueError("样本批次长度不一致")
                if any(len(value) != control_size for value in control_values.values()):
                    raise ValueError("控制批次长度不一致")
                for index in range(sample_count, sample_count + size):
                    for folder, suffix in (("front_rgb", ".jpg"), ("front_depth", ".png"), ("down_depth", ".png"), ("bev", ".npy")):
                        sensor = path / folder / f"{index:06d}{suffix}"
                        if not sensor.is_file() or sensor.stat().st_size == 0:
                            raise ValueError(f"传感器文件缺失：{sensor}")
                for name in ("front_intrinsics", "down_intrinsics"):
                    if name in data:
                        intrinsics[name] = data[name].copy()
            for name, value in state_values.items():
                if size:
                    samples.setdefault(name, []).append(value)
            for name, value in control_values.items():
                if control_size:
                    controls.setdefault(name, []).append(value)
            sample_count += size
            control_count += control_size
        except Exception as exc:
            recovery_errors.append(f"recovery stopped at {chunk.name}: {exc}")
            break
    state_arrays = {name: np.concatenate(values) for name, values in samples.items()}
    state_arrays.update(intrinsics)
    control_arrays = {name: np.concatenate(values) for name, values in controls.items()}
    _atomic_savez(path / "state.npz", state_arrays)
    _atomic_savez(path / "controls.npz", control_arrays)
    meta.update(
        status="incomplete", success=False, goal_world_xy=None,
        failure_reason=reason, sample_count=sample_count, control_count=control_count,
        valid=False, validation_errors=["episode is incomplete", *recovery_errors],
    )
    if sample_count:
        meta["end_time_sec"] = float(state_arrays["sample_time_sec"][-1])
        meta["end_pose_world_xyyaw"] = _json_vector(state_arrays["pose_world_xyyaw"][-1])
        meta["duration_sec"] = max(0.0, meta["end_time_sec"] - float(meta["start_time_sec"]))
    _atomic_json(path / "meta.json", meta)
    return path


def _next_episode_index(scene_dir: Path) -> int:
    maximum = 0
    for child in scene_dir.iterdir():
        name = child.name.lstrip(".")
        if not name.startswith("episode_"):
            continue
        suffix = name[len("episode_"):].split(".", 1)[0]
        if suffix.isdigit():
            maximum = max(maximum, int(suffix))
    return maximum + 1


def _json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _repository_revision() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi
