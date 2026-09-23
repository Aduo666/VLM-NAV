from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

from .transforms import relative_goal_xyyaw


DEPTH_INVALID_MM = np.uint16(0)
DEPTH_NO_RETURN_MM = np.uint16(65535)
DEPTH_MAX_FINITE_MM = 65534


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
    map_version: int


class EpisodeRecorder:
    """Streams heavy sensor data to disk and keeps only small state arrays in RAM."""

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
        for name in ("front_rgb", "front_depth", "down_depth", "bev"):
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
            "map_version": [],
        }
        self._meta_seed = {
            "episode_id": f"{index:06d}",
            "scene_name": self.scene_path.stem,
            "usd_path": str(self.scene_path),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_revision": _repository_revision(),
            "start_pose_world_xyyaw": _json_vector(start_pose_world),
            "start_position_world_xyz": _json_vector(start_position_world_xyz),
            "start_time_sec": float(start_time_sec),
            "depth_encoding": {
                "format": "uint16_png_millimeters",
                "invalid_value": int(DEPTH_INVALID_MM),
                "no_return_value": int(DEPTH_NO_RETURN_MM),
            },
            "bev_channel_order": ["occupied", "free", "unknown"],
        }
        if extra_meta:
            self._meta_seed.update(extra_meta)
        return temp_dir

    def append(self, sample: SampleRecord) -> int:
        if not self._active or self._episode_dir is None:
            raise RuntimeError("Recorder is not active")
        index = len(self._state["sample_time_sec"])
        stem = f"{index:06d}"

        future = self._executor.submit(
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
        self._pending.append(future)
        if len(self._pending) >= self.max_pending_writes:
            self._pending.popleft().result()

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
            "pose_world_xyyaw": np.asarray(sample.pose_world_xyyaw, dtype=np.float32),
            "world_position_xyz": np.asarray(sample.world_position_xyz, dtype=np.float32),
            "world_orientation_wxyz": np.asarray(sample.world_orientation_wxyz, dtype=np.float32),
            "measured_velocity_vw": np.asarray(sample.measured_velocity_vw, dtype=np.float32),
            "human_action_vw": np.asarray(sample.human_action_vw, dtype=np.float32),
            "applied_action_vw": np.asarray(sample.applied_action_vw, dtype=np.float32),
            "steering_position": np.asarray(sample.steering_position, dtype=np.float32),
            "steering_velocity": np.asarray(sample.steering_velocity, dtype=np.float32),
            "wheel_velocity": np.asarray(sample.wheel_velocity, dtype=np.float32),
            "front_base_from_camera": np.asarray(sample.front_base_from_camera, dtype=np.float32),
            "down_base_from_camera": np.asarray(sample.down_base_from_camera, dtype=np.float32),
            "map_version": int(sample.map_version),
        }
        for key, value in values.items():
            self._state[key].append(value)
        return index

    def finish(
        self,
        *,
        success: bool,
        end_pose_world: np.ndarray,
        end_time_sec: float,
        failure_reason: str | None = None,
    ) -> Path:
        if not self._active or self._episode_dir is None or self._final_dir is None:
            raise RuntimeError("Recorder is not active")
        self._flush_pending()
        count = len(self._state["sample_time_sec"])
        poses = np.asarray(self._state["pose_world_xyyaw"], dtype=np.float32).reshape(count, 3)
        end_pose = np.asarray(end_pose_world, dtype=np.float32).reshape(3)
        if success:
            goal_body = np.asarray(
                [relative_goal_xyyaw(pose, end_pose) for pose in poses],
                dtype=np.float32,
            ).reshape(count, 3)
            goal_world = end_pose
        else:
            goal_body = np.full((count, 3), np.nan, dtype=np.float32)
            goal_world = None

        arrays = {
            key: np.asarray(values)
            for key, values in self._state.items()
        }
        arrays["goal_body_xyyaw"] = goal_body
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
        np.savez_compressed(self._episode_dir / "state.npz", **arrays)

        start_time = float(self._meta_seed["start_time_sec"])
        meta = {
            **self._meta_seed,
            "success": bool(success),
            "failure_reason": None if success else str(failure_reason or "unspecified"),
            "goal_pose_world_xyyaw": None if goal_world is None else _json_vector(goal_world),
            "end_pose_world_xyyaw": _json_vector(end_pose),
            "end_time_sec": float(end_time_sec),
            "duration_sec": max(0.0, float(end_time_sec) - start_time),
            "sample_count": int(count),
            "valid": True,
            "validation_errors": [],
        }
        (self._episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        errors = validate_episode_directory(self._episode_dir)
        if errors:
            meta["valid"] = False
            meta["validation_errors"] = errors
            (self._episode_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if self._final_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {self._final_dir}")
        self._episode_dir.rename(self._final_dir)
        result = self._final_dir
        self._clear()
        return result

    def discard(self) -> None:
        self._flush_pending(ignore_errors=True)
        if self._episode_dir is not None and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir)
        self._clear()

    def close(self) -> None:
        self._flush_pending(ignore_errors=True)
        self._executor.shutdown(wait=True, cancel_futures=False)

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
                "goal_body_xyyaw",
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
                    "goal_body_xyyaw": (count, 3),
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
                finite_keys = sequence_keys - {"goal_body_xyyaw"}
                finite_keys |= {"front_intrinsics", "down_intrinsics"}
                for key in sorted(finite_keys):
                    if not np.isfinite(np.asarray(data[key], dtype=np.float64)).all():
                        errors.append(f"{key} contains non-finite values")
                if bool(meta.get("success")):
                    goal = np.asarray(data["goal_body_xyyaw"], dtype=np.float64)
                    if not np.isfinite(goal).all():
                        errors.append("successful episode contains non-finite relative goal labels")
                    collection = meta.get("collection", {})
                    if isinstance(collection, dict) and len(goal):
                        position_tolerance = float(
                            collection.get("success_goal_position_tolerance_m", 0.15)
                        )
                        yaw_tolerance = np.radians(float(
                            collection.get("success_goal_yaw_tolerance_deg", 10.0)
                        ))
                        if np.linalg.norm(goal[-1, :2]) > position_tolerance:
                            errors.append("last relative goal position exceeds tolerance")
                        if abs(float(goal[-1, 2])) > yaw_tolerance:
                            errors.append("last relative goal yaw exceeds tolerance")
                else:
                    goal = np.asarray(data["goal_body_xyyaw"], dtype=np.float64)
                    if not np.isnan(goal).all():
                        errors.append("failed episode relative goal labels must be NaN")

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
    depth = encoded.astype(np.float32) / 1000.0
    depth[encoded == DEPTH_INVALID_MM] = np.nan
    depth[encoded == DEPTH_NO_RETURN_MM] = np.inf
    return depth


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
    depth = np.asarray(depth_m, dtype=np.float32)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    finite = np.isfinite(depth) & (depth > 0.0)
    mm = np.rint(depth[finite] * 1000.0)
    encoded[finite] = np.clip(mm, 1, DEPTH_MAX_FINITE_MM).astype(np.uint16)
    encoded[np.isposinf(depth)] = DEPTH_NO_RETURN_MM
    ok = cv2.imwrite(
        str(path),
        encoded,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    )
    if not ok:
        raise IOError(f"Failed to write depth PNG: {path}")


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
