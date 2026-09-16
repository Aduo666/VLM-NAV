"""Strict HTTP client for the official NavDP point-goal server.

NavDP coordinates follow the public benchmark contract: +X is forward and +Y
is left, in metres.  The returned critic values are relative model preferences;
they are deliberately not interpreted as collision or safety probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np


@dataclass(frozen=True)
class NavDPClientConfig:
    base_url: str
    timeout_sec: float
    stop_threshold: float
    jpeg_quality: int
    max_abs_coordinate_m: float
    max_point_jump_m: float
    min_path_length_m: float
    model_match_tolerance_m: float

    @classmethod
    def from_mapping(cls, data: dict) -> "NavDPClientConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"NavDP server config requires exactly: {sorted(expected)}")
        config = cls(
            base_url=str(data["base_url"]).strip().rstrip("/"),
            timeout_sec=float(data["timeout_sec"]),
            stop_threshold=float(data["stop_threshold"]),
            jpeg_quality=int(data["jpeg_quality"]),
            max_abs_coordinate_m=float(data["max_abs_coordinate_m"]),
            max_point_jump_m=float(data["max_point_jump_m"]),
            min_path_length_m=float(data["min_path_length_m"]),
            model_match_tolerance_m=float(data["model_match_tolerance_m"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("NavDP base_url must start with http:// or https://")
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ValueError("NavDP timeout_sec must be positive")
        if not math.isfinite(self.stop_threshold):
            raise ValueError("NavDP stop_threshold must be finite")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("NavDP jpeg_quality must be in [1, 100]")
        for name in (
            "max_abs_coordinate_m",
            "max_point_jump_m",
            "min_path_length_m",
            "model_match_tolerance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"NavDP {name} must be finite and positive")


@dataclass(frozen=True)
class NavDPCandidate:
    index: int
    trajectory_xy: np.ndarray
    value: float | None
    is_model_choice: bool
    server_selected_only: bool = False


@dataclass(frozen=True)
class NavDPResult:
    candidates: tuple[NavDPCandidate, ...]
    model_choice_index: int
    latency_sec: float

    @property
    def model_choice(self) -> NavDPCandidate:
        return self.candidates[self.model_choice_index]


class NavDPClient:
    """One-environment client matching NavDP's public Flask API."""

    def __init__(self, config: NavDPClientConfig) -> None:
        config.validate()
        self.config = config

    def reset(self, camera_intrinsics: np.ndarray) -> str:
        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("NavDP reset requires finite 3x3 camera intrinsics")
        response = self._post_json(
            "/navigator_reset",
            {
                "intrinsic": intrinsics.tolist(),
                "stop_threshold": self.config.stop_threshold,
                "batch_size": 1,
            },
        )
        algorithm = str(response.get("algo", "")).strip()
        if not algorithm:
            raise RuntimeError("NavDP reset response omitted 'algo'")
        return algorithm

    def pointgoal_step(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        goal_forward_m: float,
        goal_left_m: float,
    ) -> NavDPResult:
        rgb_image, depth_image = _validate_observation(rgb, depth_m)
        goal = np.asarray([goal_forward_m, goal_left_m], dtype=np.float64)
        if not np.isfinite(goal).all():
            raise ValueError("NavDP point goal must be finite")

        ok, encoded_rgb = cv2.imencode(
            ".jpg",
            rgb_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode NavDP RGB input")
        depth_uint16 = np.clip(
            np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0) * 10000.0,
            0.0,
            65535.0,
        ).astype(np.uint16)
        ok, encoded_depth = cv2.imencode(".png", depth_uint16)
        if not ok:
            raise RuntimeError("Failed to PNG-encode NavDP depth input")

        start = time.monotonic()
        payload = self._post_multipart(
            "/pointgoal_step",
            fields={
                "goal_data": json.dumps(
                    {
                        "goal_x": [float(goal[0])],
                        "goal_y": [float(goal[1])],
                    }
                ),
                "rgb_time": repr(time.time()),
                "depth_time": repr(time.time()),
            },
            files={
                "image": ("image.jpg", "image/jpeg", encoded_rgb.tobytes()),
                "depth": ("depth.png", "image/png", encoded_depth.tobytes()),
            },
        )
        return normalize_navdp_response(
            payload,
            latency_sec=time.monotonic() - start,
            config=self.config,
        )

    def _post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        return self._request(
            path,
            body,
            content_type="application/json",
        )

    def _post_multipart(self, path: str, *, fields: dict[str, str], files: dict) -> dict:
        boundary = f"----captainav-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                (
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(value).encode("utf-8"),
                    b"\r\n",
                )
            )
        for name, (filename, mime_type, content) in files.items():
            chunks.extend(
                (
                    f"--{boundary}\r\n".encode(),
                    (
                        f'Content-Disposition: form-data; name="{name}"; '
                        f'filename="{filename}"\r\n'
                    ).encode(),
                    f"Content-Type: {mime_type}\r\n\r\n".encode(),
                    bytes(content),
                    b"\r\n",
                )
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        return self._request(
            path,
            b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def _request(self, path: str, body: bytes, *, content_type: str) -> dict:
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers={"Content-Type": content_type, "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"NavDP HTTP {exc.code} at {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"NavDP request failed at {path}: {exc}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"NavDP returned invalid JSON at {path}: {text[:500]}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"NavDP returned non-object JSON at {path}")
        return payload


def normalize_navdp_response(
    payload: dict,
    *,
    latency_sec: float,
    config: NavDPClientConfig,
) -> NavDPResult:
    """Validate one-batch NavDP output without guessing malformed dimensions."""
    missing = {"trajectory", "all_trajectory", "all_values"} - set(payload)
    if missing:
        raise RuntimeError(f"NavDP response missing fields: {sorted(missing)}")
    selected = _remove_single_batch_axis(np.asarray(payload["trajectory"], dtype=np.float64), 2, "trajectory")
    all_paths = _remove_single_batch_axis(
        np.asarray(payload["all_trajectory"], dtype=np.float64), 3, "all_trajectory"
    )
    values = np.asarray(payload["all_values"], dtype=np.float64)
    while values.ndim > 1 and values.shape[0] == 1:
        values = values[0]
    values = values.reshape(-1)
    if len(all_paths) == 0:
        raise RuntimeError("NavDP returned no candidate trajectories")
    if values.size != len(all_paths) or not np.isfinite(values).all():
        raise RuntimeError(
            f"NavDP candidate/value mismatch paths={len(all_paths)} values={values.shape}"
        )

    selected_xy = _validate_path(selected, "trajectory", config)
    paths_xy = [
        _validate_path(path, f"all_trajectory[{index}]", config)
        for index, path in enumerate(all_paths)
    ]
    errors = np.asarray(
        [_trajectory_distance(selected_xy, path) for path in paths_xy],
        dtype=np.float64,
    )
    model_index = int(np.argmin(errors))
    selected_only = bool(errors[model_index] > config.model_match_tolerance_m)
    if selected_only:
        model_index = len(paths_xy)
        paths_xy.append(selected_xy)
        candidate_values: list[float | None] = [float(value) for value in values]
        candidate_values.append(None)
    else:
        candidate_values = [float(value) for value in values]

    candidates = tuple(
        NavDPCandidate(
            index=index,
            trajectory_xy=np.ascontiguousarray(path, dtype=np.float64),
            value=candidate_values[index],
            is_model_choice=index == model_index,
            server_selected_only=selected_only and index == model_index,
        )
        for index, path in enumerate(paths_xy)
    )
    return NavDPResult(
        candidates=candidates,
        model_choice_index=model_index,
        latency_sec=float(latency_sec),
    )


def _remove_single_batch_axis(array: np.ndarray, target_ndim: int, name: str) -> np.ndarray:
    while array.ndim > target_ndim and array.shape[0] == 1:
        array = array[0]
    if array.ndim != target_ndim:
        raise RuntimeError(f"NavDP {name} has unsupported shape {array.shape}")
    return array


def _validate_path(path: np.ndarray, name: str, config: NavDPClientConfig) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise RuntimeError(f"NavDP {name} must be Nx2+, got {values.shape}")
    values = values[:, :2]
    if not np.isfinite(values).all():
        raise RuntimeError(f"NavDP {name} contains non-finite coordinates")
    if float(np.max(np.abs(values))) > config.max_abs_coordinate_m:
        raise RuntimeError(f"NavDP {name} exceeds coordinate bound")
    segments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    if float(np.max(segments)) > config.max_point_jump_m:
        raise RuntimeError(f"NavDP {name} contains an implausible point jump")
    if float(np.sum(segments)) < config.min_path_length_m:
        raise RuntimeError(f"NavDP {name} is shorter than the configured minimum")
    return values


def _trajectory_distance(first: np.ndarray, second: np.ndarray) -> float:
    count = max(len(first), len(second), 2)
    samples = np.linspace(0.0, 1.0, count)
    first_resampled = np.column_stack(
        [np.interp(samples, np.linspace(0.0, 1.0, len(first)), first[:, axis]) for axis in (0, 1)]
    )
    second_resampled = np.column_stack(
        [np.interp(samples, np.linspace(0.0, 1.0, len(second)), second[:, axis]) for axis in (0, 1)]
    )
    return float(np.sqrt(np.mean(np.sum((first_resampled - second_resampled) ** 2, axis=1))))


def _validate_observation(rgb: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"NavDP RGB must be HxWx3+, got {image.shape}")
    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
    image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError(f"NavDP RGB/depth mismatch rgb={image.shape} depth={depth.shape}")
    valid_ratio = float(np.mean(np.isfinite(depth) & (depth > 0.0)))
    if valid_ratio < 0.01:
        raise ValueError(f"NavDP depth has too few valid pixels: ratio={valid_ratio:.4f}")
    return image, np.ascontiguousarray(depth)
