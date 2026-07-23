from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observation_adapter import Pose2D


@dataclass(frozen=True)
class TurnHeadingEstimatorConfig:
    """Robot-independent free-space heading estimator used by typed TURN steps."""

    heading_half_fov_deg: float = 60.0
    heading_step_deg: float = 5.0
    lidar_bin_half_width_deg: float = 3.0
    min_corridor_width_deg: float = 15.0
    min_clearance_m: float = 0.60
    max_clearance_m: float = 4.0


@dataclass(frozen=True)
class TurnHeadingEstimate:
    valid: bool
    reason: str
    local_heading_deg: float = 0.0
    world_yaw_rad: float = 0.0
    fused_clearance_m: float = 0.0
    lidar_clearance_m: float = 0.0
    map_clearance_m: float = 0.0
    map_known_ratio: float = 0.0
    corridor_width_deg: float = 0.0
    score: float = 0.0


class TurnHeadingEstimator:
    """Estimate the centerline of the safest visible corridor.

    Frame assumptions:
    - LiDAR endpoints use robot-local ``+X forward, +Y left``.
    - ``Pose2D`` and the occupancy snapshot share the odom/world frame.
    - Occupancy rows increase with world ``+Y`` and columns with world ``+X``.

    The estimator deliberately does not use navigation action candidates.
    It first creates a heading-clearance profile from LiDAR, validates each
    direction against inflated global occupancy, then chooses the center of a
    contiguous safe heading cluster.
    """

    def __init__(self, config: TurnHeadingEstimatorConfig) -> None:
        self._config = config

    def estimate(
        self,
        *,
        robot_pose: Pose2D,
        beam_endpoints_local_xy: np.ndarray | None,
        occupancy_snapshot: Any | None,
    ) -> TurnHeadingEstimate:
        endpoints = _valid_endpoints(beam_endpoints_local_xy)
        snapshot = _valid_snapshot(occupancy_snapshot)
        if endpoints.shape[0] == 0 and snapshot is None:
            return TurnHeadingEstimate(valid=False, reason="no_lidar_or_occupancy")

        half_fov = max(5.0, float(self._config.heading_half_fov_deg))
        step_deg = max(1.0, float(self._config.heading_step_deg))
        headings = np.arange(-half_fov, half_fov + 0.5 * step_deg, step_deg, dtype=np.float32)
        lidar_angles, lidar_ranges = _polar_endpoints(endpoints)

        samples: list[dict[str, float | bool]] = []
        for heading_deg in headings:
            lidar_clearance = _lidar_clearance(
                heading_deg=float(heading_deg),
                angles_deg=lidar_angles,
                ranges_m=lidar_ranges,
                bin_half_width_deg=max(
                    0.5,
                    float(self._config.lidar_bin_half_width_deg),
                ),
                max_clearance_m=max(0.2, float(self._config.max_clearance_m)),
            )
            map_clearance, known_ratio = _map_clearance(
                snapshot=snapshot,
                robot_pose=robot_pose,
                local_heading_deg=float(heading_deg),
                max_clearance_m=max(0.2, float(self._config.max_clearance_m)),
            )
            fused_clearance = _fuse_clearance(
                lidar_clearance=lidar_clearance,
                map_clearance=map_clearance,
            )
            valid = bool(
                np.isfinite(fused_clearance)
                and float(fused_clearance) >= float(self._config.min_clearance_m)
            )
            samples.append(
                {
                    "heading_deg": float(heading_deg),
                    "lidar_clearance_m": float(lidar_clearance),
                    "map_clearance_m": float(map_clearance),
                    "map_known_ratio": float(known_ratio),
                    "fused_clearance_m": float(fused_clearance),
                    "valid": valid,
                }
            )

        safe_clusters = _contiguous_safe_clusters(samples)
        min_width = max(step_deg, float(self._config.min_corridor_width_deg))
        safe_clusters = [
            cluster
            for cluster in safe_clusters
            if _cluster_width_deg(cluster=cluster, step_deg=step_deg) >= min_width
        ]
        if not safe_clusters:
            return TurnHeadingEstimate(valid=False, reason="no_safe_corridor_cluster")

        best_cluster = max(
            safe_clusters,
            key=lambda cluster: _cluster_score(
                cluster=cluster,
                half_fov_deg=half_fov,
                step_deg=step_deg,
                max_clearance_m=max(0.2, float(self._config.max_clearance_m)),
            ),
        )
        first_heading = float(best_cluster[0]["heading_deg"])
        last_heading = float(best_cluster[-1]["heading_deg"])
        center_heading = 0.5 * (first_heading + last_heading)
        center_sample = min(
            best_cluster,
            key=lambda sample: abs(float(sample["heading_deg"]) - center_heading),
        )
        corridor_width = _cluster_width_deg(cluster=best_cluster, step_deg=step_deg)
        score = _cluster_score(
            cluster=best_cluster,
            half_fov_deg=half_fov,
            step_deg=step_deg,
            max_clearance_m=max(0.2, float(self._config.max_clearance_m)),
        )
        world_yaw = _wrap_angle_rad(
            float(robot_pose.yaw_rad) + float(np.deg2rad(center_heading))
        )
        return TurnHeadingEstimate(
            valid=True,
            reason="ok",
            local_heading_deg=float(center_heading),
            world_yaw_rad=float(world_yaw),
            fused_clearance_m=float(center_sample["fused_clearance_m"]),
            lidar_clearance_m=float(center_sample["lidar_clearance_m"]),
            map_clearance_m=float(center_sample["map_clearance_m"]),
            map_known_ratio=float(center_sample["map_known_ratio"]),
            corridor_width_deg=float(corridor_width),
            score=float(score),
        )


def _valid_endpoints(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=np.float32)
    endpoints = np.asarray(value, dtype=np.float32)
    if endpoints.ndim != 2 or endpoints.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    endpoints = endpoints[:, :2]
    finite = np.isfinite(endpoints).all(axis=1)
    ranges = np.linalg.norm(endpoints, axis=1)
    return np.ascontiguousarray(endpoints[finite & (ranges > 0.05)], dtype=np.float32)


def _valid_snapshot(snapshot: Any | None) -> Any | None:
    if snapshot is None:
        return None
    occupied = np.asarray(getattr(snapshot, "inflated_occupied", None))
    free = np.asarray(getattr(snapshot, "free_mask", None))
    if occupied.ndim != 2 or free.shape != occupied.shape:
        return None
    if occupied.size == 0:
        return None
    return snapshot


def _polar_endpoints(endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if endpoints.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    angles = np.degrees(np.arctan2(endpoints[:, 1], endpoints[:, 0])).astype(np.float32)
    ranges = np.linalg.norm(endpoints, axis=1).astype(np.float32)
    return angles, ranges


def _lidar_clearance(
    *,
    heading_deg: float,
    angles_deg: np.ndarray,
    ranges_m: np.ndarray,
    bin_half_width_deg: float,
    max_clearance_m: float,
) -> float:
    if angles_deg.size == 0:
        return float("nan")
    delta = np.abs(_wrap_angle_deg_array(angles_deg - float(heading_deg)))
    selected = ranges_m[delta <= float(bin_half_width_deg)]
    if selected.size == 0:
        return float("nan")
    # A low percentile rejects isolated long miss beams while remaining less
    # sensitive than a raw minimum to one noisy return.
    return float(np.clip(np.percentile(selected, 20.0), 0.0, max_clearance_m))


def _map_clearance(
    *,
    snapshot: Any | None,
    robot_pose: Pose2D,
    local_heading_deg: float,
    max_clearance_m: float,
) -> tuple[float, float]:
    if snapshot is None:
        return float("nan"), 0.0
    occupied = np.asarray(snapshot.inflated_occupied, dtype=bool)
    free = np.asarray(snapshot.free_mask, dtype=bool)
    resolution = max(0.01, float(snapshot.resolution_m))
    world_heading = float(robot_pose.yaw_rad) + float(np.deg2rad(local_heading_deg))
    cos_h = float(np.cos(world_heading))
    sin_h = float(np.sin(world_heading))
    distances = np.arange(resolution, max_clearance_m + 0.5 * resolution, resolution)
    known_count = 0
    sample_count = 0
    last_clear_distance = 0.0
    for distance_m in distances:
        world_x = float(robot_pose.x) + float(distance_m) * cos_h
        world_y = float(robot_pose.y) + float(distance_m) * sin_h
        col = int(np.floor((world_x - float(snapshot.origin_x)) / resolution))
        row = int(np.floor((world_y - float(snapshot.origin_y)) / resolution))
        if row < 0 or col < 0 or row >= occupied.shape[0] or col >= occupied.shape[1]:
            break
        sample_count += 1
        if bool(occupied[row, col]):
            break
        known_count += int(bool(free[row, col]))
        last_clear_distance = float(distance_m)
    known_ratio = 0.0 if sample_count <= 0 else float(known_count) / float(sample_count)
    return float(last_clear_distance), float(known_ratio)


def _fuse_clearance(*, lidar_clearance: float, map_clearance: float) -> float:
    lidar_valid = bool(np.isfinite(lidar_clearance))
    map_valid = bool(np.isfinite(map_clearance))
    if lidar_valid and map_valid:
        return float(min(lidar_clearance, map_clearance))
    if lidar_valid:
        return float(lidar_clearance)
    if map_valid:
        return float(map_clearance)
    return float("nan")


def _contiguous_safe_clusters(
    samples: list[dict[str, float | bool]],
) -> list[list[dict[str, float | bool]]]:
    clusters: list[list[dict[str, float | bool]]] = []
    current: list[dict[str, float | bool]] = []
    for sample in samples:
        if bool(sample["valid"]):
            current.append(sample)
            continue
        if current:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    return clusters


def _cluster_width_deg(
    *,
    cluster: list[dict[str, float | bool]],
    step_deg: float,
) -> float:
    if not cluster:
        return 0.0
    return (
        float(cluster[-1]["heading_deg"])
        - float(cluster[0]["heading_deg"])
        + float(step_deg)
    )


def _cluster_score(
    *,
    cluster: list[dict[str, float | bool]],
    half_fov_deg: float,
    step_deg: float,
    max_clearance_m: float,
) -> float:
    clearances = np.asarray(
        [float(sample["fused_clearance_m"]) for sample in cluster],
        dtype=np.float32,
    )
    known = np.asarray(
        [float(sample["map_known_ratio"]) for sample in cluster],
        dtype=np.float32,
    )
    center = 0.5 * (
        float(cluster[0]["heading_deg"]) + float(cluster[-1]["heading_deg"])
    )
    width_ratio = np.clip(
        _cluster_width_deg(cluster=cluster, step_deg=step_deg)
        / max(1.0, 2.0 * half_fov_deg),
        0.0,
        1.0,
    )
    clearance_score = float(np.median(clearances)) / max(0.1, max_clearance_m)
    center_score = 1.0 - min(1.0, abs(center) / max(1.0, half_fov_deg))
    return float(
        0.55 * clearance_score
        + 0.20 * float(width_ratio)
        + 0.15 * float(np.mean(known))
        + 0.10 * center_score
    )


def _wrap_angle_deg_array(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) + 180.0) % 360.0 - 180.0


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)
