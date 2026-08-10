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
    map_clearance_m: float = 0.0
    map_known_ratio: float = 0.0
    corridor_width_deg: float = 0.0
    score: float = 0.0


class TurnHeadingEstimator:
    """Estimate the centerline of the safest visible corridor.

    Frame assumptions:
    - ``Pose2D`` and the occupancy snapshot share the odom/world frame.
    - Occupancy rows increase with world ``+Y`` and columns with world ``+X``.

    The estimator deliberately does not use navigation action candidates.
    It creates a heading-clearance profile from the RGB-D occupancy map and
    chooses the center of a contiguous safe heading cluster.
    """

    def __init__(self, config: TurnHeadingEstimatorConfig) -> None:
        self._config = config

    def estimate(
        self,
        *,
        robot_pose: Pose2D,
        occupancy_snapshot: Any | None,
    ) -> TurnHeadingEstimate:
        snapshot = _valid_snapshot(occupancy_snapshot)
        if snapshot is None:
            return TurnHeadingEstimate(valid=False, reason="no_occupancy")

        half_fov = max(5.0, float(self._config.heading_half_fov_deg))
        step_deg = max(1.0, float(self._config.heading_step_deg))
        headings = np.arange(-half_fov, half_fov + 0.5 * step_deg, step_deg, dtype=np.float32)
        samples: list[dict[str, float | bool]] = []
        for heading_deg in headings:
            map_clearance, known_ratio = _map_clearance(
                snapshot=snapshot,
                robot_pose=robot_pose,
                local_heading_deg=float(heading_deg),
                max_clearance_m=max(0.2, float(self._config.max_clearance_m)),
            )
            fused_clearance = map_clearance
            valid = bool(
                np.isfinite(fused_clearance)
                and float(fused_clearance) >= float(self._config.min_clearance_m)
            )
            samples.append(
                {
                    "heading_deg": float(heading_deg),
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
            map_clearance_m=float(center_sample["map_clearance_m"]),
            map_known_ratio=float(center_sample["map_known_ratio"]),
            corridor_width_deg=float(corridor_width),
            score=float(score),
        )


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


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)
