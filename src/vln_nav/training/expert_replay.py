from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import CanonicalEpisode, ReplayPose


@dataclass(frozen=True)
class ExpertReplayConfig:
    path_spacing_m: float = 0.25
    max_turn_step_deg: float = 15.0
    linear_speed_mps: float = 0.75
    angular_speed_radps: float = 0.75
    final_hold_sec: float = 0.25


def build_expert_replay(
    episode: CanonicalEpisode,
    *,
    config: ExpertReplayConfig | None = None,
) -> list[ReplayPose]:
    """Convert an InteriorNav path into deterministic camera replay poses.

    InteriorNav provides route points, not expert controls or per-point camera
    headings. We therefore resample the polyline, derive heading from the path
    tangent, and insert pure in-place camera turns no larger than
    ``max_turn_step_deg``. The timestamps are synthetic and are used only to
    make perception scheduling and debug playback deterministic.
    """

    cfg = config or ExpertReplayConfig()
    path = np.asarray(episode.reference_path_xyz, dtype=np.float64)
    sampled, progress, source_segments = _resample_path(
        path,
        spacing_m=max(0.02, float(cfg.path_spacing_m)),
    )
    initial_yaw = quaternion_wxyz_to_yaw(episode.start_rotation_wxyz)
    max_turn_rad = float(np.deg2rad(max(1.0, float(cfg.max_turn_step_deg))))
    linear_speed = max(0.02, float(cfg.linear_speed_mps))
    angular_speed = max(0.02, float(cfg.angular_speed_radps))

    records: list[ReplayPose] = []
    timestamp = 0.0
    current_yaw = float(initial_yaw)

    def append_pose(
        *,
        position: np.ndarray,
        yaw: float,
        path_progress: float,
        source_segment: int,
        motion_type: str,
        delta_distance: float,
        delta_yaw: float,
    ) -> None:
        records.append(
            ReplayPose(
                frame_index=len(records),
                timestamp_sec=float(timestamp),
                position_xyz=tuple(float(value) for value in position[:3]),
                yaw_rad=float(_wrap_angle(yaw)),
                path_progress=float(np.clip(path_progress, 0.0, 1.0)),
                source_segment_index=int(source_segment),
                motion_type=str(motion_type),
                delta_distance_m=float(delta_distance),
                delta_yaw_rad=float(delta_yaw),
            )
        )

    append_pose(
        position=sampled[0],
        yaw=current_yaw,
        path_progress=float(progress[0]),
        source_segment=int(source_segments[0]),
        motion_type="START",
        delta_distance=0.0,
        delta_yaw=0.0,
    )

    for index in range(sampled.shape[0] - 1):
        current = sampled[index]
        following = sampled[index + 1]
        delta_xy = following[:2] - current[:2]
        distance = float(np.linalg.norm(delta_xy))
        if distance <= 1.0e-8:
            continue
        target_yaw = float(np.arctan2(delta_xy[1], delta_xy[0]))
        yaw_error = float(_wrap_angle(target_yaw - current_yaw))
        turn_count = int(np.ceil(abs(yaw_error) / max_turn_rad))
        if turn_count > 0:
            turn_step = yaw_error / float(turn_count)
            for _ in range(turn_count):
                timestamp += abs(turn_step) / angular_speed
                current_yaw = float(_wrap_angle(current_yaw + turn_step))
                append_pose(
                    position=current,
                    yaw=current_yaw,
                    path_progress=float(progress[index]),
                    source_segment=int(source_segments[index]),
                    motion_type="TURN_LEFT" if turn_step > 0.0 else "TURN_RIGHT",
                    delta_distance=0.0,
                    delta_yaw=turn_step,
                )
        timestamp += distance / linear_speed
        current_yaw = target_yaw
        append_pose(
            position=following,
            yaw=current_yaw,
            path_progress=float(progress[index + 1]),
            source_segment=int(source_segments[index + 1]),
            motion_type="MOVE",
            delta_distance=distance,
            delta_yaw=0.0,
        )

    timestamp += max(0.0, float(cfg.final_hold_sec))
    append_pose(
        position=sampled[-1],
        yaw=current_yaw,
        path_progress=1.0,
        source_segment=int(source_segments[-1]),
        motion_type="STOP",
        delta_distance=0.0,
        delta_yaw=0.0,
    )
    return records


def quaternion_wxyz_to_yaw(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _resample_path(
    path_xyz: np.ndarray,
    *,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(path_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("reference path must contain at least two XYZ points")
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    keep = np.concatenate([[True], segment_lengths > 1.0e-8])
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("reference path has zero planar length")
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    distances = np.arange(0.0, total_length, spacing_m, dtype=np.float64)
    if distances.size == 0 or total_length - float(distances[-1]) > 1.0e-8:
        distances = np.append(distances, total_length)
    sampled = np.empty((distances.shape[0], 3), dtype=np.float64)
    for axis in range(3):
        sampled[:, axis] = np.interp(distances, cumulative, points[:, axis])
    source_segments = np.searchsorted(cumulative[1:], distances, side="right")
    source_segments = np.clip(source_segments, 0, points.shape[0] - 2).astype(np.int32)
    progress = distances / max(total_length, 1.0e-8)
    return sampled, progress, source_segments


def _wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))
