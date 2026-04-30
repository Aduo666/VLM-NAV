from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .observation_adapter import Pose2D, VLNObservation
from ..runtime.config import PROJECT_ROOT


@dataclass
class OccupancyGridConfig:
    enabled: bool = True
    resolution_m: float = 0.08
    size_m: float = 24.0
    update_period_sec: float = 0.12
    depth_sample_stride_px: int = 10
    depth_roi_vertical_range: tuple[float, float] = (0.38, 0.72)
    min_depth_m: float = 0.20
    max_depth_m: float = 5.0
    occupied_log_odds: float = 0.85
    free_log_odds: float = -0.35
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0
    occupied_threshold: float = 1.2
    free_threshold: float = -0.5
    occupied_min_neighbors: int = 3
    decay_rate: float = 0.990
    inflation_radius_m: float = 0.26
    robot_radius_m: float = 0.32
    path_corridor_radius_m: float = 0.30
    turn_clearance_radius_m: float = 0.36
    unknown_is_traversable: bool = True
    unknown_cell_cost: float = 0.10
    max_astar_iterations: int = 7000
    max_path_detour_ratio: float = 2.2
    debug_enabled: bool = True
    debug_image_path: str = "assets/occupancy_grid.png"
    debug_update_period_sec: float = 0.5
    debug_width: int = 900
    debug_height: int = 900
    debug_view_size_m: float = 8.0
    debug_show_free_space: bool = False


@dataclass
class OccupancyGridSnapshot:
    resolution_m: float
    size_m: float
    log_odds: np.ndarray
    inflated_occupied: np.ndarray
    free_mask: np.ndarray


@dataclass
class _CandidateEvaluation:
    valid: bool
    reason: str
    path_length_m: float
    blocked_cell_count: int
    goal_world_xy: tuple[float, float]


class OccupancyGridMapper:
    """Local odom-frame occupancy grid used for local reachability checks.

    Frame convention:
    - Observation pose is odom/base pose in meters.
    - Robot local +X is forward.
    - Robot local +Y is left.
    - Image u increases to the right, so image-left pixels project to +Y.

    This is not full SLAM. It is a lightweight local planning layer.
    LiDAR observations are preferred. Depth observations remain available for
    fallback when LiDAR is disabled.
    """

    def __init__(self, *, config: OccupancyGridConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("occupancy_grid")
        self._resolution_m = max(0.03, float(config.resolution_m))
        self._size_m = max(4.0, float(config.size_m))
        self._cell_count = int(round(self._size_m / self._resolution_m))
        if self._cell_count % 2 == 1:
            self._cell_count += 1
        self._half_size_m = 0.5 * float(self._cell_count) * self._resolution_m
        self._log_odds = np.zeros((self._cell_count, self._cell_count), dtype=np.float32)
        self._last_update_time_sec = -float("inf")
        self._last_debug_save_time_sec = -float("inf")
        self._last_filter_signature: tuple[int, int, tuple[str, ...]] | None = None
        self._last_lidar_points_local_xy = np.zeros((0, 2), dtype=np.float32)
        self._debug_path = Path(config.debug_image_path)
        if not self._debug_path.is_absolute():
            self._debug_path = PROJECT_ROOT / self._debug_path

    @property
    def output_path(self) -> Path:
        return self._debug_path

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    def update(self, *, observation: VLNObservation) -> None:
        if not bool(self._config.enabled):
            return
        timestamp_sec = float(observation.meta.get("timestamp_sec", 0.0))
        if timestamp_sec - float(self._last_update_time_sec) < max(0.02, float(self._config.update_period_sec)):
            return
        depth = observation.depth
        if depth is None:
            return
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim != 2 or depth_np.size == 0:
            return

        self._log_odds *= float(np.clip(self._config.decay_rate, 0.90, 1.0))

        height, width = depth_np.shape[:2]
        v0, v1 = self._depth_roi(height)
        stride = max(2, int(self._config.depth_sample_stride_px))
        min_depth = max(0.02, float(self._config.min_depth_m))
        max_depth = max(min_depth + 0.10, float(self._config.max_depth_m))
        half_fov_rad = float(np.deg2rad(float(observation.camera.get("hfov_deg", 90.0))) * 0.5)
        robot_xy = (float(observation.pose.x), float(observation.pose.y))
        cos_yaw = float(np.cos(observation.pose.yaw_rad))
        sin_yaw = float(np.sin(observation.pose.yaw_rad))

        occupied_updates = 0
        free_updates = 0
        for v in range(v0, v1, stride):
            for u in range(0, width, stride):
                distance = float(depth_np[v, u])
                if not np.isfinite(distance) or distance < min_depth or distance > max_depth:
                    continue
                normalized_u = (float(u) - float(width - 1) * 0.5) / max(float(width - 1) * 0.5, 1.0)
                bearing_rad = float(normalized_u * half_fov_rad)
                forward_m = distance * float(np.cos(bearing_rad))
                left_m = -distance * float(np.sin(bearing_rad))
                if forward_m <= 0.05:
                    continue
                world_x = robot_xy[0] + cos_yaw * forward_m - sin_yaw * left_m
                world_y = robot_xy[1] + sin_yaw * forward_m + cos_yaw * left_m
                free_count, occupied_count = self._mark_ray(
                    start_xy=robot_xy,
                    end_xy=(float(world_x), float(world_y)),
                    endpoint_distance_m=float(np.hypot(forward_m, left_m)),
                )
                free_updates += free_count
                occupied_updates += occupied_count

        self._last_update_time_sec = timestamp_sec
        if occupied_updates > 0:
            self._logger.debug(
                "[Occupancy] update free_cells=%s occupied_cells=%s occupied_total=%s",
                free_updates,
                occupied_updates,
                int(np.count_nonzero(self._occupied_mask())),
            )

    def update_from_lidar(
        self,
        *,
        robot_pose: Pose2D,
        points_local_xy: np.ndarray,
        timestamp_sec: float,
    ) -> None:
        if not bool(self._config.enabled):
            return
        if float(timestamp_sec) - float(self._last_update_time_sec) < max(0.02, float(self._config.update_period_sec)):
            return
        points = np.asarray(points_local_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
            return
        self._last_lidar_points_local_xy = points[:, :2].copy()

        self._log_odds *= float(np.clip(self._config.decay_rate, 0.90, 1.0))
        robot_xy = (float(robot_pose.x), float(robot_pose.y))
        cos_yaw = float(np.cos(robot_pose.yaw_rad))
        sin_yaw = float(np.sin(robot_pose.yaw_rad))
        min_depth = max(0.02, float(self._config.min_depth_m))
        max_depth = max(min_depth + 0.10, float(self._config.max_depth_m))

        occupied_updates = 0
        free_updates = 0
        for forward_m, left_m in points[:, :2]:
            forward = float(forward_m)
            left = float(left_m)
            distance = float(np.hypot(forward, left))
            if not np.isfinite(distance) or distance < min_depth or distance > max_depth:
                continue
            world_x = robot_xy[0] + cos_yaw * forward - sin_yaw * left
            world_y = robot_xy[1] + sin_yaw * forward + cos_yaw * left
            free_count, occupied_count = self._mark_ray(
                start_xy=robot_xy,
                end_xy=(float(world_x), float(world_y)),
                endpoint_distance_m=distance,
            )
            free_updates += free_count
            occupied_updates += occupied_count

        self._last_update_time_sec = float(timestamp_sec)
        if occupied_updates > 0:
            self._logger.debug(
                "[Occupancy] lidar_update free_cells=%s occupied_cells=%s occupied_total=%s",
                free_updates,
                occupied_updates,
                int(np.count_nonzero(self._occupied_mask())),
            )

    def filter_candidates(self, *, observation: VLNObservation, candidate_set: CandidateSet) -> CandidateSet:
        if not bool(self._config.enabled):
            return candidate_set
        filtered: list[WaypointCandidate] = []
        blocked_reasons: list[str] = []
        original_valid_count = int(candidate_set.valid_count)
        for candidate in candidate_set.candidates:
            if not bool(candidate.valid):
                filtered.append(candidate)
                continue
            evaluation = self._evaluate_candidate(observation=observation, candidate=candidate)
            depth_support = dict(candidate.depth_support)
            depth_support.update(
                {
                    "occupancy_valid": 1.0 if evaluation.valid else 0.0,
                    "occupancy_path_length_m": float(evaluation.path_length_m),
                    "occupancy_blocked_cells": float(evaluation.blocked_cell_count),
                    "occupancy_reason": evaluation.reason,
                }
            )
            if evaluation.valid:
                path_penalty = min(0.25, max(0.0, evaluation.path_length_m - candidate.travel_distance_m) * 0.12)
                filtered.append(
                    replace(
                        candidate,
                        clearance_score=float(np.clip(float(candidate.clearance_score) - path_penalty, 0.0, 1.0)),
                        depth_support=depth_support,
                    )
                )
            else:
                blocked_reasons.append(evaluation.reason)
                filtered.append(
                    replace(
                        candidate,
                        valid=False,
                        clearance_score=0.0,
                        depth_support=depth_support,
                    )
                )

        valid_count = sum(1 for candidate in filtered if candidate.valid)
        if original_valid_count > 0 and valid_count == 0:
            filtered = self._restore_best_turn_candidate(candidate_set=CandidateSet(filtered, 0, {}))
            valid_count = sum(1 for candidate in filtered if candidate.valid)
            if valid_count > 0:
                blocked_reasons.append("restored_best_turn_candidate")

        signature = (
            original_valid_count,
            int(valid_count),
            tuple(sorted(set(blocked_reasons))[:4]),
        )
        if signature != self._last_filter_signature:
            self._logger.info(
                "[OccupancyPlanner] candidates valid=%s->%s blocked_reasons=%s",
                original_valid_count,
                int(valid_count),
                ",".join(signature[2]) if signature[2] else "none",
            )
            self._last_filter_signature = signature

        debug = dict(candidate_set.predictor_debug)
        debug["occupancy_valid_before"] = original_valid_count
        debug["occupancy_valid_after"] = int(valid_count)
        debug["occupancy_blocked_reasons"] = blocked_reasons[:8]
        return CandidateSet(candidates=filtered, valid_count=int(valid_count), predictor_debug=debug)

    def save_debug(self, *, observation: VLNObservation, candidate_set: CandidateSet) -> Path | None:
        if not bool(self._config.enabled) or not bool(self._config.debug_enabled):
            return None
        timestamp_sec = float(observation.meta.get("timestamp_sec", 0.0))
        if timestamp_sec - float(self._last_debug_save_time_sec) < max(0.05, float(self._config.debug_update_period_sec)):
            return None
        image = self._render_debug(observation=observation, candidate_set=candidate_set)
        _save_image_atomic(image, self._debug_path)
        self._last_debug_save_time_sec = timestamp_sec
        return self._debug_path

    def snapshot(self) -> OccupancyGridSnapshot | None:
        if not bool(self._config.enabled):
            return None
        inflated = self._inflated_occupied_mask()
        free = self._log_odds <= float(self._config.free_threshold)
        return OccupancyGridSnapshot(
            resolution_m=float(self._resolution_m),
            size_m=float(self._cell_count) * float(self._resolution_m),
            log_odds=self._log_odds.copy(),
            inflated_occupied=inflated.copy(),
            free_mask=free.copy(),
        )

    def _evaluate_candidate(self, *, observation: VLNObservation, candidate: WaypointCandidate) -> _CandidateEvaluation:
        goal_world_xy = _local_to_world_xy(robot_pose=observation.pose, local_xy=candidate.local_goal_xy)
        if not self._inside_world(goal_world_xy[0], goal_world_xy[1]):
            return _CandidateEvaluation(False, "goal_outside_grid", 0.0, 0, goal_world_xy)

        if abs(float(candidate.heading_delta_deg)) <= 12.0:
            blocked_cells = self._corridor_blocked_count(
                start_xy=(float(observation.pose.x), float(observation.pose.y)),
                end_xy=goal_world_xy,
                radius_m=float(self._config.path_corridor_radius_m),
            )
            if blocked_cells > 0:
                return _CandidateEvaluation(False, "corridor_occupied", 0.0, int(blocked_cells), goal_world_xy)
            path = self._astar_path(
                start_xy=(float(observation.pose.x), float(observation.pose.y)),
                goal_xy=goal_world_xy,
            )
            if path is None:
                return _CandidateEvaluation(False, "astar_unreachable", 0.0, 0, goal_world_xy)
            straight = max(1e-3, float(np.hypot(goal_world_xy[0] - observation.pose.x, goal_world_xy[1] - observation.pose.y)))
            path_length = self._path_length_m(path)
            max_allowed = straight * max(1.0, float(self._config.max_path_detour_ratio)) + 0.25
            if path_length > max_allowed:
                return _CandidateEvaluation(False, "astar_detour_too_large", path_length, 0, goal_world_xy)
            return _CandidateEvaluation(True, "reachable", path_length, 0, goal_world_xy)

        blocked_near_robot = self._disc_blocked_count(
            center_xy=(float(observation.pose.x), float(observation.pose.y)),
            radius_m=float(self._config.turn_clearance_radius_m),
        )
        if blocked_near_robot > 0:
            return _CandidateEvaluation(False, "turn_footprint_occupied", 0.0, int(blocked_near_robot), goal_world_xy)
        return _CandidateEvaluation(True, "turn_in_place_clear", 0.0, 0, goal_world_xy)

    def _restore_best_turn_candidate(self, *, candidate_set: CandidateSet) -> list[WaypointCandidate]:
        candidates = list(candidate_set.candidates)
        turn_candidates = [
            candidate
            for candidate in candidates
            if abs(float(candidate.heading_delta_deg)) > 12.0
            and str(candidate.depth_support.get("occupancy_reason", "")) != "turn_footprint_occupied"
        ]
        if not turn_candidates:
            return candidates
        selected = max(turn_candidates, key=lambda item: (float(item.clearance_score), abs(float(item.heading_delta_deg))))
        restored: list[WaypointCandidate] = []
        for candidate in candidates:
            if int(candidate.slot_id) == int(selected.slot_id):
                support = dict(candidate.depth_support)
                support["occupancy_reason"] = "restored_best_turn_candidate"
                support["occupancy_valid"] = 1.0
                restored.append(
                    replace(
                        candidate,
                        valid=True,
                        clearance_score=max(0.05, float(selected.clearance_score)),
                        travel_distance_m=max(0.45, float(selected.travel_distance_m)),
                        depth_support=support,
                    )
                )
            else:
                restored.append(candidate)
        return restored

    def _mark_ray(
        self,
        *,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        endpoint_distance_m: float,
    ) -> tuple[int, int]:
        sx, sy = [float(value) for value in start_xy]
        ex, ey = [float(value) for value in end_xy]
        if not self._inside_world(sx, sy):
            return 0, 0
        distance = max(1e-6, float(endpoint_distance_m))
        max_free_distance = max(0.0, distance - max(self._resolution_m, float(self._config.robot_radius_m) * 0.35))
        free_updates = 0
        occupied_updates = 0
        sample_count = max(1, int(max_free_distance / self._resolution_m))
        for idx in range(1, sample_count + 1):
            alpha = min(1.0, (float(idx) * self._resolution_m) / distance)
            wx = sx + (ex - sx) * alpha
            wy = sy + (ey - sy) * alpha
            cell = self._world_to_cell(wx, wy)
            if cell is None:
                continue
            self._add_log_odds(cell, float(self._config.free_log_odds))
            free_updates += 1
        endpoint_cell = self._world_to_cell(ex, ey)
        if endpoint_cell is not None:
            self._add_log_odds(endpoint_cell, float(self._config.occupied_log_odds))
            occupied_updates += 1
        return free_updates, occupied_updates

    def _astar_path(
        self,
        *,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> list[tuple[int, int]] | None:
        start = self._world_to_cell(start_xy[0], start_xy[1])
        goal = self._world_to_cell(goal_xy[0], goal_xy[1])
        if start is None or goal is None:
            return None
        blocked = self._inflated_occupied_mask()
        if self._cell_blocked(blocked, goal):
            return None
        if self._cell_blocked(blocked, start):
            blocked[start[0], start[1]] = False

        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        heappush(open_heap, (0.0, 0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}
        visited: set[tuple[int, int]] = set()
        counter = 0
        max_iterations = max(200, int(self._config.max_astar_iterations))
        while open_heap and len(visited) < max_iterations:
            _, _, current = heappop(open_heap)
            if current in visited:
                continue
            if current == goal:
                return _reconstruct_path(came_from, current)
            visited.add(current)
            for neighbor, step_cost in self._neighbors(current):
                if neighbor in visited or self._cell_blocked(blocked, neighbor):
                    continue
                unknown_cost = float(self._config.unknown_cell_cost) if not self._cell_free(neighbor) else 0.0
                tentative = float(g_score[current]) + step_cost + max(0.0, unknown_cost)
                if tentative >= float(g_score.get(neighbor, float("inf"))):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                priority = tentative + _cell_distance(neighbor, goal)
                heappush(open_heap, (priority, counter, neighbor))
        return None

    def _corridor_blocked_count(
        self,
        *,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        radius_m: float,
    ) -> int:
        blocked = self._inflated_occupied_mask()
        distance = max(1e-6, float(np.hypot(float(end_xy[0]) - float(start_xy[0]), float(end_xy[1]) - float(start_xy[1]))))
        sample_step = max(self._resolution_m, float(radius_m) * 0.45)
        sample_count = max(1, int(np.ceil(distance / sample_step)))
        blocked_count = 0
        for idx in range(sample_count + 1):
            alpha = float(idx) / float(sample_count)
            wx = float(start_xy[0]) + (float(end_xy[0]) - float(start_xy[0])) * alpha
            wy = float(start_xy[1]) + (float(end_xy[1]) - float(start_xy[1])) * alpha
            blocked_count += self._disc_blocked_count(center_xy=(wx, wy), radius_m=radius_m, blocked_mask=blocked)
            if blocked_count > 0:
                return blocked_count
        return blocked_count

    def _disc_blocked_count(
        self,
        *,
        center_xy: tuple[float, float],
        radius_m: float,
        blocked_mask: np.ndarray | None = None,
    ) -> int:
        center = self._world_to_cell(center_xy[0], center_xy[1])
        if center is None:
            return 1
        blocked = self._inflated_occupied_mask() if blocked_mask is None else blocked_mask
        radius_cells = max(1, int(np.ceil(float(radius_m) / self._resolution_m)))
        row, col = center
        count = 0
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                rr = row + dr
                cc = col + dc
                if rr < 0 or rr >= self._cell_count or cc < 0 or cc >= self._cell_count:
                    count += 1
                    continue
                count += int(blocked[rr, cc])
        return int(count)

    def _inflated_occupied_mask(self) -> np.ndarray:
        occupied = self._occupied_mask()
        radius_cells = max(0, int(np.ceil(float(self._config.inflation_radius_m) / self._resolution_m)))
        if radius_cells <= 0 or not np.any(occupied):
            return occupied
        inflated = occupied.copy()
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr == 0 and dc == 0:
                    continue
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                inflated |= _shift_mask(occupied, dr, dc)
        return inflated

    def _occupied_mask(self) -> np.ndarray:
        raw = self._log_odds >= float(self._config.occupied_threshold)
        min_neighbors = max(1, int(self._config.occupied_min_neighbors))
        if min_neighbors <= 1 or not np.any(raw):
            return raw
        return raw & (_neighbor_count(raw) >= min_neighbors)

    def _cell_free(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        if row < 0 or row >= self._cell_count or col < 0 or col >= self._cell_count:
            return False
        if bool(self._config.unknown_is_traversable):
            return bool(self._log_odds[row, col] <= float(self._config.occupied_threshold))
        return bool(self._log_odds[row, col] <= float(self._config.free_threshold))

    def _cell_blocked(self, blocked: np.ndarray, cell: tuple[int, int]) -> bool:
        row, col = cell
        if row < 0 or row >= self._cell_count or col < 0 or col >= self._cell_count:
            return True
        return bool(blocked[row, col])

    def _neighbors(self, cell: tuple[int, int]):
        row, col = cell
        for dr, dc, cost in (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (1, 1, 1.414),
        ):
            rr = row + dr
            cc = col + dc
            if 0 <= rr < self._cell_count and 0 <= cc < self._cell_count:
                yield (rr, cc), float(cost)

    def _path_length_m(self, path: list[tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        length_cells = 0.0
        for first, second in zip(path[:-1], path[1:]):
            length_cells += _cell_distance(first, second)
        return float(length_cells * self._resolution_m)

    def _add_log_odds(self, cell: tuple[int, int], value: float) -> None:
        row, col = cell
        self._log_odds[row, col] = float(
            np.clip(
                float(self._log_odds[row, col]) + float(value),
                float(self._config.log_odds_min),
                float(self._config.log_odds_max),
            )
        )

    def _world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        if not self._inside_world(x_m, y_m):
            return None
        col = int(np.floor((float(x_m) + self._half_size_m) / self._resolution_m))
        row = int(np.floor((float(y_m) + self._half_size_m) / self._resolution_m))
        if row < 0 or row >= self._cell_count or col < 0 or col >= self._cell_count:
            return None
        return row, col

    def _cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = (float(col) + 0.5) * self._resolution_m - self._half_size_m
        y = (float(row) + 0.5) * self._resolution_m - self._half_size_m
        return float(x), float(y)

    def _inside_world(self, x_m: float, y_m: float) -> bool:
        return abs(float(x_m)) < self._half_size_m and abs(float(y_m)) < self._half_size_m

    def _depth_roi(self, image_height: int) -> tuple[int, int]:
        top, bottom = self._config.depth_roi_vertical_range
        top = float(np.clip(top, 0.0, 0.95))
        bottom = float(np.clip(bottom, top + 0.02, 1.0))
        v0 = int(round(top * float(image_height)))
        v1 = int(round(bottom * float(image_height)))
        return max(0, v0), max(v0 + 1, min(int(image_height), v1))

    def _render_debug(self, *, observation: VLNObservation, candidate_set: CandidateSet) -> Image.Image:
        width = max(360, int(self._config.debug_width))
        height = max(360, int(self._config.debug_height))
        view_size_m = max(2.0, float(self._config.debug_view_size_m))
        image = self._render_binary_grid_image(
            width=width,
            height=height,
            center_x=float(observation.pose.x),
            center_y=float(observation.pose.y),
            yaw_rad=float(observation.pose.yaw_rad),
            view_size_m=view_size_m,
        )
        draw = ImageDraw.Draw(image)
        transform = _DebugTransform(
            width=width,
            height=height,
            center_x=float(observation.pose.x),
            center_y=float(observation.pose.y),
            yaw_rad=float(observation.pose.yaw_rad),
            size_m=view_size_m,
        )
        self._draw_debug_grid(draw=draw, transform=transform)
        self._draw_debug_candidates(draw=draw, transform=transform, observation=observation, candidate_set=candidate_set)
        self._draw_debug_robot(draw=draw, transform=transform, pose=observation.pose)
        draw.rectangle((0, 0, width, 64), fill=(235, 235, 235))
        title_font = _load_font(22)
        label_font = _load_font(16)
        draw.text((12, 8), "ROBOT-CENTERED LIDAR OCCUPANCY MAP", fill=(0, 0, 0), font=title_font)
        draw.text(
            (12, 34),
            "up=front | image-left=robot-left | black=occupied white=free gray=unknown cyan=current lidar yellow=candidate",
            fill=(30, 30, 30),
            font=label_font,
        )
        return image

    def _render_binary_grid_image(
        self,
        *,
        width: int,
        height: int,
        center_x: float,
        center_y: float,
        yaw_rad: float,
        view_size_m: float,
    ) -> Image.Image:
        transform = _DebugTransform(
            width=int(width),
            height=int(height),
            center_x=float(center_x),
            center_y=float(center_y),
            yaw_rad=float(yaw_rad),
            size_m=float(view_size_m),
        )
        header_h = 64
        canvas = np.full((int(height), int(width), 3), 185, dtype=np.uint8)
        canvas[:header_h, :, :] = 235
        occupied = self._occupied_mask()
        free = self._log_odds <= float(self._config.free_threshold)

        y_indices, x_indices = np.indices((int(height) - header_h, int(width)), dtype=np.float32)
        x_indices = x_indices
        y_indices = y_indices + float(header_h)
        local_left = -(x_indices - float(width) * 0.5) / max(transform.scale, 1e-6)
        local_forward = (float(header_h) + float(height - header_h) * 0.5 - y_indices) / max(transform.scale, 1e-6)
        cos_yaw = float(np.cos(float(yaw_rad)))
        sin_yaw = float(np.sin(float(yaw_rad)))
        world_x = float(center_x) + cos_yaw * local_forward - sin_yaw * local_left
        world_y = float(center_y) + sin_yaw * local_forward + cos_yaw * local_left
        cols = np.floor((world_x + self._half_size_m) / self._resolution_m).astype(np.int32)
        rows = np.floor((world_y + self._half_size_m) / self._resolution_m).astype(np.int32)
        inside = (rows >= 0) & (rows < self._cell_count) & (cols >= 0) & (cols < self._cell_count)
        sampled_occupied = np.zeros_like(inside, dtype=bool)
        sampled_free = np.zeros_like(inside, dtype=bool)
        sampled_occupied[inside] = occupied[rows[inside], cols[inside]]
        sampled_free[inside] = free[rows[inside], cols[inside]]
        body = canvas[header_h:, :, :]
        body[sampled_free] = 255
        body[sampled_occupied] = 0
        return Image.fromarray(canvas, mode="RGB")

    def _draw_debug_grid(self, *, draw: ImageDraw.ImageDraw, transform: "_DebugTransform") -> None:
        half = float(transform.size_m) * 0.5
        for left_m in np.arange(np.floor(-half), np.ceil(half) + 0.01, 1.0):
            x0, y0 = transform.local_to_px(-half, float(left_m))
            x1, y1 = transform.local_to_px(half, float(left_m))
            draw.line((x0, y0, x1, y1), fill=(145, 145, 145), width=1)
        for forward_m in np.arange(np.floor(-half), np.ceil(half) + 0.01, 1.0):
            x0, y0 = transform.local_to_px(float(forward_m), -half)
            x1, y1 = transform.local_to_px(float(forward_m), half)
            draw.line((x0, y0, x1, y1), fill=(145, 145, 145), width=1)

    def _draw_debug_cells(self, *, draw: ImageDraw.ImageDraw, transform: "_DebugTransform") -> None:
        occupied = self._occupied_mask()
        inflated = self._inflated_occupied_mask()
        inflated_only = inflated & ~occupied
        inflated_boundary = inflated_only & (_neighbor_count(inflated_only) < 9)
        free = self._log_odds <= float(self._config.free_threshold)
        row_min, row_max, col_min, col_max = self._visible_cell_bounds(transform)
        step = max(1, int(np.floor(0.04 / self._resolution_m)))
        for row in range(row_min, row_max + 1, step):
            for col in range(col_min, col_max + 1, step):
                wx, wy = self._cell_to_world(row, col)
                x, y = transform.to_px(wx, wy)
                radius = max(1, int(round(transform.scale * self._resolution_m * 0.55)))
                if occupied[row, col]:
                    color = (210, 45, 35)
                elif inflated_boundary[row, col]:
                    color = (170, 92, 46)
                elif free[row, col] and bool(self._config.debug_show_free_space):
                    color = (20, 54, 42)
                else:
                    continue
                draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)

    def _visible_cell_bounds(self, transform: "_DebugTransform") -> tuple[int, int, int, int]:
        x0 = transform.center_x - 0.5 * transform.size_m
        x1 = transform.center_x + 0.5 * transform.size_m
        y0 = transform.center_y - 0.5 * transform.size_m
        y1 = transform.center_y + 0.5 * transform.size_m
        min_cell = self._world_to_cell(max(-self._half_size_m + 1e-3, x0), max(-self._half_size_m + 1e-3, y0))
        max_cell = self._world_to_cell(min(self._half_size_m - 1e-3, x1), min(self._half_size_m - 1e-3, y1))
        if min_cell is None or max_cell is None:
            return 0, self._cell_count - 1, 0, self._cell_count - 1
        return min_cell[0], max_cell[0], min_cell[1], max_cell[1]

    def _draw_debug_candidates(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        transform: "_DebugTransform",
        observation: VLNObservation,
        candidate_set: CandidateSet,
    ) -> None:
        rx, ry = transform.local_to_px(0.0, 0.0)
        for candidate in candidate_set.candidates:
            local_forward, local_left = [float(value) for value in candidate.local_goal_xy[:2]]
            x, y = transform.local_to_px(local_forward, local_left)
            color = (255, 225, 55) if candidate.valid else (120, 120, 120)
            draw.line((rx, ry, x, y), fill=color, width=2)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline=(255, 255, 255), width=2)
            draw.text((x - 5, y - 9), str(candidate.slot_id), fill=(0, 0, 0), font=_load_font(16))
            reason = str(candidate.depth_support.get("occupancy_reason", ""))
            if reason and reason != "reachable":
                draw.text((x + 12, y - 9), reason, fill=color, font=_load_font(14))

        self._draw_latest_lidar_points(draw=draw, transform=transform)

    def _draw_latest_lidar_points(self, *, draw: ImageDraw.ImageDraw, transform: "_DebugTransform") -> None:
        points = np.asarray(self._last_lidar_points_local_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
            return
        for forward_m, left_m in points[:: max(1, int(len(points) / 220))]:
            x, y = transform.local_to_px(float(forward_m), float(left_m))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 190, 255))

    def _draw_debug_robot(self, *, draw: ImageDraw.ImageDraw, transform: "_DebugTransform", pose: Pose2D) -> None:
        _ = pose
        x, y = transform.local_to_px(0.0, 0.0)
        scale = 22.0
        tip = (x, y - int(round(scale)))
        left = (x - int(round(scale * 0.70)), y + int(round(scale * 0.70)))
        right = (x + int(round(scale * 0.70)), y + int(round(scale * 0.70)))
        draw.polygon([tip, left, right], fill=(80, 180, 255), outline=(220, 245, 255))


@dataclass
class _DebugTransform:
    width: int
    height: int
    center_x: float
    center_y: float
    yaw_rad: float
    size_m: float

    @property
    def scale(self) -> float:
        return min(float(self.width), float(self.height - 54)) / max(1e-6, float(self.size_m))

    def to_px(self, x_m: float, y_m: float) -> tuple[int, int]:
        dx = float(x_m) - float(self.center_x)
        dy = float(y_m) - float(self.center_y)
        cos_yaw = float(np.cos(self.yaw_rad))
        sin_yaw = float(np.sin(self.yaw_rad))
        forward_m = cos_yaw * dx + sin_yaw * dy
        left_m = -sin_yaw * dx + cos_yaw * dy
        return self.local_to_px(forward_m, left_m)

    def local_to_px(self, forward_m: float, left_m: float) -> tuple[int, int]:
        header_h = 64.0
        x = int(round(float(self.width) * 0.5 - float(left_m) * self.scale))
        y = int(round(header_h + float(self.height - header_h) * 0.5 - float(forward_m) * self.scale))
        return x, y

    def cell_rect(self, x_m: float, y_m: float, resolution_m: float) -> tuple[int, int, int, int]:
        half = 0.5 * float(resolution_m)
        x0, y0 = self.to_px(float(x_m) - half, float(y_m) + half)
        x1, y1 = self.to_px(float(x_m) + half, float(y_m) - half)
        return min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1


def _local_to_world_xy(*, robot_pose: Pose2D, local_xy: list[float] | tuple[float, float]) -> tuple[float, float]:
    forward_m, left_m = [float(value) for value in local_xy[:2]]
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    world_x = float(robot_pose.x) + cos_yaw * forward_m - sin_yaw * left_m
    world_y = float(robot_pose.y) + sin_yaw * forward_m + cos_yaw * left_m
    return float(world_x), float(world_y)


def _shift_mask(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    src_r0 = max(0, -int(row_shift))
    src_r1 = min(rows, rows - int(row_shift))
    dst_r0 = max(0, int(row_shift))
    dst_r1 = min(rows, rows + int(row_shift))
    src_c0 = max(0, -int(col_shift))
    src_c1 = min(cols, cols - int(col_shift))
    dst_c0 = max(0, int(col_shift))
    dst_c1 = min(cols, cols + int(col_shift))
    if src_r0 < src_r1 and src_c0 < src_c1:
        result[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return result


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, int(size))
        except Exception:
            continue
    return ImageFont.load_default()


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    count = np.zeros(mask.shape, dtype=np.int16)
    for row_shift in (-1, 0, 1):
        for col_shift in (-1, 0, 1):
            count += _shift_mask(mask, row_shift, col_shift).astype(np.int16)
    return count


def _cell_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return float(np.hypot(float(first[0] - second[0]), float(first[1] - second[1])))


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _save_image_atomic(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{output_path.name}.", suffix=".tmp", dir=str(output_path.parent))
    os.close(fd)
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
