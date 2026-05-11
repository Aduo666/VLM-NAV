from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .landmark_map import LandmarkMapEntry
from .observation_adapter import Pose2D, VLNObservation
from ..runtime.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OccupancyGridConfig:
    enabled: bool = True
    resolution_m: float = 0.08
    size_m: float = 80.0
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
    decay_rate: float = 1.0
    inflation_radius_m: float = 0.26
    robot_radius_m: float = 0.32
    path_corridor_radius_m: float = 0.30
    turn_clearance_radius_m: float = 0.36
    unknown_is_traversable: bool = True
    unknown_cell_cost: float = 0.10
    max_astar_iterations: int = 7000
    max_path_detour_ratio: float = 2.2
    debug_enabled: bool = True
    debug_image_path: str = "assets/local_map.png"
    debug_update_period_sec: float = 0.5
    debug_width: int = 900
    debug_height: int = 900
    debug_view_size_m: float = 8.0
    debug_show_free_space: bool = False
    semantic_planner_enabled: bool = True
    semantic_goal_step_m: float = 0.90
    semantic_goal_min_step_m: float = 0.45
    semantic_goal_lateral_search_m: float = 0.80


@dataclass
class OccupancyGridSnapshot:
    resolution_m: float
    size_m: float
    log_odds: np.ndarray
    inflated_occupied: np.ndarray
    free_mask: np.ndarray
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class _CandidateEvaluation:
    valid: bool
    reason: str
    path_length_m: float
    blocked_cell_count: int
    goal_world_xy: tuple[float, float]


@dataclass
class SemanticSubgoalPlan:
    valid: bool
    reason: str
    target_label: str = ""
    target_world_xy: tuple[float, float] | None = None
    target_local_xy: tuple[float, float] = (0.0, 0.0)
    approach_world_xy: tuple[float, float] | None = None
    subgoal_world_xy: tuple[float, float] | None = None
    subgoal_local_xy: tuple[float, float] = (0.0, 0.0)
    path_length_m: float = 0.0
    distance_to_target_m: float = 0.0
    bearing_deg: float = 0.0
    path_world_xy: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Global Occupancy Map
# ---------------------------------------------------------------------------


class GlobalOccupancyMap:
    """Global log-odds occupancy grid in the odom/world coordinate system.

    Frame convention (unchanged from the original mapper):
    - Observation pose is odom/base pose in meters.
    - Robot local +X is forward, +Y is left.
    - Image u increases to the right, so image-left pixels project to +Y.

    Key differences from the old local grid:
    - Grid origin is tracked in world coordinates and can be dynamically
      expanded as the robot navigates beyond the current boundaries.
    - No decay by default (``decay_rate=1.0``).
    - LiDAR rays are processed in batch via numpy (Bresenham-style free/hit
      marking without a Python loop per sample).
    """

    def __init__(self, *, config: OccupancyGridConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("occupancy_grid")
        self._resolution_m = max(0.03, float(config.resolution_m))
        self._initial_size_m = max(4.0, float(config.size_m))
        self._cell_count = int(round(self._initial_size_m / self._resolution_m))
        if self._cell_count % 2 == 1:
            self._cell_count += 1

        # World-space origin of cell (0, 0).  For a fresh grid centred on the
        # world origin the corner sits at (-half, -half).
        half = 0.5 * float(self._cell_count) * self._resolution_m
        self._origin_x: float = -half
        self._origin_y: float = -half

        self._log_odds = np.zeros((self._cell_count, self._cell_count), dtype=np.float32)
        self._last_update_time_sec = -float("inf")
        self._last_debug_save_time_sec = -float("inf")
        self._last_filter_signature: tuple[int, int, tuple[str, ...]] | None = None
        self._last_lidar_points_local_xy = np.zeros((0, 2), dtype=np.float32)
        self._last_semantic_plan: SemanticSubgoalPlan | None = None
        self._last_semantic_entries: list[LandmarkMapEntry] = []
        self._debug_path = Path(config.debug_image_path)
        if not self._debug_path.is_absolute():
            self._debug_path = PROJECT_ROOT / self._debug_path

    # -- properties ----------------------------------------------------------

    @property
    def output_path(self) -> Path:
        return self._debug_path

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    # -- coordinate helpers --------------------------------------------------

    def _world_to_cell_xy(
        self, x_m: np.ndarray, y_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert world (x, y) arrays to integer (row, col) arrays."""
        col = np.floor((x_m - self._origin_x) / self._resolution_m).astype(np.intp)
        row = np.floor((y_m - self._origin_y) / self._resolution_m).astype(np.intp)
        return row, col

    def _world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        col = int(np.floor((float(x_m) - self._origin_x) / self._resolution_m))
        row = int(np.floor((float(y_m) - self._origin_y) / self._resolution_m))
        if row < 0 or row >= self._cell_count or col < 0 or col >= self._cell_count:
            return None
        return row, col

    def _cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self._origin_x + (float(col) + 0.5) * self._resolution_m
        y = self._origin_y + (float(row) + 0.5) * self._resolution_m
        return float(x), float(y)

    def _inside_world(self, x_m: float, y_m: float) -> bool:
        col = (float(x_m) - self._origin_x) / self._resolution_m
        row = (float(y_m) - self._origin_y) / self._resolution_m
        return 0.0 <= col < float(self._cell_count) and 0.0 <= row < float(self._cell_count)

    # -- dynamic expansion ---------------------------------------------------

    def _expand_if_needed(self, robot_x: float, robot_y: float) -> None:
        """Double the grid if the robot is within 20% of any boundary."""
        margin = 0.2 * float(self._cell_count) * self._resolution_m
        x_min = self._origin_x
        y_min = self._origin_y
        x_max = self._origin_x + float(self._cell_count) * self._resolution_m
        y_max = self._origin_y + float(self._cell_count) * self._resolution_m
        needs_expand = (
            robot_x - x_min < margin
            or x_max - robot_x < margin
            or robot_y - y_min < margin
            or y_max - robot_y < margin
        )
        if not needs_expand:
            return
        old_count = self._cell_count
        new_count = old_count * 2
        new_grid = np.zeros((new_count, new_count), dtype=np.float32)
        # Place old data in the centre of the new grid.
        offset = old_count // 2
        new_grid[offset : offset + old_count, offset : offset + old_count] = self._log_odds
        self._origin_x -= float(offset) * self._resolution_m
        self._origin_y -= float(offset) * self._resolution_m
        self._cell_count = new_count
        self._log_odds = new_grid
        self._logger.info(
            "[OccupancyGrid] expanded %d->%d cells  origin=(%.1f, %.1f)",
            old_count,
            new_count,
            self._origin_x,
            self._origin_y,
        )

    # -- depth-based update (fallback) ---------------------------------------

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

        robot_x = float(observation.pose.x)
        robot_y = float(observation.pose.y)
        self._expand_if_needed(robot_x, robot_y)

        height, width = depth_np.shape[:2]
        v0, v1 = self._depth_roi(height)
        stride = max(2, int(self._config.depth_sample_stride_px))
        min_depth = max(0.02, float(self._config.min_depth_m))
        max_depth = max(min_depth + 0.10, float(self._config.max_depth_m))
        half_fov_rad = float(np.deg2rad(float(observation.camera.get("hfov_deg", 90.0))) * 0.5)
        cos_yaw = float(np.cos(observation.pose.yaw_rad))
        sin_yaw = float(np.sin(observation.pose.yaw_rad))

        # Vectorised pixel -> world conversion.
        vs = np.arange(v0, v1, stride, dtype=np.float32)
        us = np.arange(0, width, stride, dtype=np.float32)
        if vs.size == 0 or us.size == 0:
            return
        uu, vv = np.meshgrid(us, vs)
        dists = depth_np[vv.astype(int), uu.astype(int)]
        valid = np.isfinite(dists) & (dists >= min_depth) & (dists <= max_depth)
        if not np.any(valid):
            return
        norm_u = (uu - float(width - 1) * 0.5) / max(float(width - 1) * 0.5, 1.0)
        bearings = norm_u * half_fov_rad
        forward_m = dists * np.cos(bearings)
        left_m = -dists * np.sin(bearings)
        fwd_valid = forward_m > 0.05
        valid = valid & fwd_valid
        if not np.any(valid):
            return
        world_xs = robot_x + cos_yaw * forward_m - sin_yaw * left_m
        world_ys = robot_y + sin_yaw * forward_m + cos_yaw * left_m

        start_xy = np.array([robot_x, robot_y], dtype=np.float64)
        end_xy = np.stack([world_xs[valid], world_ys[valid]], axis=-1).astype(np.float64)
        endpoint_dists = np.hypot(forward_m[valid], left_m[valid]).astype(np.float64)
        hit_mask = np.ones(end_xy.shape[0], dtype=bool)

        n_free, n_occ = self._batch_mark_rays(
            start_xy=start_xy,
            end_xy_arr=end_xy,
            endpoint_distances=endpoint_dists,
            endpoint_hit_mask=hit_mask,
        )
        self._last_update_time_sec = timestamp_sec
        if n_occ > 0:
            self._logger.debug(
                "[Occupancy] update free_cells=%s occupied_cells=%s occupied_total=%s",
                n_free,
                n_occ,
                int(np.count_nonzero(self._occupied_mask())),
            )

    # -- LiDAR-based update --------------------------------------------------

    def update_from_lidar(
        self,
        *,
        robot_pose: Pose2D,
        points_local_xy: np.ndarray,
        beam_endpoints_local_xy: np.ndarray | None = None,
        beam_hit_mask: np.ndarray | None = None,
        timestamp_sec: float,
    ) -> None:
        if not bool(self._config.enabled):
            return
        if float(timestamp_sec) - float(self._last_update_time_sec) < max(
            0.02, float(self._config.update_period_sec)
        ):
            return
        beam_endpoints = (
            None
            if beam_endpoints_local_xy is None
            else np.asarray(beam_endpoints_local_xy, dtype=np.float32)
        )
        hit_mask = (
            None
            if beam_hit_mask is None
            else np.asarray(beam_hit_mask, dtype=bool).reshape(-1)
        )
        if (
            beam_endpoints is not None
            and beam_endpoints.ndim == 2
            and beam_endpoints.shape[1] >= 2
            and beam_endpoints.size > 0
        ):
            endpoints = beam_endpoints[:, :2]
            hits = (
                hit_mask
                if hit_mask is not None and hit_mask.shape[0] == endpoints.shape[0]
                else np.ones((endpoints.shape[0],), dtype=bool)
            )
        else:
            points = np.asarray(points_local_xy, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
                return
            endpoints = points[:, :2]
            hits = np.ones((endpoints.shape[0],), dtype=bool)
        if endpoints.ndim != 2 or endpoints.shape[0] == 0:
            return
        self._last_lidar_points_local_xy = endpoints.copy()

        robot_x = float(robot_pose.x)
        robot_y = float(robot_pose.y)
        self._expand_if_needed(robot_x, robot_y)

        cos_yaw = float(np.cos(robot_pose.yaw_rad))
        sin_yaw = float(np.sin(robot_pose.yaw_rad))
        min_depth = max(0.02, float(self._config.min_depth_m))
        max_depth = max(min_depth + 0.10, float(self._config.max_depth_m))

        # Vectorised local -> world conversion.
        fwd = endpoints[:, 0].astype(np.float64)
        lft = endpoints[:, 1].astype(np.float64)
        world_xs = float(robot_x) + cos_yaw * fwd - sin_yaw * lft
        world_ys = float(robot_y) + sin_yaw * fwd + cos_yaw * lft

        distances = np.hypot(world_xs - robot_x, world_ys - robot_y)
        valid = np.isfinite(distances) & (distances >= min_depth) & (distances <= max_depth)
        if not np.any(valid):
            self._last_update_time_sec = float(timestamp_sec)
            return
        valid_idx = np.where(valid)[0]

        start_xy = np.array([robot_x, robot_y], dtype=np.float64)
        end_xy_arr = np.stack([world_xs[valid_idx], world_ys[valid_idx]], axis=-1).astype(np.float64)
        endpoint_distances = distances[valid_idx].astype(np.float64)
        endpoint_hit_mask = hits[valid_idx].astype(bool)

        n_free, n_occ = self._batch_mark_rays(
            start_xy=start_xy,
            end_xy_arr=end_xy_arr,
            endpoint_distances=endpoint_distances,
            endpoint_hit_mask=endpoint_hit_mask,
        )

        self._last_update_time_sec = float(timestamp_sec)
        if n_occ > 0:
            self._logger.debug(
                "[Occupancy] lidar_update free_cells=%s occupied_cells=%s occupied_total=%s",
                n_free,
                n_occ,
                int(np.count_nonzero(self._occupied_mask())),
            )

    # -- vectorised ray marking ----------------------------------------------

    def _batch_mark_rays(
        self,
        *,
        start_xy: np.ndarray,
        end_xy_arr: np.ndarray,
        endpoint_distances: np.ndarray,
        endpoint_hit_mask: np.ndarray,
    ) -> tuple[int, int]:
        """Bresenham-style batch ray marking via numpy.

        For each beam:
        - Intermediate cells along the ray are marked FREE.
        - The endpoint cell is marked OCCUPIED only when the beam is a hit.
        - No-hit / max-range beams only clear free space.

        Returns (n_free_cells_updated, n_occupied_cells_updated).
        """
        n = end_xy_arr.shape[0]
        if n == 0:
            return 0, 0
        sx, sy = float(start_xy[0]), float(start_xy[1])
        dx = end_xy_arr[:, 0] - sx
        dy = end_xy_arr[:, 1] - sy

        beam_dists = np.sqrt(dx * dx + dy * dy)
        min_beam = max(0.05, float(self._config.min_depth_m))
        valid_beam = beam_dists >= min_beam
        if not np.any(valid_beam):
            return 0, 0

        vb = np.where(valid_beam)[0]
        dx_v = dx[vb]
        dy_v = dy[vb]
        dist_v = beam_dists[vb]
        dir_x = dx_v / dist_v
        dir_y = dy_v / dist_v

        # Sample spacing and count along the free region.
        min_sample_dist = max(self._resolution_m, float(self._config.robot_radius_m) * 0.35)
        max_free_dist = np.maximum(0.0, dist_v - min_sample_dist)
        n_samples = np.maximum(1, (max_free_dist / self._resolution_m).astype(np.intp))

        if int(n_samples.max()) <= 0:
            return 0, 0

        max_s = int(n_samples.max())
        idx = np.arange(max_s, dtype=np.float64)
        sample_dist = (idx + 1.0) * self._resolution_m
        in_range = sample_dist[np.newaxis, :] <= max_free_dist[:, np.newaxis]

        # World positions of all samples.
        wx = sx + dir_x[:, np.newaxis] * sample_dist[np.newaxis, :]
        wy = sy + dir_y[:, np.newaxis] * sample_dist[np.newaxis, :]

        rows, cols = self._world_to_cell_xy(wx, wy)
        in_grid = (rows >= 0) & (rows < self._cell_count) & (cols >= 0) & (cols < self._cell_count)
        valid_mask = in_grid & in_range

        if np.any(valid_mask):
            r = rows[valid_mask].ravel()
            c = cols[valid_mask].ravel()
            np.add.at(
                self._log_odds,
                (r, c),
                np.float32(float(self._config.free_log_odds)),
            )
        n_free = int(np.count_nonzero(valid_mask))

        # Mark occupied endpoint cells (only for hit beams).
        hits_v = endpoint_hit_mask[vb]
        hit_mask = hits_v & (dist_v > 0.0)
        if np.any(hit_mask):
            hx = end_xy_arr[vb[hit_mask], 0]
            hy = end_xy_arr[vb[hit_mask], 1]
            h_rows, h_cols = self._world_to_cell_xy(hx, hy)
            h_valid = (h_rows >= 0) & (h_rows < self._cell_count) & (h_cols >= 0) & (h_cols < self._cell_count)
            if np.any(h_valid):
                hr = h_rows[h_valid].ravel()
                hc = h_cols[h_valid].ravel()
                np.add.at(
                    self._log_odds,
                    (hr, hc),
                    np.float32(float(self._config.occupied_log_odds)),
                )
        n_occupied = int(np.sum(hit_mask)) if np.any(hit_mask) else 0

        # Clamp to configured bounds.
        np.clip(
            self._log_odds,
            float(self._config.log_odds_min),
            float(self._config.log_odds_max),
            out=self._log_odds,
        )
        return n_free, n_occupied

    # -- candidate filtering -------------------------------------------------

    def filter_candidates(
        self, *, observation: VLNObservation, candidate_set: CandidateSet
    ) -> CandidateSet:
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
                path_penalty = min(
                    0.25,
                    max(0.0, evaluation.path_length_m - candidate.travel_distance_m) * 0.12,
                )
                filtered.append(
                    replace(
                        candidate,
                        clearance_score=float(
                            np.clip(float(candidate.clearance_score) - path_penalty, 0.0, 1.0)
                        ),
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
            filtered = self._restore_best_turn_candidate(
                candidate_set=CandidateSet(filtered, 0, {})
            )
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
        return CandidateSet(
            candidates=filtered, valid_count=int(valid_count), predictor_debug=debug
        )

    # -- debug output --------------------------------------------------------

    def save_debug(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        semantic_entries: list[LandmarkMapEntry] | None = None,
        semantic_plan: SemanticSubgoalPlan | None = None,
    ) -> Path | None:
        if not bool(self._config.enabled) or not bool(self._config.debug_enabled):
            return None
        timestamp_sec = float(observation.meta.get("timestamp_sec", 0.0))
        if timestamp_sec - float(self._last_debug_save_time_sec) < max(
            0.05, float(self._config.debug_update_period_sec)
        ):
            return None
        self._last_semantic_entries = list(semantic_entries or [])
        self._last_semantic_plan = semantic_plan
        image = self._render_debug(
            observation=observation,
            candidate_set=candidate_set,
            semantic_entries=self._last_semantic_entries,
            semantic_plan=self._last_semantic_plan,
        )
        _save_image_atomic(image, self._debug_path)
        self._last_debug_save_time_sec = timestamp_sec
        return self._debug_path

    # -- snapshot (for navigation_map_visualizer) ----------------------------

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
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )

    # -- local submap (for candidate filtering & A*) -------------------------

    def get_local_submap(
        self, robot_pose: Pose2D, size_m: float
    ) -> dict[str, Any]:
        """Return a local view of the global grid centred on *robot_pose*.

        Returns a dict with keys: ``log_odds``, ``resolution_m``,
        ``origin_x``, ``origin_y``, ``rows``, ``cols``.
        Cells outside the global grid are filled with 0 (unknown).
        """
        half = float(size_m) * 0.5
        cx = float(robot_pose.x)
        cy = float(robot_pose.y)
        origin_x = cx - half
        origin_y = cy - half
        n_cells = max(1, int(round(float(size_m) / self._resolution_m)))
        sub = np.zeros((n_cells, n_cells), dtype=np.float32)

        # Map sub-map cells into global grid coordinates.
        sub_ox = (origin_x - self._origin_x) / self._resolution_m
        sub_oy = (origin_y - self._origin_y) / self._resolution_m
        for sr in range(n_cells):
            for sc in range(n_cells):
                gr = int(np.floor(sub_oy + sr))
                gc = int(np.floor(sub_ox + sc))
                if 0 <= gr < self._cell_count and 0 <= gc < self._cell_count:
                    sub[sr, sc] = self._log_odds[gr, gc]
        return {
            "log_odds": sub,
            "resolution_m": self._resolution_m,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "rows": n_cells,
            "cols": n_cells,
        }

    # -- semantic planner ----------------------------------------------------

    def plan_toward_landmark(
        self,
        *,
        observation: VLNObservation,
        landmark_entry: LandmarkMapEntry | None,
        stop_distance_m: float,
    ) -> SemanticSubgoalPlan:
        if not bool(self._config.enabled) or not bool(
            self._config.semantic_planner_enabled
        ):
            return SemanticSubgoalPlan(valid=False, reason="semantic_planner_disabled")
        if landmark_entry is None or not bool(landmark_entry.world_xy):
            return SemanticSubgoalPlan(valid=False, reason="no_landmark_anchor")

        robot_xy = np.asarray(
            [float(observation.pose.x), float(observation.pose.y)], dtype=np.float32
        )
        target_world = np.asarray(landmark_entry.world_xy[:2], dtype=np.float32)
        target_local = _world_to_local_xy(
            robot_pose=observation.pose, world_xy=target_world
        )
        distance_to_target_m = float(
            np.hypot(float(target_local[0]), float(target_local[1]))
        )
        bearing_deg = float(
            np.degrees(
                np.arctan2(float(target_local[1]), max(1.0e-6, float(target_local[0])))
            )
        )
        if distance_to_target_m <= max(0.05, float(stop_distance_m)):
            plan = SemanticSubgoalPlan(
                valid=False,
                reason="within_stop_distance",
                target_label=str(landmark_entry.label),
                target_world_xy=(float(target_world[0]), float(target_world[1])),
                target_local_xy=(float(target_local[0]), float(target_local[1])),
                distance_to_target_m=float(distance_to_target_m),
                bearing_deg=float(bearing_deg),
            )
            self._last_semantic_plan = plan
            return plan

        direction_world = target_world - robot_xy
        norm = float(np.linalg.norm(direction_world))
        if norm <= 1.0e-6:
            return SemanticSubgoalPlan(valid=False, reason="zero_direction_to_target")
        unit_world = direction_world / norm
        left_normal_world = np.asarray(
            [-float(unit_world[1]), float(unit_world[0])], dtype=np.float32
        )
        standoff_m = max(0.20, float(stop_distance_m))
        approach_distance_m = max(0.10, distance_to_target_m - standoff_m)
        base_approach_world = robot_xy + unit_world * float(approach_distance_m)
        lateral_search_m = max(0.0, float(self._config.semantic_goal_lateral_search_m))
        lateral_offsets = [0.0]
        for offset in (0.25, 0.50, 0.80, 1.10):
            if offset <= lateral_search_m + 1.0e-6:
                lateral_offsets.extend([offset, -offset])
        radial_scales = (1.00, 0.85, 0.70, 0.55)

        candidate_goals_world: list[tuple[float, float]] = []
        seen_goal_keys: set[tuple[int, int]] = set()
        for radial_scale in radial_scales:
            scaled_world = robot_xy + unit_world * float(
                max(0.10, approach_distance_m * float(radial_scale))
            )
            for lateral_offset in lateral_offsets:
                goal_world = scaled_world + left_normal_world * float(lateral_offset)
                goal_xy = (float(goal_world[0]), float(goal_world[1]))
                key = (int(round(goal_xy[0] * 20.0)), int(round(goal_xy[1] * 20.0)))
                if key in seen_goal_keys or not self._inside_world(goal_xy[0], goal_xy[1]):
                    continue
                seen_goal_keys.add(key)
                candidate_goals_world.append(goal_xy)

        best_plan: SemanticSubgoalPlan | None = None
        for goal_world_xy in candidate_goals_world:
            path = self._astar_path(
                start_xy=(float(observation.pose.x), float(observation.pose.y)),
                goal_xy=goal_world_xy,
            )
            if path is None:
                continue
            path_length_m = self._path_length_m(path)
            if path_length_m <= 1.0e-6:
                continue
            subgoal_step_m = float(
                np.clip(
                    min(
                        path_length_m,
                        max(0.0, distance_to_target_m - standoff_m),
                    ),
                    float(self._config.semantic_goal_min_step_m),
                    float(self._config.semantic_goal_step_m),
                )
            )
            subgoal_world_xy = self._path_point_world(
                path=path, distance_along_path_m=subgoal_step_m
            )
            subgoal_local_xy = _world_to_local_xy(
                robot_pose=observation.pose, world_xy=subgoal_world_xy
            )
            if float(subgoal_local_xy[0]) < 0.02 and abs(float(subgoal_local_xy[1])) < 0.02:
                continue
            score = float(
                path_length_m
                + 0.25 * abs(float(goal_world_xy[0]) - float(base_approach_world[0]))
                + 0.25 * abs(float(goal_world_xy[1]) - float(base_approach_world[1]))
            )
            candidate_plan = SemanticSubgoalPlan(
                valid=True,
                reason="ok",
                target_label=str(landmark_entry.label),
                target_world_xy=(float(target_world[0]), float(target_world[1])),
                target_local_xy=(float(target_local[0]), float(target_local[1])),
                approach_world_xy=(float(goal_world_xy[0]), float(goal_world_xy[1])),
                subgoal_world_xy=(float(subgoal_world_xy[0]), float(subgoal_world_xy[1])),
                subgoal_local_xy=(float(subgoal_local_xy[0]), float(subgoal_local_xy[1])),
                path_length_m=float(path_length_m),
                distance_to_target_m=float(distance_to_target_m),
                bearing_deg=float(bearing_deg),
                path_world_xy=[
                    self._cell_to_world(row, col) for row, col in path
                ],
            )
            if best_plan is None or score < (
                float(best_plan.path_length_m)
                + 0.25
                * abs(float(best_plan.approach_world_xy[0]) - float(base_approach_world[0]))
                + 0.25
                * abs(float(best_plan.approach_world_xy[1]) - float(base_approach_world[1]))
            ):
                best_plan = candidate_plan

        if best_plan is None:
            best_plan = SemanticSubgoalPlan(
                valid=False,
                reason="no_reachable_semantic_subgoal",
                target_label=str(landmark_entry.label),
                target_world_xy=(float(target_world[0]), float(target_world[1])),
                target_local_xy=(float(target_local[0]), float(target_local[1])),
                approach_world_xy=(float(base_approach_world[0]), float(base_approach_world[1])),
                distance_to_target_m=float(distance_to_target_m),
                bearing_deg=float(bearing_deg),
            )
        self._last_semantic_plan = best_plan
        return best_plan

    # -- internal helpers (unchanged logic) ----------------------------------

    def _evaluate_candidate(
        self, *, observation: VLNObservation, candidate: WaypointCandidate
    ) -> _CandidateEvaluation:
        goal_world_xy = _local_to_world_xy(
            robot_pose=observation.pose, local_xy=candidate.local_goal_xy
        )
        if not self._inside_world(goal_world_xy[0], goal_world_xy[1]):
            return _CandidateEvaluation(
                False, "goal_outside_grid", 0.0, 0, goal_world_xy
            )

        if abs(float(candidate.heading_delta_deg)) <= 12.0:
            blocked_cells = self._corridor_blocked_count(
                start_xy=(float(observation.pose.x), float(observation.pose.y)),
                end_xy=goal_world_xy,
                radius_m=float(self._config.path_corridor_radius_m),
            )
            if blocked_cells > 0:
                return _CandidateEvaluation(
                    False, "corridor_occupied", 0.0, int(blocked_cells), goal_world_xy
                )
            path = self._astar_path(
                start_xy=(float(observation.pose.x), float(observation.pose.y)),
                goal_xy=goal_world_xy,
            )
            if path is None:
                return _CandidateEvaluation(
                    False, "astar_unreachable", 0.0, 0, goal_world_xy
                )
            straight = max(
                1e-3,
                float(
                    np.hypot(
                        goal_world_xy[0] - observation.pose.x,
                        goal_world_xy[1] - observation.pose.y,
                    )
                ),
            )
            path_length = self._path_length_m(path)
            max_allowed = straight * max(1.0, float(self._config.max_path_detour_ratio)) + 0.25
            if path_length > max_allowed:
                return _CandidateEvaluation(
                    False, "astar_detour_too_large", path_length, 0, goal_world_xy
                )
            return _CandidateEvaluation(
                True, "reachable", path_length, 0, goal_world_xy
            )

        blocked_near_robot = self._disc_blocked_count(
            center_xy=(float(observation.pose.x), float(observation.pose.y)),
            radius_m=float(self._config.turn_clearance_radius_m),
        )
        if blocked_near_robot > 0:
            return _CandidateEvaluation(
                False,
                "turn_footprint_occupied",
                0.0,
                int(blocked_near_robot),
                goal_world_xy,
            )
        return _CandidateEvaluation(
            True, "turn_in_place_clear", 0.0, 0, goal_world_xy
        )

    def _restore_best_turn_candidate(
        self, *, candidate_set: CandidateSet
    ) -> list[WaypointCandidate]:
        candidates = list(candidate_set.candidates)
        turn_candidates = [
            candidate
            for candidate in candidates
            if abs(float(candidate.heading_delta_deg)) > 12.0
            and str(candidate.depth_support.get("occupancy_reason", ""))
            != "turn_footprint_occupied"
        ]
        if not turn_candidates:
            return candidates
        selected = max(
            turn_candidates,
            key=lambda item: (
                float(item.clearance_score),
                abs(float(item.heading_delta_deg)),
            ),
        )
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
                unknown_cost = (
                    float(self._config.unknown_cell_cost)
                    if not self._cell_free(neighbor)
                    else 0.0
                )
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
        distance = max(
            1e-6,
            float(
                np.hypot(
                    float(end_xy[0]) - float(start_xy[0]),
                    float(end_xy[1]) - float(start_xy[1]),
                )
            ),
        )
        sample_step = max(self._resolution_m, float(radius_m) * 0.45)
        sample_count = max(1, int(np.ceil(distance / sample_step)))
        blocked_count = 0
        for idx in range(sample_count + 1):
            alpha = float(idx) / float(sample_count)
            wx = float(start_xy[0]) + (float(end_xy[0]) - float(start_xy[0])) * alpha
            wy = float(start_xy[1]) + (float(end_xy[1]) - float(start_xy[1])) * alpha
            blocked_count += self._disc_blocked_count(
                center_xy=(wx, wy), radius_m=radius_m, blocked_mask=blocked
            )
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
        radius_cells = max(
            0,
            int(np.ceil(float(self._config.inflation_radius_m) / self._resolution_m)),
        )
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

    def _path_point_world(
        self,
        *,
        path: list[tuple[int, int]],
        distance_along_path_m: float,
    ) -> tuple[float, float]:
        if not path:
            return 0.0, 0.0
        if len(path) == 1:
            return self._cell_to_world(path[0][0], path[0][1])
        remaining = max(0.0, float(distance_along_path_m))
        previous_xy = self._cell_to_world(path[0][0], path[0][1])
        for cell in path[1:]:
            current_xy = self._cell_to_world(cell[0], cell[1])
            segment_m = float(
                np.hypot(
                    float(current_xy[0]) - float(previous_xy[0]),
                    float(current_xy[1]) - float(previous_xy[1]),
                )
            )
            if segment_m <= 1.0e-6:
                previous_xy = current_xy
                continue
            if remaining <= segment_m:
                alpha = remaining / segment_m
                world_x = float(previous_xy[0]) + alpha * (
                    float(current_xy[0]) - float(previous_xy[0])
                )
                world_y = float(previous_xy[1]) + alpha * (
                    float(current_xy[1]) - float(previous_xy[1])
                )
                return float(world_x), float(world_y)
            remaining -= segment_m
            previous_xy = current_xy
        return previous_xy

    def _depth_roi(self, image_height: int) -> tuple[int, int]:
        top, bottom = self._config.depth_roi_vertical_range
        top = float(np.clip(top, 0.0, 0.95))
        bottom = float(np.clip(bottom, top + 0.02, 1.0))
        v0 = int(round(top * float(image_height)))
        v1 = int(round(bottom * float(image_height)))
        return max(0, v0), max(v0 + 1, min(int(image_height), v1))

    # -- debug rendering (vectorised) ----------------------------------------

    def _render_debug(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        semantic_entries: list[LandmarkMapEntry] | None = None,
        semantic_plan: SemanticSubgoalPlan | None = None,
    ) -> Image.Image:
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
        self._draw_debug_candidates(
            draw=draw,
            transform=transform,
            observation=observation,
            candidate_set=candidate_set,
        )
        self._draw_semantic_overlays(
            draw=draw,
            transform=transform,
            observation=observation,
            semantic_entries=semantic_entries or [],
            semantic_plan=semantic_plan,
        )
        self._draw_debug_robot(draw=draw, transform=transform, pose=observation.pose)

        # Header bar with title and legend.
        draw.rectangle((0, 0, width, 64), fill=(235, 235, 235))
        title_font = _load_font(22)
        label_font = _load_font(16)
        draw.text((12, 8), "LOCAL ROBOT-CENTERED OCCUPANCY MAP", fill=(0, 0, 0), font=title_font)
        legend = (
            "up=front | image-left=robot-left | "
            "black=occupied  white=free  gray=unknown  "
            "cyan=lidar  yellow=candidate  magenta=potential  orange=confirmed  green=reached/subgoal"
        )
        draw.text((12, 34), legend, fill=(30, 30, 30), font=label_font)

        # Bottom-right legend swatches.
        swatch_x = width - 250
        swatch_y = height - 30
        for colour, label_text in (
            ((0, 0, 0), "occupied"),
            ((255, 255, 255), "free"),
            ((185, 185, 185), "unknown"),
        ):
            draw.rectangle(
                (swatch_x, swatch_y, swatch_x + 14, swatch_y + 14),
                fill=colour,
                outline=(100, 100, 100),
            )
            draw.text((swatch_x + 18, swatch_y - 2), label_text, fill=(30, 30, 30), font=label_font)
            swatch_x += 80

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

        body_h = int(height) - header_h
        y_indices, x_indices = np.indices((body_h, int(width)), dtype=np.float32)
        y_indices = y_indices + float(header_h)
        local_left = -(x_indices - float(width) * 0.5) / max(transform.scale, 1e-6)
        local_forward = (
            float(header_h) + float(body_h) * 0.5 - y_indices
        ) / max(transform.scale, 1e-6)
        cos_yaw = float(np.cos(float(yaw_rad)))
        sin_yaw = float(np.sin(float(yaw_rad)))
        world_x = float(center_x) + cos_yaw * local_forward - sin_yaw * local_left
        world_y = float(center_y) + sin_yaw * local_forward + cos_yaw * local_left
        rows, cols = self._world_to_cell_xy(world_x, world_y)
        inside = (
            (rows >= 0)
            & (rows < self._cell_count)
            & (cols >= 0)
            & (cols < self._cell_count)
        )
        sampled_occupied = np.zeros_like(inside, dtype=bool)
        sampled_free = np.zeros_like(inside, dtype=bool)
        sampled_occupied[inside] = occupied[rows[inside], cols[inside]]
        sampled_free[inside] = free[rows[inside], cols[inside]]
        body = canvas[header_h:, :, :]
        body[sampled_free] = 255
        body[sampled_occupied] = 0
        return Image.fromarray(canvas, mode="RGB")

    def _draw_debug_grid(
        self, *, draw: ImageDraw.ImageDraw, transform: "_DebugTransform"
    ) -> None:
        half = float(transform.size_m) * 0.5
        for left_m in np.arange(np.floor(-half), np.ceil(half) + 0.01, 1.0):
            x0, y0 = transform.local_to_px(-half, float(left_m))
            x1, y1 = transform.local_to_px(half, float(left_m))
            draw.line((x0, y0, x1, y1), fill=(145, 145, 145), width=1)
        for forward_m in np.arange(np.floor(-half), np.ceil(half) + 0.01, 1.0):
            x0, y0 = transform.local_to_px(float(forward_m), -half)
            x1, y1 = transform.local_to_px(float(forward_m), half)
            draw.line((x0, y0, x1, y1), fill=(145, 145, 145), width=1)

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
            draw.ellipse(
                (x - 10, y - 10, x + 10, y + 10),
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
            draw.text((x - 5, y - 9), str(candidate.slot_id), fill=(0, 0, 0), font=_load_font(16))
            reason = str(candidate.depth_support.get("occupancy_reason", ""))
            if reason and reason != "reachable":
                draw.text((x + 12, y - 9), reason, fill=color, font=_load_font(14))

        self._draw_latest_lidar_points(draw=draw, transform=transform)

    def _draw_latest_lidar_points(
        self, *, draw: ImageDraw.ImageDraw, transform: "_DebugTransform"
    ) -> None:
        points = np.asarray(self._last_lidar_points_local_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
            return
        for forward_m, left_m in points[:: max(1, int(len(points) / 220))]:
            x, y = transform.local_to_px(float(forward_m), float(left_m))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 190, 255))

    def _draw_debug_robot(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        transform: "_DebugTransform",
        pose: Pose2D,
    ) -> None:
        _ = pose
        x, y = transform.local_to_px(0.0, 0.0)
        scale = 22.0
        tip = (x, y - int(round(scale)))
        left = (x - int(round(scale * 0.70)), y + int(round(scale * 0.70)))
        right = (x + int(round(scale * 0.70)), y + int(round(scale * 0.70)))
        draw.polygon([tip, left, right], fill=(80, 180, 255), outline=(220, 245, 255))

    def _draw_semantic_overlays(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        transform: "_DebugTransform",
        observation: VLNObservation,
        semantic_entries: list[LandmarkMapEntry],
        semantic_plan: SemanticSubgoalPlan | None,
    ) -> None:
        _ = observation
        anchor_font = _load_font(16)
        for entry in semantic_entries:
            if not bool(entry.world_xy):
                continue
            x, y = transform.to_px(float(entry.world_xy[0]), float(entry.world_xy[1]))
            evidence = str(getattr(entry, "evidence_level", "")).strip().lower()
            if bool(entry.reached):
                color = (80, 235, 125)
            elif evidence == "confirmed":
                color = (245, 175, 40)
            else:
                color = (220, 60, 220)
            draw.ellipse(
                (x - 10, y - 10, x + 10, y + 10),
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
            draw.text(
                (x + 12, y - 9),
                f"{entry.label} {evidence or entry.status} {entry.confidence:.2f}",
                fill=color,
                font=anchor_font,
            )

        plan = semantic_plan
        if plan is None:
            return
        if plan.path_world_xy:
            path_px = [
                transform.to_px(float(x_m), float(y_m))
                for x_m, y_m in plan.path_world_xy
            ]
            if len(path_px) >= 2:
                draw.line(path_px, fill=(60, 220, 60), width=3)
        if plan.subgoal_world_xy is not None:
            sx, sy = transform.to_px(
                float(plan.subgoal_world_xy[0]), float(plan.subgoal_world_xy[1])
            )
            draw.ellipse(
                (sx - 8, sy - 8, sx + 8, sy + 8),
                fill=(60, 220, 60),
                outline=(255, 255, 255),
                width=2,
            )
            draw.text((sx + 10, sy - 9), "subgoal", fill=(20, 120, 20), font=anchor_font)


# Backward-compatible alias.
OccupancyGridMapper = GlobalOccupancyMap


# ---------------------------------------------------------------------------
# Debug transform (unchanged)
# ---------------------------------------------------------------------------


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
        y = int(
            round(header_h + float(self.height - header_h) * 0.5 - float(forward_m) * self.scale)
        )
        return x, y

    def cell_rect(
        self, x_m: float, y_m: float, resolution_m: float
    ) -> tuple[int, int, int, int]:
        half = 0.5 * float(resolution_m)
        x0, y0 = self.to_px(float(x_m) - half, float(y_m) + half)
        x1, y1 = self.to_px(float(x_m) + half, float(y_m) - half)
        return min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1


# ---------------------------------------------------------------------------
# Free functions (unchanged)
# ---------------------------------------------------------------------------


def _local_to_world_xy(
    *, robot_pose: Pose2D, local_xy: list[float] | tuple[float, float]
) -> tuple[float, float]:
    forward_m, left_m = [float(value) for value in local_xy[:2]]
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    world_x = float(robot_pose.x) + cos_yaw * forward_m - sin_yaw * left_m
    world_y = float(robot_pose.y) + sin_yaw * forward_m + cos_yaw * left_m
    return float(world_x), float(world_y)


def _world_to_local_xy(
    *,
    robot_pose: Pose2D,
    world_xy: list[float] | tuple[float, float] | np.ndarray,
) -> tuple[float, float]:
    delta = np.asarray(world_xy, dtype=np.float32).reshape(2) - np.asarray(
        [float(robot_pose.x), float(robot_pose.y)], dtype=np.float32
    )
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    forward_m = cos_yaw * float(delta[0]) + sin_yaw * float(delta[1])
    left_m = -sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
    return float(forward_m), float(left_m)


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
    fd, temp_path = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
