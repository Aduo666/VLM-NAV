from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .observation_adapter import VLNObservation


@dataclass
class WaypointCandidate:
    slot_id: int
    label: str
    valid: bool
    local_goal_xy: list[float]
    heading_delta_deg: float
    travel_distance_m: float
    clearance_score: float
    depth_support: dict[str, float]
    image_uv: tuple[int, int]


@dataclass
class CandidateSet:
    candidates: list[WaypointCandidate] = field(default_factory=list)
    valid_count: int = 0
    predictor_debug: dict = field(default_factory=dict)


@dataclass
class CandidateWaypointPredictorConfig:
    image_width: int
    image_height: int
    horizontal_fov_deg: float
    roi_vertical_range: tuple[float, float]
    valid_depth_range: tuple[float, float]
    clearance_threshold_m: float
    slot_bearings_deg: tuple[float, ...]
    min_travel_distance_m: float
    preferred_distance_m: float
    max_travel_distance_m: float
    min_free_ratio: float
    column_half_width_px: int
    adaptive_enabled: bool = True
    adaptive_count: int = 5
    adaptive_min_spacing_deg: float = 18.0
    adaptive_smoothing_px: int = 15


class CandidateWaypointPredictor:
    def __init__(self, config: CandidateWaypointPredictorConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger.getChild("candidate_waypoint_predictor")
        self._cx = (float(config.image_width) - 1.0) * 0.5
        self._hfov_rad = np.deg2rad(float(config.horizontal_fov_deg))

    def set_horizontal_fov_deg(self, value: float) -> None:
        self._config.horizontal_fov_deg = float(value)
        self._hfov_rad = np.deg2rad(float(value))

    def predict(self, observation: VLNObservation) -> CandidateSet:
        depth = observation.depth
        if depth is None or np.asarray(depth).ndim != 2:
            return CandidateSet(
                candidates=self._empty_candidates(),
                valid_count=0,
                predictor_debug={"reason": "depth_unavailable"},
            )

        depth_np = np.asarray(depth, dtype=np.float32)
        roi = self._compute_roi(depth_np.shape[0])
        if roi is None:
            return CandidateSet(
                candidates=self._empty_candidates(),
                valid_count=0,
                predictor_debug={"reason": "invalid_roi"},
            )
        v_start, v_end = roi

        valid_min, valid_max = [float(value) for value in self._config.valid_depth_range]
        candidate_bearings_deg = self._base_bearings_deg()
        if bool(self._config.adaptive_enabled):
            candidate_bearings_deg = self._adaptive_bearings_deg(
                depth_image=depth_np,
                v_start=v_start,
                v_end=v_end,
                valid_min=valid_min,
                valid_max=valid_max,
            )
        candidates: list[WaypointCandidate] = []
        valid_count = 0
        for slot_id, bearing_deg in enumerate(candidate_bearings_deg):
            candidate = self._candidate_for_slot(
                depth_image=depth_np,
                bearing_deg=float(bearing_deg),
                slot_id=slot_id,
                v_start=v_start,
                v_end=v_end,
                valid_min=valid_min,
                valid_max=valid_max,
            )
            candidates.append(candidate)
            valid_count += int(candidate.valid)

        return CandidateSet(
            candidates=candidates,
            valid_count=valid_count,
            predictor_debug={
                "roi": [int(v_start), int(v_end)],
                "valid_count": int(valid_count),
                "candidate_bearings_deg": [round(float(value), 1) for value in candidate_bearings_deg],
            },
        )

    def _empty_candidates(self) -> list[WaypointCandidate]:
        result: list[WaypointCandidate] = []
        for slot_id, bearing_deg in enumerate(self._base_bearings_deg()):
            result.append(
                WaypointCandidate(
                    slot_id=slot_id,
                    label=_slot_label(float(bearing_deg)),
                    valid=False,
                    local_goal_xy=[0.0, 0.0],
                    heading_delta_deg=float(bearing_deg),
                    travel_distance_m=0.0,
                    clearance_score=0.0,
                    depth_support={"free_ratio": 0.0, "min_depth": 0.0, "median_depth": 0.0},
                    image_uv=(int(self._cx), self._config.image_height // 2),
                )
            )
        return result

    def _candidate_for_slot(
        self,
        *,
        depth_image: np.ndarray,
        bearing_deg: float,
        slot_id: int,
        v_start: int,
        v_end: int,
        valid_min: float,
        valid_max: float,
    ) -> WaypointCandidate:
        width = depth_image.shape[1]
        u_center = self._bearing_to_column(bearing_deg, width)
        half_width = max(2, int(self._config.column_half_width_px))
        u0 = max(0, u_center - half_width)
        u1 = min(width, u_center + half_width + 1)
        patch = depth_image[v_start:v_end, u0:u1]
        patch = np.asarray(patch, dtype=np.float32)
        valid_mask = np.isfinite(patch) & (patch >= valid_min) & (patch <= valid_max)
        if not np.any(valid_mask):
            return WaypointCandidate(
                slot_id=slot_id,
                label=_slot_label(bearing_deg),
                valid=False,
                local_goal_xy=[0.0, 0.0],
                heading_delta_deg=float(bearing_deg),
                travel_distance_m=0.0,
                clearance_score=0.0,
                depth_support={"free_ratio": 0.0, "min_depth": 0.0, "median_depth": 0.0},
                image_uv=(u_center, (v_start + v_end) // 2),
            )

        valid_patch = patch[valid_mask]
        safe_mask = valid_mask & (patch >= float(self._config.clearance_threshold_m))
        free_ratio = float(np.mean(safe_mask))
        median_depth = float(np.median(valid_patch))
        min_depth = float(np.percentile(valid_patch, 20.0))
        travel_distance_m = float(
            np.clip(
                min(median_depth, float(self._config.preferred_distance_m)),
                float(self._config.min_travel_distance_m),
                float(self._config.max_travel_distance_m),
            )
        )
        heading_rad = float(np.deg2rad(bearing_deg))
        local_goal_xy = [
            float(travel_distance_m * np.cos(heading_rad)),
            float(travel_distance_m * np.sin(heading_rad)),
        ]
        clearance_score = float(
            np.clip(
                0.55 * free_ratio
                + 0.45
                * np.exp(
                    -0.5
                    * (
                        (travel_distance_m - float(self._config.preferred_distance_m))
                        / max(0.25, float(self._config.max_travel_distance_m - self._config.min_travel_distance_m))
                    )
                    ** 2
                ),
                0.0,
                1.0,
            )
        )
        valid = bool(
            free_ratio >= float(self._config.min_free_ratio)
            and median_depth >= float(self._config.min_travel_distance_m)
            and local_goal_xy[0] > 0.10
        )
        return WaypointCandidate(
            slot_id=slot_id,
            label=_slot_label(bearing_deg),
            valid=valid,
            local_goal_xy=local_goal_xy if valid else [0.0, 0.0],
            heading_delta_deg=float(bearing_deg),
            travel_distance_m=float(travel_distance_m if valid else 0.0),
            clearance_score=float(clearance_score if valid else 0.0),
            depth_support={
                "free_ratio": free_ratio,
                "min_depth": min_depth,
                "median_depth": median_depth,
            },
            image_uv=(u_center, (v_start + v_end) // 2),
        )

    def _compute_roi(self, image_height: int) -> tuple[int, int] | None:
        top_ratio, bottom_ratio = [float(value) for value in self._config.roi_vertical_range]
        top_ratio = float(np.clip(top_ratio, 0.0, 0.95))
        bottom_ratio = float(np.clip(bottom_ratio, 0.05, 1.0))
        v_start = int(round(top_ratio * float(image_height)))
        v_end = int(round(bottom_ratio * float(image_height)))
        if v_end - v_start < 4:
            return None
        return v_start, v_end

    def _base_bearings_deg(self) -> list[float]:
        configured = [float(value) for value in self._config.slot_bearings_deg]
        count = max(3, int(self._config.adaptive_count))
        if len(configured) == count:
            return configured
        if not configured:
            max_bearing = 75.0
            return [float(value) for value in np.linspace(max_bearing, -max_bearing, count)]
        left_bound = max(configured)
        right_bound = min(configured)
        return [float(value) for value in np.linspace(left_bound, right_bound, count)]

    def _adaptive_bearings_deg(
        self,
        *,
        depth_image: np.ndarray,
        v_start: int,
        v_end: int,
        valid_min: float,
        valid_max: float,
    ) -> list[float]:
        base_bearings = self._base_bearings_deg()
        if depth_image.ndim != 2 or depth_image.size == 0:
            return base_bearings
        depth_profile = self._depth_profile(
            depth_image=depth_image,
            v_start=v_start,
            v_end=v_end,
            valid_min=valid_min,
            valid_max=valid_max,
        )
        if depth_profile is None:
            return base_bearings
        width = int(depth_image.shape[1])
        if width <= 4:
            return base_bearings
        base_columns = [self._bearing_to_column(value, width) for value in base_bearings]
        search_radius_cols = max(6, int(round(width * 0.10)))
        min_spacing_cols = max(
            4,
            int(round(width * float(self._config.adaptive_min_spacing_deg) / max(10.0, float(self._config.horizontal_fov_deg)))),
        )
        chosen_columns: list[int] = []
        for index, base_column in enumerate(base_columns):
            lower = max(0, base_column - search_radius_cols)
            upper = min(width - 1, base_column + search_radius_cols)
            if lower > upper:
                lower = upper = int(np.clip(base_column, 0, width - 1))
            window = depth_profile[lower : upper + 1]
            if window.size == 0 or not np.isfinite(window).any():
                best_column = int(np.clip(base_column, 0, width - 1))
            else:
                local_offset = int(np.nanargmax(window))
                best_column = lower + local_offset
            if chosen_columns:
                best_column = max(best_column, chosen_columns[-1] + min_spacing_cols)
            remaining = len(base_columns) - index - 1
            max_allowed = (width - 1) - remaining * min_spacing_cols
            best_column = int(np.clip(best_column, 0, max_allowed))
            chosen_columns.append(best_column)
        return [self._column_to_bearing(column, width) for column in chosen_columns]

    def _bearing_to_column(self, bearing_deg: float, image_width: int) -> int:
        bearing_rad = float(np.deg2rad(bearing_deg))
        normalized = np.tan(bearing_rad) / max(np.tan(self._hfov_rad * 0.5), 1e-6)
        # Robot local +Y is left, while image u increases to the right.
        u = int(round(self._cx - normalized * self._cx))
        return max(0, min(int(image_width) - 1, u))

    def _column_to_bearing(self, column: int, image_width: int) -> float:
        normalized = (float(column) - self._cx) / max(self._cx, 1.0)
        bearing_rad = float(np.arctan(normalized * np.tan(self._hfov_rad * 0.5)))
        return float(np.rad2deg(-bearing_rad))

    def _depth_profile(
        self,
        *,
        depth_image: np.ndarray,
        v_start: int,
        v_end: int,
        valid_min: float,
        valid_max: float,
    ) -> np.ndarray | None:
        patch = np.asarray(depth_image[v_start:v_end, :], dtype=np.float32)
        if patch.ndim != 2 or patch.size == 0:
            return None
        valid_mask = np.isfinite(patch) & (patch >= float(valid_min)) & (patch <= float(valid_max))
        if not np.any(valid_mask):
            return None
        column_score = np.zeros(patch.shape[1], dtype=np.float32)
        for column in range(patch.shape[1]):
            valid_values = patch[:, column][valid_mask[:, column]]
            if valid_values.size == 0:
                column_score[column] = 0.0
                continue
            column_score[column] = float(np.percentile(valid_values, 60.0))
        smooth_window = max(3, int(self._config.adaptive_smoothing_px))
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        return np.convolve(column_score, kernel, mode="same")


def _slot_label(bearing_deg: float) -> str:
    if bearing_deg <= -60.0:
        return "HARD_RIGHT"
    if bearing_deg < -10.0:
        return "RIGHT"
    if bearing_deg >= 60.0:
        return "HARD_LEFT"
    if bearing_deg > 10.0:
        return "LEFT"
    return "CENTER"
