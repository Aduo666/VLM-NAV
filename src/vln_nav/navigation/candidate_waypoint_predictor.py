from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .observation_adapter import VLNObservation
from ..runtime.config import PROJECT_ROOT


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
    debug_image_path: str
    adaptive_enabled: bool = True
    adaptive_count: int = 5
    adaptive_min_spacing_deg: float = 18.0
    adaptive_smoothing_px: int = 15


class CandidateWaypointPredictor:
    def __init__(self, config: CandidateWaypointPredictorConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger.getChild("candidate_waypoint_predictor")
        self._debug_image_path = Path(config.debug_image_path)
        self._last_saved_debug_path: Path | None = None
        self._cx = (float(config.image_width) - 1.0) * 0.5
        self._hfov_rad = np.deg2rad(float(config.horizontal_fov_deg))

    @property
    def debug_image_path(self) -> Path:
        target = self._debug_image_path
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        return target

    @property
    def last_saved_debug_path(self) -> Path | None:
        return self._last_saved_debug_path

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

    def save_debug(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        target_observation=None,
        left_panel_image_path: str | Path | None = None,
    ) -> Path | None:
        if observation.rgb is None or np.asarray(observation.rgb).ndim != 3:
            return None
        image_rgb = np.asarray(observation.rgb, dtype=np.uint8)
        rgb_panel = Image.fromarray(image_rgb.copy())
        _overlay_fresh_target_mask(image=rgb_panel, target_observation=target_observation)
        draw = ImageDraw.Draw(rgb_panel)
        _draw_target_overlay(draw=draw, image_size=rgb_panel.size, target_observation=target_observation)
        for candidate in candidate_set.candidates:
            u, v = candidate.image_uv
            fill = (220, 40, 40) if candidate.valid else (85, 85, 85)
            radius = 12
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=fill, outline=(255, 255, 255), width=2)
            _draw_centered_text(
                draw=draw,
                xy=(u, v),
                text=str(candidate.slot_id),
                font=_debug_font(15, bold=True),
                fill=(255, 255, 255),
            )

        canvas = rgb_panel
        if left_panel_image_path is not None:
            rgb_width, rgb_height = rgb_panel.size
            costmap_width = max(360, int(rgb_height * 0.78))
            costmap_panel = _load_left_panel_image(
                image_path=left_panel_image_path,
                width=costmap_width,
                height=rgb_height,
            )
            if costmap_panel is not None:
                canvas = Image.new("RGB", (costmap_width + rgb_width, rgb_height), (0, 0, 0))
                canvas.paste(costmap_panel, (0, 0))
                canvas.paste(rgb_panel, (costmap_width, 0))

        target = self.debug_image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _save_image_atomic(canvas, target)
        self._last_saved_debug_path = target
        return target

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

    def _render_candidate_costmap_panel(
        self,
        *,
        width: int,
        height: int,
        depth: np.ndarray | None,
        candidate_set: CandidateSet,
    ) -> Image.Image:
        panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
        draw = ImageDraw.Draw(panel)
        width = int(width)
        height = int(height)
        margin = 26
        robot_x = width // 2
        robot_y = height - 62
        meters_forward = 4.0
        meters_side = 2.4
        scale = min((height - 110) / meters_forward, (width - 2 * margin) / (2.0 * meters_side))

        grid_color = (45, 55, 64)
        for meter in np.arange(0.5, meters_forward + 0.01, 0.5):
            y = int(round(robot_y - meter * scale))
            draw.line((margin, y, width - margin, y), fill=grid_color, width=1)
            draw.text((margin, y - 8), f"{meter:.1f}m", fill=(105, 118, 128), font=_debug_font(12))
        for side_meter in np.arange(-meters_side, meters_side + 0.01, 0.5):
            x = int(round(robot_x - side_meter * scale))
            draw.line((x, robot_y, x, max(28, robot_y - meters_forward * scale)), fill=grid_color, width=1)

        draw.rectangle((0, 0, width, 34), fill=(0, 0, 0))
        draw.text((12, 10), "LOCAL CANDIDATE COSTMAP", fill=(255, 255, 255), font=_debug_font(16, bold=True))
        draw.line((robot_x, robot_y, robot_x, max(28, robot_y - meters_forward * scale)), fill=(130, 145, 155), width=2)
        half_fov_rad = max(1e-6, self._hfov_rad * 0.5)
        fov_len = meters_forward * scale
        draw.line(
            (robot_x, robot_y, robot_x - int(round(np.tan(half_fov_rad) * fov_len)), int(round(robot_y - fov_len))),
            fill=(70, 82, 45),
            width=2,
        )
        draw.line(
            (robot_x, robot_y, robot_x + int(round(np.tan(half_fov_rad) * fov_len)), int(round(robot_y - fov_len))),
            fill=(70, 82, 45),
            width=2,
        )
        _draw_depth_costmap_points(
            draw=draw,
            depth=depth,
            robot_x=robot_x,
            robot_y=robot_y,
            scale=scale,
            meters_forward=meters_forward,
            meters_side=meters_side,
            horizontal_fov_rad=self._hfov_rad,
            stop_distance_m=float(self._config.clearance_threshold_m),
        )

        draw.polygon(
            [
                (robot_x, robot_y - 20),
                (robot_x - 12, robot_y + 12),
                (robot_x + 12, robot_y + 12),
            ],
            fill=(80, 180, 255),
            outline=(210, 240, 255),
        )
        draw.text((robot_x + 16, robot_y - 10), "robot", fill=(210, 240, 255), font=_debug_font(14, bold=True))

        for candidate in candidate_set.candidates:
            heading_rad = float(np.deg2rad(candidate.heading_delta_deg))
            if candidate.valid:
                forward_m = float(candidate.local_goal_xy[0])
                left_m = float(candidate.local_goal_xy[1])
                color = (50, 210, 110)
                radius = 12
            else:
                preview_distance = max(0.45, min(float(self._config.preferred_distance_m), meters_forward))
                forward_m = float(preview_distance * np.cos(heading_rad))
                left_m = float(preview_distance * np.sin(heading_rad))
                color = (115, 115, 115)
                radius = 9
            x = int(round(np.clip(robot_x - left_m * scale, margin, width - margin)))
            y = int(round(np.clip(robot_y - forward_m * scale, 42, robot_y + 20)))
            draw.line((robot_x, robot_y, x, y), fill=color, width=2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=3)
            _draw_centered_text(
                draw=draw,
                xy=(x, y),
                text=str(candidate.slot_id),
                font=_debug_font(22, bold=True),
                fill=(255, 255, 255),
            )
            _draw_candidate_tag(
                draw=draw,
                xy=(x + radius + 6, max(38, y - 18)),
                candidate=candidate,
            )

        return panel


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


def _candidate_action_text(candidate: WaypointCandidate) -> str:
    heading = float(candidate.heading_delta_deg)
    if heading > 10.0:
        return "LEFT"
    if heading < -10.0:
        return "RIGHT"
    return "FORWARD"


def _draw_candidate_tag(
    *,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    candidate: WaypointCandidate,
) -> None:
    text = f"{int(candidate.slot_id)} {_candidate_action_text(candidate)}"
    font = _debug_font(18, bold=True)
    x, y = int(xy[0]), int(xy[1])
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
    except Exception:
        w = max(80, 10 * len(text))
        h = 22
    fill = (25, 25, 25) if candidate.valid else (75, 75, 75)
    outline = (255, 255, 255) if candidate.valid else (150, 150, 150)
    draw.rectangle((x - 3, y - 3, x + w + 7, y + h + 5), fill=fill, outline=outline, width=2)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _draw_centered_text(
    *,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int],
) -> None:
    x, y = int(xy[0]), int(xy[1])
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font)
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
    except Exception:
        w = 8 * len(str(text))
        h = 14
    draw.text((x - w // 2, y - h // 2 - 1), str(text), fill=fill, font=font)


def _debug_font(size: int, *, bold: bool = False):
    names = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, int(size))
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_depth_costmap_points(
    *,
    draw: ImageDraw.ImageDraw,
    depth: np.ndarray | None,
    robot_x: int,
    robot_y: int,
    scale: float,
    meters_forward: float,
    meters_side: float,
    horizontal_fov_rad: float,
    stop_distance_m: float,
) -> None:
    if depth is None:
        return
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2 or depth_array.size == 0:
        return

    height, width = depth_array.shape[:2]
    half_fov_rad = max(1e-6, float(horizontal_fov_rad) * 0.5)
    v0 = int(round(height * 0.42))
    v1 = int(round(height * 0.92))
    step_v = max(4, height // 80)
    step_u = max(4, width // 100)
    max_range_m = min(float(meters_forward), 5.0)
    stop_distance_m = float(stop_distance_m)

    for v in range(max(0, v0), min(height, v1), step_v):
        for u in range(0, width, step_u):
            distance = float(depth_array[v, u])
            if not np.isfinite(distance) or distance <= 0.10 or distance > max_range_m:
                continue
            normalized_u = (float(u) - float(width - 1) * 0.5) / max(float(width - 1) * 0.5, 1.0)
            bearing_rad = normalized_u * half_fov_rad
            forward_m = distance * float(np.cos(bearing_rad))
            left_m = -distance * float(np.sin(bearing_rad))
            if forward_m < 0.0 or forward_m > meters_forward or abs(left_m) > meters_side:
                continue
            x = int(round(robot_x - left_m * scale))
            y = int(round(robot_y - forward_m * scale))
            if distance <= stop_distance_m:
                color = (235, 70, 55)
                radius = 2
            elif distance <= stop_distance_m + 0.45:
                color = (245, 150, 60)
                radius = 1
            else:
                color = (75, 92, 104)
                radius = 1
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def _load_left_panel_image(*, image_path: str | Path | None, width: int, height: int) -> Image.Image | None:
    if image_path is None:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        source = Image.open(path).convert("RGB")
        source.thumbnail((int(width), int(height)), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
        panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
        offset = ((int(width) - source.width) // 2, (int(height) - source.height) // 2)
        panel.paste(source, offset)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, int(width), 34), fill=(0, 0, 0))
        draw.text((12, 10), "LOCAL OCCUPANCY MAP / CANDIDATES", fill=(255, 255, 255))
        return panel
    except Exception:
        return None


def _overlay_fresh_target_mask(*, image: Image.Image, target_observation) -> None:
    if not _visible_target_with_bbox(target_observation):
        return
    mask = getattr(target_observation, "mask", None)
    if mask is None:
        return
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.ndim != 2 or mask_np.shape[0] != image.height or mask_np.shape[1] != image.width:
        return
    mask_alpha = Image.fromarray(np.asarray(mask_np, dtype=np.uint8) * 95, mode="L")
    green = Image.new("RGB", image.size, (40, 255, 110))
    blended = Image.blend(image, green, 0.36)
    image.paste(blended, (0, 0), mask_alpha)


def _draw_target_overlay(*, draw: ImageDraw.ImageDraw, image_size: tuple[int, int], target_observation) -> None:
    width, height = [int(value) for value in image_size]
    if target_observation is None:
        draw.rectangle((0, height - 34, width, height), fill=(0, 0, 0))
        draw.text((12, height - 24), "target grounding: unavailable", fill=(255, 210, 90))
        return

    label = str(getattr(target_observation, "label", "target"))
    reason = str(getattr(target_observation, "reason", "unknown"))
    source = str(getattr(target_observation, "source", "unknown"))
    segmentation_source = str(getattr(target_observation, "segmentation_source", "none"))
    depth_m = getattr(target_observation, "depth_m", None)
    depth_text = "none" if depth_m is None else f"{float(depth_m):.2f}m"
    if _visible_target_with_bbox(target_observation):
        bbox = getattr(target_observation, "bbox", None)
        if bbox is not None:
            xmin, ymin, xmax, ymax = [int(value) for value in bbox]
            xmin = max(0, min(width - 1, xmin))
            ymin = max(0, min(height - 1, ymin))
            xmax = max(0, min(width, xmax))
            ymax = max(0, min(height, ymax))
            color = (40, 255, 110)
            draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=4)
            center_uv = getattr(target_observation, "center_uv", None)
            if center_uv is not None:
                center_u, center_v = [int(round(float(value))) for value in center_uv]
                draw.line((center_u - 12, center_v, center_u + 12, center_v), fill=color, width=3)
                draw.line((center_u, center_v - 12, center_u, center_v + 12), fill=color, width=3)
            freshness = "fresh" if _fresh_visible_target(target_observation) else "reused"
            text = f"{freshness} grounded {label} depth={depth_text} seg={segmentation_source} source={source}"
            draw.rectangle((xmin, max(0, ymin - 26), min(width, xmin + 520), ymin), fill=(0, 0, 0))
            draw.text((xmin + 4, max(0, ymin - 21)), text[:95], fill=color)
            return

    status = (
        f"target grounding: not overlaid label={label} reason={reason} "
        f"seg={segmentation_source} source={source} depth={depth_text}"
    )
    draw.rectangle((0, height - 38, width, height), fill=(0, 0, 0))
    draw.text((12, height - 26), status[:125], fill=(255, 130, 90))


def _fresh_visible_target(target_observation) -> bool:
    if target_observation is None:
        return False
    if not bool(getattr(target_observation, "visible", False)):
        return False
    if getattr(target_observation, "bbox", None) is None:
        return False
    source = str(getattr(target_observation, "source", "")).lower()
    reason = str(getattr(target_observation, "reason", "")).lower()
    return "reused" not in source and reason == "ok"


def _visible_target_with_bbox(target_observation) -> bool:
    if target_observation is None:
        return False
    if not bool(getattr(target_observation, "visible", False)):
        return False
    return getattr(target_observation, "bbox", None) is not None


def _save_image_atomic(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
