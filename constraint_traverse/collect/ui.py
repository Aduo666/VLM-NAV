from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .bev import BevConfig, body_xy_to_bev_pixel


@dataclass(frozen=True)
class PanelStatus:
    mode: str
    episode_id: str
    sample_count: int
    elapsed_sec: float
    message: str


class CollectorPanel:
    def __init__(self, ui, *, image_size_px: int, command_sink) -> None:
        self._ui = ui
        self._command_sink = command_sink
        self._size = int(image_size_px)
        self._provider = ui.ByteImageProvider()
        initial = np.full((self._size, self._size, 4), 128, dtype=np.uint8)
        initial[..., 3] = 255
        self._provider.set_data_array(
            np.ascontiguousarray(initial).reshape(-1),
            [self._size, self._size],
        )

        self.control_window = ui.Window(
            "Constraint Traverse Collector",
            width=330,
            height=245,
            visible=True,
        )
        with self.control_window.frame:
            with ui.VStack(spacing=5):
                self._state_label = ui.Label("State: IDLE", height=24)
                self._episode_label = ui.Label("Episode: -", height=24)
                self._sample_label = ui.Label("Samples: 0", height=24)
                self._message_label = ui.Label("Ready", word_wrap=True, height=45)
                ui.Button(
                    "START DEMO",
                    height=32,
                    clicked_fn=lambda: command_sink("START"),
                )
                with ui.HStack(height=32):
                    ui.Button(
                        "SUCCESS",
                        clicked_fn=lambda: command_sink("SUCCESS"),
                    )
                    ui.Button(
                        "FAILURE",
                        clicked_fn=lambda: command_sink("FAILURE"),
                    )
                with ui.HStack(height=32):
                    ui.Button(
                        "DISCARD",
                        clicked_fn=lambda: command_sink("DISCARD"),
                    )
                    ui.Button(
                        "RESET TO START",
                        clicked_fn=lambda: command_sink("RESET_TO_START"),
                    )

        self.bev_window = ui.Window(
            "Episode BEV",
            width=self._size + 20,
            height=self._size + 20,
            visible=True,
        )
        with self.bev_window.frame:
            self._image = ui.ImageWithProvider(
                self._provider,
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
            )

    def update_status(self, status: PanelStatus) -> None:
        self._state_label.text = f"State: {status.mode}"
        self._episode_label.text = f"Episode: {status.episode_id or '-'}"
        self._sample_label.text = (
            f"Samples: {status.sample_count}  Time: {status.elapsed_sec:.1f}s"
        )
        self._message_label.text = status.message

    def update_bev(
        self,
        bev: np.ndarray,
        *,
        config: BevConfig,
        start_body_xy: np.ndarray | None,
        goal_body_xy: np.ndarray | None,
        trajectory_body_xy: np.ndarray | None,
    ) -> None:
        image = render_bev_preview(
            bev,
            config=config,
            start_body_xy=start_body_xy,
            goal_body_xy=goal_body_xy,
            trajectory_body_xy=trajectory_body_xy,
        )
        image = image.resize((self._size, self._size), resample=Image.Resampling.NEAREST)
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self._provider.set_data_array(
            np.ascontiguousarray(rgba).reshape(-1),
            [self._size, self._size],
        )

    def shutdown(self) -> None:
        self.control_window.visible = False
        self.bev_window.visible = False


def render_bev_preview(
    bev: np.ndarray,
    *,
    config: BevConfig,
    start_body_xy: np.ndarray | None = None,
    goal_body_xy: np.ndarray | None = None,
    trajectory_body_xy: np.ndarray | None = None,
) -> Image.Image:
    data = np.asarray(bev, dtype=np.uint8)
    if data.shape != (3, config.height, config.width):
        raise ValueError(
            f"BEV shape {data.shape} does not match {(3, config.height, config.width)}"
        )

    occupied = data[0] > 0
    free = data[1] > 0
    rgb = np.full((config.height, config.width, 3), 130, dtype=np.uint8)
    rgb[free] = (238, 238, 238)
    rgb[occupied] = (25, 25, 25)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    if trajectory_body_xy is not None:
        pixels = [
            body_xy_to_bev_pixel(point, config)
            for point in np.asarray(trajectory_body_xy, dtype=np.float64).reshape(-1, 2)
        ]
        pixels = [point for point in pixels if point is not None]
        if len(pixels) >= 2:
            draw.line(pixels, fill=(0, 210, 220), width=2)

    _draw_marker(draw, start_body_xy, config, fill=(40, 190, 60), radius=3)
    _draw_marker(draw, goal_body_xy, config, fill=(225, 55, 55), radius=3)
    _draw_marker(draw, np.asarray([0.0, 0.0]), config, fill=(55, 105, 235), radius=3)

    robot_pixel = body_xy_to_bev_pixel(np.asarray([0.0, 0.0]), config)
    heading_pixel = body_xy_to_bev_pixel(
        np.asarray([min(0.4, 0.5 * config.forward_m), 0.0]),
        config,
    )
    if robot_pixel is not None and heading_pixel is not None:
        draw.line((*robot_pixel, *heading_pixel), fill=(55, 105, 235), width=2)
    return image


def _draw_marker(draw, body_xy, config: BevConfig, *, fill, radius: int) -> None:
    if body_xy is None:
        return
    pixel = body_xy_to_bev_pixel(np.asarray(body_xy, dtype=np.float64), config)
    if pixel is None:
        return
    x, y = pixel
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
    )
