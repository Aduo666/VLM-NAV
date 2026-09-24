from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .bev import BevConfig, body_xy_to_bev_pixel


_NEAREST_RESAMPLE = getattr(Image, "Resampling", Image).NEAREST


@dataclass(frozen=True)
class PanelStatus:
    mode: str
    episode_id: str
    sample_count: int
    elapsed_sec: float
    robot_placed: bool
    goal_selected: bool
    can_reset_start: bool


class CollectorPanel:
    def __init__(self, ui, *, image_size_px: int, command_sink) -> None:
        self._ui = ui
        self._command_sink = command_sink
        self._size = int(image_size_px)
        self._latest_bev: np.ndarray | None = None
        self._latest_bev_config: BevConfig | None = None
        self._provider = ui.ByteImageProvider()
        initial = np.full((self._size, self._size, 4), 128, dtype=np.uint8)
        initial[..., 3] = 255
        self._provider.set_data_array(
            np.ascontiguousarray(initial).reshape(-1),
            [self._size, self._size],
        )

        self.control_window = ui.Window(
            "Collector Controls",
            width=330,
            height=300,
            visible=True,
        )
        with self.control_window.frame:
            with ui.VStack(spacing=5):
                self._state_label = ui.Label("State: IDLE", height=24)
                self._episode_label = ui.Label("Episode: -", height=24)
                self._sample_label = ui.Label("Samples: 0", height=24)
                self._time_label = ui.Label("Time: 0.0s", height=24)
                self._step_label = ui.Label("Step 1: Press BORN", height=24)
                self._keys_label = ui.Label(
                    "Drive: Arrow keys or W/S/A/D",
                    height=24,
                )
                self._born_button = ui.Button(
                    "BORN",
                    height=32,
                    clicked_fn=lambda: command_sink("BORN"),
                )
                self._reset_map_button = ui.Button(
                    "RESET MAP",
                    height=32,
                    clicked_fn=lambda: command_sink("RESET_MAP"),
                )
                self._start_button = ui.Button(
                    "START DEMO",
                    height=32,
                    clicked_fn=lambda: command_sink("START"),
                )
                with ui.HStack(height=32):
                    self._success_button = ui.Button(
                        "SUCCESS",
                        clicked_fn=lambda: command_sink("SUCCESS"),
                    )
                    self._discard_button = ui.Button(
                        "DISCARD",
                        clicked_fn=lambda: command_sink("DISCARD"),
                    )
                self._reset_start_button = ui.Button(
                    "RESET TO START",
                    height=32,
                    clicked_fn=lambda: command_sink("RESET_TO_START"),
                )

        self.bev_window = ui.Window(
            "Goal Map",
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
                pixel_aligned=True,
                opaque_for_mouse_events=True,
            )
            self._image.set_mouse_pressed_fn(self._on_map_pressed)

    def update_status(self, status: PanelStatus) -> None:
        self._state_label.text = f"State: {status.mode}"
        self._episode_label.text = f"Episode: {status.episode_id or '-'}"
        self._sample_label.text = f"Samples: {status.sample_count}"
        self._time_label.text = f"Time: {status.elapsed_sec:.1f}s"

        idle = status.mode == "IDLE"
        placing_born = status.mode == "PLACING_BORN"
        recording = status.mode == "RECORDING"
        self._born_button.enabled = idle or placing_born
        self._reset_map_button.enabled = idle and status.robot_placed
        self._start_button.enabled = (
            idle and status.robot_placed and status.goal_selected
        )
        self._success_button.enabled = recording
        self._discard_button.enabled = recording
        self._reset_start_button.enabled = (
            status.can_reset_start and (idle or recording)
        )
        if placing_born:
            self._step_label.text = "Step 1: Drag robot, press BORN again"
        elif not status.robot_placed:
            self._step_label.text = "Step 1: Press BORN"
        elif recording:
            self._step_label.text = "Step 4: Drive to target, press SUCCESS"
        elif not status.goal_selected:
            self._step_label.text = "Step 2: Click free space on Goal Map"
        else:
            self._step_label.text = "Step 3: Stop robot, press START DEMO"

    def update_bev(
        self,
        bev: np.ndarray,
        *,
        config: BevConfig,
        start_body_xy: np.ndarray | None,
        goal_body_xy: np.ndarray | None,
        trajectory_body_xy: np.ndarray | None,
    ) -> None:
        self._latest_bev = np.asarray(bev, dtype=np.uint8).copy()
        self._latest_bev_config = config
        image = render_bev_preview(
            bev,
            config=config,
            start_body_xy=start_body_xy,
            goal_body_xy=goal_body_xy,
            trajectory_body_xy=trajectory_body_xy,
            image_size_px=self._size,
        )
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self._provider.set_data_array(
            np.ascontiguousarray(rgba).reshape(-1),
            [self._size, self._size],
        )

    def shutdown(self) -> None:
        self.control_window.visible = False
        self.bev_window.visible = False

    def _on_map_pressed(
        self,
        x: float,
        y: float,
        button: int,
        _modifiers,
    ) -> None:
        if (
            button != 0
            or self._latest_bev is None
            or self._latest_bev_config is None
        ):
            return
        config = self._latest_bev_config
        width = max(float(self._image.computed_width), 1.0)
        height = max(float(self._image.computed_height), 1.0)
        column = int(np.clip(
            (float(x) - float(self._image.screen_position_x))
            / width
            * config.width,
            0,
            config.width - 1,
        ))
        row = int(np.clip(
            (float(y) - float(self._image.screen_position_y))
            / height
            * config.height,
            0,
            config.height - 1,
        ))
        if not bool(self._latest_bev[1, row, column] > 0):
            return
        forward = config.forward_m - (float(row) + 0.5) * config.resolution_m
        left = config.left_m - (float(column) + 0.5) * config.resolution_m
        self._command_sink(("SET_GOAL_BODY", forward, left))


def render_bev_preview(
    bev: np.ndarray,
    *,
    config: BevConfig,
    start_body_xy: np.ndarray | None = None,
    goal_body_xy: np.ndarray | None = None,
    trajectory_body_xy: np.ndarray | None = None,
    image_size_px: int | None = None,
) -> Image.Image:
    data = np.asarray(bev, dtype=np.uint8)
    if data.shape != (3, config.height, config.width):
        raise ValueError(
            f"BEV shape {data.shape} does not match {(3, config.height, config.width)}"
        )

    occupied = data[0] > 0
    free = data[1] > 0
    rgb = np.full((config.height, config.width, 3), (108, 110, 112), dtype=np.uint8)
    rgb[free] = (238, 237, 232)
    rgb[occupied] = (28, 29, 29)
    image = Image.fromarray(rgb, mode="RGB")
    output_size = config.width if image_size_px is None else int(image_size_px)
    if output_size <= 0:
        raise ValueError("image_size_px must be positive")
    if image.size != (output_size, output_size):
        image = image.resize(
            (output_size, output_size),
            resample=_NEAREST_RESAMPLE,
        )
    draw = ImageDraw.Draw(image)

    if trajectory_body_xy is not None:
        pixels = [
            _display_bev_pixel(point, config, output_size)
            for point in np.asarray(trajectory_body_xy, dtype=np.float64).reshape(-1, 2)
        ]
        pixels = [point for point in pixels if point is not None]
        if len(pixels) >= 2:
            draw.line(pixels, fill=(0, 190, 205), width=2)

    _draw_marker(
        draw,
        start_body_xy,
        config,
        output_size,
        fill=(45, 185, 75),
        radius=4,
    )
    _draw_marker(
        draw,
        goal_body_xy,
        config,
        output_size,
        fill=(225, 65, 65),
        radius=4,
    )
    robot_pixel = _display_bev_pixel(
        np.asarray([0.0, 0.0]),
        config,
        output_size,
    )
    if robot_pixel is not None:
        x, y = robot_pixel
        draw.polygon(
            [(x, y - 8), (x - 4, y + 4), (x + 4, y + 4)],
            fill=(55, 92, 168),
        )
    return image


def _display_bev_pixel(
    body_xy,
    config: BevConfig,
    output_size: int,
) -> tuple[int, int] | None:
    pixel = body_xy_to_bev_pixel(np.asarray(body_xy, dtype=np.float64), config)
    if pixel is None:
        return None
    scale_x = float(output_size) / float(config.width)
    scale_y = float(output_size) / float(config.height)
    return (
        int(round((float(pixel[0]) + 0.5) * scale_x - 0.5)),
        int(round((float(pixel[1]) + 0.5) * scale_y - 0.5)),
    )


def _draw_marker(
    draw,
    body_xy,
    config: BevConfig,
    output_size: int,
    *,
    fill,
    radius: int,
) -> None:
    if body_xy is None:
        return
    pixel = _display_bev_pixel(body_xy, config, output_size)
    if pixel is None:
        return
    x, y = pixel
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
    )
