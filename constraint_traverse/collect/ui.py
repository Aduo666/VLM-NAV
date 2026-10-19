from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vln_nav.active_perception.bev_goal_map import BevGoalMap


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
        self._goal_map: BevGoalMap | None = None
        self._map_pose: np.ndarray | None = None
        self._provider = ui.ByteImageProvider()
        initial = np.empty((self._size, self._size, 4), dtype=np.uint8)
        initial[:, :, :3] = (108, 110, 112)
        initial[:, :, 3] = 255
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
                    "Curve: hold ↑+←/→ or W+A/D    Spin: ←/→ or A/D",
                    height=30,
                    word_wrap=True,
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
        self._start_button.enabled = idle and status.robot_placed
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
            self._step_label.text = "Drive through, stop, press SUCCESS"
        else:
            self._step_label.text = "Stop robot, press START DEMO; map hint is optional"

    def update_map(self, goal_map: BevGoalMap, *, robot_world_xyyaw: np.ndarray) -> None:
        self._goal_map = goal_map
        self._map_pose = np.asarray(robot_world_xyyaw, dtype=np.float64).reshape(3).copy()
        rgba = np.asarray(goal_map.rgba, dtype=np.uint8)
        height, width = rgba.shape[:2]
        self._provider.set_data_array(
            np.ascontiguousarray(rgba).reshape(-1),
            [width, height],
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
        if button != 0 or self._goal_map is None or self._map_pose is None:
            return
        width = max(float(self._image.computed_width), 1.0)
        height = max(float(self._image.computed_height), 1.0)
        column = int(np.clip(
            (float(x) - float(self._image.screen_position_x))
            / width
            * self._goal_map.selectable_mask.shape[1],
            0,
            self._goal_map.selectable_mask.shape[1] - 1,
        ))
        row = int(np.clip(
            (float(y) - float(self._image.screen_position_y))
            / height
            * self._goal_map.selectable_mask.shape[0],
            0,
            self._goal_map.selectable_mask.shape[0] - 1,
        ))
        if not bool(self._goal_map.selectable_mask[row, column]):
            return
        forward, left = self._goal_map.pixel_to_local(column, row)
        cosine, sine = np.cos(self._map_pose[2]), np.sin(self._map_pose[2])
        self._command_sink((
            "SET_HINT_WORLD",
            float(self._map_pose[0] + cosine * forward - sine * left),
            float(self._map_pose[1] + sine * forward + cosine * left),
        ))
