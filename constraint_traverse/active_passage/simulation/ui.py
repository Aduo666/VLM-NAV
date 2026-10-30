from __future__ import annotations

import time

import numpy as np

from ..cameras import CAMERA_PATHS, DEFAULT_MAP_SOURCES


class ExperimentPanel:
    def __init__(self, ui, sink, image_size, map_sources=DEFAULT_MAP_SOURCES):
        self.sink, self.size = sink, int(image_size)
        self.goal_map = self.map_pose = None
        self._map_dirty = True
        self._last_map_refresh = -float("inf")
        self._sync_sources = False
        self.provider = ui.ByteImageProvider()
        self.controls = ui.Window("Active Passage Controls", width=350, height=300)
        self.controls.position_x, self.controls.position_y = 10, 60
        with self.controls.frame:
            with ui.VStack(spacing=6):
                self.source_models = {}
                with ui.HStack(height=25):
                    ui.Label("Map fusion", width=70)
                    for name in CAMERA_PATHS:
                        model = ui.SimpleBoolModel(name in map_sources)
                        ui.CheckBox(model, width=22)
                        ui.Label(name, width=48)
                        model.add_value_changed_fn(self._sources_changed)
                        self.source_models[name] = model
                self.place = ui.Button("PLACE ROBOT", height=30, clicked_fn=lambda: sink("PLACE"))
                self.save = ui.Button("SAVE START", height=30, clicked_fn=lambda: sink("SAVE_START"))
                self.navigate = ui.Button("START NAVIGATION", height=32, clicked_fn=lambda: sink("NAVIGATE"))
                with ui.HStack(height=32):
                    self.start = ui.Button("START RECORDING", clicked_fn=lambda: sink("START"))
                    ui.Button("STOP", clicked_fn=lambda: sink("STOP"))
                self.reset = ui.Button("RESET TRIAL", height=30, clicked_fn=lambda: sink("RESET_TRIAL"))
                self.reset_map = ui.Button("RESET MAP", height=30, clicked_fn=lambda: sink("RESET_MAP"))
        self.map_window = ui.Window("Active Passage Map", width=self.size + 20, height=self.size + 45)
        self.map_window.position_x, self.map_window.position_y = 380, 60
        with self.map_window.frame:
            self.image = ui.ImageWithProvider(self.provider, width=ui.Fraction(1), height=ui.Fraction(1),
                                              fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                                              opaque_for_mouse_events=True)
            self.image.set_mouse_pressed_fn(self._click)

    def request_map_refresh(self):
        self._map_dirty = True

    def _sources_changed(self, model):
        if not self._sync_sources:
            self.sink(("MAP_SOURCES", tuple(name for name, value in self.source_models.items() if value.as_bool)))

    def _click(self, x, y, button, modifiers):
        if button != 0 or self.goal_map is None:
            return
        column = int((x - self.image.screen_position_x) / max(self.image.computed_width, 1) * self.size)
        row = int((y - self.image.screen_position_y) / max(self.image.computed_height, 1) * self.size)
        if not (0 <= column < self.size and 0 <= row < self.size):
            return
        forward, left = self.goal_map.pixel_to_local(column, row)
        position, yaw, floor_z = self.map_pose
        cosine, sine = np.cos(yaw), np.sin(yaw)
        self.sink(("GOAL", np.asarray([position[0] + cosine * forward - sine * left,
                                      position[1] + sine * forward + cosine * left, floor_z])))

    def update(self, runtime):
        recording = runtime.experiment.recorder is not None
        placing = runtime.placing
        self.place.text = "CONFIRM PLACEMENT" if placing else "PLACE ROBOT"
        self.place.enabled = self.save.enabled = not recording
        self.start.enabled = not recording and not placing and runtime.goal is not None
        self.navigate.enabled = self.start.enabled
        self.reset.enabled = not placing and runtime.experiment.saved is not None
        self.reset_map.enabled = not recording and not placing
        self._sync_sources = True
        try:
            for name, model in self.source_models.items():
                if model.as_bool != (name in runtime.fusion.sources):
                    model.set_value(name in runtime.fusion.sources)
        finally:
            self._sync_sources = False
        now = time.monotonic()
        period = float(runtime.navigation["map_selector"]["render_period_sec"])
        if self._map_dirty or now - self._last_map_refresh >= period:
            self.goal_map, self.map_pose = runtime.render_map()
            rgba = self.goal_map.rgba
            self.provider.set_data_array(np.ascontiguousarray(rgba).reshape(-1), [rgba.shape[1], rgba.shape[0]])
            self._map_dirty = False
            self._last_map_refresh = now

    def close(self):
        self.controls.visible = self.map_window.visible = False
