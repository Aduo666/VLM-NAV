from __future__ import annotations

import numpy as np
from pathlib import Path

from ..cameras import CAMERA_PATHS, DEFAULT_MAP_SOURCES


def chinese_text_style():
    for filename in ("/usr/share/fonts/MyFonts/YaHei Consolas Hybrid 1.12 For Powerline.ttf",
                     "/usr/share/fonts/MyFonts/simkai.ttf"):
        if Path(filename).is_file():
            return {"font": filename, "font_size": 14}
    raise FileNotFoundError("界面需要支持中文的 TTF 字体，请安装微软雅黑或楷体字体")


class ExperimentPanel:
    def __init__(self, ui, sink, image_size, map_sources=DEFAULT_MAP_SOURCES):
        self.sink, self.size = sink, int(image_size)
        self.goal_map = self.map_pose = None
        self._sync_sources = False
        text_style = chinese_text_style()
        self.provider = ui.ByteImageProvider()
        self.controls = ui.Window("Active Passage Controls", width=350, height=410)
        self.controls.position_x, self.controls.position_y = 10, 60
        with self.controls.frame:
            with ui.VStack(spacing=6):
                self.status = ui.Label("READY", height=28, style=text_style)
                self.detail = ui.Label("", height=42, word_wrap=True, style=text_style)
                self.goal_label = ui.Label("Goal: --", height=26, style=text_style)
                self.source_models = {}
                with ui.HStack(height=25):
                    ui.Label("地图融合", width=70, style=text_style)
                    for name in CAMERA_PATHS:
                        model = ui.SimpleBoolModel(name in map_sources)
                        ui.CheckBox(model, width=22)
                        ui.Label(name, width=48, style=text_style)
                        model.add_value_changed_fn(self._sources_changed)
                        self.source_models[name] = model
                self.map_stats = ui.Label("", height=28, style=text_style)
                self.place = ui.Button("PLACE ROBOT", height=30, clicked_fn=lambda: sink("PLACE"))
                self.save = ui.Button("SAVE START", height=30, clicked_fn=lambda: sink("SAVE_START"))
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
        self.providers = {}
        self.camera_window = ui.Window("Active Passage RGB", width=660, height=210)
        self.camera_window.position_x, self.camera_window.position_y = 10, 550
        with self.camera_window.frame:
            with ui.HStack(spacing=4):
                for name in CAMERA_PATHS:
                    with ui.VStack():
                        ui.Label(name, height=20)
                        provider = ui.ByteImageProvider()
                        ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                        self.providers[name] = provider

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
        self.status.text = "PLACING" if placing else runtime.experiment.status
        self.detail.text = runtime.message
        self.goal_label.text = "Goal: --" if runtime.goal is None else "Goal: " + ", ".join(f"{x:.2f}" for x in runtime.goal)
        self.place.text = "CONFIRM PLACEMENT" if placing else "PLACE ROBOT"
        self.place.enabled = self.save.enabled = not recording
        self.start.enabled = not recording and not placing and runtime.goal is not None
        self.reset.enabled = not placing and runtime.experiment.saved is not None
        self.reset_map.enabled = not recording and not placing
        self._sync_sources = True
        try:
            for name, model in self.source_models.items():
                if model.as_bool != (name in runtime.fusion.sources):
                    model.set_value(name in runtime.fusion.sources)
        finally:
            self._sync_sources = False
        counts = runtime.fusion.source_cell_counts()
        self.map_stats.text = f"地图 v{runtime.local_map.version} " + " ".join(f"{name}:{value}" for name, value in counts.items())
        self.goal_map, self.map_pose = runtime.render_map()
        rgba = self.goal_map.rgba
        self.provider.set_data_array(np.ascontiguousarray(rgba).reshape(-1), [rgba.shape[1], rgba.shape[0]])
        for observation in runtime.observations:
            frame = observation.frame
            rgba = np.concatenate((frame.rgb, np.full((*frame.rgb.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
            self.providers[frame.frame_id].set_data_array(rgba.reshape(-1), [rgba.shape[1], rgba.shape[0]])

    def close(self):
        self.controls.visible = self.map_window.visible = self.camera_window.visible = False
