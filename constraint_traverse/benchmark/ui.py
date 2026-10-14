from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


class MapRenderer:
    def __init__(self, geometry, image_size_px=520):
        self.geometry, self.size = geometry, int(image_size_px)
        self.half = geometry.scene.half_size_m

    def world_to_pixel(self, points):
        points = np.asarray(points).reshape(-1, 2)
        return np.column_stack(((points[:, 0] / (2 * self.half) + 0.5) * self.size,
                                (0.5 - points[:, 1] / (2 * self.half)) * self.size))

    def pixel_to_world(self, column, row):
        return np.asarray([(column / self.size - 0.5) * 2 * self.half,
                           (0.5 - row / self.size) * 2 * self.half])

    def render(self, pose, goal=None, prediction=None, candidates=None, actual=None, reference=None):
        image = Image.new("RGBA", (self.size, self.size), (240, 241, 243, 255))
        draw = ImageDraw.Draw(image)
        for value in np.arange(-self.half, self.half + 0.01, 0.5):
            px = self.world_to_pixel([[value, -self.half], [value, self.half]])
            py = self.world_to_pixel([[-self.half, value], [self.half, value]])
            draw.line([tuple(v) for v in px], fill=(215, 218, 223), width=1)
            draw.line([tuple(v) for v in py], fill=(215, 218, 223), width=1)
        for bounds in self.geometry.scene.obstacles:
            coordinates = self.world_to_pixel([[bounds[0], bounds[2]], [bounds[1], bounds[3]]])
            draw.rectangle([tuple(coordinates.min(axis=0)), tuple(coordinates.max(axis=0))], fill=(47, 52, 60))
        for points, color, width in ((reference, (139, 121, 182), 2), (actual, (35, 123, 208), 2),
                                     (prediction, (15, 159, 115), 3)):
            if points is not None and len(points) >= 2:
                draw.line([tuple(v) for v in self.world_to_pixel(np.asarray(points)[:, :2])], fill=color, width=width)
        if candidates is not None:
            for candidate in candidates:
                draw.line([tuple(v) for v in self.world_to_pixel(candidate[:, :2])], fill=(180, 201, 189), width=1)
            if prediction is not None and len(prediction) > 1:
                draw.line([tuple(v) for v in self.world_to_pixel(prediction[:, :2])], fill=(15, 159, 115), width=3)
        shape = self.geometry.body_at(pose)
        polygons = [shape] if shape.geom_type == "Polygon" else shape.geoms
        for polygon in polygons:
            draw.polygon([tuple(v) for v in self.world_to_pixel(polygon.exterior.coords)], outline=(38, 82, 134))
        start = self.world_to_pixel([pose[:2]])[0]
        tip = self.world_to_pixel([[pose[0] + 0.18 * math.cos(pose[2]), pose[1] + 0.18 * math.sin(pose[2])]])[0]
        draw.line([tuple(start), tuple(tip)], fill=(38, 82, 134), width=3)
        if goal is not None:
            x, y = self.world_to_pixel([goal])[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(218, 64, 73))
        return np.asarray(image, dtype=np.uint8)


class BenchmarkPanel:
    def __init__(self, ui, renderer, command_sink, method="MPPI", navigation_condition="goal"):
        self.ui, self.renderer, self.sink = ui, renderer, command_sink
        self.provider = ui.ByteImageProvider()
        self.controls = ui.Window("Passage Controls", width=330, height=290)
        with self.controls.frame:
            with ui.VStack(spacing=6):
                self.status = ui.Label("State: READY", height=24)
                self.result = ui.Label("Result: -", height=24)
                self.planner = ui.ComboBox(0 if method == "MPPI" else 1, "MPPI", "KMPPI", height=28)
                self.condition = ui.ComboBox(0 if navigation_condition == "goal" else 1, "Goal", "Reference Path", height=28)
                self.born = ui.Button("BORN", height=30, clicked_fn=lambda: command_sink("BORN"))
                with ui.HStack(height=30):
                    self.start = ui.Button("START", clicked_fn=self._start)
                    ui.Button("STOP", clicked_fn=lambda: command_sink("STOP"))
                ui.Button("RESET TRIAL", height=30, clicked_fn=lambda: command_sink("RESET_TRIAL"))
                ui.Button("RESET MAP", height=30, clicked_fn=lambda: command_sink("RESET_MAP"))
        self.map_window = ui.Window("Passage Map", width=renderer.size + 20, height=renderer.size + 40)
        with self.map_window.frame:
            self.image = ui.ImageWithProvider(self.provider, width=ui.Fraction(1), height=ui.Fraction(1),
                                              fill_policy=ui.IwpFillPolicy.IWP_STRETCH, opaque_for_mouse_events=True)
            self.image.set_mouse_pressed_fn(self._click)

    def _start(self):
        method_index = self.planner.model.get_item_value_model().as_int
        condition_index = self.condition.model.get_item_value_model().as_int
        self.sink(("START", ("MPPI", "KMPPI")[method_index], ("goal", "reference")[condition_index]))

    def _click(self, x, y, button, _modifiers):
        if button != 0:
            return
        column = (x - self.image.screen_position_x) / max(self.image.computed_width, 1) * self.renderer.size
        row = (y - self.image.screen_position_y) / max(self.image.computed_height, 1) * self.renderer.size
        if 0 <= column < self.renderer.size and 0 <= row < self.renderer.size:
            self.sink(("GOAL", self.renderer.pixel_to_world(column, row)))

    def update(self, rgba, mode, result):
        self.status.text = f"State: {mode}"
        self.result.text = f"Result: {result}"
        self.start.enabled = mode == "READY"
        self.born.enabled = mode in {"READY", "PLACING"}
        self.planner.enabled = self.condition.enabled = mode == "READY"
        self.provider.set_data_array(np.ascontiguousarray(rgba).reshape(-1), [rgba.shape[1], rgba.shape[0]])

    def close(self):
        self.controls.visible = self.map_window.visible = False
