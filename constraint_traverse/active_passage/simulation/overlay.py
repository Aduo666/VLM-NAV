from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw


def draw_navigation(image, navigator, position, yaw):
    canvas = Image.fromarray(image.rgba)
    draw = ImageDraw.Draw(canvas)
    size = image.rgba.shape[0]
    cosine, sine = np.cos(yaw), np.sin(yaw)

    def line(points, color, width):
        points = np.asarray(points)
        if len(points) < 2:
            return
        delta = points[:, :2] - position[:2]
        forward = cosine * delta[:, 0] + sine * delta[:, 1]
        left = -sine * delta[:, 0] + cosine * delta[:, 1]
        pixels = np.column_stack(((0.5 - left / image.view_size_m) * size,
                                  (1 - (forward + image.rear_view_m) / image.view_size_m) * size))
        draw.line([tuple(point) for point in pixels], fill=color, width=width)

    if navigator.path is not None:
        line(navigator.path.world_xy, (175, 100, 235, 255), 2)
    if navigator.sampling.last_plan is not None:
        line(navigator.sampling.last_plan.prediction[:, :2], (30, 200, 70, 255), 3)
    line(navigator.actual, (45, 140, 250, 255), 2)
    return replace(image, rgba=np.asarray(canvas))
