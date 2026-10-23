from __future__ import annotations

from pathlib import Path

import numpy as np

from constraint_traverse.collect.bev import BevConfig
from vln_nav.active_perception.bev_goal_map import render_bev_goal_map


class EvaluationMapPanel:
    def __init__(self, ui, *, display_config: dict):
        self.view_size_m = float(display_config["view_size_m"])
        self.rear_view_m = float(display_config["rear_view_m"])
        self.image_size_px = int(display_config["image_size_px"])
        self._goal_map = None
        self._provider = ui.ByteImageProvider()
        initial = np.empty((self.image_size_px, self.image_size_px, 4), dtype=np.uint8)
        initial[:, :, :3] = (108, 110, 112)
        initial[:, :, 3] = 255
        self._provider.set_data_array(initial.reshape(-1), [self.image_size_px, self.image_size_px])
        self.window = ui.Window("Evaluation Map", width=self.image_size_px + 28,
                                height=self.image_size_px + 120, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=5):
                self._episode_label = ui.Label("Episode: -", height=24)
                self._time_label = ui.Label("Time: 0.0 s | Goal: -", height=24)
                self._map_label = ui.Label("Map age: -", height=24)
                self._image = ui.ImageWithProvider(self._provider, width=ui.Fraction(1),
                    height=ui.Fraction(1), fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    pixel_aligned=True)

    def update(self, *, snapshot, bev_config: BevConfig, robot_world_xyz, robot_yaw_rad: float,
               floor_z_m: float, goal_body_xy, episode: Path, elapsed_sec: float,
               timeout_sec: float, map_age_sec: float):
        self._goal_map = render_bev_goal_map(snapshot, robot_world_xyz=robot_world_xyz,
            robot_yaw_rad=robot_yaw_rad, view_size_m=self.view_size_m, rear_view_m=self.rear_view_m,
            image_size_px=self.image_size_px, goal_local_xy=goal_body_xy,
            obstacle_min_relative_z_m=bev_config.obstacle_min_height_m,
            obstacle_max_relative_z_m=bev_config.obstacle_max_height_m,
            projection_reference_z_m=floor_z_m)
        self._provider.set_data_array(self._goal_map.rgba.reshape(-1), [self.image_size_px, self.image_size_px])
        self._episode_label.text = f"Episode: {episode.parent.name}/{episode.name}"
        self._time_label.text = (f"Time: {elapsed_sec:.1f}/{timeout_sec:.0f} s"
                                 f" | Goal: {np.linalg.norm(goal_body_xy):.2f} m")
        forward, left = map(float, goal_body_xy)
        visible = (-self.rear_view_m < forward <= self.view_size_m - self.rear_view_m
                   and -self.view_size_m / 2 < left <= self.view_size_m / 2)
        self._map_label.text = (f"Map age: {map_age_sec:.2f} s"
                                f" | View: {self.view_size_m:g} x {self.view_size_m:g} m"
                                + ("" if visible else " | Goal outside view"))

    def shutdown(self):
        self.window.destroy()

