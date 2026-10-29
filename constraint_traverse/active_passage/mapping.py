from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np

from constraint_traverse.rgbd_map import SOURCE_BITS
from vln_nav.active_perception.contracts import TimedDepthObservation

from .cameras import CAMERA_PATHS, DEFAULT_MAP_SOURCES, normalize_map_sources
from .contracts import RealObservation


class ObservationMapFusion:
    def __init__(self, local_map, sources=DEFAULT_MAP_SOURCES):
        self.local_map = local_map
        self.sources = normalize_map_sources(sources)
        self._last = {}
        self._enabled_after = {}

    def set_sources(self, sources, *, timestamp_sec):
        selected = normalize_map_sources(sources)
        if not math.isfinite(timestamp_sec):
            raise ValueError("融合开关时间必须为有限数值")
        if selected == self.sources:
            return False
        for source in set(selected) - set(self.sources):
            self._enabled_after[source] = float(timestamp_sec)
        self.sources = selected
        return True

    def reset_history(self):
        self._last.clear()
        self._enabled_after.clear()

    def snapshot(self):
        return deepcopy(dict(sources=self.sources, last=self._last, enabled_after=self._enabled_after))

    def restore(self, state):
        selected = normalize_map_sources(state["sources"])
        self.sources = selected
        self._last = deepcopy(state["last"])
        self._enabled_after = deepcopy(state["enabled_after"])

    def source_cell_counts(self):
        return {source: int(np.count_nonzero(self.local_map.source_mask & SOURCE_BITS[source]))
                for source in CAMERA_PATHS}

    def integrate(self, observations, *, floor_offset_m):
        observations = tuple(observations)
        if not all(isinstance(value, RealObservation) for value in observations):
            raise TypeError("地图融合只接受真实观测 RealObservation")
        if not math.isfinite(floor_offset_m):
            raise ValueError("地面高度偏移必须为有限数值")
        if any(value.frame.frame_id not in CAMERA_PATHS for value in observations):
            raise ValueError("真实观测的相机来源无效")
        selected = []
        history = deepcopy(self._last)
        for observation in sorted(observations, key=lambda value: value.frame.timestamp_sec):
            frame = observation.frame
            source, timestamp = frame.frame_id, frame.timestamp_sec
            if source not in self.sources or timestamp < self._enabled_after.get(source, -math.inf) - 1e-6:
                continue
            previous_time, previous_ids = history.get(source, (-math.inf, set()))
            if timestamp < previous_time - 1e-6:
                continue
            if abs(timestamp - previous_time) <= 1e-6:
                if observation.observation_id in previous_ids:
                    continue
                ids = previous_ids | {observation.observation_id}
            else:
                ids = {observation.observation_id}
            history[source] = (float(timestamp), ids)
            selected.append(observation)
        if not selected:
            return None
        latest = selected[-1]
        position = latest.robot.world_from_base[:3, 3]
        depths = [TimedDepthObservation(source=value.frame.frame_id, timestamp_sec=value.frame.timestamp_sec,
                    depth_m=value.frame.depth_m, intrinsics=value.frame.intrinsics,
                    camera_to_world=value.frame.world_from_camera) for value in selected]
        update = self.local_map.integrate_batch(depths, center_world_xy=position[:2],
                                               floor_z_m=float(position[2]) - floor_offset_m)
        self._last = history
        return dict(**asdict(update), timestamp_sec=float(latest.frame.timestamp_sec),
                    enabled_sources=list(self.sources),
                    observations=[dict(observation_id=value.observation_id, source=value.frame.frame_id,
                                       timestamp_sec=float(value.frame.timestamp_sec)) for value in selected])
