from __future__ import annotations

from copy import deepcopy

import numpy as np

from constraint_traverse.collect.sampling import (
    PhysicsHistory,
    PhysicsState,
    capture_aligned_frame,
    pair_aligned_frames,
)
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap


class InvalidDepthFrame(RuntimeError):
    def __init__(self, frames, valid_ratios):
        self.frames = frames
        self.valid_ratios = np.asarray(valid_ratios).copy()
        super().__init__(f"相机有效深度比例不足：{self.valid_ratios.tolist()}")


class MapSampler:
    def __init__(self, local_map: LocalVoxelMap, max_skew_sec: float,
                 min_valid_ratio: float, integration_period_sec: float):
        if not np.isfinite(max_skew_sec) or max_skew_sec < 0:
            raise ValueError("相机时间差上限必须为有限非负数")
        if not np.isfinite(min_valid_ratio) or not 0 <= min_valid_ratio <= 1:
            raise ValueError("有效深度比例下限必须位于 0 到 1 之间")
        if not np.isfinite(integration_period_sec) or integration_period_sec <= 0:
            raise ValueError("地图融合周期必须为有限正数")
        self.local_map = local_map
        self.max_skew_sec = float(max_skew_sec)
        self.min_valid_ratio = float(min_valid_ratio)
        self.integration_period_sec = float(integration_period_sec)
        self.history = PhysicsHistory()
        timestamp = local_map.snapshot().timestamp_sec
        self.reset(start_time_sec=max(0.0, timestamp), map_timestamp_sec=timestamp)

    def reset(self, *, start_time_sec: float, map_timestamp_sec: float):
        if not np.isfinite(start_time_sec) or np.isnan(map_timestamp_sec) or np.isposinf(map_timestamp_sec):
            raise ValueError("回合开始时间或初始地图时间无效")
        if map_timestamp_sec > start_time_sec + 1e-6:
            raise ValueError("初始地图时间晚于回合开始时间")
        self.history.clear()
        self.start_time_sec = float(start_time_sec)
        self.last_integration_sec = float(map_timestamp_sec)
        self.last_integration_available_sec = -np.inf
        self.last_poll_sec = -np.inf
        self.last_capture_sec = {source: -np.inf for source in ("front", "down")}
        self.last_available_sec = {source: -np.inf for source in ("front", "down")}
        self._accepted_capture_sec = dict(self.last_capture_sec)
        self._accepted_available_sec = dict(self.last_available_sec)
        self.valid_ratios = np.zeros(2, dtype=np.float64)
        self.fusion_count = 0
        self._pending = {source: None for source in ("front", "down")}

    def append_state(self, state: PhysicsState):
        # 保存独立数组，后续物理状态更新不会改写历史位姿。
        self.history.append(deepcopy(state))

    def poll(self, front_camera, down_camera, *, now: float) -> bool:
        if not np.isfinite(now) or now < self.last_poll_sec - 1e-6:
            raise ValueError("相机可用时间无效或发生倒退，请在新回合调用 reset")
        self.last_poll_sec = float(now)
        for source, camera in (("front", front_camera), ("down", down_camera)):
            captured = capture_aligned_frame(source=source, camera=camera, history=self.history,
                                             after_sec=self.last_capture_sec[source])
            if captured is None:
                continue
            frame, _ = captured
            if frame.timestamp_sec < self.start_time_sec - 1e-6 or frame.timestamp_sec > now + 1e-6:
                continue
            self._pending[source] = captured
            self.last_capture_sec[source] = float(frame.timestamp_sec)
            self.last_available_sec[source] = float(now)
        pair = pair_aligned_frames(self._pending["front"], self._pending["down"],
            max_skew_sec=self.max_skew_sec, not_before_sec=self.start_time_sec)
        if pair is None:
            return False
        self._pending = {source: None for source in ("front", "down")}
        frames = (pair.front, pair.down)
        self.valid_ratios[:] = [np.mean(np.isfinite(frame.depth_m) & (frame.depth_m > 0)) for frame in frames]
        if np.any(self.valid_ratios < self.min_valid_ratio):
            # 无效图像不提交时间戳，同一物理时刻重新渲染后仍可读取。
            self.last_capture_sec = dict(self._accepted_capture_sec)
            self.last_available_sec = dict(self._accepted_available_sec)
            raise InvalidDepthFrame(frames, self.valid_ratios)
        self._accepted_capture_sec = dict(self.last_capture_sec)
        self._accepted_available_sec = dict(self.last_available_sec)
        capture_time = max(frame.timestamp_sec for frame in frames)
        if capture_time - self.last_integration_sec < self.integration_period_sec - 1e-9:
            return False
        observations = [TimedDepthObservation(source=frame.source, timestamp_sec=frame.timestamp_sec,
            depth_m=frame.depth_m, intrinsics=frame.intrinsics,
            camera_to_world=frame.world_from_camera) for frame in frames]
        self.local_map.integrate_batch(observations, center_world_xyz=pair.state.position)
        self.last_integration_sec = float(capture_time)
        self.last_integration_available_sec = max(self.last_available_sec.values())
        self.fusion_count += 1
        return True
