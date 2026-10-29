from __future__ import annotations

import numpy as np

from .recording import TrialRecorder
from .state import require_reset_accuracy


class Experiment:
    def __init__(self, backend, output_root, metadata, *, timeout_sec=120.0):
        self.backend, self.output_root, self.metadata = backend, output_root, metadata
        self.timeout_sec = float(timeout_sec)
        if not np.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ValueError("试验超时必须为有限正数")
        self.saved = None
        self.recorder = None
        self.last_record = None
        self.last_reset_errors = None
        self.status = "READY"

    def save_start(self):
        if self.recorder is not None:
            raise ValueError("保存起点需要先停止记录")
        self.saved = self.backend.capture_snapshot()
        return self.saved

    def start(self):
        if self.recorder is not None:
            raise ValueError("试验已经开始记录")
        if self.backend.goal is None:
            raise ValueError("请在地图上选择目标位置")
        self.saved = self.backend.capture_snapshot()
        metadata = dict(self.metadata, goal_world_xyz=self.backend.goal.tolist(),
                        joint_names=list(self.saved.robot.joint_names), reset_errors=self.last_reset_errors)
        self.recorder = TrialRecorder(self.output_root, metadata, self.saved)
        self.status = "RECORDING"
        self.last_record = self.recorder.path
        self.recorder.observations(self.saved.observations)
        return self.last_record

    def stop(self, reason="STOPPED", *, details=None):
        self.backend.stop_motion()
        if self.recorder is not None:
            self.recorder.finish(reason, self.backend.timestamp, details=details)
            self.recorder = None
        self.status = reason

    def sample(self, robot, requested, applied, observations, *, physical_time_sec):
        if self.recorder is None:
            return
        self.recorder.state(robot, requested, applied, physical_time_sec=physical_time_sec)
        self.recorder.observations(observations)
        if robot.timestamp_sec - self.recorder.metadata["start_timestamp_sec"] >= self.timeout_sec:
            self.stop("TIMEOUT")

    def reset(self):
        if self.saved is None:
            raise ValueError("尚未保存试验起点")
        self.stop("RESET_BY_USER")
        errors = self.backend.restore_snapshot(self.saved)
        self.last_reset_errors = errors
        try:
            require_reset_accuracy(errors)
        except RuntimeError:
            self.status = "RESET_FAILED"
            raise
        self.status = "READY"
        return errors

    def close(self, reason="USER_EXIT", *, details=None):
        self.stop(reason, details=details)
