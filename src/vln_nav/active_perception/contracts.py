from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TimedDepthObservation:
    source: str
    timestamp_sec: float
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray

    def validate(self) -> None:
        depth = np.asarray(self.depth_m)
        if not self.source.strip() or not math.isfinite(float(self.timestamp_sec)):
            raise ValueError("Depth observation requires a source and finite timestamp")
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError(f"Depth observation must be HxW, got {depth.shape}")
        if np.asarray(self.intrinsics).shape != (3, 3):
            raise ValueError("Camera intrinsics must be 3x3")
        transform = np.asarray(self.camera_to_world)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_world must be a finite 4x4 transform")
