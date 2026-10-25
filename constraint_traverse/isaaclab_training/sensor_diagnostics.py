from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np


def finite_values(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, None).tolist()


def save_sensor_diagnostic(directory, *, metadata, frames, arrays):
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = "sensor_failure_" + uuid4().hex
    metadata = dict(metadata, cameras={})
    arrays = dict(arrays)
    for source, frame in frames.items():
        depth = np.asarray(frame["distance_to_image_plane"])
        finite = depth[np.isfinite(depth)]
        size = max(1, depth.size)
        metadata["cameras"][source] = dict(depth_shape=list(depth.shape),
            timestamp_sec=float(frame["rendering_time"]),
            valid_ratio=float(np.count_nonzero(np.isfinite(depth) & (depth > 0)) / size),
            zero_ratio=float(np.count_nonzero(depth == 0) / size),
            nan_ratio=float(np.count_nonzero(np.isnan(depth)) / size),
            inf_ratio=float(np.count_nonzero(np.isinf(depth)) / size),
            finite_min_m=float(finite.min()) if finite.size else None,
            finite_max_m=float(finite.max()) if finite.size else None)
        arrays[source + "_raw_depth_m"] = depth
        if isinstance(frame.get("rgb"), np.ndarray):
            arrays[source + "_rgb"] = frame["rgb"]
    metadata["arrays_file"] = name + ".npz"
    np.savez_compressed(directory / metadata["arrays_file"], **arrays)
    path = directory / (name + ".json")
    path.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return str(path)
