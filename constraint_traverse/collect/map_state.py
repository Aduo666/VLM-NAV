from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from vln_nav.active_perception.local_geometry import LocalVoxelSnapshot


def snapshot_arrays(snapshot: LocalVoxelSnapshot) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(getattr(snapshot, item.name)).copy()
        for item in fields(LocalVoxelSnapshot)
    }


def load_snapshot(path: Path) -> LocalVoxelSnapshot:
    with np.load(path, allow_pickle=False) as data:
        values = {item.name: data[item.name].copy() for item in fields(LocalVoxelSnapshot)}
    for name in ("resolution_m", "timestamp_sec"):
        values[name] = float(values[name])
    values["version"] = int(values["version"])
    return LocalVoxelSnapshot(**values)


def verify_episode_bevs(path: Path) -> list[str]:
    from .bev import BevConfig, local_voxel_config_from_mapping, snapshot_to_robot_bev
    from .recorder import decode_depth_png
    from vln_nav.active_perception.contracts import TimedDepthObservation
    from vln_nav.active_perception.local_geometry import LocalVoxelMap

    errors = []
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        local_map = LocalVoxelMap(local_voxel_config_from_mapping(meta["voxel_map"]))
        local_map.restore(load_snapshot(path / "initial_map.npz"))
        config = BevConfig.from_mapping(meta["bev"])
        with np.load(path / "state.npz", allow_pickle=False) as state:
            for index in range(len(state["sample_time_sec"])):
                if bool(state["map_integrated"][index]):
                    observations = [
                        TimedDepthObservation(
                            source=source,
                            timestamp_sec=float(state[f"{source}_timestamp_sec"][index]),
                            depth_m=decode_depth_png(path / f"{source}_depth" / f"{index:06d}.png"),
                            intrinsics=state[f"{source}_intrinsics"],
                            camera_to_world=state[f"{source}_world_from_camera"][index],
                        )
                        for source in ("front", "down")
                    ]
                    local_map.integrate_batch(observations, center_world_xyz=state["world_position_xyz"][index])
                snapshot = local_map.snapshot()
                rebuilt = snapshot_to_robot_bev(
                    snapshot, robot_world_xyyaw=state["pose_world_xyyaw"][index],
                    floor_z_m=float(state["floor_z_m"][index]), config=config,
                )
                saved = np.load(path / "bev" / f"{index:06d}.npy", allow_pickle=False)
                if snapshot.version != int(state["map_version"][index]) or not np.array_equal(rebuilt, saved):
                    errors.append(f"BEV {index:06d} differs from reconstructed map")
                if len(errors) >= 20:
                    break
    except Exception as exc:
        errors.append(f"BEV reconstruction failed: {exc}")
    return errors
