from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pickle
import uuid

import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


class TrialRecorder:
    def __init__(self, output_root, metadata, snapshot):
        self.path = Path(output_root) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir()
        self.metadata = dict(metadata, schema_version=1, execution_mode="manual",
                             automatic_planner=False, start_timestamp_sec=snapshot.robot.timestamp_sec)
        self.counts = dict(states=0, observations=0, events=0)
        self.last_time = float(snapshot.robot.timestamp_sec)
        self.closed = False
        self.stream = (self.path / "records.jsonl").open("w", encoding="utf-8")
        # 快照仅保存本程序产生的数据，恢复入口只使用当前会话持有的快照。
        with (self.path / "initial_state.pkl").open("wb") as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(self.path / "meta.json", {**self.metadata, "status": "RECORDING", **self.counts})

    def _append(self, value):
        if self.closed:
            raise RuntimeError("试验记录已经结束")
        self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def state(self, robot, requested, applied, *, physical_time_sec):
        timestamp = float(robot.timestamp_sec)
        if timestamp < self.last_time - 1e-6:
            raise ValueError("同一条试验记录的时间不能倒退")
        self._append(dict(kind="state", timestamp_sec=timestamp, physical_time_sec=float(physical_time_sec),
                          world_from_base=robot.world_from_base.tolist(), base_twist=robot.base_twist.tolist(),
                          joint_names=list(robot.joint_names), joint_positions=robot.joint_positions.tolist(),
                          joint_velocities=robot.joint_velocities.tolist(),
                          requested_command=np.asarray(requested).tolist(), applied_command=np.asarray(applied).tolist()))
        self.last_time = timestamp
        self.counts["states"] += 1

    def observations(self, observations):
        for observation in observations:
            frame, robot = observation.frame, observation.robot
            index = self.counts["observations"]
            relative = f"frames/{index:06d}.npz"
            np.savez_compressed(self.path / relative, rgb=frame.rgb, depth_m=frame.depth_m,
                                robot_mask=(np.zeros(frame.depth_m.shape, dtype=bool) if frame.robot_mask is None
                                            else frame.robot_mask),
                                timestamp_sec=frame.timestamp_sec, intrinsics=frame.intrinsics,
                                world_from_camera=frame.world_from_camera, world_from_base=robot.world_from_base,
                                base_twist=robot.base_twist, joint_positions=robot.joint_positions,
                                joint_velocities=robot.joint_velocities,
                                joint_names=np.asarray(robot.joint_names, dtype=np.str_))
            self._append(dict(kind="observation", observation_id=observation.observation_id,
                              camera=frame.frame_id, timestamp_sec=frame.timestamp_sec, file=relative))
            self.counts["observations"] += 1

    def event(self, name, timestamp, **values):
        self._append(dict(kind="event", name=name, timestamp_sec=float(timestamp), **values))
        self.counts["events"] += 1

    def finish(self, reason, timestamp, *, details=None):
        if self.closed:
            return
        self.event("FINISH", timestamp, reason=reason, details=details or {})
        self.stream.close()
        self.closed = True
        write_json(self.path / "meta.json", {**self.metadata, **self.counts, "status": reason,
                   "end_timestamp_sec": float(timestamp),
                   "duration_sec": max(0.0, float(timestamp) - self.metadata["start_timestamp_sec"]),
                   "details": details or {}})


def inspect_record(path):
    path = Path(path)
    with (path / "meta.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    counts = dict(states=0, observations=0, events=0)
    previous = -np.inf
    recorded_observations, map_observations = set(), set()
    with (path / "records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] == "state":
                if record["timestamp_sec"] < previous - 1e-6:
                    raise ValueError("状态记录时间倒退")
                previous = record["timestamp_sec"]
                counts["states"] += 1
            elif record["kind"] == "observation":
                with np.load(path / record["file"], allow_pickle=False) as frame:
                    if frame["rgb"].shape[:2] != frame["depth_m"].shape:
                        raise ValueError("记录中的 RGB 与深度尺寸不一致")
                    if "robot_mask" in frame and (frame["robot_mask"].shape != frame["depth_m"].shape
                                                   or frame["robot_mask"].dtype != np.bool_):
                        raise ValueError("机器人过滤标记的尺寸或类型不一致")
                    if abs(float(frame["timestamp_sec"]) - record["timestamp_sec"]) > 1e-6:
                        raise ValueError("图像索引与实际文件时间不一致")
                counts["observations"] += 1
                recorded_observations.add((record["observation_id"], record["camera"], record["timestamp_sec"]))
            elif record["kind"] == "event":
                counts["events"] += 1
                if record["name"] == "MAP_UPDATE":
                    map_observations.update((row["observation_id"], row["source"], row["timestamp_sec"])
                                            for row in record["update"]["observations"])
            else:
                raise ValueError("试验记录类型无效")
    if any(metadata[name] != count for name, count in counts.items()):
        raise ValueError("试验记录数量与索引不一致")
    if not map_observations <= recorded_observations:
        raise ValueError("地图融合引用了没有保存或来源、时间不一致的真实观测")
    return dict(metadata, verified_counts=counts)

