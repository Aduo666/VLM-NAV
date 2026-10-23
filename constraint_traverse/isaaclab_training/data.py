from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from constraint_traverse.collect.bev import BevConfig, local_voxel_config_from_mapping, snapshot_to_robot_bev
from constraint_traverse.collect.map_state import load_snapshot
from constraint_traverse.collect.recorder import decode_depth_png, validate_episode_directory
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap
from vln_nav.control.fine_path_tracker import fit_agx_command
from vln_nav.control.trajectory import TrajectoryConfig, VelocityCommand, wheel_targets


ACTION_SCALE = np.array([0.8, 0.5], dtype=np.float32)
BEV_SHAPE = (3, 80, 80)
BEV_SIZE = int(np.prod(BEV_SHAPE))
STATE_SIZE = 15
OBS_SIZE = BEV_SIZE + STATE_SIZE
CACHE_VERSION = 1


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: value.copy() for key, value in archive.items()}


def discover(root: Path) -> list[Path]:
    paths = [root] if (root / "meta.json").is_file() else sorted(root.glob("*/episode_*/meta.json"))
    paths = [p.parent if p.name == "meta.json" else p for p in paths]
    if not paths:
        raise ValueError(f"没有找到完整的 Episode：{root}")
    return paths


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    inputs = [path / name for name in ("meta.json", "state.npz", "controls.npz", "initial_map.npz")]
    for folder, suffix in (("front_depth", "png"), ("down_depth", "png"), ("bev", "npy")):
        inputs.extend(sorted((path / folder).glob(f"*.{suffix}")))
    for item in inputs:
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def trajectory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("state.npz", "controls.npz"):
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def state_vector(goal, pose, velocity, steering, wheels, previous, map_age) -> np.ndarray:
    delta = np.asarray(goal) - np.asarray(pose)[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    relative = [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
    return np.concatenate((relative, velocity, steering, wheels, previous, [map_age])).astype(np.float32)


def map_age(timestamp: float, map_timestamp: float) -> float:
    return float(np.clip(timestamp - map_timestamp, 0.0, 60.0))


class Episode:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.controls = read_npz(self.path / "controls.npz")
        self.frames = read_npz(self.path / "state.npz")
        self.initial_map = load_snapshot(self.path / "initial_map.npz")
        if not self.meta.get("valid") or not self.meta.get("success") or self.meta.get("status") != "complete":
            raise ValueError(f"Episode 状态不符合训练要求：{self.path}")
        limits = self.meta["teleop"]
        if not np.allclose([limits["max_linear_mps"], limits["max_angular_radps"]], ACTION_SCALE):
            raise ValueError(f"示范的速度范围不一致：{self.path}")
        if BevConfig.from_mapping(self.meta["bev"]).height != 80 or BevConfig.from_mapping(self.meta["bev"]).width != 80:
            raise ValueError("当前网络要求 80×80 BEV")
        self.dt = float(self.meta["physics_dt_sec"])
        self.decisions = np.arange(0, len(self.controls["physics_step"]), 4)
        steps = self.controls["physics_step"]
        if not np.all(np.diff(steps) == 1):
            raise ValueError("物理步记录存在缺失")
        phase = (steps - steps[0]) % 4
        changes = np.any(np.abs(np.diff(self.controls["human_action_vw"], axis=0)) > 1e-10, axis=1)
        if np.any(phase[1:][changes] != 0):
            raise ValueError("人工指令变化未发生在控制更新点")
        if np.any(np.abs(self.controls["human_action_vw"]) > ACTION_SCALE + 1e-6):
            raise ValueError("人工指令超过约定的训练动作范围")
        self.floor_offset = float(self.frames["world_position_xyz"][0, 2] - self.frames["floor_z_m"][0])

    def build(self, *, verify_bevs: bool = True) -> dict[str, np.ndarray]:
        local_map = LocalVoxelMap(local_voxel_config_from_mapping(self.meta["voxel_map"]))
        local_map.restore(self.initial_map)
        config = BevConfig.from_mapping(self.meta["bev"])
        frames, controls = self.frames, self.controls
        cursor = 0
        verified = 0

        def consume(index):
            nonlocal verified
            if frames["map_integrated"][index]:
                observations = [TimedDepthObservation(
                    source=source,
                    timestamp_sec=float(frames[f"{source}_timestamp_sec"][index]),
                    depth_m=decode_depth_png(self.path / f"{source}_depth" / f"{index:06d}.png"),
                    intrinsics=frames[f"{source}_intrinsics"],
                    camera_to_world=frames[f"{source}_world_from_camera"][index],
                ) for source in ("front", "down")]
                local_map.integrate_batch(observations, center_world_xyz=frames["world_position_xyz"][index])
            if verify_bevs:
                snapshot = local_map.snapshot()
                bev = snapshot_to_robot_bev(snapshot, robot_world_xyyaw=frames["pose_world_xyyaw"][index],
                                            floor_z_m=float(frames["floor_z_m"][index]), config=config)
                saved = np.load(self.path / "bev" / f"{index:06d}.npy", allow_pickle=False)
                if not np.array_equal(bev, saved) or snapshot.version != frames["map_version"][index]:
                    raise ValueError(f"BEV 重建不一致：{self.path}，帧 {index}")
                verified += 1

        bevs, states, labels = [], [], []
        previous = np.zeros(2, dtype=np.float32)
        for index in self.decisions:
            # 采集循环在提交本步指令之后才融合本步收到的相机帧。
            while cursor < len(frames["physics_step"]) and frames["action_physics_step"][cursor] < controls["physics_step"][index]:
                consume(cursor)
                cursor += 1
            snapshot = local_map.snapshot()
            pose = controls["pose_world_xyyaw"][index]
            bevs.append(snapshot_to_robot_bev(snapshot, robot_world_xyyaw=pose,
                floor_z_m=float(controls["world_position_xyz"][index, 2]) - self.floor_offset, config=config))
            states.append(state_vector(self.meta["goal_world_xy"], pose,
                controls["measured_velocity_vw"][index], controls["steering_position"][index],
                controls["wheel_velocity"][index], previous,
                map_age(float(controls["timestamp_sec"][index]), snapshot.timestamp_sec)))
            previous = controls["human_action_vw"][index].astype(np.float32)
            labels.append(previous / ACTION_SCALE)
        while cursor < len(frames["physics_step"]):
            consume(cursor)
            cursor += 1
        labels = np.asarray(labels, dtype=np.float32)
        weights = np.ones(len(labels), dtype=np.float32)
        moving = np.flatnonzero(np.any(np.abs(labels) > 1e-6, axis=1))
        if len(moving):
            weights[:moving[0]] = 0.2
            weights[min(moving[-1] + 6, len(weights)):] = 0.2
        return dict(bev=np.asarray(bevs), state=np.asarray(states), action=labels, weight=weights,
                    control_index=self.decisions, verified_frames=np.asarray(verified))


def prepare(root: Path, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    entries, fingerprints, all_states = [], set(), []
    configuration = None
    for path in discover(root):
        episode = Episode(path)
        digest = fingerprint(path)
        trajectory_digest = trajectory_fingerprint(path)
        if trajectory_digest in fingerprints:
            raise ValueError(f"发现重复示范：{path}")
        fingerprints.add(trajectory_digest)
        current = {key: episode.meta[key] for key in ("physics_dt_sec", "controller", "teleop", "cameras", "bev", "voxel_map", "map_integration_period_sec")}
        if configuration is None:
            configuration = current
        elif configuration != current:
            raise ValueError(f"采集配置不一致：{path}")
        filename = f"{path.parent.name}_{path.name}_{digest[:12]}.npz"
        target = cache / filename
        if target.is_file():
            arrays = read_npz(target)
        else:
            errors = validate_episode_directory(path)
            if errors:
                raise ValueError(f"{path}：{errors[:5]}")
            arrays = episode.build()
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(target)
        if int(arrays["verified_frames"]) != int(episode.meta["sample_count"]):
            raise ValueError("缓存缺少完整 BEV 验收记录")
        validate_sample_interfaces(episode, arrays)
        all_states.append(arrays["state"])
        entries.append(dict(path=str(path), cache=filename, fingerprint=digest, trajectory_fingerprint=trajectory_digest,
            frames=int(episode.meta["sample_count"]), decisions=len(arrays["action"]),
            physics_dt_sec=episode.dt))
        print(f"已检查 {len(entries)} 条：{path.parent.name}/{path.name}，{len(arrays['action'])} 个决策", flush=True)
    states = np.concatenate(all_states)
    if not np.isfinite(states).all():
        raise ValueError("车辆状态包含非有限值")
    manifest = dict(version=CACHE_VERSION, root=str(root.resolve()), episodes=entries, configuration=configuration,
        state_mean=states.mean(axis=0).tolist(), state_std=np.maximum(states.std(axis=0), 0.05).tolist(),
        frames=sum(e["frames"] for e in entries), decisions=sum(e["decisions"] for e in entries))
    (cache / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest(cache: Path) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != CACHE_VERSION:
        raise ValueError("训练缓存版本不匹配")
    return manifest


def validate_manifest_sources(manifest: dict) -> None:
    root = Path(manifest["root"])
    actual = {str(path.resolve()) for path in discover(root)}
    expected = {entry["path"] for entry in manifest["episodes"]}
    if actual != expected:
        raise ValueError("原始示范目录的 Episode 数量或路径已变化，请重新运行 prepare")
    for entry in manifest["episodes"]:
        if fingerprint(Path(entry["path"])) != entry["fingerprint"]:
            raise ValueError(f"原始示范已变化，请重新运行 prepare：{entry['path']}")


def validate_sample_interfaces(episode: Episode, arrays: dict) -> None:
    controls, frames = episode.controls, episode.frames
    indices = episode.decisions
    np.testing.assert_array_equal(arrays["control_index"], indices)
    np.testing.assert_allclose(arrays["action"], controls["human_action_vw"][indices] / ACTION_SCALE, atol=2e-7)
    previous = np.concatenate((np.zeros((1, 2)), controls["human_action_vw"][indices[:-1]]))
    np.testing.assert_allclose(arrays["state"][:, 12:14], previous, atol=2e-7)
    fused = np.flatnonzero(frames["map_integrated"])
    available = np.searchsorted(frames["action_physics_step"][fused], controls["physics_step"][indices], side="left") - 1
    timestamps = np.full(len(indices), episode.initial_map.timestamp_sec)
    valid = available >= 0
    timestamps[valid] = frames["sample_time_sec"][fused[available[valid]]]
    expected_age = np.clip(controls["timestamp_sec"][indices] - timestamps, 0, 60)
    np.testing.assert_allclose(arrays["state"][:, -1], expected_age, atol=2e-6)
    config = TrajectoryConfig.from_mapping(episode.meta["controller"])
    applied = []
    for values in controls["human_action_vw"]:
        desired = VelocityCommand(float(values[0]), float(values[1]))
        command = wheel_targets(fit_agx_command(desired, config), config)[2]
        applied.append([command.linear_mps, command.angular_radps])
    np.testing.assert_allclose(applied, controls["applied_action_vw"], atol=2e-7)
