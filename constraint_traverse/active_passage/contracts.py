from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

import numpy as np


def _array(value, *, shape=None, dtype=np.float64, finite=True):
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"数组形状应为 {shape}，实际为 {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError("数组需要包含有限数值")
    result.setflags(write=False)
    return result


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("标识符需要使用非空字符串")
    return value


def _version(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("版本号必须为非负整数")
    return int(value)


def _transform(value):
    result = _array(value, shape=(4, 4))
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("位姿需要使用有效的刚体变换矩阵")
    return result


@dataclass(frozen=True, eq=False)
class TaskSpec:
    task_id: str
    instruction: str
    passage_id: str
    goal_world_xyz: np.ndarray
    goal_tolerance_m: float

    def __post_init__(self):
        for value in (self.task_id, self.instruction, self.passage_id):
            _identifier(value)
        if not math.isfinite(self.goal_tolerance_m) or self.goal_tolerance_m <= 0:
            raise ValueError("目标位置容差必须为正数")
        object.__setattr__(self, "goal_world_xyz", _array(self.goal_world_xyz, shape=(3,)))


@dataclass(frozen=True, eq=False)
class RobotState:
    timestamp_sec: float
    world_from_base: np.ndarray
    base_twist: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    def __post_init__(self):
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("机器人时间戳必须为有限数值")
        names = tuple(self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError("关节名称不能重复")
        for name in names:
            _identifier(name)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "world_from_base", _transform(self.world_from_base))
        # 速度顺序为世界坐标中的线速度 xyz 和角速度 xyz。
        object.__setattr__(self, "base_twist", _array(self.base_twist, shape=(6,)))
        for field in ("joint_positions", "joint_velocities"):
            object.__setattr__(self, field, _array(getattr(self, field), shape=(len(names),)))


@dataclass(frozen=True, eq=False)
class CameraFrame:
    frame_id: str
    timestamp_sec: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    robot_mask: np.ndarray | None = None

    def __post_init__(self):
        _identifier(self.frame_id)
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("相机时间戳必须为有限数值")
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) == 0:
            raise ValueError("RGB 图像必须使用非空的 uint8 H×W×3 数组")
        depth = _array(self.depth_m, shape=rgb.shape[:2], dtype=np.float32, finite=False)
        if (depth[np.isfinite(depth)] < 0).any() or np.isneginf(depth).any():
            raise ValueError("深度值不能为负数")
        intrinsics = _array(self.intrinsics, shape=(3, 3))
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0, 0, 1]):
            raise ValueError("相机内参无效")
        object.__setattr__(self, "rgb", _array(rgb, dtype=np.uint8))
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "world_from_camera", _transform(self.world_from_camera))
        if self.robot_mask is not None:
            mask = np.asarray(self.robot_mask)
            if mask.shape != depth.shape or mask.dtype != np.bool_:
                raise ValueError("机器人像素过滤标记必须为与深度同尺寸的布尔数组")
            object.__setattr__(self, "robot_mask", _array(mask, dtype=bool))


@dataclass(frozen=True, eq=False)
class RealObservation:
    observation_id: str
    frame: CameraFrame
    robot: RobotState

    def __post_init__(self):
        _identifier(self.observation_id)
        if not isinstance(self.frame, CameraFrame) or not isinstance(self.robot, RobotState):
            raise TypeError("真实观测需要相机帧和机器人状态")
        if abs(self.frame.timestamp_sec - self.robot.timestamp_sec) > 1e-6:
            raise ValueError("真实观测与机器人状态需要对应同一采集时刻")


@dataclass(frozen=True, eq=False)
class PredictedObservation:
    prediction_id: str
    hypothesis_id: str
    frame: CameraFrame

    def __post_init__(self):
        _identifier(self.prediction_id)
        _identifier(self.hypothesis_id)
        if not isinstance(self.frame, CameraFrame):
            raise TypeError("预测观测需要相机帧")


@dataclass(frozen=True, eq=False)
class VoxelGeometry:
    occupancy: np.ndarray
    origin_world_xyz: np.ndarray
    resolution_m: float

    def __post_init__(self):
        values = np.asarray(self.occupancy)
        # 体素按 z、y、x 排列；未知为 -1，空闲为 0，占用为 1。
        if values.ndim != 3 or min(values.shape) == 0 or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("体素几何需要非空的三维占用状态数组")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("体素分辨率必须为正数")
        object.__setattr__(self, "occupancy", _array(values, dtype=np.int8))
        object.__setattr__(self, "origin_world_xyz", _array(self.origin_world_xyz, shape=(3,)))


@dataclass(frozen=True, eq=False)
class ObservedGeometry:
    grid: VoxelGeometry
    observation_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.grid, VoxelGeometry):
            raise TypeError("观测几何需要数据形式的体素网格")
        ids = tuple(self.observation_ids)
        if not ids:
            raise ValueError("观测几何需要记录真实观测来源")
        for value in ids:
            _identifier(value)
        object.__setattr__(self, "observation_ids", ids)


@dataclass(frozen=True, eq=False)
class GroundTruthGeometry:
    scene_variant_id: str
    grid: VoxelGeometry

    def __post_init__(self):
        _identifier(self.scene_variant_id)
        if not isinstance(self.grid, VoxelGeometry):
            raise TypeError("完整真实几何需要数据形式的体素网格")


@dataclass(frozen=True, eq=False)
class GeometryHypothesis:
    hypothesis_id: str
    grid: VoxelGeometry
    weight: float

    def __post_init__(self):
        _identifier(self.hypothesis_id)
        if not isinstance(self.grid, VoxelGeometry):
            raise TypeError("环境假设需要数据形式的体素网格")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("环境假设权重必须为非负数")


@dataclass(frozen=True, eq=False)
class BeliefSnapshot:
    version: int
    observations: tuple[RealObservation, ...]
    observed_geometry: ObservedGeometry | None = None
    hypotheses: tuple[GeometryHypothesis, ...] = ()

    def __post_init__(self):
        observations, hypotheses = tuple(self.observations), tuple(self.hypotheses)
        object.__setattr__(self, "version", _version(self.version))
        if not all(isinstance(value, RealObservation) for value in observations):
            raise TypeError("真实观测历史只接受 RealObservation")
        if self.observed_geometry is not None and not isinstance(self.observed_geometry, ObservedGeometry):
            raise TypeError("实际地图只接受 ObservedGeometry")
        if not all(isinstance(value, GeometryHypothesis) for value in hypotheses):
            raise TypeError("环境假设类型无效")
        if len({value.hypothesis_id for value in hypotheses}) != len(hypotheses):
            raise ValueError("环境假设标识符不能重复")
        ids = {value.observation_id for value in observations}
        if len(ids) != len(observations):
            raise ValueError("真实观测标识符不能重复")
        if self.observed_geometry is not None and not set(self.observed_geometry.observation_ids) <= ids:
            raise ValueError("观测几何包含没有记录的真实观测来源")
        if hypotheses and not math.isclose(sum(value.weight for value in hypotheses), 1, rel_tol=1e-9):
            raise ValueError("环境假设权重之和必须为 1")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "hypotheses", hypotheses)


@dataclass(frozen=True, eq=False)
class TraversalPlan:
    task_id: str
    poses_world_xyyaw: np.ndarray
    motion_direction: np.ndarray
    belief_version: int

    def __post_init__(self):
        _identifier(self.task_id)
        poses = _array(self.poses_world_xyyaw)
        directions = np.asarray(self.motion_direction)
        if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2:
            raise ValueError("通行路径需要至少两个世界坐标平面位姿")
        if directions.shape != (len(poses),) or not np.isin(directions, [-1, 0, 1]).all():
            raise ValueError("通行路径需要对应的运动方向")
        if ((directions[1:] == 0) & (np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) > 1e-6)).any():
            raise ValueError("原地旋转或保持姿态的运动段不能包含平移")
        object.__setattr__(self, "poses_world_xyyaw", poses)
        object.__setattr__(self, "motion_direction", _array(directions, dtype=np.int8))
        object.__setattr__(self, "belief_version", _version(self.belief_version))


@dataclass(frozen=True, eq=False)
class ArmObservationPlan:
    joint_names: tuple[str, ...]
    outward_positions: np.ndarray
    return_positions: np.ndarray
    sample_period_sec: float
    belief_version: int
    safety_passed: bool

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("机械臂观察需要明确且不重复的关节名称")
        for name in names:
            _identifier(name)
        outward, returning = _array(self.outward_positions), _array(self.return_positions)
        if any(value.ndim != 2 or value.shape[1] != len(names) or len(value) < 2 for value in (outward, returning)):
            raise ValueError("观察动作需要完整的伸出与收回关节轨迹")
        if not np.allclose(outward[-1], returning[0], atol=1e-6, rtol=0):
            raise ValueError("伸出与收回轨迹需要在观察位置连续")
        if not np.allclose(outward[0], returning[-1], atol=1e-6, rtol=0):
            raise ValueError("观察动作需要返回起始收纳姿态")
        if not math.isfinite(self.sample_period_sec) or self.sample_period_sec <= 0:
            raise ValueError("机械臂轨迹采样周期必须为正数")
        if not isinstance(self.safety_passed, bool):
            raise TypeError("机械臂安全检查结果必须为布尔值")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "outward_positions", outward)
        object.__setattr__(self, "return_positions", returning)
        object.__setattr__(self, "belief_version", _version(self.belief_version))


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    TIMEOUT = "timeout"
    PLANNING_FAILED = "planning_failed"
    COLLISION = "collision"


@dataclass(frozen=True)
class ExecutionOutcome:
    status: ExecutionStatus
    collided: bool
    traversal_duration_sec: float
    minimum_clearance_m: float | None

    def __post_init__(self):
        if not isinstance(self.status, ExecutionStatus) or not isinstance(self.collided, bool):
            raise TypeError("执行结果需要明确的终止状态和碰撞标志")
        if self.status is ExecutionStatus.COLLISION and not self.collided:
            raise ValueError("碰撞终止状态需要对应碰撞事件")
        if not math.isfinite(self.traversal_duration_sec) or self.traversal_duration_sec < 0:
            raise ValueError("通行执行时间必须为非负数")
        if self.minimum_clearance_m is not None:
            if math.isnan(self.minimum_clearance_m) or self.minimum_clearance_m == -math.inf:
                raise ValueError("最小间距不能为 NaN 或负无穷")

    @property
    def completed(self):
        return self.status is ExecutionStatus.COMPLETED


@dataclass(frozen=True, eq=False)
class LossSamples:
    losses: np.ndarray
    environment_ids: tuple[str, ...]
    execution_noise_ids: tuple[str, ...]
    weights: np.ndarray | None = None
    observation_sample_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        losses = _array(self.losses)
        environments, noises = tuple(self.environment_ids), tuple(self.execution_noise_ids)
        if losses.ndim != 1 or len(losses) == 0 or len(environments) != len(losses) or len(noises) != len(losses):
            raise ValueError("每个损失样本需要环境与执行误差标识符")
        for value in environments + noises:
            _identifier(value)
        if self.observation_sample_ids is not None:
            ids = tuple(self.observation_sample_ids)
            if len(ids) != len(losses):
                raise ValueError("观测样本标识符需要与损失样本对应")
            for value in ids:
                _identifier(value)
            object.__setattr__(self, "observation_sample_ids", ids)
        weights = np.ones(len(losses)) if self.weights is None else np.asarray(self.weights, dtype=np.float64)
        if weights.shape != losses.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("样本权重必须为对应的非负有限数值")
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("样本权重之和必须为正数")
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "environment_ids", environments)
        object.__setattr__(self, "execution_noise_ids", noises)
        object.__setattr__(self, "weights", _array(weights / total))


@dataclass(frozen=True)
class RiskSummary:
    expectation: float
    cvar: float
    alpha: float
    beta: float

    @property
    def total(self):
        return self.expectation + self.beta * self.cvar


@dataclass(frozen=True)
class ObservationValue:
    before: RiskSummary
    after: RiskSummary
    observation_cost: float

    @property
    def value(self):
        return self.before.total - self.after.total - self.observation_cost


class TaskProvider(Protocol):
    def current_task(self) -> TaskSpec: ...


class BeliefUpdater(Protocol):
    def update(self, observation: RealObservation) -> BeliefSnapshot: ...
    def snapshot(self) -> BeliefSnapshot: ...


class TraversalPlanner(Protocol):
    def plan(self, task: TaskSpec, belief: BeliefSnapshot, robot: RobotState) -> TraversalPlan | None: ...


class InspectionPlanner(Protocol):
    def plan_round_trip(self, robot: RobotState, belief: BeliefSnapshot, observe_positions: np.ndarray) -> ArmObservationPlan: ...


class OutcomeEvaluator(Protocol):
    def execution_loss(self, outcome: ExecutionOutcome) -> float: ...
    def risk(self, samples: LossSamples) -> RiskSummary: ...
    def observation_value(self, before: LossSamples, after: LossSamples, *, observation_cost: float) -> ObservationValue: ...
