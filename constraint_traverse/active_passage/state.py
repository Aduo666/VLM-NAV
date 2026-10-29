from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import random
import sys

import numpy as np

from .contracts import RealObservation, RobotState


@dataclass
class RandomState:
    python: tuple
    numpy: tuple
    generator: dict
    torch_cpu: np.ndarray | None = None
    torch_cuda: list[np.ndarray] | None = None

    @classmethod
    def capture(cls, generator):
        result = cls(deepcopy(random.getstate()), deepcopy(np.random.get_state()),
                     deepcopy(generator.bit_generator.state))
        torch = sys.modules.get("torch")
        if torch is not None:
            result.torch_cpu = torch.get_rng_state().cpu().numpy().copy()
            if torch.cuda.is_initialized():
                result.torch_cuda = [value.cpu().numpy().copy() for value in torch.cuda.get_rng_state_all()]
        return result

    def restore(self, generator):
        random.setstate(deepcopy(self.python))
        np.random.set_state(deepcopy(self.numpy))
        generator.bit_generator.state = deepcopy(self.generator)
        torch = sys.modules.get("torch")
        if self.torch_cpu is not None:
            if torch is None:
                raise RuntimeError("恢复随机状态时缺少原先使用的 Torch")
            torch.set_rng_state(torch.from_numpy(self.torch_cpu.copy()))
        if self.torch_cuda is not None:
            torch.cuda.set_rng_state_all([torch.from_numpy(value.copy()) for value in self.torch_cuda])


@dataclass
class TrialSnapshot:
    robot: RobotState
    scene_transforms: dict
    rigid_bodies: dict
    controller: dict
    teleop: dict
    map_state: dict
    observations: tuple[RealObservation, ...]
    goal_world_xyz: np.ndarray | None
    random: RandomState
    sensor_sequence: int
    sensor_time_sec: float
    map_time_sec: float
    fusion_state: dict = field(default_factory=dict)


def robot_errors(expected: RobotState, actual: RobotState):
    if expected.joint_names != actual.joint_names:
        raise ValueError("恢复后的关节名称或排列发生变化")
    rotation = expected.world_from_base[:3, :3].T @ actual.world_from_base[:3, :3]
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return dict(position_m=float(np.linalg.norm(expected.world_from_base[:3, 3] - actual.world_from_base[:3, 3])),
                orientation_rad=angle,
                linear_velocity_mps=float(np.linalg.norm(expected.base_twist[:3] - actual.base_twist[:3])),
                angular_velocity_radps=float(np.linalg.norm(expected.base_twist[3:] - actual.base_twist[3:])),
                joint_position_rad=float(np.max(np.abs(expected.joint_positions - actual.joint_positions), initial=0)),
                joint_velocity_radps=float(np.max(np.abs(expected.joint_velocities - actual.joint_velocities), initial=0)))


def require_reset_accuracy(errors):
    limits = dict(position_m=0.01, orientation_rad=np.deg2rad(1), linear_velocity_mps=0.01,
                  angular_velocity_radps=0.02, joint_position_rad=np.deg2rad(1), joint_velocity_radps=0.02,
                  rigid_body_error=1e-4, scene_transform_error=1e-8, controller_error=1e-6,
                  map_error=0.0)
    if set(errors) != set(limits) or any(not np.isfinite(value) or value > limits[name]
                                       for name, value in errors.items()):
        raise RuntimeError(f"试验状态恢复未通过验收：{errors}")

