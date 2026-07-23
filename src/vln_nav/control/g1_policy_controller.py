from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vln_nav.runtime.robot_profile import G1ControllerProfile, PrimitiveProfile


_TERM_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


class G1PolicyController:
    """TorchScript locomotion policy adapter for the local Unitree G1 asset."""

    def __init__(
        self,
        *,
        robot: Any,
        available_joint_names: list[str],
        controller_profile: G1ControllerProfile,
        primitive_profile: PrimitiveProfile,
        policy_decimation: int,
        logger: logging.Logger,
    ) -> None:
        self._robot = robot
        self._profile = controller_profile
        self._primitives = primitive_profile
        self._available_joint_names = list(available_joint_names)
        self._decimation = max(1, int(policy_decimation))
        self._logger = logger.getChild("g1_policy")
        self._device = torch.device(controller_profile.inference_device)
        self._deploy = _load_deploy_config(controller_profile.deploy_config_path)
        self._policy = torch.jit.load(str(controller_profile.policy_path), map_location=self._device)
        self._policy.eval()
        self._command_min, self._command_max = _velocity_command_ranges(self._deploy)
        self._angular_command_limit = max(
            float(self._primitives.turn_command),
            abs(float(self._command_min[2])),
            abs(float(self._command_max[2])),
        )
        self._validate_primitive_commands()
        self._backward_command_mps = min(
            0.5 * float(self._primitives.forward_command),
            abs(float(self._command_min[0])),
        )

        sdk_names = list(controller_profile.sdk_joint_names)
        joint_ids_map = _numeric_vector(self._deploy, "joint_ids_map", length=29, dtype=np.int64)
        if sorted(joint_ids_map.tolist()) != list(range(29)):
            raise ValueError("G1 deploy joint_ids_map must be a permutation of [0, 28]")
        self._policy_joint_names = [sdk_names[int(index)] for index in joint_ids_map]
        name_to_index = {name: index for index, name in enumerate(available_joint_names)}
        missing = [name for name in self._policy_joint_names if name not in name_to_index]
        if missing:
            raise RuntimeError(
                "G1 articulation does not match deploy.yaml; missing policy joints: "
                f"{missing}. Available joints: {available_joint_names}"
            )
        self._joint_indices = np.asarray(
            [name_to_index[name] for name in self._policy_joint_names], dtype=np.int64
        )

        self._default_joint_pos = _numeric_vector(self._deploy, "default_joint_pos", length=29)
        # Deployment gains are exported in Unitree SDK order while policy state/actions
        # use the Isaac articulation order described by joint_ids_map.
        self._stiffness = _numeric_vector(self._deploy, "stiffness", length=29)[joint_ids_map]
        self._damping = _numeric_vector(self._deploy, "damping", length=29)[joint_ids_map]
        action_cfg = _mapping(self._deploy, "actions", "deploy")
        joint_action_cfg = _mapping(action_cfg, "JointPositionAction", "deploy.actions")
        self._action_scale = _numeric_vector(joint_action_cfg, "scale", length=29)
        self._action_offset = _numeric_vector(joint_action_cfg, "offset", length=29)

        observation_cfg = _mapping(self._deploy, "observations", "deploy")
        self._history: dict[str, deque[np.ndarray]] = {}
        self._term_scale: dict[str, np.ndarray] = {}
        for term_name in _TERM_ORDER:
            term_cfg = _mapping(observation_cfg, term_name, "deploy.observations")
            history_length = int(term_cfg.get("history_length", 1))
            if history_length <= 0:
                raise ValueError(f"G1 observation history_length must be positive: {term_name}")
            scale_value = term_cfg.get("scale", 1.0)
            self._term_scale[term_name] = np.asarray(scale_value, dtype=np.float32).reshape(-1)
            self._history[term_name] = deque(maxlen=history_length)

        self._last_action = np.zeros(29, dtype=np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        self._physics_step_count = 0
        self._policy_eval_count = 0
        self._history_initialized = False
        self._configure_pd_gains(available_joint_names)
        self._initialize_joint_state()
        self._apply_joint_targets(self._default_joint_pos)
        self._logger.info(
            "[G1Policy] loaded=%s input_dim=480 output_dim=29 device=%s policy_joints=%s decimation=%s",
            controller_profile.policy_path,
            self._device,
            self._policy_joint_names,
            self._decimation,
        )

    @property
    def standup_duration_sec(self) -> float:
        return float(self._profile.standup_duration_sec)

    @property
    def command_limits(self) -> dict[str, tuple[float, float]]:
        return {
            "linear_x_mps": (float(self._command_min[0]), float(self._command_max[0])),
            "linear_y_mps": (float(self._command_min[1]), float(self._command_max[1])),
            "angular_z_radps": (
                -float(self._angular_command_limit),
                float(self._angular_command_limit),
            ),
        }

    def apply(self, action_name: str, *, speed_scale: float = 1.0) -> tuple[float, float]:
        action = str(action_name).strip().upper()
        speed = float(np.clip(speed_scale, 0.0, 1.0))
        if action == "FORWARD":
            self.set_velocity_command(
                linear_x_mps=float(self._primitives.forward_command) * speed,
                angular_z_radps=0.0,
            )
        elif action == "BACKWARD":
            self.set_velocity_command(
                linear_x_mps=-float(self._backward_command_mps) * speed,
                angular_z_radps=0.0,
            )
        elif action == "TURN_LEFT":
            self.set_velocity_command(
                linear_x_mps=0.0,
                angular_z_radps=float(self._primitives.turn_command) * speed,
            )
        elif action == "TURN_RIGHT":
            self.set_velocity_command(
                linear_x_mps=0.0,
                angular_z_radps=-float(self._primitives.turn_command) * speed,
            )
        elif action == "STOP":
            self._command.fill(0.0)
        else:
            raise RuntimeError(
                f"G1 does not support action {action!r}; expected FORWARD, BACKWARD, "
                "TURN_LEFT, TURN_RIGHT, or STOP"
            )

        return float(self._command[0]), float(self._command[2])

    def set_velocity_command(
        self,
        *,
        linear_x_mps: float,
        angular_z_radps: float,
    ) -> tuple[float, float]:
        self._command[:] = np.asarray(
            [
                np.clip(float(linear_x_mps), self._command_min[0], self._command_max[0]),
                0.0,
                np.clip(
                    float(angular_z_radps),
                    -self._angular_command_limit,
                    self._angular_command_limit,
                ),
            ],
            dtype=np.float32,
        )
        return float(self._command[0]), float(self._command[2])

    def reset(self) -> None:
        self._command.fill(0.0)
        self._last_action.fill(0.0)
        self._physics_step_count = 0
        self._policy_eval_count = 0
        self._history_initialized = False
        for history in self._history.values():
            history.clear()
        self._configure_pd_gains(self._available_joint_names)
        self._initialize_joint_state()
        self._apply_joint_targets(self._default_joint_pos)

    def advance_physics(self) -> None:
        """Evaluate/apply the locomotion policy once per actual physics step."""
        if self._physics_step_count % self._decimation == 0:
            self._run_policy()
        else:
            self._apply_joint_targets(self._action_offset + self._action_scale * self._last_action)
        self._physics_step_count += 1

    def track_local_target(self, **_: Any) -> tuple[float, float]:
        raise RuntimeError("TRACK_TARGET is not supported by the unitree_g1_29dof profile")

    def readback(self) -> dict[str, Any]:
        joint_positions = np.asarray(self._robot.get_joint_positions(), dtype=np.float32).reshape(-1)
        policy_joint_positions = joint_positions[self._joint_indices]
        joint_targets = self._action_offset + self._action_scale * self._last_action
        return {
            "command_velocity": {
                "linear_x_mps": float(self._command[0]),
                "linear_y_mps": float(self._command[1]),
                "angular_z_radps": float(self._command[2]),
            },
            "policy_action": {
                name: float(value) for name, value in zip(self._policy_joint_names, self._last_action)
            },
            "joint_tracking": {
                "max_error_rad": float(np.max(np.abs(joint_targets - policy_joint_positions))),
                "mean_error_rad": float(np.mean(np.abs(joint_targets - policy_joint_positions))),
            },
        }

    def _run_policy(self) -> None:
        terms = self._observation_terms()
        term_ranges = {
            name: (float(np.min(value)), float(np.max(value)))
            for name, value in terms.items()
        }
        if self._policy_eval_count == 0:
            self._logger.info("[G1PolicyInput] first_frame_ranges=%s", term_ranges)
        if not self._history_initialized:
            for term_name, value in terms.items():
                history = self._history[term_name]
                for _ in range(int(history.maxlen or 1)):
                    history.append(value.copy())
            self._history_initialized = True
        else:
            for term_name, value in terms.items():
                self._history[term_name].append(value.copy())

        observation = np.concatenate(
            [np.concatenate(list(self._history[name]), axis=0) for name in _TERM_ORDER], axis=0
        ).astype(np.float32, copy=False)
        if observation.shape != (480,):
            raise RuntimeError(f"G1 policy observation shape mismatch: {observation.shape}, expected (480,)")
        if not np.all(np.isfinite(observation)):
            invalid_terms = [
                name for name, value in terms.items() if not np.all(np.isfinite(value))
            ]
            raise RuntimeError(
                f"G1 policy observation contains non-finite values in terms: {invalid_terms}"
            )
        with torch.inference_mode():
            output = self._policy(torch.from_numpy(observation).unsqueeze(0).to(self._device))
        if isinstance(output, (tuple, list)):
            output = output[0]
        action = np.asarray(output.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if action.shape != (29,):
            raise RuntimeError(f"G1 policy action shape mismatch: {action.shape}, expected (29,)")
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                "G1 policy returned non-finite action values; "
                f"observation_term_ranges={term_ranges}"
            )
        if float(np.max(np.abs(action))) > 100.0:
            raise RuntimeError(
                "G1 policy action exceeded the safety bound: "
                f"max_abs={float(np.max(np.abs(action))):.3g} "
                f"observation_term_ranges={term_ranges}"
            )
        self._last_action = action
        self._policy_eval_count += 1
        self._apply_joint_targets(self._action_offset + self._action_scale * action)

    def _observation_terms(self) -> dict[str, np.ndarray]:
        joint_positions = np.asarray(self._robot.get_joint_positions(), dtype=np.float32).reshape(-1)
        joint_velocities = np.asarray(self._robot.get_joint_velocities(), dtype=np.float32).reshape(-1)
        if joint_positions.size <= int(np.max(self._joint_indices)):
            raise RuntimeError("G1 articulation returned an incomplete joint position vector")
        policy_joint_pos = joint_positions[self._joint_indices]
        policy_joint_vel = joint_velocities[self._joint_indices]

        _, orientation_wxyz = self._robot.get_world_pose()
        rotation_world_from_body = _rotation_matrix_wxyz(np.asarray(orientation_wxyz, dtype=np.float32))
        angular_velocity_world = np.asarray(self._robot.get_angular_velocity(), dtype=np.float32).reshape(3)
        angular_velocity_body = rotation_world_from_body.T @ angular_velocity_world
        projected_gravity = rotation_world_from_body.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)

        raw_terms = {
            "base_ang_vel": angular_velocity_body,
            "projected_gravity": projected_gravity,
            "velocity_commands": self._command,
            "joint_pos_rel": policy_joint_pos - self._default_joint_pos,
            "joint_vel_rel": policy_joint_vel,
            "last_action": self._last_action,
        }
        scaled: dict[str, np.ndarray] = {}
        for name, value in raw_terms.items():
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            scale = self._term_scale[name]
            if scale.size == 1:
                vector = vector * float(scale[0])
            elif scale.size == vector.size:
                vector = vector * scale
            else:
                raise ValueError(
                    f"G1 observation scale mismatch for {name}: scale={scale.size} value={vector.size}"
                )
            scaled[name] = vector
        return scaled

    def _configure_pd_gains(self, available_joint_names: list[str]) -> None:
        articulation_controller = self._robot.get_articulation_controller()
        current_kps, current_kds = articulation_controller.get_gains()
        kps = np.asarray(current_kps, dtype=np.float32).reshape(-1)
        kds = np.asarray(current_kds, dtype=np.float32).reshape(-1)
        if kps.size != len(available_joint_names) or kds.size != len(available_joint_names):
            raise RuntimeError(
                "G1 articulation gain vector does not match the detected articulation joint count"
            )
        kps[self._joint_indices] = self._stiffness
        kds[self._joint_indices] = self._damping
        articulation_controller.set_gains(kps=kps, kds=kds, save_to_usd=False)

    def _validate_primitive_commands(self) -> None:
        forward = float(self._primitives.forward_command)
        turn = float(self._primitives.turn_command)
        if not 0.0 <= forward <= float(self._command_max[0]):
            raise ValueError(
                "G1 forward_command is outside the policy training range: "
                f"value={forward:.3f} range=[0.000,{float(self._command_max[0]):.3f}]"
            )
        turn_limit = min(
            abs(float(self._command_min[2])),
            abs(float(self._command_max[2])),
        )
        if turn < 0.0:
            raise ValueError(f"G1 turn_command must be non-negative: value={turn:.3f}")
        if turn > turn_limit:
            self._logger.warning(
                "[G1Policy] configured turn_command=%.3f rad/s exceeds the exported "
                "training range limit=%.3f rad/s; preserving the configured VLN command",
                turn,
                turn_limit,
            )

    def _initialize_joint_state(self) -> None:
        self._robot.set_joint_positions(
            np.asarray(self._default_joint_pos, dtype=np.float32),
            joint_indices=self._joint_indices,
        )
        self._robot.set_joint_velocities(
            np.zeros(29, dtype=np.float32),
            joint_indices=self._joint_indices,
        )

    def _apply_joint_targets(self, targets: np.ndarray) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._robot.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(targets, dtype=np.float32),
                joint_indices=self._joint_indices,
            )
        )


def _load_deploy_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"G1 deploy config must be a YAML mapping: {path}")
    return payload


def _mapping(payload: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a YAML mapping")
    return value


def _velocity_command_ranges(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    commands = _mapping(payload, "commands", "deploy")
    base_velocity = _mapping(commands, "base_velocity", "deploy.commands")
    ranges = _mapping(base_velocity, "ranges", "deploy.commands.base_velocity")
    values = []
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        bounds = np.asarray(ranges.get(key), dtype=np.float32).reshape(-1)
        if bounds.size != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(
                f"G1 deploy commands.base_velocity.ranges.{key} must be [minimum, maximum]"
            )
        values.append(bounds)
    stacked = np.stack(values, axis=0)
    return stacked[:, 0].copy(), stacked[:, 1].copy()


def _numeric_vector(
    payload: dict[str, Any], key: str, *, length: int, dtype: Any = np.float32
) -> np.ndarray:
    value = np.asarray(payload.get(key), dtype=dtype).reshape(-1)
    if value.size != int(length):
        raise ValueError(f"G1 deploy {key} must contain {length} values, got {value.size}")
    return value


def _rotation_matrix_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if quaternion.size < 4:
        raise ValueError("Quaternion must contain four values in wxyz order")
    norm = float(np.linalg.norm(quaternion[:4]))
    if norm <= 1.0e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quaternion[:4] / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
