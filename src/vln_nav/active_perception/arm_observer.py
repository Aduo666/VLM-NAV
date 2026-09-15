"""Finite arm-viewpoint execution with explicit joint ownership and checks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from vln_nav.active_perception.contracts import ArmMotionVerification, ExecutionAssessment
from vln_nav.active_perception.local_geometry import LocalVoxelSnapshot, OCCUPIED, UNKNOWN


@dataclass(frozen=True)
class ArmSafetyConfig:
    link_radius_m: float
    sample_spacing_m: float
    interpolation_step_rad: float
    trusted_min_xyz: tuple[float, float, float]
    trusted_max_xyz: tuple[float, float, float]
    vehicle_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]

    def validate(self) -> None:
        positive = (self.link_radius_m, self.sample_spacing_m, self.interpolation_step_rad)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("Arm sweep radius, spacing, and interpolation step must be positive")
        trusted_min = np.asarray(self.trusted_min_xyz, dtype=np.float64)
        trusted_max = np.asarray(self.trusted_max_xyz, dtype=np.float64)
        if trusted_min.shape != (3,) or trusted_max.shape != (3,) or not (
            np.isfinite(trusted_min).all() and np.isfinite(trusted_max).all()
        ) or np.any(trusted_min >= trusted_max):
            raise ValueError("Trusted arm activity bounds must be finite, ordered XYZ triples")
        for lower, upper in self.vehicle_boxes:
            low = np.asarray(lower, dtype=np.float64)
            high = np.asarray(upper, dtype=np.float64)
            if low.shape != (3,) or high.shape != (3,) or not (
                np.isfinite(low).all() and np.isfinite(high).all()
            ) or np.any(low >= high):
                raise ValueError("Vehicle collision boxes must contain finite, ordered XYZ bounds")


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class UrdfArmKinematics:
    """Minimal URDF FK used only for conservative viewpoint-path sampling."""

    def __init__(self, urdf_path: Path, *, base_link: str, tip_link: str, joint_names: tuple[str, ...]) -> None:
        root = ET.parse(urdf_path).getroot()
        by_child: dict[str, _Joint] = {}
        for node in root.findall("joint"):
            parent_node, child_node = node.find("parent"), node.find("child")
            if parent_node is None or child_node is None:
                continue
            origin = node.find("origin")
            axis = node.find("axis")
            limit = node.find("limit")
            joint = _Joint(
                name=str(node.get("name")),
                joint_type=str(node.get("type")),
                parent=str(parent_node.get("link")),
                child=str(child_node.get("link")),
                xyz=_triplet(None if origin is None else origin.get("xyz"), (0, 0, 0)),
                rpy=_triplet(None if origin is None else origin.get("rpy"), (0, 0, 0)),
                axis=_triplet(None if axis is None else axis.get("xyz"), (1, 0, 0)),
                lower=-math.inf if limit is None or limit.get("lower") is None else float(limit.get("lower")),
                upper=math.inf if limit is None or limit.get("upper") is None else float(limit.get("upper")),
            )
            if joint.child in by_child:
                raise ValueError(f"URDF child link has multiple parent joints: {joint.child}")
            by_child[joint.child] = joint
        chain: list[_Joint] = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"No URDF chain from {base_link} to {tip_link}; stopped at {link}")
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        self.chain = tuple(reversed(chain))
        movable = tuple(joint.name for joint in self.chain if joint.joint_type != "fixed")
        if movable != joint_names:
            raise ValueError(f"URDF movable chain {movable} does not match configured joints {joint_names}")
        self.joint_names = joint_names
        self._joint_by_name = {joint.name: joint for joint in self.chain}

    def validate_positions(self, positions_rad: np.ndarray) -> None:
        q = np.asarray(positions_rad, dtype=np.float64).reshape(-1)
        if q.size != len(self.joint_names) or not np.isfinite(q).all():
            raise ValueError("Arm joint vector has wrong shape or non-finite values")
        for name, value in zip(self.joint_names, q):
            joint = self._joint_by_name[name]
            if value < joint.lower - 1e-6 or value > joint.upper + 1e-6:
                raise ValueError(f"Arm pose exceeds {name} limit: {value:.3f} rad")

    def joint_limits_rad(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(self._joint_by_name[name].lower), float(self._joint_by_name[name].upper))
            for name in self.joint_names
        )

    def link_segments(self, positions_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.validate_positions(positions_rad)
        q_by_name = dict(zip(self.joint_names, map(float, positions_rad)))
        transform = np.eye(4, dtype=np.float64)
        pivots: list[np.ndarray] = []
        for joint in self.chain:
            transform = transform @ _origin_transform(joint.xyz, joint.rpy)
            if joint.joint_type != "fixed":
                pivots.append(transform[:3, 3].copy())
                transform = transform @ _axis_rotation(joint.axis, q_by_name[joint.name])
        if len(pivots) < 2:
            raise ValueError("Arm chain does not provide enough pivots for collision capsules")
        # Extend the last wrist capsule slightly beyond joint6. This remains an
        # approximation; the acceptance UI reports it as such.
        tool = transform @ np.array([0.0, 0.0, 0.12, 1.0])
        starts = np.asarray(pivots, dtype=np.float64)
        ends = np.vstack([starts[1:], tool[:3]])
        # Several industrial-arm URDFs place consecutive rotation axes at the
        # same pivot. Such a zero-length kinematic segment is not a link body.
        keep = np.linalg.norm(ends - starts, axis=1) > 0.02
        return starts[keep], ends[keep]


class ArmMotionVerifier:
    def __init__(self, kinematics: UrdfArmKinematics, config: ArmSafetyConfig) -> None:
        config.validate()
        self.kinematics = kinematics
        self.config = config

    def verify(
        self,
        start_rad: np.ndarray,
        target_rad: np.ndarray,
        snapshot: LocalVoxelSnapshot,
        base_to_world: np.ndarray,
    ) -> ArmMotionVerification:
        start = np.asarray(start_rad, dtype=np.float64).reshape(-1)
        target = np.asarray(target_rad, dtype=np.float64).reshape(-1)
        base_to_world = np.asarray(base_to_world, dtype=np.float64)
        if base_to_world.shape != (4, 4) or not np.isfinite(base_to_world).all():
            raise ValueError("base_to_world must be a finite 4x4 transform")
        self.kinematics.validate_positions(start)
        self.kinematics.validate_positions(target)
        count = max(1, int(math.ceil(np.max(np.abs(target - start)) / self.config.interpolation_step_rad)))
        unknown_count = occupied_count = 0
        checked_states = 0
        for alpha in np.linspace(0.0, 1.0, count + 1):
            checked_states += 1
            starts, ends = self.kinematics.link_segments(start + alpha * (target - start))
            if _has_self_collision(starts, ends, self.config.link_radius_m):
                return ArmMotionVerification(ExecutionAssessment.BLOCKED, "predicted_arm_self_collision",
                                             checked_states, unknown_count, occupied_count)
            local_points = _sample_capsules(starts, ends, self.config.link_radius_m, self.config.sample_spacing_m)
            if _inside_any_box(local_points, self.config.vehicle_boxes).any():
                point = local_points[np.flatnonzero(_inside_any_box(local_points, self.config.vehicle_boxes))[0]]
                return ArmMotionVerification(ExecutionAssessment.BLOCKED, "predicted_robot_self_collision",
                                             checked_states, unknown_count, occupied_count,
                                             tuple(map(float, _transform_points(point[None], base_to_world)[0])))
            world_points = _transform_points(local_points, base_to_world)
            states = snapshot.query(world_points)
            occupied = np.flatnonzero(states == OCCUPIED)
            if len(occupied):
                occupied_count += int(len(occupied))
                return ArmMotionVerification(ExecutionAssessment.BLOCKED, "occupied_voxel_in_arm_sweep",
                                             checked_states, unknown_count, occupied_count,
                                             tuple(map(float, world_points[occupied[0]])))
            trusted = _inside_box(local_points, self.config.trusted_min_xyz, self.config.trusted_max_xyz)
            unknown = np.flatnonzero((states == UNKNOWN) & ~trusted)
            unknown_count += int(len(unknown))
            if len(unknown):
                return ArmMotionVerification(ExecutionAssessment.UNCERTAIN, "unknown_voxel_in_arm_sweep",
                                             checked_states, unknown_count, occupied_count,
                                             tuple(map(float, world_points[unknown[0]])))
        return ArmMotionVerification(ExecutionAssessment.EXECUTABLE, "arm_sweep_clear",
                                     checked_states, unknown_count, occupied_count)


class ArmPoseController:
    """Dynamic-Control position executor; selection and safety stay external."""

    def __init__(self, *, articulation_path: str, available_joint_names: list[str], joint_names: tuple[str, ...],
                 max_speed_radps: float, tolerance_rad: float) -> None:
        if not math.isfinite(max_speed_radps) or max_speed_radps <= 0:
            raise ValueError("Arm maximum speed must be positive")
        if not math.isfinite(tolerance_rad) or tolerance_rad <= 0:
            raise ValueError("Arm position tolerance must be positive")
        self.joint_names = tuple(joint_names)
        missing = set(self.joint_names) - set(available_joint_names)
        if missing:
            raise ValueError(f"Configured arm joints are missing from the articulation: {sorted(missing)}")
        from omni.isaac.dynamic_control import _dynamic_control

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._articulation = self._dc.get_articulation(str(articulation_path))
        if self._articulation is None:
            raise RuntimeError(f"Invalid articulation for arm control: {articulation_path}")
        self._dofs = []
        for name in self.joint_names:
            dof = self._dc.find_articulation_dof(self._articulation, name)
            if dof is None:
                raise RuntimeError(f"Dynamic Control did not find arm joint: {name}")
            self._dofs.append(dof)
        self.max_speed = float(max_speed_radps)
        self.tolerance = float(tolerance_rad)
        self.target: np.ndarray | None = None
        self._start: np.ndarray | None = None
        self._elapsed_sec = 0.0
        self._duration_sec = 0.0

    def current(self) -> np.ndarray:
        return np.asarray([self._dc.get_dof_position(dof) for dof in self._dofs], dtype=np.float64)

    def command(self, target_rad: np.ndarray) -> None:
        target = np.asarray(target_rad, dtype=np.float64).reshape(-1)
        if target.size != len(self._dofs) or not np.isfinite(target).all():
            raise ValueError("Invalid arm position target")
        current = self.current()
        self.target = target.copy()
        self._start = current.copy()
        self._elapsed_sec = 0.0
        maximum_delta = float(np.max(np.abs(target - current)))
        # Minimum-jerk interpolation has a peak normalized velocity of 1.875.
        self._duration_sec = 0.0 if maximum_delta <= self.tolerance else max(
            0.20,
            1.875 * maximum_delta / self.max_speed,
        )

    def stop_at_current(self) -> None:
        """Replace any old target with the measured position immediately."""
        current = self.current()
        self._apply(current)
        self.target = None
        self._start = None
        self._elapsed_sec = 0.0
        self._duration_sec = 0.0

    def max_error_rad(self) -> float:
        if self.target is None:
            return 0.0
        return float(np.max(np.abs(self.target - self.current())))

    def update(self, dt: float) -> bool:
        if self.target is None:
            return True
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Arm control dt must be positive")
        current = self.current()
        error = self.target - current
        self._elapsed_sec += float(dt)
        if self._start is None or self._duration_sec <= 0:
            desired = self.target
            trajectory_complete = True
        else:
            phase = min(1.0, self._elapsed_sec / self._duration_sec)
            blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
            desired = self._start + blend * (self.target - self._start)
            trajectory_complete = phase >= 1.0
        self._apply(desired)
        reached = bool(trajectory_complete and np.max(np.abs(error)) <= self.tolerance)
        if reached:
            self.target = None
            self._start = None
        return reached

    def _apply(self, positions: np.ndarray) -> None:
        self._dc.wake_up_articulation(self._articulation)
        for dof, value in zip(self._dofs, positions):
            self._dc.set_dof_position_target(dof, float(value))
            self._dc.set_dof_velocity_target(dof, 0.0)


def _triplet(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(default if text is None else [float(value) for value in text.split()], dtype=np.float64)


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy)
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    result = np.eye(4)
    result[:3, :3] = [[t*x*x+c, t*x*y-s*z, t*x*z+s*y],
                     [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
                     [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]
    return result


def _sample_capsules(starts: np.ndarray, ends: np.ndarray, radius: float, spacing: float) -> np.ndarray:
    result: list[np.ndarray] = []
    offsets = np.asarray([(0, 0, 0), (radius, 0, 0), (-radius, 0, 0),
                          (0, radius, 0), (0, -radius, 0), (0, 0, radius), (0, 0, -radius)])
    for start, end in zip(starts, ends):
        count = max(2, int(math.ceil(np.linalg.norm(end - start) / spacing)) + 1)
        centers = start + np.linspace(0, 1, count)[:, None] * (end - start)
        result.append((centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3))
    return np.concatenate(result, axis=0)


def _inside_box(points: np.ndarray, lower, upper) -> np.ndarray:
    return np.all((points >= np.asarray(lower)) & (points <= np.asarray(upper)), axis=1)


def _inside_any_box(points: np.ndarray, boxes) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for lower, upper in boxes:
        result |= _inside_box(points, lower, upper)
    return result


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def _has_self_collision(starts: np.ndarray, ends: np.ndarray, radius: float) -> bool:
    for i in range(len(starts)):
        for j in range(i + 2, len(starts)):
            if _segment_distance(starts[i], ends[i], starts[j], ends[j]) < 2 * radius:
                return True
    return False


def _segment_distance(p1, q1, p2, q2) -> float:
    # Closest distance between finite 3-D segments.
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-12:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= 1e-12:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = 0.0 if abs(denom) <= 1e-12 else np.clip((b * f - c * e) / denom, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))
