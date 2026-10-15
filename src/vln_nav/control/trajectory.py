from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class WheelGeometry:
    drive_joint: str
    steering_joint: str
    x_m: float
    y_m: float
    radius_m: float
    steering_sign: float
    drive_sign: float


@dataclass(frozen=True)
class TrajectoryConfig:
    wheels: tuple[WheelGeometry, ...]
    max_linear_mps: float
    max_angular_radps: float
    max_acceleration_mps2: float
    max_lateral_acceleration_mps2: float
    max_wheel_radps: float
    max_steering_rad: float
    steering_alignment_rad: float
    lookahead_m: float
    arrival_distance_m: float
    max_path_error_m: float
    timeout_sec: float

    @classmethod
    def from_mapping(cls, data: dict) -> "TrajectoryConfig":
        expected = {f.name for f in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"AGX continuous requires exactly: {sorted(expected)}")
        wheel_fields = {f.name for f in fields(WheelGeometry)}
        wheels = []
        for row in data["wheels"]:
            if set(row) != wheel_fields:
                raise ValueError(f"Invalid continuous wheel fields: {row.keys()}")
            values = {k: (str(v) if k.endswith("joint") else float(v)) for k, v in row.items()}
            wheels.append(WheelGeometry(**values))
        config = cls(wheels=tuple(wheels), **{k: float(data[k]) for k in expected - {"wheels"}})
        config.validate()
        return config

    def validate(self) -> None:
        for f in fields(self):
            if f.name != "wheels":
                value = getattr(self, f.name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"continuous.{f.name} must be finite and positive")
        if len(self.wheels) != 4 or len({w.drive_joint for w in self.wheels}) != 4 or len({w.steering_joint for w in self.wheels}) != 4:
            raise ValueError("Continuous AGX requires four distinct drive/steering pairs")
        for w in self.wheels:
            if not all(math.isfinite(v) for v in (w.x_m, w.y_m, w.radius_m)) or w.radius_m <= 0:
                raise ValueError("Invalid wheel geometry")
            if w.steering_sign not in (-1, 1) or w.drive_sign not in (-1, 1):
                raise ValueError("Wheel joint signs must be -1 or +1")
        if not self.arrival_distance_m < self.lookahead_m <= self.max_path_error_m:
            raise ValueError("Require arrival_distance < lookahead <= max_path_error")
        if not self.steering_alignment_rad < self.max_steering_rad <= math.pi / 2:
            raise ValueError("Require alignment < steering limit <= pi/2")


@dataclass(frozen=True)
class VelocityCommand:
    linear_mps: float = 0.0
    angular_radps: float = 0.0


@dataclass(frozen=True)
class TrackingResult:
    command: VelocityCommand
    status: str
    remaining_m: float
    cross_track_m: float
    target_xy: tuple[float, float]
    nominal_command: VelocityCommand = VelocityCommand()


def wheel_targets(command: VelocityCommand, config: TrajectoryConfig):
    """Rigid-body twist -> each wheel's rolling vector; not differential drive."""
    v, omega = command.linear_mps, command.angular_radps
    if not math.isfinite(v) or not math.isfinite(omega):
        raise ValueError("Non-finite body velocity")
    scale = min(1.0, config.max_linear_mps / max(abs(v), 1e-12),
                config.max_angular_radps / max(abs(omega), 1e-12))
    v, omega = v * scale, omega * scale
    drives, steering = {}, {}
    for wheel in config.wheels:
        vx, vy = v - omega * wheel.y_m, omega * wheel.x_m
        angle = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-10 else 0.0
        speed = math.hypot(vx, vy)
        # Equivalent wheel orientation plus reversed rolling avoids a pi rotation.
        if angle > math.pi / 2:
            angle -= math.pi
            speed = -speed
        elif angle < -math.pi / 2:
            angle += math.pi
            speed = -speed
        if abs(angle) > config.max_steering_rad + 1e-9:
            raise ValueError("Requested curvature exceeds AGX steering limit")
        steering[wheel.steering_joint] = angle / wheel.steering_sign
        drives[wheel.drive_joint] = speed / wheel.radius_m / wheel.drive_sign
    motor_scale = min(1.0, config.max_wheel_radps / max(max(map(abs, drives.values())), 1e-12))
    return ({k: x * motor_scale for k, x in drives.items()}, steering,
            VelocityCommand(v * motor_scale, omega * motor_scale))


class AgxVelocityAdapter:
    """Write a smooth body command while steering converges to its target.

    A four-wheel-steering chassis must usually roll a little while changing a
    large steering angle.  Hard-zeroing the drive wheels until every steering
    joint is aligned can therefore deadlock against tire contact friction.
    PathTracker already ramps velocity from zero, so the transient remains
    bounded without a separate stop gate.
    """
    def __init__(self, drive, config: TrajectoryConfig):
        config.validate()
        self.drive, self.config = drive, config
        self.status = "STOPPED"
        self._steering_target_rad: dict[str, float] = {}
        self._steering_measured_rad: dict[str, float] = {}
        self._max_steering_error_rad = 0.0

    def apply(self, command: VelocityCommand) -> VelocityCommand:
        try:
            motors, steering, applied = wheel_targets(command, self.config)
            measured = self.drive.readback()["steering_position"]
            errors = [abs(measured[name] - value) for name, value in steering.items()]
            if not all(math.isfinite(e) for e in errors):
                raise ValueError("Non-finite steering readback")
            self._steering_target_rad = dict(steering)
            self._steering_measured_rad = {
                name: float(measured[name]) for name in steering
            }
            self._max_steering_error_rad = max(errors, default=0.0)
            self.status = "TRACKING"
            if self._max_steering_error_rad > self.config.steering_alignment_rad:
                self.status = "STEERING_TRANSIENT"
            if command == VelocityCommand():
                self.status = "STOPPED"
            self.drive.apply_continuous_joint_targets(motors, steering)
            return applied
        except Exception:
            self.stop()
            raise

    def steering_diagnostics(self) -> dict[str, object]:
        return {
            "status": self.status,
            "max_error_rad": float(self._max_steering_error_rad),
            "target_rad": dict(self._steering_target_rad),
            "measured_rad": dict(self._steering_measured_rad),
        }

    def stop(self) -> None:
        self.drive.apply_continuous_joint_targets(
            {w.drive_joint: 0.0 for w in self.config.wheels},
            {w.steering_joint: 0.0 for w in self.config.wheels},
        )
        self.status = "STOPPED"


class PathTracker:
    def __init__(
        self,
        path_xy,
        config: TrajectoryConfig,
        *,
        initial_speed_mps: float = 0.0,
    ):
        config.validate()
        if not math.isfinite(float(initial_speed_mps)) or float(initial_speed_mps) < 0:
            raise ValueError("initial_speed_mps 必须为有限非负数")
        path = np.asarray(path_xy, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError("Path must have at least two finite XY points")
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        path = path[np.r_[True, lengths > 1e-6]]
        if len(path) < 2:
            raise ValueError("Path has zero length")
        self.path, self.config = path, config
        self.lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(self.lengths)]
        self.progress = self.elapsed = 0.0
        self.speed = min(float(initial_speed_mps), float(config.max_linear_mps))
        self.finished = False

    def feedback(self, applied: VelocityCommand) -> None:
        self.speed = abs(applied.linear_mps)

    def _point(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.path[:, axis]) for axis in (0, 1)])

    def step(self, pose, dt: float) -> TrackingResult:
        x, y, yaw = map(float, pose)
        if not all(math.isfinite(a) for a in (x, y, yaw, dt)) or dt <= 0:
            raise ValueError("Pose and positive dt must be finite")
        cfg = self.config
        self.elapsed += dt
        position = np.array([x, y])
        start = max(0, int(np.searchsorted(self.s, self.progress, side="right")) - 1)
        end_s = min(self.s[-1], self.progress + max(2 * cfg.lookahead_m, cfg.max_linear_mps * dt))
        best = (float("inf"), self.progress)
        for i in range(start, len(self.lengths)):
            if self.s[i] > end_s:
                break
            direction = self.path[i + 1] - self.path[i]
            fraction = float(np.dot(position - self.path[i], direction) / self.lengths[i] ** 2)
            s = float(np.clip(self.s[i] + fraction * self.lengths[i],
                              max(self.progress, self.s[i]), min(end_s, self.s[i + 1])))
            distance = float(np.linalg.norm(position - self._point(s)))
            if distance < best[0]:
                best = (distance, s)
        error, self.progress = best
        remaining = float(self.s[-1] - self.progress)
        target = self._point(min(self.s[-1], self.progress + cfg.lookahead_m))
        delta = target - position
        left = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
        distance_to_end = float(np.linalg.norm(position - self.path[-1]))
        if remaining <= cfg.arrival_distance_m and distance_to_end <= cfg.arrival_distance_m:
            self.finished = True
        status = "ARRIVED" if self.finished else "TRACKING"
        command = VelocityCommand()
        nominal = VelocityCommand()
        if status == "TRACKING":
            curvature = 2.0 * left / max(float(np.dot(delta, delta)), 1e-9)
            speed_limit = min(cfg.max_linear_mps,
                        math.sqrt(cfg.max_lateral_acceleration_mps2 / max(abs(curvature), 1e-9)),
                        cfg.max_angular_radps / max(abs(curvature), 1e-9),
                        1.5 * distance_to_end,
                        math.sqrt(2 * cfg.max_acceleration_mps2 * max(0, remaining)))
            nominal = VelocityCommand(speed_limit, speed_limit * curvature)
            speed = min(speed_limit, self.speed + cfg.max_acceleration_mps2 * dt)
            command = VelocityCommand(speed, speed * curvature)
            self.speed = speed
        else:
            self.speed = 0.0
        return TrackingResult(command, status, remaining, error, tuple(map(float, target)), nominal)
