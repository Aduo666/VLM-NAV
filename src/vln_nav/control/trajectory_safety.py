"""RGB-D acceptance safety: raw occupancy + swept rectangular body.

No VLM, obstacle oracle, inflation, or autonomous steering/recovery. This is a
static-scene experimental guard, not a safety-rated controller. World XY and
yaw use the same fixed frame as the camera projection and path tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from vln_nav.control.trajectory import VelocityCommand


@dataclass(frozen=True)
class TrajectorySafetyConfig:
    front_m: float
    rear_m: float
    half_width_m: float
    margin_m: float
    sensor_timeout_sec: float
    wall_timeout_sec: float
    reaction_time_sec: float
    preview_time_sec: float
    braking_deceleration_mps2: float
    braking_deceleration_radps2: float
    release_good_frames: int
    minimum_motion_mps: float
    min_valid_depth_ratio: float
    hold_log_period_sec: float

    @classmethod
    def from_mapping(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError("Invalid AGX trajectory safety configuration fields")
        cfg = cls(**{key: float(value) for key, value in data.items()})
        for f in fields(cls):
            value = getattr(cfg, f.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"safety.{f.name} must be finite and positive")
        if cfg.min_valid_depth_ratio > 1 or int(cfg.release_good_frames) != cfg.release_good_frames:
            raise ValueError("Invalid depth ratio or release frame count")
        return cfg


@dataclass(frozen=True)
class SafetyDecision:
    command: VelocityCommand
    status: str
    reason: str
    scale: float
    sensor_age_sec: float
    sweep: np.ndarray
    blocker_world_xy: tuple[float, float] | None = None


def footprint_cells(snapshot, pose, config, *, margin=True):
    """Conservative cell/rectangle overlap (cell half-diagonal added once)."""
    x, y, yaw = map(float, pose)
    res = float(snapshot.resolution_m)
    pad = (config.margin_m if margin else 0.0) + res / math.sqrt(2)
    radius = math.hypot(max(config.front_m, config.rear_m) + pad, config.half_width_m + pad)
    cols = np.arange(math.floor((x - radius - snapshot.origin_x) / res),
                     math.ceil((x + radius - snapshot.origin_x) / res) + 1)
    rows = np.arange(math.floor((y - radius - snapshot.origin_y) / res),
                     math.ceil((y + radius - snapshot.origin_y) / res) + 1)
    cc, rr = np.meshgrid(cols, rows)
    dx = snapshot.origin_x + (cc + .5) * res - x
    dy = snapshot.origin_y + (rr + .5) * res - y
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    keep = ((forward <= config.front_m + pad) & (forward >= -config.rear_m - pad)
            & (np.abs(left) <= config.half_width_m + pad))
    return rr[keep], cc[keep]


def stopping_sweep(pose, command, config, resolution_m, *, sensor_age_sec=0.0, preview=True):
    """Sample reaction + optional preview + proportional braking, including corners."""
    v, w = command.linear_mps, command.angular_radps
    if abs(v) + abs(w) < 1e-10:
        return np.asarray([pose], dtype=float)
    radius = math.hypot(max(config.front_m, config.rear_m), config.half_width_m) + config.margin_m
    dt = min(.05, resolution_m / (4 * max(abs(v) + abs(w) * radius, .01)))
    reaction = config.reaction_time_sec + max(0.0, sensor_age_sec)
    if preview:
        reaction += config.preview_time_sec
    brake = max(abs(v) / config.braking_deceleration_mps2,
                abs(w) / config.braking_deceleration_radps2)
    state = np.asarray(pose, dtype=float).copy()
    result = [state.copy()]
    t = 0.0
    while t < reaction + brake:
        step = min(dt, reaction + brake - t)
        factor = 1.0 if t < reaction else max(0.0, 1.0 - (t - reaction) / max(brake, 1e-9))
        theta = w * factor * step
        if abs(w) > 1e-9:
            state[0] += (v / w) * (math.sin(state[2] + theta) - math.sin(state[2]))
            state[1] -= (v / w) * (math.cos(state[2] + theta) - math.cos(state[2]))
        else:
            state[:2] += v * factor * step * np.array([math.cos(state[2]), math.sin(state[2])])
        state[2] += theta
        result.append(state.copy())
        t += step
    return np.asarray(result)


class TrajectorySafetySupervisor:
    def __init__(self, config):
        self.config = config
        self.snapshot = None
        self.timestamp = self.wall_timestamp = None
        self.seed_pose = None
        self.blocked = False
        self.fresh_frames = 0
        self.good_frames = 0
        self.frame_number = 0
        self.checked_frame = -1
        self.blocker_xy = None

    def reset(self, initial_pose):
        # Empty-floor test bootstrap ONLY: the initial placement plus margin must
        # be verified clear by the operator. Never seed new space as the car moves.
        self.__init__(self.config)
        self.seed_pose = tuple(initial_pose)

    def update(self, snapshot, *, timestamp, wall_timestamp):
        if self.timestamp is not None and timestamp < self.timestamp:
            return
        if self.timestamp is None or timestamp > self.timestamp:
            if self.timestamp is None or timestamp - self.timestamp > self.config.sensor_timeout_sec or wall_timestamp - self.wall_timestamp > self.config.wall_timeout_sec:
                self.fresh_frames = 0
            self.fresh_frames += 1
            self.frame_number += 1
        self.snapshot = snapshot
        self.timestamp, self.wall_timestamp = float(timestamp), float(wall_timestamp)
        if self.seed_pose is not None:
            rows, cols = footprint_cells(snapshot, self.seed_pose, self.config)
            inside = (rows >= 0) & (rows < snapshot.free_mask.shape[0]) & (cols >= 0) & (cols < snapshot.free_mask.shape[1])
            rr, cc = rows[inside], cols[inside]
            snapshot.free_mask[rr, cc] |= ~snapshot.occupied_mask[rr, cc]

    def _check(self, sweep):
        snapshot = self.snapshot
        self.blocker_xy = None
        for pose in sweep:
            rows, cols = footprint_cells(snapshot, pose, self.config)
            if ((rows < 0).any() or (cols < 0).any() or (rows >= snapshot.free_mask.shape[0]).any()
                    or (cols >= snapshot.free_mask.shape[1]).any()):
                return "outside_map"
            if snapshot.occupied_mask[rows, cols].any():
                index = np.flatnonzero(snapshot.occupied_mask[rows, cols])[0]
                self.blocker_xy = (snapshot.origin_x + (cols[index] + .5) * snapshot.resolution_m,
                                   snapshot.origin_y + (rows[index] + .5) * snapshot.resolution_m)
                return "obstacle_in_sweep"
            if not snapshot.free_mask[rows, cols].all():
                index = np.flatnonzero(~snapshot.free_mask[rows, cols])[0]
                self.blocker_xy = (snapshot.origin_x + (cols[index] + .5) * snapshot.resolution_m,
                                   snapshot.origin_y + (rows[index] + .5) * snapshot.resolution_m)
                return "unknown_in_sweep"
        return "clear"

    def evaluate(self, command, *, pose, measured, now, wall_now, probe_command=None):
        # Probe the unconstrained tracking speed, not last frame's already-scaled
        # acceleration ramp. Otherwise repeated scaling converges to zero motion.
        probe = command if probe_command is None else probe_command
        empty = np.asarray([pose], dtype=float)
        age = float("inf") if self.timestamp is None else now - self.timestamp

        def hold(reason, sweep=empty, blocker=None):
            self.blocked, self.good_frames = True, 0
            self.checked_frame = self.frame_number
            return SafetyDecision(VelocityCommand(), "HOLD", reason, 0.0, age, sweep, blocker)

        if not all(math.isfinite(x) for x in [*pose, command.linear_mps, command.angular_radps,
                                             measured.linear_mps, measured.angular_radps,
                                             probe.linear_mps, probe.angular_radps, now, wall_now]):
            return hold("invalid_motion")
        if max(abs(command.linear_mps), abs(command.angular_radps), abs(measured.linear_mps), abs(measured.angular_radps), abs(probe.linear_mps), abs(probe.angular_radps)) > 5:
            return hold("motion_out_of_bounds")
        if self.snapshot is None:
            return hold("waiting_rgbd")
        if age > self.config.sensor_timeout_sec or age < -.01 or wall_now - self.wall_timestamp > self.config.wall_timeout_sec:
            self.fresh_frames = 0
            return hold("stale_rgbd")
        if self.fresh_frames < self.config.release_good_frames:
            return SafetyDecision(VelocityCommand(), "HOLD", "waiting_stable_rgbd", 0.0, age, empty)
        # Measured inertia can be greater than the new command; do not only check
        # a near-zero desired velocity while the robot is still moving fast.
        current_sweep = stopping_sweep(pose, measured, self.config, self.snapshot.resolution_m,
                                      sensor_age_sec=age, preview=False)
        reason = self._check(current_sweep)
        if reason != "clear":
            return hold("measured_" + reason, current_sweep, self.blocker_xy)
        rejected_reason = "clear"
        rejected_sweep = current_sweep
        rejected_xy = None
        radius = math.hypot(max(self.config.front_m, self.config.rear_m), self.config.half_width_m)
        probe_motion = abs(probe.linear_mps) + radius * abs(probe.angular_radps)
        commanded_motion = abs(command.linear_mps) + radius * abs(command.angular_radps)
        if probe_motion > 1e-9 and (
            abs(probe.linear_mps * command.angular_radps - probe.angular_radps * command.linear_mps) > 1e-6
            or probe.linear_mps * command.linear_mps + probe.angular_radps * command.angular_radps < -1e-9
        ):
            return hold("probe_command_direction_mismatch")
        for scale in (1.0, .75, .5, .25):
            if 0 < probe_motion * scale < self.config.minimum_motion_mps:
                continue
            candidate = VelocityCommand(probe.linear_mps * scale, probe.angular_radps * scale)
            # Use the full permitted sensor-age budget, keeping geometric
            # acceptance stable as a fresh frame ages between control ticks.
            sweep = stopping_sweep(pose, candidate, self.config, self.snapshot.resolution_m,
                                   sensor_age_sec=self.config.sensor_timeout_sec)
            reason = self._check(sweep)
            if reason != "clear":
                if rejected_reason == "clear":
                    rejected_reason, rejected_sweep, rejected_xy = reason, sweep, self.blocker_xy
                continue
            if self.blocked:
                if self.checked_frame != self.frame_number:
                    self.good_frames += 1
                    self.checked_frame = self.frame_number
                if self.good_frames < self.config.release_good_frames:
                    return SafetyDecision(VelocityCommand(), "HOLD", "waiting_clear_path", 0.0, age, sweep)
                self.blocked = False
            factor = min(1.0, probe_motion * scale / max(commanded_motion, 1e-12))
            applied = VelocityCommand(command.linear_mps * factor, command.angular_radps * factor)
            return SafetyDecision(applied, "CLEAR" if scale == 1 else "SLOW",
                                  rejected_reason, scale, age, sweep, rejected_xy)
        return hold(rejected_reason if rejected_reason != "clear" else "no_useful_safe_speed", rejected_sweep, rejected_xy)
