from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import uuid

import numpy as np


@dataclass
class TrialSnapshot:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    last_command: np.ndarray
    goal: np.ndarray
    seed: int

    @classmethod
    def capture(cls, robot, last_command, goal, seed):
        position, orientation = robot.get_world_pose()
        return cls(*[np.asarray(value).copy() for value in (
            position, orientation, robot.get_linear_velocity(), robot.get_angular_velocity(),
            robot.get_joint_positions(), robot.get_joint_velocities(), last_command, goal,
        )], int(seed))

    def restore(self, robot):
        robot.set_world_pose(position=self.position, orientation=self.orientation)
        robot.set_joint_positions(self.joint_positions)
        robot.set_joint_velocities(self.joint_velocities)
        robot.set_linear_velocity(self.linear_velocity)
        robot.set_angular_velocity(self.angular_velocity)

    def errors(self, robot):
        restored = TrialSnapshot.capture(robot, self.last_command, self.goal, self.seed)
        return {
            name: float(np.max(np.abs(getattr(restored, name) - getattr(self, name))))
            for name in ("position", "orientation", "linear_velocity", "angular_velocity", "joint_positions", "joint_velocities")
        }

    def as_dict(self):
        return {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in self.__dict__.items()}


class TrialRecord:
    def __init__(self, output_root: Path, metadata: dict):
        trial_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.path = Path(output_root) / trial_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "plans").mkdir()
        self.metadata = metadata
        self.samples = []
        self.plan_stats = []
        self.contacts = []
        self._write_meta({"status": "RUNNING"})

    def add_plan(self, result, timestamp_sec):
        index = len(self.plan_stats)
        np.savez_compressed(
            self.path / "plans" / f"{index:06d}.npz",
            timestamp_sec=timestamp_sec, command=result.command, controls=result.controls,
            prediction=result.prediction, candidates=result.candidates,
            candidate_controls=result.candidate_controls, candidate_costs=result.candidate_costs,
        )
        self.plan_stats.append({
            "time_sec": float(timestamp_sec), "valid": bool(result.valid), "reason": result.reason,
            "min_clearance_m": float(result.min_clearance_m),
            "planning_sec": float(result.planning_sec), "optimizer_sec": float(result.optimizer_sec),
        })

    def add_sample(self, timestamp_sec, state, requested, applied, clearance_m):
        self.samples.append((float(timestamp_sec), np.asarray(state).copy(), np.asarray(requested).copy(),
                             np.asarray(applied).copy(), float(clearance_m)))

    def finish(self, reason, *, passed, goal_reached, contacts, reset_errors=None):
        self.contacts = list(contacts)
        if self.samples:
            timestamps, states, requested, applied, clearance = zip(*self.samples)
            np.savez_compressed(self.path / "state.npz", time_sec=timestamps, state=states,
                                requested_command=requested, applied_command=applied, clearance_m=clearance)
        else:
            np.savez_compressed(self.path / "state.npz", time_sec=np.empty(0), state=np.empty((0, 17)),
                                requested_command=np.empty((0, 2)), applied_command=np.empty((0, 2)), clearance_m=np.empty(0))
        latencies = np.asarray([row["planning_sec"] for row in self.plan_stats])
        metrics = {
            "status": reason, "success": reason == "SUCCESS", "completely_passed": bool(passed),
            "goal_reached": bool(goal_reached), "collision": bool(contacts),
            "duration_sec": float(self.samples[-1][0]) if self.samples else 0.0,
            "sample_count": len(self.samples), "planning_cycles": len(self.plan_stats),
            "min_clearance_m": min((row[-1] for row in self.samples), default=None),
            "planning_mean_sec": float(latencies.mean()) if latencies.size else None,
            "planning_p95_sec": float(np.percentile(latencies, 95)) if latencies.size else None,
            "planning_deadline_misses": int(np.sum(latencies > self.metadata["config"]["planner"]["dt_sec"])),
            "reset_errors": reset_errors,
        }
        self._write_meta(metrics)
        with (self.path / "events.json").open("w", encoding="utf-8") as handle:
            json.dump({"plans": self.plan_stats, "contacts": contacts}, handle, indent=2, ensure_ascii=False)
        return metrics

    def _write_meta(self, metrics):
        with (self.path / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump({**self.metadata, **metrics}, handle, indent=2, ensure_ascii=False, allow_nan=False)
