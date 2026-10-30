from __future__ import annotations

import math
import time

import numpy as np

from constraint_traverse.reference_path import ReferencePath
from ..recording import inspect_record
from .verify_navigation import place


def turn_routes():
    angle = np.linspace(0, math.pi / 2, 41)
    quarter = np.column_stack((-1.6 + np.sin(angle), -0.5 + 1 - np.cos(angle)))
    exit_xy = np.column_stack((np.full(12, -0.6), np.linspace(0.55, 1.3, 12)))
    left = np.vstack((quarter, exit_xy))
    yaw = np.r_[angle, np.full(12, math.pi / 2)]
    right = left * [1, -1]
    first = np.column_stack((-1.5 + np.sin(angle), -0.5 + 1 - np.cos(angle)))
    second = np.column_stack((-0.5 + 1 - np.cos(angle[1:]), 0.5 + np.sin(angle[1:])))
    exit_s = np.column_stack((np.linspace(0.55, 1.3, 12), np.full(12, 1.5)))
    s_xy = np.vstack((first, second, exit_s))
    s_yaw = np.r_[angle, math.pi / 2 - angle[1:], np.zeros(12)]
    reverse = np.column_stack((np.linspace(-0.8, -1.6, 17), np.full(17, -0.5)))
    reversal = np.vstack((reverse, left[1:]))
    for name, points, headings, directions in (
        ("left_90", left, yaw, np.ones(len(left), dtype=np.int8)),
        ("right_90", right, -yaw, np.ones(len(right), dtype=np.int8)),
        ("s_turn", s_xy, s_yaw, np.ones(len(s_xy), dtype=np.int8)),
        ("reverse_turn", reversal, np.r_[np.zeros(len(reverse)), yaw[1:]],
         np.r_[np.full(len(reverse), -1, dtype=np.int8), np.ones(len(left) - 1, dtype=np.int8)])):
        directions[0] = 0
        yield name, points, headings, directions


def path_error(points, xy):
    delta = np.diff(points, axis=0)
    fraction = np.clip(((xy - points[:-1]) * delta).sum(-1) / np.maximum((delta * delta).sum(-1), 1e-12), 0, 1)
    return float(np.linalg.norm(xy - (points[:-1] + fraction[:, None] * delta), axis=-1).min())


def verify_turns(runtime):
    frames = 0
    runtime.reset_map()
    for x in (-2.8, 0.0, 2.8):
        for y in (-2.8, 0.0, 2.8):
            for yaw in (0, math.pi / 2, math.pi, -math.pi / 2):
                place(runtime, x, y, yaw)
                for _ in range(2):
                    runtime.observations = runtime.sensors.capture()
                    runtime.update_map(runtime.observations)
                    frames += len(runtime.observations)
    results = {}
    for name, points, headings, directions in turn_routes():
        place(runtime, *points[0], headings[0])
        for _ in range(60):
            runtime.step([0, 0])
        runtime.goal = np.r_[points[-1], 0]
        runtime.experiment.metadata["test_scene"] = dict(name=name, reference_source="FIXED_TRACKING_DIAGNOSTIC",
            map_source="REAL_RGBD", unknown_policy="blocked")
        initial = runtime.experiment.save_start()
        results[name] = []
        for repeat in range(runtime.args.repeats):
            runtime.experiment.saved = initial
            runtime.experiment.reset()
            runtime.start_navigation()
            navigator = runtime.navigator
            reference = ReferencePath(points, headings, directions, points[-1], "FIXED_TRACKING_DIAGNOSTIC",
                repeat + 1, runtime.timestamp, runtime.local_map.version)
            grid = runtime.local_map.snapshot().grid
            navigator._check_poses(grid, np.column_stack((points, headings)))
            navigator.path = reference
            navigator.signature = reference.map_signature(grid, radius_m=navigator.path_planner.checker.bounding_radius_m)
            navigator.events.append(("REFERENCE_PATH", dict(path=reference, diagnostics=dict(fixed_diagnostic=True))))
            maximum_error = 0.0
            direction_violations = 0
            esses = []
            seen = None
            started = time.monotonic()
            while runtime.experiment.recorder is not None:
                runtime.step()
                state = navigator.sampling.read_state(runtime.robot, runtime.drive)
                maximum_error = max(maximum_error, path_error(points, state[:2]))
                if navigator.sampling.reference is not None:
                    index = min(navigator.sampling.reference_cursor, len(directions) - 1)
                    direction = directions[index]
                    direction_violations += int(runtime.requested[0] * direction < -1e-5
                        or (direction != 0 and abs(runtime.requested[0]) < 1e-6 and abs(runtime.requested[1]) > 1e-5))
                plan = navigator.sampling.last_plan
                if plan is not None and plan is not seen:
                    esses.append(plan.diagnostics["ess"])
                    seen = plan
                if runtime.panel is not None:
                    runtime.panel.update(runtime)
                if time.monotonic() - started > 600:
                    runtime.experiment.stop("WALL_TIMEOUT")
            record = inspect_record(runtime.experiment.last_record)
            result = dict(status=record["status"], maximum_lateral_error_m=maximum_error,
                direction_violations=direction_violations, ess_median=float(np.median(esses)) if esses else 0,
                records=record["verified_counts"])
            result["passed"] = record["status"] == "SUCCESS" and maximum_error <= 0.05 and direction_violations == 0
            results[name].append(result)
            runtime.logger.info("[TurnAcceptance] route=%s repeat=%d result=%s", name, repeat + 1, result)
    return dict(turns=results, actual_rgbd_frames=frames,
                passed=all(result["passed"] for values in results.values() for result in values))
