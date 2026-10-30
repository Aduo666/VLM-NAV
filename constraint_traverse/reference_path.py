from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


def angle_difference(target, current):
    return (np.asarray(target) - current + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class ReferenceWindow:
    world_xy: np.ndarray
    world_yaw_rad: np.ndarray
    direction: int
    cursor: int
    reaches_path_end: bool

    @property
    def length_m(self):
        return float(np.linalg.norm(np.diff(self.world_xy, axis=0), axis=1).sum())


@dataclass(frozen=True)
class ReferencePath:
    world_xy: np.ndarray
    world_yaw_rad: np.ndarray
    motion_direction: np.ndarray
    goal_world_xy: np.ndarray
    source: str
    request_id: int
    created_sim_time_sec: float
    map_version: int
    partial: bool = False

    def __post_init__(self):
        directions = np.asarray(self.motion_direction)
        if not np.isin(directions, [-1, 0, 1]).all():
            raise ValueError("参考路径方向必须为前进、后退或原地旋转")
        arrays = {
            "world_xy": np.asarray(self.world_xy, dtype=np.float64).copy(),
            "world_yaw_rad": np.asarray(self.world_yaw_rad, dtype=np.float64).copy(),
            "motion_direction": np.asarray(self.motion_direction, dtype=np.int8).copy(),
            "goal_world_xy": np.asarray(self.goal_world_xy, dtype=np.float64).copy(),
        }
        points = arrays["world_xy"]
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("参考路径需要至少两个二维位置")
        if arrays["world_yaw_rad"].shape != (len(points),) or arrays["motion_direction"].shape != (len(points),):
            raise ValueError("参考路径的位置、朝向和方向长度必须一致")
        if arrays["goal_world_xy"].shape != (2,) or not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("参考路径和目标必须包含有限坐标")
        if not np.isin(arrays["motion_direction"], [-1, 0, 1]).all():
            raise ValueError("参考路径方向必须为前进、后退或原地旋转")
        if not math.isfinite(self.created_sim_time_sec):
            raise ValueError("参考路径时间必须为有限数值")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_positions(cls, world_xy, *, initial_yaw_rad, **metadata):
        points = np.asarray(world_xy, dtype=np.float64)
        differences = np.diff(points, axis=0)
        lengths = np.linalg.norm(differences, axis=1)
        moving = np.flatnonzero(lengths > 1.0e-6)
        if not len(moving):
            raise ValueError("二维参考路径没有有效运动段")
        first = differences[moving[0]]
        direction = 1 if first @ np.asarray([math.cos(initial_yaw_rad), math.sin(initial_yaw_rad)]) >= 0 else -1
        headings = np.empty(len(points))
        headings[0] = initial_yaw_rad
        directions = np.zeros(len(points), dtype=np.int8)
        for index, delta in enumerate(differences, 1):
            headings[index] = (math.atan2(delta[1], delta[0]) + (math.pi if direction < 0 else 0)
                               if lengths[index - 1] > 1.0e-6 else headings[index - 1])
            directions[index] = direction if lengths[index - 1] > 1.0e-6 else 0
        return cls(points, headings, directions, **metadata)

    def window(self, pose, *, lookahead_m, cursor=1, position_tolerance_m=0.04,
               yaw_tolerance_rad=math.radians(5), stopped=True):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(lookahead_m) or lookahead_m <= 0:
            raise ValueError("前瞻路径需要有效位姿和正数距离")
        cursor = max(1, int(cursor))
        if cursor < len(self.world_xy) and self.motion_direction[cursor] != 0:
            direction = self.motion_direction[cursor]
            best, distance_ahead = float("inf"), 0.0
            nearest = cursor
            for index in range(cursor, len(self.world_xy)):
                if self.motion_direction[index] != direction or distance_ahead > 0.5:
                    break
                start, end = self.world_xy[index - 1:index + 1]
                delta = end - start
                fraction = float(np.clip((pose[:2] - start) @ delta / max(delta @ delta, 1e-12), 0, 1))
                projected = start + fraction * delta
                distance = float(np.linalg.norm(pose[:2] - projected))
                heading = self.world_yaw_rad[index - 1] + fraction * float(angle_difference(
                    self.world_yaw_rad[index], self.world_yaw_rad[index - 1]))
                if distance < best and abs(float(angle_difference(heading, pose[2]))) < math.pi / 3:
                    best, nearest = distance, index
                distance_ahead += float(np.linalg.norm(delta)) * (1 - fraction)
            cursor = nearest
        while cursor < len(self.world_xy):
            start, end = self.world_xy[cursor - 1:cursor + 1]
            segment = end - start
            length = float(np.linalg.norm(segment))
            direction = int(self.motion_direction[cursor])
            if direction == 0:
                reached = (np.linalg.norm(pose[:2] - end) <= position_tolerance_m
                           and abs(float(angle_difference(self.world_yaw_rad[cursor], pose[2]))) <= yaw_tolerance_rad)
            else:
                along = float((pose[:2] - start) @ segment / max(length * length, 1.0e-12))
                lateral = np.linalg.norm(pose[:2] - (start + along * segment))
                reached = (np.linalg.norm(pose[:2] - end) <= position_tolerance_m
                           or (along >= 1 and lateral <= 2 * position_tolerance_m))
            if not reached:
                break
            if cursor + 1 < len(self.world_xy) and self.motion_direction[cursor + 1] != direction and not stopped:
                break
            cursor += 1
        if cursor >= len(self.world_xy):
            return None
        direction = int(self.motion_direction[cursor])
        if direction == 0:
            return ReferenceWindow(self.world_xy[cursor - 1:cursor + 1],
                                   self.world_yaw_rad[cursor - 1:cursor + 1], 0, cursor,
                                   cursor == len(self.world_xy) - 1)
        start, end = self.world_xy[cursor - 1:cursor + 1]
        delta = end - start
        fraction = float(np.clip((pose[:2] - start) @ delta / max(delta @ delta, 1.0e-12), 0, 1))
        points = [start + fraction * delta]
        headings = [self.world_yaw_rad[cursor - 1] + fraction * float(angle_difference(
            self.world_yaw_rad[cursor], self.world_yaw_rad[cursor - 1]))]
        remaining = float(lookahead_m)
        last_index = cursor - 1
        for index in range(cursor, len(self.world_xy)):
            if self.motion_direction[index] != direction:
                break
            distance = float(np.linalg.norm(self.world_xy[index] - points[-1]))
            ratio = min(1.0, remaining / max(distance, 1.0e-12))
            points.append(points[-1] + ratio * (self.world_xy[index] - points[-1]))
            headings.append(headings[-1] + ratio * float(angle_difference(self.world_yaw_rad[index], headings[-1])))
            remaining -= distance
            last_index = index
            if remaining <= 0:
                break
        return ReferenceWindow(np.asarray(points), np.asarray(headings), direction, cursor,
                               last_index == len(self.world_xy) - 1 and remaining >= 0)

    def needs_extension(self, pose, *, lookahead_m):
        window = self.window(pose, lookahead_m=lookahead_m)
        return window is None or (window.reaches_path_end and window.length_m < 0.5 * lookahead_m)

    def map_signature(self, grid, *, radius_m):
        minimum = self.world_xy.min(axis=0) - radius_m
        maximum = self.world_xy.max(axis=0) + radius_m
        first = np.floor((minimum - [grid.origin_x_m, grid.origin_y_m]) / grid.resolution_m).astype(int)
        last = np.ceil((maximum - [grid.origin_x_m, grid.origin_y_m]) / grid.resolution_m).astype(int)
        x0, y0 = max(0, first[0]), max(0, first[1])
        x1, y1 = min(grid.shape[1], last[0] + 1), min(grid.shape[0], last[1] + 1)
        digest = hashlib.blake2b(digest_size=16)
        for mask in (grid.occupied_mask, grid.unknown_mask):
            rows, columns = np.nonzero(mask[y0:y1, x0:x1])
            world_cells = np.column_stack((columns + x0 + round(grid.origin_x_m / grid.resolution_m),
                                           rows + y0 + round(grid.origin_y_m / grid.resolution_m))).astype(np.int64)
            digest.update(np.asarray([len(world_cells)], dtype=np.int64).tobytes())
            digest.update(world_cells.tobytes())
        return digest.digest()
