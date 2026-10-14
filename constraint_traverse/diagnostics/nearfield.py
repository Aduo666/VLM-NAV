from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from PIL import Image, ImageDraw

from vln_nav.active_perception.camera_geometry import CameraFrameGeometry
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap, LocalVoxelMapConfig
from vln_nav.navigation.fine_local_planner import FineLocalPlanner, FinePlannerConfig
from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    PlanningGrid,
    clear_current_footprint,
    planning_grid_from_snapshot,
)


def build_stride_comparison(
    *,
    frame_pairs: list[tuple[CameraFrameGeometry, CameraFrameGeometry]],
    map_config: LocalVoxelMapConfig,
    fine_config: FinePlannerConfig,
    collision_checker: FootprintCollisionChecker,
    robot_world_xyz: np.ndarray,
    robot_yaw_rad: float,
    floor_z_m: float,
    obstacle_min_height_m: float,
    obstacle_max_height_m: float,
) -> tuple[dict[int, dict], dict]:
    if not frame_pairs:
        raise ValueError("至少需要一组同步的前视与底盘下视深度帧")
    robot = np.asarray(robot_world_xyz, dtype=np.float64).reshape(3)
    results = {}
    for stride in (10, 5):
        voxel_map = LocalVoxelMap(replace(map_config, pixel_stride=stride))
        voxel_map.reset(robot)
        updates = []
        for front, down in frame_pairs:
            if front.source != "front" or down.source != "down":
                raise ValueError("建图只接受前视与底盘下视相机")
            observations = [
                TimedDepthObservation(
                    source=frame.source,
                    timestamp_sec=frame.timestamp_sec,
                    depth_m=frame.depth_m,
                    intrinsics=frame.intrinsics,
                    camera_to_world=frame.world_from_camera,
                )
                for frame in (front, down)
            ]
            update = voxel_map.integrate_batch(
                observations,
                center_world_xyz=robot,
            )
            updates.append({
                "version": update.version,
                "rays": dict(update.source_ray_counts),
                "source_voxel_updates": {
                    source: {"free": free, "occupied": occupied}
                    for source, free, occupied
                    in update.source_voxel_update_counts
                },
            })
        snapshot = voxel_map.snapshot()
        raw_grid = planning_grid_from_snapshot(
            snapshot,
            floor_z_m=floor_z_m,
            obstacle_min_height_m=obstacle_min_height_m,
            obstacle_max_height_m=obstacle_max_height_m,
        )
        planning_grid = clear_current_footprint(
            raw_grid,
            x_m=float(robot[0]),
            y_m=float(robot[1]),
            yaw_rad=float(robot_yaw_rad),
            collision_checker=collision_checker,
        )
        results[stride] = {
            "snapshot": snapshot,
            "raw_grid": raw_grid,
            "planning_grid": planning_grid,
            "updates": updates,
        }
    start = (float(robot[0]), float(robot[1]), float(robot_yaw_rad))
    planner = FineLocalPlanner(fine_config, collision_checker)
    actions = planner.inspect_initial_actions(
        grid=results[10]["planning_grid"],
        start_pose_world=start,
    )
    straight = next(
        action for action in actions if action.action_name == "forward_2"
    )
    physical_unknown = set(straight.physical_unknown_linear_indices)
    cells = []
    grid10 = results[10]["planning_grid"]
    for linear in straight.unknown_linear_indices:
        row, column = divmod(int(linear), grid10.shape[1])
        world_x = grid10.origin_x_m + (column + 0.5) * grid10.resolution_m
        world_y = grid10.origin_y_m + (row + 0.5) * grid10.resolution_m
        row5, column5, inside5 = results[5]["planning_grid"].world_to_cell(
            world_x, world_y
        )
        grid5 = results[5]["planning_grid"]
        cell = {
            "linear_index_stride_10": int(linear),
            "world_xy_m": [world_x, world_y],
            "footprint_layer": (
                "physical" if linear in physical_unknown else "safety_margin"
            ),
            "stride_10": grid_cell_state(grid10, row, column),
            "stride_5": (
                grid_cell_state(grid5, row5, column5)
                if inside5 else "OUTSIDE_MAP"
            ),
            "source_bits_stride_10": source_bits_for_cell(
                results[10]["snapshot"],
                row,
                column,
                floor_z_m=floor_z_m,
                min_height_m=obstacle_min_height_m,
                max_height_m=obstacle_max_height_m,
            ),
            "source_bits_stride_5": source_bits_for_cell(
                results[5]["snapshot"],
                row5,
                column5,
                floor_z_m=floor_z_m,
                min_height_m=obstacle_min_height_m,
                max_height_m=obstacle_max_height_m,
            ) if inside5 else 0,
        }
        cell["camera_projection"] = {
            frame.source: project_cell_to_depth(
                frame,
                world_x=world_x,
                world_y=world_y,
                floor_z_m=floor_z_m,
            )
            for frame in frame_pairs[-1]
        }
        cells.append(cell)
    action_records = [
        {
            "name": action.action_name,
            "result": action.result,
            "endpoint_world_xyyaw": list(action.endpoint_world_xyyaw),
            "unknown_cells": len(action.unknown_linear_indices),
            "unknown_area_m2": action.unknown_area_m2,
            "part_unknown_counts": dict(action.part_unknown_counts),
            "physical_unknown_count": action.physical_unknown_count,
            "safety_margin_unknown_count": action.safety_margin_unknown_count,
            "physical_collision": action.physical_collision,
            "safety_margin_collision": action.safety_margin_collision,
        }
        for action in actions
    ]
    return results, {
        "cells": cells,
        "actions": action_records,
        "physical_unknown_count": len(physical_unknown),
        "margin_only_unknown_count": len(cells) - len(physical_unknown),
    }


def newly_covered_straight_cells(
    *,
    grid: PlanningGrid,
    checker: FootprintCollisionChecker,
    start_pose_world: tuple[float, float, float],
    distance_m: float,
    sample_step_m: float,
    margin_m: float,
) -> np.ndarray:
    x, y, yaw = map(float, start_pose_world)
    initial, _ = checker.covered_linear_indices(
        grid,
        x_m=x,
        y_m=y,
        yaw_rad=yaw,
        margin_m=margin_m,
        clearing_only=True,
    )
    steps = max(1, int(math.ceil(distance_m / sample_step_m)))
    swept = []
    for distance in np.linspace(distance_m / steps, distance_m, steps):
        cells, _ = checker.covered_linear_indices(
            grid,
            x_m=x + distance * math.cos(yaw),
            y_m=y + distance * math.sin(yaw),
            yaw_rad=yaw,
            margin_m=margin_m,
        )
        swept.append(cells)
    return np.setdiff1d(
        np.unique(np.concatenate(swept)),
        initial,
        assume_unique=True,
    )


def grid_cell_state(grid: PlanningGrid, row: int, column: int) -> str:
    if grid.occupied_mask[row, column]:
        return "OCCUPIED"
    if grid.free_mask[row, column]:
        return "FREE"
    return "UNKNOWN"


def source_bits_for_cell(
    snapshot,
    row: int,
    column: int,
    *,
    floor_z_m: float,
    min_height_m: float,
    max_height_m: float,
) -> int:
    centers = snapshot.origin_world_xyz[2] + (
        np.arange(snapshot.state.shape[0], dtype=np.float64) + 0.5
    ) * snapshot.resolution_m
    band = (
        (centers >= floor_z_m + min_height_m)
        & (centers <= floor_z_m + max_height_m)
    )
    return int(np.bitwise_or.reduce(snapshot.source_mask[band, row, column]))


def project_cell_to_depth(
    frame: CameraFrameGeometry,
    *,
    world_x: float,
    world_y: float,
    floor_z_m: float,
) -> list[dict]:
    camera_from_world = np.linalg.inv(frame.world_from_camera)
    rows, columns = frame.depth_m.shape
    samples = []
    for height in (0.05, 0.15, 0.30, 0.60):
        point = camera_from_world @ np.asarray([
            world_x, world_y, floor_z_m + height, 1.0
        ])
        depth = float(point[2])
        record = {"relative_height_m": height, "camera_depth_m": depth}
        if depth <= 0:
            record["visibility"] = "BEHIND_CAMERA"
        else:
            u = float(frame.intrinsics[0, 0] * point[0] / depth
                      + frame.intrinsics[0, 2])
            v = float(frame.intrinsics[1, 1] * point[1] / depth
                      + frame.intrinsics[1, 2])
            record["pixel_uv"] = [u, v]
            if not (0 <= u < columns and 0 <= v < rows):
                record["visibility"] = "OUTSIDE_IMAGE"
            else:
                sampled = float(frame.depth_m[int(v), int(u)])
                record["measured_depth_m"] = sampled
                if not math.isfinite(sampled):
                    record["visibility"] = "INVALID_DEPTH"
                elif sampled < depth - 0.05:
                    record["visibility"] = "NEARER_SURFACE"
                else:
                    record["visibility"] = "DEPTH_REACHES_POINT"
        samples.append(record)
    return samples


def render_comparison(
    results: dict[int, dict],
    cells: list[dict],
    *,
    robot_world_xy: tuple[float, float],
) -> Image.Image:
    grid10 = results[10]["raw_grid"]
    grid5 = results[5]["raw_grid"]
    cell_size = 4
    panels = []
    for stride, grid in ((10, grid10), (5, grid5)):
        rgb = np.full((*grid.shape, 3), (100, 103, 107), dtype=np.uint8)
        rgb[grid.free_mask] = (230, 231, 222)
        rgb[grid.occupied_mask] = (38, 42, 48)
        for cell in cells:
            row, column, inside = grid.world_to_cell(*cell["world_xy_m"])
            if inside:
                rgb[row, column] = (
                    (235, 69, 124)
                    if cell["footprint_layer"] == "physical"
                    else (241, 163, 69)
                )
        robot_row, robot_column, inside = grid.world_to_cell(*robot_world_xy)
        if inside:
            rgb[robot_row, robot_column] = (64, 112, 199)
        panel = Image.fromarray(np.flipud(rgb), mode="RGB").resize(
            (grid.shape[1] * cell_size, grid.shape[0] * cell_size),
            Image.NEAREST,
        )
        panels.append(panel)
    width, height = panels[0].size
    image = Image.new("RGB", (2 * width, height + 42), (31, 35, 40))
    image.paste(panels[0], (0, 30))
    image.paste(panels[1], (width, 30))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "RGB-D stride=10", fill=(240, 240, 240))
    draw.text((width + 8, 8), "RGB-D stride=5", fill=(240, 240, 240))
    draw.text(
        (8, height + 33),
        "PINK physical unknown / ORANGE margin-only / BLUE robot",
        fill=(240, 240, 240),
    )
    return image
