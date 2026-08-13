from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .geometry import world_to_robot_xy


@dataclass(frozen=True)
class FrontierExtractorConfig:
    min_cluster_cells: int = 6
    max_frontiers: int = 24
    information_radius_m: float = 1.0
    max_distance_m: float = 12.0
    persistent_match_radius_m: float = 0.75
    approach_offset_m: float = 0.40
    approach_search_radius_m: float = 0.80


@dataclass(frozen=True)
class Frontier:
    frontier_id: int
    boundary_world_xy: tuple[float, float]
    navigation_goal_world_xy: tuple[float, float]
    boundary_cell_rc: tuple[int, int]
    navigation_goal_cell_rc: tuple[int, int]
    cell_count: int
    information_gain: int
    distance_m: float
    bearing_rad: float
    local_forward_m: float
    local_left_m: float
    path_cost_m: float
    obstacle_clearance_m: float
    feature: tuple[float, ...]


class FrontierExtractor:
    """Extract safe and reachable free/unknown boundaries.

    A frontier exposes two different points:

    * ``boundary_world_xy`` is an actual known-free cell adjacent to unknown.
    * ``navigation_goal_world_xy`` is a reachable free cell moved inward from
      that boundary and outside the inflated obstacle mask.

    Keeping those meanings separate prevents a connected-component centroid
    from being treated as a traversable navigation goal.
    """

    def __init__(self, config: FrontierExtractorConfig | None = None) -> None:
        self.config = config or FrontierExtractorConfig()

    def extract(
        self,
        *,
        occupied_mask: np.ndarray,
        free_mask: np.ndarray,
        inflated_occupied_mask: np.ndarray,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
        robot_pose: tuple[float, float, float],
    ) -> list[Frontier]:
        occupied = np.asarray(occupied_mask, dtype=bool)
        free = np.asarray(free_mask, dtype=bool)
        inflated = np.asarray(inflated_occupied_mask, dtype=bool)
        if occupied.shape != free.shape or occupied.shape != inflated.shape or occupied.ndim != 2:
            raise ValueError(
                "occupied_mask, free_mask, and inflated_occupied_mask must have equal [H, W] shapes"
            )
        resolution = float(resolution_m)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(f"resolution_m must be positive; got {resolution_m!r}")

        unknown = ~(occupied | free)
        traversable = free & ~inflated
        robot_cell = _world_to_cell(
            (float(robot_pose[0]), float(robot_pose[1])),
            resolution_m=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            shape=free.shape,
        )
        if robot_cell is None:
            return []
        if not traversable[robot_cell]:
            robot_cell = _nearest_true_cell(traversable, robot_cell, max_radius_cells=8)
        if robot_cell is None:
            return []

        path_steps = _grid_path_steps(traversable=traversable, start=robot_cell)
        reachable = path_steps >= 0
        raw_frontier_mask = free & _adjacent_to(unknown)
        frontier_mask = raw_frontier_mask & traversable & reachable
        components = [
            component
            for component in _connected_components(frontier_mask)
            if component.shape[0] >= max(1, int(self.config.min_cluster_cells))
        ]

        robot_x, robot_y, robot_yaw = [float(value) for value in robot_pose]
        info_radius_cells = max(1, int(round(self.config.information_radius_m / resolution)))
        approach_offset_cells = max(1.0, float(self.config.approach_offset_m) / resolution)
        approach_search_cells = max(
            approach_offset_cells,
            float(self.config.approach_search_radius_m) / resolution,
        )
        occupied_cells = np.argwhere(occupied)
        frontiers: list[Frontier] = []
        for component in components:
            boundary_cell = _component_medoid_cell(component)
            navigation_goal_cell = _select_navigation_goal_cell(
                boundary_cell=boundary_cell,
                component=component,
                unknown=unknown,
                traversable=traversable,
                reachable=reachable,
                robot_cell=robot_cell,
                approach_offset_cells=approach_offset_cells,
                search_radius_cells=approach_search_cells,
            )
            if navigation_goal_cell is None:
                continue

            boundary_world = _cell_to_world(
                boundary_cell,
                resolution_m=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            navigation_goal_world = _cell_to_world(
                navigation_goal_cell,
                resolution_m=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            local = world_to_robot_xy(
                np.asarray([navigation_goal_world], dtype=np.float32),
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw_rad=robot_yaw,
            )[0]
            distance = float(np.hypot(float(local[0]), float(local[1])))
            if distance > max(0.5, float(self.config.max_distance_m)):
                continue
            bearing = float(np.arctan2(float(local[1]), float(local[0])))
            information_gain = _unknown_count_near(
                unknown=unknown,
                row=boundary_cell[0],
                col=boundary_cell[1],
                radius=info_radius_cells,
            )
            path_cost_m = float(path_steps[navigation_goal_cell]) * resolution
            obstacle_clearance_m = _clearance_to_occupied(
                cell=navigation_goal_cell,
                occupied_cells=occupied_cells,
                resolution_m=resolution,
            )
            feature = (
                float(local[0]),
                float(local[1]),
                distance,
                float(np.sin(bearing)),
                float(np.cos(bearing)),
                float(np.log1p(component.shape[0])),
                float(np.log1p(information_gain)),
                path_cost_m,
                obstacle_clearance_m,
            )
            frontiers.append(
                Frontier(
                    frontier_id=-1,
                    boundary_world_xy=boundary_world,
                    navigation_goal_world_xy=navigation_goal_world,
                    boundary_cell_rc=boundary_cell,
                    navigation_goal_cell_rc=navigation_goal_cell,
                    cell_count=int(component.shape[0]),
                    information_gain=int(information_gain),
                    distance_m=distance,
                    bearing_rad=bearing,
                    local_forward_m=float(local[0]),
                    local_left_m=float(local[1]),
                    path_cost_m=path_cost_m,
                    obstacle_clearance_m=obstacle_clearance_m,
                    feature=feature,
                )
            )
        frontiers.sort(key=lambda item: (-item.information_gain, item.path_cost_m))
        return frontiers[: max(1, int(self.config.max_frontiers))]


class FrontierTracker:
    """Assign stable IDs by matching the physical free/unknown boundary."""

    def __init__(self, *, match_radius_m: float = 0.75) -> None:
        self._match_radius_m = max(0.10, float(match_radius_m))
        self._next_id = 0
        self._previous: dict[int, tuple[float, float]] = {}

    def update(self, frontiers: list[Frontier]) -> list[Frontier]:
        available = set(self._previous)
        current: dict[int, tuple[float, float]] = {}
        result: list[Frontier] = []
        for frontier in frontiers:
            best_id = None
            best_distance = float("inf")
            point = np.asarray(frontier.boundary_world_xy, dtype=np.float32)
            for frontier_id in available:
                distance = float(
                    np.linalg.norm(point - np.asarray(self._previous[frontier_id], dtype=np.float32))
                )
                if distance <= self._match_radius_m and distance < best_distance:
                    best_id = int(frontier_id)
                    best_distance = distance
            if best_id is None:
                best_id = int(self._next_id)
                self._next_id += 1
            else:
                available.remove(best_id)
            current[best_id] = frontier.boundary_world_xy
            result.append(replace(frontier, frontier_id=best_id))
        self._previous = current
        return result


def _component_medoid_cell(component: np.ndarray) -> tuple[int, int]:
    """Return an actual component cell closest to its arithmetic center."""

    center = np.mean(component, axis=0)
    index = int(np.argmin(np.sum(np.square(component.astype(np.float64) - center), axis=1)))
    return int(component[index, 0]), int(component[index, 1])


def _select_navigation_goal_cell(
    *,
    boundary_cell: tuple[int, int],
    component: np.ndarray,
    unknown: np.ndarray,
    traversable: np.ndarray,
    reachable: np.ndarray,
    robot_cell: tuple[int, int],
    approach_offset_cells: float,
    search_radius_cells: float,
) -> tuple[int, int] | None:
    boundary = np.asarray(boundary_cell, dtype=np.float64)
    local_unknown = _nearby_mask_cells(
        unknown,
        center=boundary_cell,
        radius=max(2, int(round(0.5 * approach_offset_cells))),
    )
    if local_unknown.size:
        # Array coordinates are [row, col]. Unknown -> boundary points inward.
        inward = boundary - np.mean(local_unknown, axis=0)
    else:
        inward = np.asarray(robot_cell, dtype=np.float64) - boundary
    norm = float(np.linalg.norm(inward))
    if norm < 1.0e-6:
        component_center = np.mean(component, axis=0)
        inward = np.asarray(robot_cell, dtype=np.float64) - component_center
        norm = float(np.linalg.norm(inward))
    if norm < 1.0e-6:
        return boundary_cell

    desired = boundary + inward / norm * float(approach_offset_cells)
    radius = int(np.ceil(search_radius_cells))
    row0 = max(0, int(np.floor(desired[0])) - radius)
    row1 = min(traversable.shape[0], int(np.ceil(desired[0])) + radius + 1)
    col0 = max(0, int(np.floor(desired[1])) - radius)
    col1 = min(traversable.shape[1], int(np.ceil(desired[1])) + radius + 1)
    valid = traversable[row0:row1, col0:col1] & reachable[row0:row1, col0:col1]
    candidates = np.argwhere(valid)
    if candidates.size == 0:
        return boundary_cell if traversable[boundary_cell] and reachable[boundary_cell] else None
    candidates[:, 0] += row0
    candidates[:, 1] += col0
    distances_to_desired = np.linalg.norm(candidates.astype(np.float64) - desired[None, :], axis=1)
    distances_to_boundary = np.linalg.norm(candidates.astype(np.float64) - boundary[None, :], axis=1)
    # Prefer the inward target, while avoiding a snap far away from this frontier.
    score = distances_to_desired + 0.20 * np.abs(distances_to_boundary - approach_offset_cells)
    chosen = candidates[int(np.argmin(score))]
    return int(chosen[0]), int(chosen[1])


def _grid_path_steps(*, traversable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    steps = np.full(traversable.shape, -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque([start])
    steps[start] = 0
    while queue:
        row, col = queue.popleft()
        next_step = int(steps[row, col]) + 1
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + delta_row
            next_col = col + delta_col
            if (
                0 <= next_row < traversable.shape[0]
                and 0 <= next_col < traversable.shape[1]
                and traversable[next_row, next_col]
                and steps[next_row, next_col] < 0
            ):
                steps[next_row, next_col] = next_step
                queue.append((next_row, next_col))
    return steps


def _adjacent_to(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    output = np.zeros(mask.shape, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            if row_offset == 1 and col_offset == 1:
                continue
            output |= padded[
                row_offset : row_offset + mask.shape[0],
                col_offset : col_offset + mask.shape[1],
            ]
    return output


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros(active.shape, dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = np.nonzero(active)
    for start_row, start_col in zip(rows.tolist(), cols.tolist()):
        if visited[start_row, start_col]:
            continue
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        cells: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if (
                        0 <= next_row < active.shape[0]
                        and 0 <= next_col < active.shape[1]
                        and active[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
        components.append(np.asarray(cells, dtype=np.int32))
    return components


def _unknown_count_near(*, unknown: np.ndarray, row: int, col: int, radius: int) -> int:
    row0 = max(0, int(row) - int(radius))
    row1 = min(unknown.shape[0], int(row) + int(radius) + 1)
    col0 = max(0, int(col) - int(radius))
    col1 = min(unknown.shape[1], int(col) + int(radius) + 1)
    return int(np.count_nonzero(unknown[row0:row1, col0:col1]))


def _nearby_mask_cells(mask: np.ndarray, *, center: tuple[int, int], radius: int) -> np.ndarray:
    row, col = center
    row0 = max(0, row - radius)
    row1 = min(mask.shape[0], row + radius + 1)
    col0 = max(0, col - radius)
    col1 = min(mask.shape[1], col + radius + 1)
    cells = np.argwhere(mask[row0:row1, col0:col1])
    if cells.size:
        cells[:, 0] += row0
        cells[:, 1] += col0
    return cells


def _clearance_to_occupied(
    *,
    cell: tuple[int, int],
    occupied_cells: np.ndarray,
    resolution_m: float,
) -> float:
    if occupied_cells.size == 0:
        return float("inf")
    delta = occupied_cells.astype(np.float64) - np.asarray(cell, dtype=np.float64)[None, :]
    return float(np.min(np.linalg.norm(delta, axis=1))) * float(resolution_m)


def _world_to_cell(
    point_xy: tuple[float, float],
    *,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    col = int(np.floor((float(point_xy[0]) - float(origin_x)) / float(resolution_m)))
    row = int(np.floor((float(point_xy[1]) - float(origin_y)) / float(resolution_m)))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    return row, col


def _cell_to_world(
    cell: tuple[int, int],
    *,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float]:
    row, col = cell
    return (
        float(origin_x) + (float(col) + 0.5) * float(resolution_m),
        float(origin_y) + (float(row) + 0.5) * float(resolution_m),
    )


def _nearest_true_cell(
    mask: np.ndarray,
    center: tuple[int, int],
    *,
    max_radius_cells: int,
) -> tuple[int, int] | None:
    row, col = center
    for radius in range(1, max(1, int(max_radius_cells)) + 1):
        row0 = max(0, row - radius)
        row1 = min(mask.shape[0], row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(mask.shape[1], col + radius + 1)
        candidates = np.argwhere(mask[row0:row1, col0:col1])
        if candidates.size:
            offsets = candidates.astype(np.float64) - np.asarray(
                [row - row0, col - col0], dtype=np.float64
            )
            candidate = candidates[int(np.argmin(np.sum(np.square(offsets), axis=1)))]
            return int(candidate[0] + row0), int(candidate[1] + col0)
    return None
