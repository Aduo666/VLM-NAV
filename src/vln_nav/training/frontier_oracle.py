from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frontier import Frontier
from .interiornav import nearest_path_index
from .schemas import FrontierToken


@dataclass(frozen=True)
class FrontierOracleConfig:
    route_sample_spacing_m: float = 0.04
    max_crossing_distance_m: float = 0.80
    max_route_distance_m: float = 1.00
    path_cost_weight: float = 0.03
    softmax_temperature_m: float = 0.50
    min_selection_probability: float = 0.45
    min_cost_margin_m: float = 0.10


@dataclass(frozen=True)
class FrontierOracleResult:
    frontier_tokens: tuple[FrontierToken, ...]
    frontier_value_targets: tuple[float, ...]
    selected_frontier_id: int | None
    navigation_mode: str
    reason: str
    confidence: float
    nearest_reference_path_index: int
    first_unknown_world_xy: tuple[float, float] | None


def label_frontiers_from_reference_route(
    *,
    frontiers: list[Frontier],
    occupied_mask: np.ndarray,
    free_mask: np.ndarray,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
    robot_xy: tuple[float, float],
    reference_path_xy: np.ndarray,
    config: FrontierOracleConfig | None = None,
) -> FrontierOracleResult:
    """Create privileged labels without exposing them to the student input.

    A hard selection is emitted only when a reachable frontier boundary aligns
    with the first future free-to-unknown route crossing. There is deliberately
    no "nearest future route" fallback: an unreliable Oracle label is more
    harmful than an unlabeled frame.
    """

    oracle_config = config or FrontierOracleConfig()
    occupied = np.asarray(occupied_mask, dtype=bool)
    free = np.asarray(free_mask, dtype=bool)
    route = np.asarray(reference_path_xy, dtype=np.float64)
    if occupied.shape != free.shape or occupied.ndim != 2:
        raise ValueError("occupied_mask and free_mask must have equal [H,W] shapes")
    if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] < 2:
        raise ValueError("reference_path_xy must have shape [N, >=2] with N > 0")

    nearest_index = nearest_path_index(route, robot_xy)
    future_route = _densify_polyline(
        route[nearest_index:, :2],
        spacing_m=max(0.01, float(oracle_config.route_sample_spacing_m)),
    )
    first_unknown = _first_free_to_unknown_crossing(
        route_xy=future_route,
        occupied_mask=occupied,
        free_mask=free,
        resolution_m=resolution_m,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    tokens = tuple(_frontier_token(frontier) for frontier in frontiers)
    values = np.zeros((len(frontiers),), dtype=np.float64)
    progress = float(nearest_index) / max(1.0, float(route.shape[0] - 1))
    if progress >= 0.98:
        return _result(
            tokens=tokens,
            values=values,
            selected_id=None,
            mode="ARRIVED",
            reason="reference_route_complete",
            confidence=1.0,
            nearest_index=nearest_index,
            first_unknown=first_unknown,
        )
    if not frontiers:
        return _result(
            tokens=tokens,
            values=values,
            selected_id=None,
            mode="KEEP_ACTIVE_GOAL",
            reason="no_reachable_frontier",
            confidence=0.0,
            nearest_index=nearest_index,
            first_unknown=first_unknown,
        )
    if first_unknown is None:
        return _result(
            tokens=tokens,
            values=values,
            selected_id=None,
            mode="KEEP_ACTIVE_GOAL",
            reason="no_route_frontier_crossing",
            confidence=0.0,
            nearest_index=nearest_index,
            first_unknown=None,
        )

    raw_costs = np.full((len(frontiers),), np.inf, dtype=np.float64)
    crossing = np.asarray(first_unknown, dtype=np.float64)
    for index, frontier in enumerate(frontiers):
        boundary = np.asarray(frontier.boundary_world_xy, dtype=np.float64)
        crossing_distance = float(np.linalg.norm(boundary - crossing))
        route_distance = float(np.min(np.linalg.norm(future_route - boundary[None, :], axis=1)))
        if crossing_distance > float(oracle_config.max_crossing_distance_m):
            continue
        if route_distance > float(oracle_config.max_route_distance_m):
            continue
        raw_costs[index] = (
            crossing_distance
            + 0.20 * route_distance
            + float(oracle_config.path_cost_weight) * float(frontier.path_cost_m)
        )

    finite_indices = np.flatnonzero(np.isfinite(raw_costs))
    if finite_indices.size == 0:
        return _result(
            tokens=tokens,
            values=values,
            selected_id=None,
            mode="KEEP_ACTIVE_GOAL",
            reason="no_aligned_route_frontier",
            confidence=0.0,
            nearest_index=nearest_index,
            first_unknown=first_unknown,
        )

    logits = -raw_costs[finite_indices] / max(
        0.05,
        float(oracle_config.softmax_temperature_m),
    )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= max(float(np.sum(probabilities)), 1.0e-12)
    best_order = np.argsort(raw_costs[finite_indices])
    best_index = int(finite_indices[int(best_order[0])])
    best_probability = float(probabilities[int(np.where(finite_indices == best_index)[0][0])])
    cost_margin = float("inf")
    if best_order.size > 1:
        second_index = int(finite_indices[int(best_order[1])])
        cost_margin = float(raw_costs[second_index] - raw_costs[best_index])

    reliable = best_probability >= float(oracle_config.min_selection_probability)
    if np.isfinite(cost_margin):
        reliable = reliable and cost_margin >= float(oracle_config.min_cost_margin_m)
    if not reliable:
        return _result(
            tokens=tokens,
            values=values,
            selected_id=None,
            mode="KEEP_ACTIVE_GOAL",
            reason="ambiguous_route_frontier",
            confidence=best_probability,
            nearest_index=nearest_index,
            first_unknown=first_unknown,
        )

    values[finite_indices] = probabilities
    return _result(
        tokens=tokens,
        values=values,
        selected_id=int(frontiers[best_index].frontier_id),
        mode="SELECT_NEW_FRONTIER",
        reason="reliable_route_frontier_crossing",
        confidence=best_probability,
        nearest_index=nearest_index,
        first_unknown=first_unknown,
    )


def _frontier_token(frontier: Frontier) -> FrontierToken:
    return FrontierToken(
        frontier_id=int(frontier.frontier_id),
        boundary_world_xy=tuple(float(value) for value in frontier.boundary_world_xy),
        navigation_goal_world_xy=tuple(
            float(value) for value in frontier.navigation_goal_world_xy
        ),
        local_forward_m=float(frontier.local_forward_m),
        local_left_m=float(frontier.local_left_m),
        distance_m=float(frontier.distance_m),
        bearing_rad=float(frontier.bearing_rad),
        cell_count=int(frontier.cell_count),
        information_gain=int(frontier.information_gain),
        path_cost_m=float(frontier.path_cost_m),
        obstacle_clearance_m=float(frontier.obstacle_clearance_m),
        reachable=True,
    )


def _result(
    *,
    tokens: tuple[FrontierToken, ...],
    values: np.ndarray,
    selected_id: int | None,
    mode: str,
    reason: str,
    confidence: float,
    nearest_index: int,
    first_unknown: tuple[float, float] | None,
) -> FrontierOracleResult:
    return FrontierOracleResult(
        frontier_tokens=tokens,
        frontier_value_targets=tuple(float(value) for value in values),
        selected_frontier_id=selected_id,
        navigation_mode=mode,
        reason=reason,
        confidence=float(confidence),
        nearest_reference_path_index=int(nearest_index),
        first_unknown_world_xy=first_unknown,
    )


def _densify_polyline(points_xy: np.ndarray, *, spacing_m: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[0] <= 1:
        return points[:, :2].copy()
    dense: list[np.ndarray] = [points[0, :2]]
    for start, end in zip(points[:-1, :2], points[1:, :2]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        steps = max(1, int(np.ceil(distance / float(spacing_m))))
        for step in range(1, steps + 1):
            dense.append(start + delta * (float(step) / float(steps)))
    return np.asarray(dense, dtype=np.float64)


def _first_free_to_unknown_crossing(
    *,
    route_xy: np.ndarray,
    occupied_mask: np.ndarray,
    free_mask: np.ndarray,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float] | None:
    seen_free = False
    for point in np.asarray(route_xy, dtype=np.float64):
        cell = _world_to_cell(
            (float(point[0]), float(point[1])),
            resolution_m=resolution_m,
            origin_x=origin_x,
            origin_y=origin_y,
            shape=free_mask.shape,
        )
        is_free = bool(cell is not None and free_mask[cell])
        is_occupied = bool(cell is not None and occupied_mask[cell])
        is_unknown = cell is None or (not is_free and not is_occupied)
        if is_free:
            seen_free = True
        elif seen_free and is_unknown:
            return float(point[0]), float(point[1])
    return None


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
