from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from vln_nav.navigation.route_corridor import (
    RouteCorridorConfig,
    build_route_corridor,
)
from vln_nav.navigation.trajectory_clearance import GeometryStatus, PlanningGrid
from vln_nav.navigation.traversability_graph import (
    TraversabilityGraph,
    TraversabilityGraphBuilder,
    TraversabilityGraphConfig,
)


@dataclass(frozen=True)
class RestrictedInterval:
    kind: str
    start_layer: int
    end_layer: int


@dataclass(frozen=True)
class TraversalTube:
    complete: bool
    pre_layer: int
    restricted_start_layer: int
    restricted_end_layer: int
    post_layer: int | None
    robust_node_mask: np.ndarray
    robust_edge_indices: np.ndarray
    pre_node_indices: np.ndarray
    post_node_indices: np.ndarray
    goal_world_pose: tuple[float, float, float]


@dataclass(frozen=True)
class RestrictedSpaceAnalysis:
    graph: TraversabilityGraph
    interval: RestrictedInterval | None
    tube: TraversalTube | None
    status: str
    reason: str

    @property
    def requires_fine_planning(self) -> bool:
        return self.status in {
            "LOCAL_BOTTLENECK",
            "CONSTRAINED_CORRIDOR",
            "UNRESOLVED",
        }


class RestrictedSpaceAnalyzer:
    def __init__(
        self,
        *,
        corridor_config: RouteCorridorConfig,
        graph_config: TraversabilityGraphConfig,
        graph_builder: TraversabilityGraphBuilder,
    ) -> None:
        corridor_config.validate()
        graph_config.validate()
        self.corridor_config = corridor_config
        self.graph_config = graph_config
        self.graph_builder = graph_builder

    def analyze(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        target_world_xy: tuple[float, float],
        route_hint_world_xy: np.ndarray | None,
    ) -> RestrictedSpaceAnalysis:
        corridor = build_route_corridor(
            grid=grid,
            start_pose_world=start_pose_world,
            target_world_xy=target_world_xy,
            route_hint_world_xy=route_hint_world_xy,
            config=self.corridor_config,
        )
        graph = self.graph_builder.build(
            grid=grid,
            corridor=corridor,
            refined=False,
        )
        interval = self._detect(graph)
        if self._needs_refinement(graph, interval):
            graph = self.graph_builder.build(
                grid=grid,
                corridor=corridor,
                refined=True,
            )
            interval = self._detect(graph)
        if interval is None:
            return RestrictedSpaceAnalysis(
                graph=graph,
                interval=None,
                tube=None,
                status="OPEN",
                reason="capacities_clear",
            )
        tube = self._build_tube(graph, interval)
        if tube is None:
            return RestrictedSpaceAnalysis(
                graph=graph,
                interval=interval,
                tube=None,
                status=interval.kind,
                reason="no_verified_safe_prefix",
            )
        reason = "complete_tube" if tube.complete else "verified_safe_prefix"
        return RestrictedSpaceAnalysis(
            graph=graph,
            interval=interval,
            tube=tube,
            status=interval.kind,
            reason=reason,
        )

    def _detect(self, graph: TraversabilityGraph) -> RestrictedInterval | None:
        profile = graph.profile
        considered = profile.s_m <= self.graph_config.trigger_lookahead_m
        observed = (
            (profile.state_observation >= self.graph_config.observation_threshold)
            & (
                profile.edge_observation
                >= self.graph_config.edge_observation_threshold
            )
        )
        local = considered & observed & (
            (profile.state_contraction <= self.graph_config.local_contraction_threshold)
            | (
                profile.transition_contraction
                <= self.graph_config.local_contraction_threshold
            )
        )
        constrained = considered & observed & (
            (profile.state_capacity <= self.graph_config.state_capacity_threshold)
            | (
                profile.transition_capacity
                <= self.graph_config.transition_capacity_threshold
            )
        )
        local_runs = _true_runs(local)
        constrained_runs = [
            run
            for run in _true_runs(constrained)
            if profile.s_m[run[1]] - profile.s_m[run[0]]
            >= self.graph_config.constrained_min_length_m
        ]
        candidates = [
            RestrictedInterval("LOCAL_BOTTLENECK", start, end)
            for start, end in local_runs
        ]
        candidates.extend(
            RestrictedInterval("CONSTRAINED_CORRIDOR", start, end)
            for start, end in constrained_runs
        )
        if candidates:
            return min(candidates, key=lambda item: (item.start_layer, item.end_layer))
        unresolved = considered & ~observed
        unresolved_runs = _true_runs(unresolved)
        if unresolved_runs:
            start, end = unresolved_runs[0]
            return RestrictedInterval("UNRESOLVED", start, end)
        return None

    def _needs_refinement(
        self,
        graph: TraversabilityGraph,
        interval: RestrictedInterval | None,
    ) -> bool:
        if graph.refined:
            return False
        profile = graph.profile
        band = self.graph_config.refinement_threshold_band
        thresholds = (
            (profile.state_capacity, self.graph_config.state_capacity_threshold),
            (
                profile.transition_capacity,
                self.graph_config.transition_capacity_threshold,
            ),
            (
                profile.state_contraction,
                self.graph_config.local_contraction_threshold,
            ),
            (
                profile.transition_contraction,
                self.graph_config.local_contraction_threshold,
            ),
        )
        near_threshold = any(
            np.any(np.isfinite(values) & (np.abs(values - threshold) <= band))
            for values, threshold in thresholds
        )
        empty_but_observed = bool(
            np.any(
                (profile.state_capacity <= 1.0e-9)
                & (
                    profile.state_observation
                    >= self.graph_config.observation_threshold
                )
            )
        )
        return bool(near_threshold or empty_but_observed or interval is not None)

    def _build_tube(
        self,
        graph: TraversabilityGraph,
        interval: RestrictedInterval,
    ) -> TraversalTube | None:
        profile = graph.profile
        pre_s = max(
            0.0,
            profile.s_m[interval.start_layer] - self.graph_config.pre_distance_m,
        )
        post_s = min(
            float(profile.s_m[-1]),
            profile.s_m[interval.end_layer] + self.graph_config.post_distance_m,
        )
        pre_layer = int(np.argmin(np.abs(profile.s_m - pre_s)))
        post_layer = int(np.argmin(np.abs(profile.s_m - post_s)))
        if post_layer <= pre_layer:
            return None
        full_through = self.graph_builder.through_mask(
            graph,
            start_layer=0,
            end_layer=post_layer,
            safe_edges=True,
        )
        complete = bool(np.any(full_through[graph.nodes_by_layer[post_layer]]))
        if complete:
            allowed = full_through & np.asarray(
                [
                    node.clearance_m >= self.graph_config.robust_node_clearance_m
                    for node in graph.nodes
                ],
                dtype=bool,
            )
            edge_indices = np.asarray(
                [
                    int(index)
                    for index in graph.safe_edge_indices
                    if graph.open_edges[int(index)].min_clearance_m
                    >= self.graph_config.robust_edge_clearance_m
                ],
                dtype=np.int64,
            )
            robust = _through_mask(
                graph,
                allowed_nodes=allowed,
                edge_indices=edge_indices,
                start_layer=0,
                end_layer=post_layer,
            )
            if np.any(robust[graph.nodes_by_layer[post_layer]]):
                return self._tube_from_mask(
                    graph=graph,
                    interval=interval,
                    node_mask=robust,
                    edge_indices=edge_indices,
                    pre_layer=pre_layer,
                    post_layer=post_layer,
                    complete=True,
                )

        for stop_layer in range(min(interval.end_layer, post_layer), 0, -1):
            partial = self.graph_builder.through_mask(
                graph,
                start_layer=0,
                end_layer=stop_layer,
                safe_edges=True,
            )
            allowed = partial & np.asarray(
                [
                    node.clearance_m >= self.graph_config.robust_node_clearance_m
                    for node in graph.nodes
                ],
                dtype=bool,
            )
            edge_indices = np.asarray(
                [
                    int(index)
                    for index in graph.safe_edge_indices
                    if graph.open_edges[int(index)].min_clearance_m
                    >= self.graph_config.robust_edge_clearance_m
                ],
                dtype=np.int64,
            )
            robust = _through_mask(
                graph,
                allowed_nodes=allowed,
                edge_indices=edge_indices,
                start_layer=0,
                end_layer=stop_layer,
            )
            if not np.any(robust[graph.nodes_by_layer[stop_layer]]):
                continue
            return self._tube_from_mask(
                graph=graph,
                interval=interval,
                node_mask=robust,
                edge_indices=edge_indices,
                pre_layer=min(pre_layer, stop_layer),
                post_layer=stop_layer,
                complete=False,
            )
        return None

    def _tube_from_mask(
        self,
        *,
        graph: TraversabilityGraph,
        interval: RestrictedInterval,
        node_mask: np.ndarray,
        edge_indices: np.ndarray,
        pre_layer: int,
        post_layer: int,
        complete: bool,
    ) -> TraversalTube:
        robust_edge_indices = np.asarray(
            [
                int(index)
                for index in edge_indices
                if node_mask[graph.open_edges[int(index)].source_index]
                and node_mask[graph.open_edges[int(index)].target_index]
            ],
            dtype=np.int64,
        )
        pre_nodes = graph.nodes_by_layer[pre_layer]
        pre_nodes = pre_nodes[node_mask[pre_nodes]]
        post_nodes = graph.nodes_by_layer[post_layer]
        post_nodes = post_nodes[node_mask[post_nodes]]
        if len(post_nodes) == 0:
            raise RuntimeError("穿越管道末端没有鲁棒状态")
        goal_index = min(
            map(int, post_nodes),
            key=lambda index: (
                abs(graph.nodes[index].lateral_m)
                + 0.5 * abs(graph.nodes[index].heading_delta_rad)
                - 0.05 * graph.nodes[index].clearance_m
            ),
        )
        return TraversalTube(
            complete=bool(complete),
            pre_layer=int(pre_layer),
            restricted_start_layer=int(interval.start_layer),
            restricted_end_layer=int(min(interval.end_layer, post_layer)),
            post_layer=int(post_layer) if complete else None,
            robust_node_mask=np.ascontiguousarray(node_mask),
            robust_edge_indices=np.ascontiguousarray(robust_edge_indices),
            pre_node_indices=np.ascontiguousarray(pre_nodes),
            post_node_indices=np.ascontiguousarray(post_nodes),
            goal_world_pose=graph.nodes[goal_index].world_pose,
        )


def _through_mask(
    graph: TraversabilityGraph,
    *,
    allowed_nodes: np.ndarray,
    edge_indices: np.ndarray,
    start_layer: int,
    end_layer: int,
) -> np.ndarray:
    node_count = len(graph.nodes)
    forward_adjacency = [[] for _ in range(node_count)]
    backward_adjacency = [[] for _ in range(node_count)]
    for edge_index in edge_indices:
        edge = graph.open_edges[int(edge_index)]
        source_layer = graph.nodes[edge.source_index].layer_index
        target_layer = graph.nodes[edge.target_index].layer_index
        if not (
            start_layer <= source_layer <= end_layer
            and start_layer <= target_layer <= end_layer
            and allowed_nodes[edge.source_index]
            and allowed_nodes[edge.target_index]
        ):
            continue
        forward_adjacency[edge.source_index].append(edge.target_index)
        backward_adjacency[edge.target_index].append(edge.source_index)
    sources = [
        int(index)
        for index in graph.nodes_by_layer[start_layer]
        if allowed_nodes[int(index)]
    ]
    destinations = [
        int(index)
        for index in graph.nodes_by_layer[end_layer]
        if allowed_nodes[int(index)]
    ]
    forward = _reachable(sources, forward_adjacency, node_count)
    backward = _reachable(destinations, backward_adjacency, node_count)
    return forward & backward & allowed_nodes


def _reachable(starts: list[int], adjacency: list[list[int]], size: int) -> np.ndarray:
    reached = np.zeros(size, dtype=bool)
    pending = deque(starts)
    for index in starts:
        reached[index] = True
    while pending:
        current = pending.popleft()
        for neighbor in adjacency[current]:
            if reached[neighbor]:
                continue
            reached[neighbor] = True
            pending.append(neighbor)
    return reached


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs
