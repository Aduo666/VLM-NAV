from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
import math

import numpy as np

from vln_nav.navigation.fine_local_planner import FinePlannerConfig
from vln_nav.navigation.route_corridor import RouteCorridor
from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    GeometryStatus,
    PlanningGrid,
)


@dataclass(frozen=True)
class TraversabilityGraphConfig:
    lateral_step_m: float
    heading_limit_deg: float
    heading_step_deg: float
    local_window_half_m: float
    endpoint_position_tolerance_m: float
    endpoint_heading_tolerance_deg: float
    refined_lateral_step_m: float
    refined_heading_step_deg: float
    observation_threshold: float
    edge_observation_threshold: float
    local_contraction_threshold: float
    local_baseline_radius_m: float
    state_capacity_threshold: float
    transition_capacity_threshold: float
    constrained_min_length_m: float
    trigger_lookahead_m: float
    robust_node_clearance_m: float
    robust_edge_clearance_m: float
    pre_distance_m: float
    post_distance_m: float
    refinement_threshold_band: float

    @classmethod
    def from_mapping(cls, data: dict) -> "TraversabilityGraphConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"traversability_graph 必须包含这些字段：{sorted(expected)}")
        config = cls(**{name: float(data[name]) for name in expected})
        config.validate()
        return config

    def validate(self) -> None:
        positive = tuple(float(getattr(self, item.name)) for item in fields(self))
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("traversability_graph 数值配置必须为有限正数")
        if self.refined_lateral_step_m >= self.lateral_step_m:
            raise ValueError("refined_lateral_step_m 必须小于 lateral_step_m")
        if self.refined_heading_step_deg >= self.heading_step_deg:
            raise ValueError("refined_heading_step_deg 必须小于 heading_step_deg")
        for name in (
            "observation_threshold",
            "edge_observation_threshold",
            "local_contraction_threshold",
            "state_capacity_threshold",
            "transition_capacity_threshold",
            "refinement_threshold_band",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 必须位于 0 至 1")
        if self.heading_step_deg > self.heading_limit_deg:
            raise ValueError("heading_step_deg 不能超过 heading_limit_deg")


@dataclass(frozen=True)
class TraversabilityNode:
    index: int
    layer_index: int
    lateral_m: float
    heading_delta_rad: float
    world_pose: tuple[float, float, float]
    status: GeometryStatus
    clearance_m: float


@dataclass(frozen=True)
class TraversabilityEdge:
    source_index: int
    target_index: int
    primitive_index: int
    status: GeometryStatus | None
    min_clearance_m: float


@dataclass(frozen=True)
class CapacityProfile:
    s_m: np.ndarray
    state_capacity: np.ndarray
    transition_capacity: np.ndarray
    state_observation: np.ndarray
    edge_observation: np.ndarray
    state_contraction: np.ndarray
    transition_contraction: np.ndarray


@dataclass(frozen=True)
class TraversabilityGraph:
    corridor: RouteCorridor
    nodes: tuple[TraversabilityNode, ...]
    open_edges: tuple[TraversabilityEdge, ...]
    kinematic_edge_indices: np.ndarray
    safe_edge_indices: np.ndarray
    nodes_by_layer: tuple[np.ndarray, ...]
    profile: CapacityProfile
    lateral_step_m: float
    heading_step_deg: float
    refined: bool


@dataclass(frozen=True)
class _Primitive:
    direction: int
    curvature_per_m: float
    spin_rad: float


class TraversabilityGraphBuilder:
    def __init__(
        self,
        *,
        config: TraversabilityGraphConfig,
        planner_config: FinePlannerConfig,
        collision_checker: FootprintCollisionChecker,
    ) -> None:
        config.validate()
        planner_config.validate()
        self.config = config
        self.planner_config = planner_config
        self.collision_checker = collision_checker
        self._primitives = tuple(
            [
                _Primitive(1, float(curvature), 0.0)
                for curvature in planner_config.curvature_samples_per_m
            ]
            + [
                _Primitive(-1, float(curvature), 0.0)
                for curvature in planner_config.curvature_samples_per_m
            ]
            + [
                _Primitive(0, 0.0, math.radians(planner_config.spin_step_deg)),
                _Primitive(0, 0.0, -math.radians(planner_config.spin_step_deg)),
            ]
        )

    def build(
        self,
        *,
        grid: PlanningGrid,
        corridor: RouteCorridor,
        refined: bool = False,
    ) -> TraversabilityGraph:
        lateral_step = (
            self.config.refined_lateral_step_m
            if refined
            else self.config.lateral_step_m
        )
        heading_step_deg = (
            self.config.refined_heading_step_deg
            if refined
            else self.config.heading_step_deg
        )
        nodes, nodes_by_layer, lookup_shape = self._sample_nodes(
            grid=grid,
            corridor=corridor,
            lateral_step_m=lateral_step,
            heading_step_deg=heading_step_deg,
        )
        open_edges = self._build_edges(
            grid=grid,
            corridor=corridor,
            nodes=nodes,
            lookup_shape=lookup_shape,
            lateral_step_m=lateral_step,
            heading_step_deg=heading_step_deg,
        )
        safe_nodes = np.asarray(
            [node.status is GeometryStatus.SAFE for node in nodes], dtype=bool
        )
        kinematic_edge_indices = np.asarray(
            [
                index
                for index, edge in enumerate(open_edges)
                if safe_nodes[edge.source_index] and safe_nodes[edge.target_index]
            ],
            dtype=np.int64,
        )
        safe_edge_indices = np.asarray(
            [
                index
                for index in kinematic_edge_indices
                if open_edges[int(index)].status is GeometryStatus.SAFE
            ],
            dtype=np.int64,
        )
        profile = self._capacity_profile(
            corridor=corridor,
            nodes=nodes,
            nodes_by_layer=nodes_by_layer,
            edges=open_edges,
            kinematic_edge_indices=kinematic_edge_indices,
            safe_edge_indices=safe_edge_indices,
        )
        return TraversabilityGraph(
            corridor=corridor,
            nodes=nodes,
            open_edges=open_edges,
            kinematic_edge_indices=np.ascontiguousarray(kinematic_edge_indices),
            safe_edge_indices=np.ascontiguousarray(safe_edge_indices),
            nodes_by_layer=nodes_by_layer,
            profile=profile,
            lateral_step_m=float(lateral_step),
            heading_step_deg=float(heading_step_deg),
            refined=bool(refined),
        )

    def through_mask(
        self,
        graph: TraversabilityGraph,
        *,
        start_layer: int,
        end_layer: int,
        safe_edges: bool,
    ) -> np.ndarray:
        if not 0 <= start_layer < end_layer < graph.corridor.layer_count:
            raise ValueError("through reachability 层范围无效")
        if safe_edges:
            allowed_nodes = np.asarray(
                [node.status is GeometryStatus.SAFE for node in graph.nodes],
                dtype=bool,
            )
            edge_indices = graph.safe_edge_indices
        else:
            allowed_nodes = np.asarray(
                [node.status is GeometryStatus.SAFE for node in graph.nodes],
                dtype=bool,
            )
            edge_indices = graph.kinematic_edge_indices
        return _through_mask(
            nodes=graph.nodes,
            edges=graph.open_edges,
            edge_indices=edge_indices,
            allowed_nodes=allowed_nodes,
            start_layer=start_layer,
            end_layer=end_layer,
        )

    def _sample_nodes(
        self,
        *,
        grid: PlanningGrid,
        corridor: RouteCorridor,
        lateral_step_m: float,
        heading_step_deg: float,
    ) -> tuple[tuple[TraversabilityNode, ...], tuple[np.ndarray, ...], tuple[int, int]]:
        lateral = _symmetric_samples(corridor.half_width_m, lateral_step_m)
        heading_deg = _symmetric_samples(
            self.config.heading_limit_deg,
            heading_step_deg,
        )
        heading = np.radians(heading_deg)
        nodes: list[TraversabilityNode] = []
        nodes_by_layer = []
        for layer in range(corridor.layer_count):
            layer_nodes = []
            for lateral_value in lateral:
                for heading_value in heading:
                    pose = corridor.world_pose(layer, lateral_value, heading_value)
                    assessment = self.collision_checker.classify_pose(
                        grid,
                        x_m=pose[0],
                        y_m=pose[1],
                        yaw_rad=pose[2],
                    )
                    index = len(nodes)
                    nodes.append(
                        TraversabilityNode(
                            index=index,
                            layer_index=layer,
                            lateral_m=float(lateral_value),
                            heading_delta_rad=float(heading_value),
                            world_pose=pose,
                            status=assessment.status,
                            clearance_m=float(assessment.min_clearance_m),
                        )
                    )
                    layer_nodes.append(index)
            nodes_by_layer.append(np.asarray(layer_nodes, dtype=np.int64))
        return tuple(nodes), tuple(nodes_by_layer), (len(lateral), len(heading))

    def _build_edges(
        self,
        *,
        grid: PlanningGrid,
        corridor: RouteCorridor,
        nodes: tuple[TraversabilityNode, ...],
        lookup_shape: tuple[int, int],
        lateral_step_m: float,
        heading_step_deg: float,
    ) -> tuple[TraversabilityEdge, ...]:
        lateral_count, heading_count = lookup_shape
        heading_step_rad = math.radians(heading_step_deg)
        edges = []
        seen = set()
        for node in nodes:
            for primitive_index, primitive in enumerate(self._primitives):
                sweep = self._sample_primitive(node.world_pose, primitive)
                endpoint = sweep[-1]
                layer, lateral, heading = corridor.coordinates(endpoint[:2], endpoint[2])
                if abs(layer - node.layer_index) > 2:
                    continue
                lateral_index = int(round(
                    (lateral + corridor.half_width_m) / lateral_step_m
                ))
                heading_index = int(round(
                    (heading + math.radians(self.config.heading_limit_deg))
                    / heading_step_rad
                ))
                if not (
                    0 <= lateral_index < lateral_count
                    and 0 <= heading_index < heading_count
                ):
                    continue
                target_index = (
                    layer * lateral_count * heading_count
                    + lateral_index * heading_count
                    + heading_index
                )
                if target_index == node.index:
                    continue
                target = nodes[target_index]
                position_error = math.hypot(
                    endpoint[0] - target.world_pose[0],
                    endpoint[1] - target.world_pose[1],
                )
                yaw_error = abs(_wrap_angle(endpoint[2] - target.world_pose[2]))
                if (
                    position_error > self.config.endpoint_position_tolerance_m
                    or yaw_error > math.radians(
                        self.config.endpoint_heading_tolerance_deg
                    )
                ):
                    continue
                key = (node.index, target_index, primitive_index)
                if key in seen:
                    continue
                seen.add(key)
                status = None
                min_clearance = 0.0
                if (
                    node.status is GeometryStatus.SAFE
                    and target.status is GeometryStatus.SAFE
                ):
                    check_poses = np.vstack((sweep[1:], np.asarray(target.world_pose)))
                    assessment = self.collision_checker.classify_swept_poses(
                        grid,
                        check_poses,
                    )
                    status = assessment.status
                    min_clearance = assessment.min_clearance_m
                edges.append(
                    TraversabilityEdge(
                        source_index=node.index,
                        target_index=target_index,
                        primitive_index=primitive_index,
                        status=status,
                        min_clearance_m=float(min_clearance),
                    )
                )
        return tuple(edges)

    def _capacity_profile(
        self,
        *,
        corridor: RouteCorridor,
        nodes: tuple[TraversabilityNode, ...],
        nodes_by_layer: tuple[np.ndarray, ...],
        edges: tuple[TraversabilityEdge, ...],
        kinematic_edge_indices: np.ndarray,
        safe_edge_indices: np.ndarray,
    ) -> CapacityProfile:
        layer_count = corridor.layer_count
        state_capacity = np.zeros(layer_count, dtype=np.float64)
        transition_capacity = np.full(layer_count, np.nan, dtype=np.float64)
        state_observation = np.zeros(layer_count, dtype=np.float64)
        edge_observation = np.full(layer_count, np.nan, dtype=np.float64)
        window_layers = max(
            1,
            int(round(
                self.config.local_window_half_m
                / max(float(np.median(np.diff(corridor.s_m))), 1.0e-6)
            )),
        )
        safe_node_mask = np.asarray(
            [node.status is GeometryStatus.SAFE for node in nodes], dtype=bool
        )
        all_node_mask = np.ones(len(nodes), dtype=bool)
        all_edge_indices = np.arange(len(edges), dtype=np.int64)
        through_masks = []
        for layer in range(layer_count):
            layer_nodes = nodes_by_layer[layer]
            unresolved = sum(
                nodes[int(index)].status is GeometryStatus.UNRESOLVED
                for index in layer_nodes
            )
            state_observation[layer] = 1.0 - unresolved / max(len(layer_nodes), 1)
            start_layer = max(0, layer - window_layers)
            end_layer = min(layer_count - 1, layer + window_layers)
            if end_layer <= start_layer:
                through_masks.append(np.zeros(len(nodes), dtype=bool))
                continue
            actual = _through_mask(
                nodes=nodes,
                edges=edges,
                edge_indices=kinematic_edge_indices,
                allowed_nodes=safe_node_mask,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            open_reference = _through_mask(
                nodes=nodes,
                edges=edges,
                edge_indices=all_edge_indices,
                allowed_nodes=all_node_mask,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            through_masks.append(actual)
            denominator = int(np.count_nonzero(open_reference[layer_nodes]))
            numerator = int(np.count_nonzero(actual[layer_nodes]))
            state_capacity[layer] = numerator / denominator if denominator else 0.0

        safe_edges = set(map(int, safe_edge_indices))
        kinematic_edges = set(map(int, kinematic_edge_indices))
        for layer in range(layer_count - 1):
            through = through_masks[layer]
            candidates = []
            unresolved_count = 0
            for edge_index in kinematic_edges:
                edge = edges[edge_index]
                source_layer = nodes[edge.source_index].layer_index
                target_layer = nodes[edge.target_index].layer_index
                if {source_layer, target_layer} != {layer, layer + 1}:
                    continue
                if not (through[edge.source_index] and through[edge.target_index]):
                    continue
                candidates.append(edge_index)
                if edge.status is GeometryStatus.UNRESOLVED:
                    unresolved_count += 1
            if candidates:
                transition_capacity[layer] = (
                    sum(index in safe_edges for index in candidates) / len(candidates)
                )
                edge_observation[layer] = 1.0 - unresolved_count / len(candidates)
            else:
                transition_capacity[layer] = 0.0
                edge_observation[layer] = 0.0
        if layer_count > 1:
            transition_capacity[-1] = transition_capacity[-2]
            edge_observation[-1] = edge_observation[-2]
        state_contraction = _relative_capacity(
            state_capacity,
            corridor.s_m,
            self.config.local_baseline_radius_m,
        )
        transition_contraction = _relative_capacity(
            transition_capacity,
            corridor.s_m,
            self.config.local_baseline_radius_m,
        )
        return CapacityProfile(
            s_m=np.ascontiguousarray(corridor.s_m.copy()),
            state_capacity=np.ascontiguousarray(state_capacity),
            transition_capacity=np.ascontiguousarray(transition_capacity),
            state_observation=np.ascontiguousarray(state_observation),
            edge_observation=np.ascontiguousarray(edge_observation),
            state_contraction=np.ascontiguousarray(state_contraction),
            transition_contraction=np.ascontiguousarray(transition_contraction),
        )

    def _sample_primitive(
        self,
        pose: tuple[float, float, float],
        primitive: _Primitive,
    ) -> np.ndarray:
        x, y, yaw = map(float, pose)
        if primitive.direction == 0:
            yaw_step = math.radians(self.planner_config.yaw_resolution_deg)
            sample_count = max(2, int(math.ceil(abs(primitive.spin_rad) / yaw_step)) + 1)
            sample_yaw = yaw + np.linspace(0.0, primitive.spin_rad, sample_count)
            sample_yaw = (sample_yaw + math.pi) % (2.0 * math.pi) - math.pi
            return np.column_stack((
                np.full_like(sample_yaw, x),
                np.full_like(sample_yaw, y),
                sample_yaw,
            ))
        sample_count = max(
            2,
            int(math.ceil(
                self.planner_config.step_length_m
                / self.planner_config.collision_sample_step_m
            )) + 1,
        )
        distances = np.linspace(
            0.0,
            primitive.direction * self.planner_config.step_length_m,
            sample_count,
        )
        curvature = primitive.curvature_per_m
        if abs(curvature) < 1.0e-9:
            sample_x = x + distances * math.cos(yaw)
            sample_y = y + distances * math.sin(yaw)
            sample_yaw = np.full_like(distances, yaw)
        else:
            sample_yaw = yaw + curvature * distances
            sample_x = x + (np.sin(sample_yaw) - math.sin(yaw)) / curvature
            sample_y = y + (-np.cos(sample_yaw) + math.cos(yaw)) / curvature
        sample_yaw = (sample_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return np.column_stack((sample_x, sample_y, sample_yaw))


def _through_mask(
    *,
    nodes: tuple[TraversabilityNode, ...],
    edges: tuple[TraversabilityEdge, ...],
    edge_indices: np.ndarray,
    allowed_nodes: np.ndarray,
    start_layer: int,
    end_layer: int,
) -> np.ndarray:
    node_count = len(nodes)
    forward_adjacency = [[] for _ in range(node_count)]
    backward_adjacency = [[] for _ in range(node_count)]
    for edge_index in edge_indices:
        edge = edges[int(edge_index)]
        source_layer = nodes[edge.source_index].layer_index
        target_layer = nodes[edge.target_index].layer_index
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
        node.index
        for node in nodes
        if node.layer_index == start_layer and allowed_nodes[node.index]
    ]
    destinations = [
        node.index
        for node in nodes
        if node.layer_index == end_layer and allowed_nodes[node.index]
    ]
    forward = _reachable(sources, forward_adjacency, node_count)
    backward = _reachable(destinations, backward_adjacency, node_count)
    return forward & backward & allowed_nodes


def _reachable(starts: list[int], adjacency: list[list[int]], size: int) -> np.ndarray:
    reached = np.zeros(size, dtype=bool)
    pending = deque(int(index) for index in starts)
    for index in starts:
        reached[int(index)] = True
    while pending:
        current = pending.popleft()
        for neighbor in adjacency[current]:
            if reached[neighbor]:
                continue
            reached[neighbor] = True
            pending.append(neighbor)
    return reached


def _relative_capacity(values: np.ndarray, s_m: np.ndarray, radius_m: float) -> np.ndarray:
    ratios = np.ones_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    for index, position in enumerate(s_m):
        neighborhood = finite & (np.abs(s_m - position) <= radius_m)
        if not np.any(neighborhood):
            ratios[index] = 1.0
            continue
        baseline = float(np.nanpercentile(values[neighborhood], 80.0))
        ratios[index] = float(values[index]) / max(baseline, 1.0e-6)
    return ratios


def _symmetric_samples(limit: float, step: float) -> np.ndarray:
    count = max(1, int(math.floor(float(limit) / float(step))))
    values = np.arange(-count, count + 1, dtype=np.float64) * float(step)
    if values[0] > -limit + 1.0e-6:
        values = np.r_[-float(limit), values]
    if values[-1] < limit - 1.0e-6:
        values = np.r_[values, float(limit)]
    return np.unique(np.clip(values, -float(limit), float(limit)))


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
