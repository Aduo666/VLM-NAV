from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .landmark_map import LandmarkMapEntry
from .observation_adapter import Pose2D


@dataclass
class TopologicalMemoryConfig:
    node_spacing_m: float = 1.0
    turn_threshold_deg: float = 40.0
    merge_radius_m: float = 0.75
    landmark_bind_radius_m: float = 1.50
    max_nodes: int = 128
    recent_node_count: int = 8


@dataclass
class TopologyNode:
    node_id: int
    world_xy: list[float]
    yaw_rad: float
    first_seen_time_sec: float
    last_seen_time_sec: float
    visit_count: int = 1
    labels: list[str] = field(default_factory=list)


class TopologicalMemory:
    def __init__(self, *, config: TopologicalMemoryConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("topology")
        self._nodes: dict[int, TopologyNode] = {}
        self._adjacency: dict[int, set[int]] = {}
        self._path_node_ids: list[int] = []
        self._next_node_id = 0
        self._last_anchor_node_id_by_label: dict[str, int] = {}
        self._last_log_signature: tuple | None = None
        self._last_robot_pose: Pose2D | None = None

    def reset(self) -> None:
        self._nodes.clear()
        self._adjacency.clear()
        self._path_node_ids.clear()
        self._next_node_id = 0
        self._last_anchor_node_id_by_label.clear()
        self._last_log_signature = None
        self._last_robot_pose = None

    def update(
        self,
        *,
        robot_pose: Pose2D,
        timestamp_sec: float,
        current_landmark: str,
        landmark_entry: LandmarkMapEntry | None,
        progress_signal: str,
    ) -> None:
        label = str(current_landmark).strip().lower()
        landmark_labels: list[str] = []
        if label:
            landmark_labels.append(label)
        if landmark_entry is not None and str(landmark_entry.label).strip():
            landmark_labels.append(str(landmark_entry.label).strip().lower())

        if not self._nodes:
            node_id = self._create_node(
                robot_pose=robot_pose,
                timestamp_sec=timestamp_sec,
                labels=[],
            )
            self._path_node_ids.append(node_id)
            self._last_robot_pose = Pose2D(float(robot_pose.x), float(robot_pose.y), float(robot_pose.yaw_rad))
            self._log_state(event="bootstrap")

        current_node_id = self.current_node_id(robot_pose=robot_pose)
        if current_node_id is None:
            current_node_id = self._path_node_ids[-1]

        if landmark_entry is not None and landmark_entry.anchor_distance_m is not None:
            if float(landmark_entry.anchor_distance_m) <= max(0.20, float(self._config.landmark_bind_radius_m)):
                self._attach_labels(node_id=current_node_id, labels=landmark_labels, timestamp_sec=timestamp_sec)
                for landmark_label in landmark_labels:
                    self._last_anchor_node_id_by_label[landmark_label] = int(current_node_id)

        should_snapshot = self._should_add_node(
            robot_pose=robot_pose,
            current_node=self._nodes[current_node_id],
            progress_signal=progress_signal,
        )
        if should_snapshot:
            target_node_id = self._find_merge_node(robot_pose=robot_pose)
            if target_node_id is None:
                target_node_id = self._create_node(robot_pose=robot_pose, timestamp_sec=timestamp_sec, labels=[])
            self._connect(self._path_node_ids[-1], target_node_id)
            if self._path_node_ids[-1] != target_node_id:
                self._path_node_ids.append(target_node_id)
            self._touch_node(target_node_id, timestamp_sec=timestamp_sec)
            if landmark_labels:
                self._attach_labels(node_id=target_node_id, labels=landmark_labels, timestamp_sec=timestamp_sec)
            current_node_id = target_node_id
            self._prune_if_needed()
            self._log_state(event="node_update")
        else:
            self._touch_node(current_node_id, timestamp_sec=timestamp_sec)

        self._last_robot_pose = Pose2D(float(robot_pose.x), float(robot_pose.y), float(robot_pose.yaw_rad))

    def summary(self, *, current_target: str, robot_pose: Pose2D) -> str:
        current_node_id = self.current_node_id(robot_pose=robot_pose)
        if current_node_id is None:
            return "topology=node_count=0"
        recent_ids = self._path_node_ids[-max(1, int(self._config.recent_node_count)) :]
        recent_text = []
        for node_id in recent_ids:
            node = self._nodes.get(int(node_id))
            if node is None:
                continue
            label_text = "/".join(node.labels[:2]) if node.labels else "-"
            recent_text.append(f"{node.node_id}:{label_text}")
        revisit_xy = self.suggest_revisit_world_xy(current_target=current_target, robot_pose=robot_pose)
        return (
            f"topology node_count={len(self._nodes)} edge_count={self.edge_count} "
            f"current_node={current_node_id} recent_path={recent_text or ['none']} "
            f"current_target_anchor_node={self._last_anchor_node_id_by_label.get(str(current_target).strip().lower(), 'none')} "
            f"revisit_world_xy={None if revisit_xy is None else [round(float(revisit_xy[0]), 2), round(float(revisit_xy[1]), 2)]}"
        )

    def current_node_id(self, *, robot_pose: Pose2D) -> int | None:
        if not self._nodes:
            return None
        best_node_id = None
        best_distance = float("inf")
        for node_id, node in self._nodes.items():
            distance = _pose_distance_m(robot_pose, node.world_xy)
            if distance < best_distance:
                best_distance = distance
                best_node_id = int(node_id)
        return best_node_id

    @property
    def edge_count(self) -> int:
        return int(sum(len(neighbors) for neighbors in self._adjacency.values()) // 2)

    def suggest_revisit_world_xy(self, *, current_target: str, robot_pose: Pose2D) -> list[float] | None:
        if not self._nodes:
            return None
        target_key = str(current_target).strip().lower()
        if target_key and target_key in self._last_anchor_node_id_by_label:
            node = self._nodes.get(int(self._last_anchor_node_id_by_label[target_key]))
            if node is not None and _pose_distance_m(robot_pose, node.world_xy) > max(0.35, float(self._config.merge_radius_m)):
                return list(node.world_xy)
        if len(self._path_node_ids) >= 2:
            current_node_id = self.current_node_id(robot_pose=robot_pose)
            for node_id in reversed(self._path_node_ids[:-1]):
                if current_node_id is not None and int(node_id) == int(current_node_id):
                    continue
                node = self._nodes.get(int(node_id))
                if node is None:
                    continue
                if _pose_distance_m(robot_pose, node.world_xy) > max(0.35, float(self._config.merge_radius_m)):
                    return list(node.world_xy)
        return None

    def _should_add_node(
        self,
        *,
        robot_pose: Pose2D,
        current_node: TopologyNode,
        progress_signal: str,
    ) -> bool:
        moved_m = _distance_xy(robot_pose.x, robot_pose.y, current_node.world_xy[0], current_node.world_xy[1])
        yaw_delta_deg = abs(
            np.degrees(_wrap_angle_rad(float(robot_pose.yaw_rad) - float(current_node.yaw_rad)))
        )
        if moved_m >= max(0.25, float(self._config.node_spacing_m)):
            return True
        if moved_m >= 0.25 and yaw_delta_deg >= max(5.0, float(self._config.turn_threshold_deg)):
            return True
        if str(progress_signal).upper() in {"STALLED", "SPINNING"} and moved_m >= 0.20:
            return True
        return False

    def _find_merge_node(self, *, robot_pose: Pose2D) -> int | None:
        merge_radius_m = max(0.10, float(self._config.merge_radius_m))
        best_node_id = None
        best_distance = float("inf")
        for node_id, node in self._nodes.items():
            distance = _pose_distance_m(robot_pose, node.world_xy)
            if distance <= merge_radius_m and distance < best_distance:
                best_distance = distance
                best_node_id = int(node_id)
        return best_node_id

    def _create_node(self, *, robot_pose: Pose2D, timestamp_sec: float, labels: list[str]) -> int:
        node_id = int(self._next_node_id)
        self._next_node_id += 1
        self._nodes[node_id] = TopologyNode(
            node_id=node_id,
            world_xy=[float(robot_pose.x), float(robot_pose.y)],
            yaw_rad=float(robot_pose.yaw_rad),
            first_seen_time_sec=float(timestamp_sec),
            last_seen_time_sec=float(timestamp_sec),
            labels=_dedupe_labels(labels),
        )
        self._adjacency.setdefault(node_id, set())
        return node_id

    def _attach_labels(self, *, node_id: int, labels: list[str], timestamp_sec: float) -> None:
        node = self._nodes.get(int(node_id))
        if node is None:
            return
        merged = _dedupe_labels(list(node.labels) + list(labels))
        node.labels = merged
        node.last_seen_time_sec = float(timestamp_sec)

    def _touch_node(self, node_id: int, *, timestamp_sec: float) -> None:
        node = self._nodes.get(int(node_id))
        if node is None:
            return
        node.visit_count += 1
        node.last_seen_time_sec = float(timestamp_sec)

    def _connect(self, first_node_id: int, second_node_id: int) -> None:
        if int(first_node_id) == int(second_node_id):
            return
        self._adjacency.setdefault(int(first_node_id), set()).add(int(second_node_id))
        self._adjacency.setdefault(int(second_node_id), set()).add(int(first_node_id))

    def _prune_if_needed(self) -> None:
        max_nodes = max(16, int(self._config.max_nodes))
        if len(self._nodes) <= max_nodes:
            return
        keep_ids = set(self._path_node_ids[-max_nodes:])
        removable_ids = [node_id for node_id in list(self._nodes.keys()) if node_id not in keep_ids]
        for node_id in removable_ids:
            self._nodes.pop(node_id, None)
            self._adjacency.pop(node_id, None)
        for node_id, neighbors in list(self._adjacency.items()):
            self._adjacency[node_id] = {neighbor for neighbor in neighbors if neighbor in self._nodes}
        self._path_node_ids = [node_id for node_id in self._path_node_ids if node_id in self._nodes]
        self._last_anchor_node_id_by_label = {
            label: node_id
            for label, node_id in self._last_anchor_node_id_by_label.items()
            if node_id in self._nodes
        }

    def _log_state(self, *, event: str) -> None:
        signature = (
            str(event),
            len(self._nodes),
            self.edge_count,
            tuple(self._path_node_ids[-4:]),
        )
        if signature == self._last_log_signature:
            return
        self._last_log_signature = signature
        self._logger.info(
            "[Topology] event=%s node_count=%s edge_count=%s recent_path=%s labels=%s",
            event,
            len(self._nodes),
            self.edge_count,
            self._path_node_ids[-6:],
            {
                node_id: self._nodes[node_id].labels
                for node_id in self._path_node_ids[-4:]
                if node_id in self._nodes and self._nodes[node_id].labels
            },
        )


def _dedupe_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = str(label).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _distance_xy(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(np.hypot(float(x0) - float(x1), float(y0) - float(y1)))


def _pose_distance_m(robot_pose: Pose2D, world_xy: list[float]) -> float:
    return _distance_xy(float(robot_pose.x), float(robot_pose.y), float(world_xy[0]), float(world_xy[1]))


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)
