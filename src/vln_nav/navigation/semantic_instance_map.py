from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rgbd_geometry_map import backproject_depth_pixels_to_world
from .yoloe_instance_segmenter import InstanceMaskObservation, color_for_label


@dataclass(frozen=True)
class SemanticInstanceMapConfig:
    min_depth_m: float = 0.30
    max_depth_m: float = 8.0
    mask_pixel_stride: int = 4
    min_valid_depth_pixels: int = 40
    min_valid_depth_ratio: float = 0.12
    point_trim_quantile: float = 0.05
    point_voxel_size_m: float = 0.05
    max_points_per_instance: int = 2000
    merge_max_centroid_distance_m: float = 0.90
    merge_min_bev_iou: float = 0.02
    merge_max_bev_gap_m: float = 0.30
    confirmed_min_observations: int = 2


@dataclass(frozen=True)
class SemanticInstanceSnapshot:
    instance_id: int
    label: str
    confidence: float
    world_xyz: tuple[float, float, float]
    world_bounds_min_xyz: tuple[float, float, float]
    world_bounds_max_xyz: tuple[float, float, float]
    observation_count: int
    first_seen_time_sec: float
    last_seen_time_sec: float
    color_rgb: tuple[int, int, int]
    confirmed: bool


@dataclass(frozen=True)
class SemanticMapUpdate:
    input_detection_count: int
    projected_detection_count: int
    rejected_depth_count: int
    created_instance_count: int
    merged_instance_count: int
    total_instance_count: int


@dataclass
class _StoredInstance:
    instance_id: int
    label: str
    confidence: float
    world_xyz: np.ndarray
    world_bounds_min_xyz: np.ndarray
    world_bounds_max_xyz: np.ndarray
    observation_count: int
    first_seen_time_sec: float
    last_seen_time_sec: float
    color_rgb: tuple[int, int, int]
    points_world: np.ndarray


@dataclass(frozen=True)
class _ProjectedDetection:
    observation: InstanceMaskObservation
    points_world: np.ndarray
    world_xyz: np.ndarray
    world_bounds_min_xyz: np.ndarray
    world_bounds_max_xyz: np.ndarray
    valid_depth_ratio: float


class SemanticInstanceMap:
    """Persistent world-frame instances projected from same-frame mask and depth."""

    def __init__(self, *, config: SemanticInstanceMapConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("semantic_instance_map")
        self._instances: dict[int, _StoredInstance] = {}
        self._next_instance_id = 1
        self._last_update = SemanticMapUpdate(0, 0, 0, 0, 0, 0)

    @property
    def last_update(self) -> SemanticMapUpdate:
        return self._last_update

    def clear(self) -> None:
        self._instances.clear()
        self._next_instance_id = 1
        self._last_update = SemanticMapUpdate(0, 0, 0, 0, 0, 0)

    def snapshots(self) -> tuple[SemanticInstanceSnapshot, ...]:
        return tuple(
            SemanticInstanceSnapshot(
                instance_id=instance.instance_id,
                label=instance.label,
                confidence=float(instance.confidence),
                world_xyz=tuple(float(value) for value in instance.world_xyz),
                world_bounds_min_xyz=tuple(
                    float(value) for value in instance.world_bounds_min_xyz
                ),
                world_bounds_max_xyz=tuple(
                    float(value) for value in instance.world_bounds_max_xyz
                ),
                observation_count=int(instance.observation_count),
                first_seen_time_sec=float(instance.first_seen_time_sec),
                last_seen_time_sec=float(instance.last_seen_time_sec),
                color_rgb=instance.color_rgb,
                confirmed=(
                    int(instance.observation_count)
                    >= max(1, int(self._config.confirmed_min_observations))
                ),
            )
            for instance in sorted(self._instances.values(), key=lambda item: item.instance_id)
        )

    def update(
        self,
        *,
        observations: list[InstanceMaskObservation],
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_view_matrix_ros: np.ndarray,
        timestamp_sec: float,
    ) -> SemanticMapUpdate:
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Semantic map expects HxW depth, got shape={depth.shape}")
        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
        view_matrix = np.asarray(camera_view_matrix_ros, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(view_matrix)):
            raise ValueError("Semantic map camera calibration contains non-finite values")
        camera_to_world = np.linalg.inv(view_matrix)

        projected_count = 0
        rejected_depth_count = 0
        created_count = 0
        merged_count = 0
        for observation in observations:
            projected = self._project_observation(
                observation=observation,
                depth=depth,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
            if projected is None:
                rejected_depth_count += 1
                continue
            projected_count += 1
            matching_id = self._find_matching_instance(projected)
            if matching_id is None:
                self._create_instance(projected, timestamp_sec=float(timestamp_sec))
                created_count += 1
            else:
                self._merge_instance(
                    self._instances[matching_id],
                    projected,
                    timestamp_sec=float(timestamp_sec),
                )
                merged_count += 1

        self._last_update = SemanticMapUpdate(
            input_detection_count=len(observations),
            projected_detection_count=projected_count,
            rejected_depth_count=rejected_depth_count,
            created_instance_count=created_count,
            merged_instance_count=merged_count,
            total_instance_count=len(self._instances),
        )
        return self._last_update

    def _project_observation(
        self,
        *,
        observation: InstanceMaskObservation,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> _ProjectedDetection | None:
        mask = np.asarray(observation.mask, dtype=bool)
        if mask.shape != depth.shape:
            self._logger.warning(
                "[SemanticMap] reject label=%s reason=mask_depth_shape mask=%s depth=%s",
                observation.label,
                mask.shape,
                depth.shape,
            )
            return None
        stride = max(1, int(self._config.mask_pixel_stride))
        sampled_mask = mask[::stride, ::stride]
        sampled_v, sampled_u = np.nonzero(sampled_mask)
        if sampled_u.size == 0:
            return None
        pixel_u = sampled_u.astype(np.intp) * stride
        pixel_v = sampled_v.astype(np.intp) * stride
        depth_values = depth[pixel_v, pixel_u].astype(np.float64)
        valid_depth = (
            np.isfinite(depth_values)
            & (depth_values >= float(self._config.min_depth_m))
            & (depth_values <= float(self._config.max_depth_m))
        )
        valid_count = int(np.count_nonzero(valid_depth))
        valid_ratio = float(valid_count / max(1, depth_values.size))
        if (
            valid_count < max(1, int(self._config.min_valid_depth_pixels))
            or valid_ratio < float(self._config.min_valid_depth_ratio)
        ):
            self._logger.info(
                "[SemanticMap] reject label=%s reason=depth valid=%s ratio=%.3f",
                observation.label,
                valid_count,
                valid_ratio,
            )
            return None

        points_world = backproject_depth_pixels_to_world(
            pixel_u=pixel_u[valid_depth],
            pixel_v=pixel_v[valid_depth],
            depth_m=depth_values[valid_depth],
            camera_intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        ).reshape(-1, 3)
        finite = np.all(np.isfinite(points_world), axis=1)
        points_world = points_world[finite]
        if points_world.shape[0] < max(1, int(self._config.min_valid_depth_pixels)):
            return None
        points_world = self._downsample_points(points_world)
        quantile = float(np.clip(self._config.point_trim_quantile, 0.0, 0.24))
        lower = np.quantile(points_world, quantile, axis=0)
        upper = np.quantile(points_world, 1.0 - quantile, axis=0)
        trimmed = points_world[np.all((points_world >= lower) & (points_world <= upper), axis=1)]
        if trimmed.shape[0] >= max(10, int(self._config.min_valid_depth_pixels) // 2):
            points_world = trimmed
        world_xyz = np.median(points_world, axis=0)
        world_bounds_min = np.quantile(points_world, quantile, axis=0)
        world_bounds_max = np.quantile(points_world, 1.0 - quantile, axis=0)
        return _ProjectedDetection(
            observation=observation,
            points_world=points_world,
            world_xyz=world_xyz,
            world_bounds_min_xyz=world_bounds_min,
            world_bounds_max_xyz=world_bounds_max,
            valid_depth_ratio=valid_ratio,
        )

    def _find_matching_instance(self, projected: _ProjectedDetection) -> int | None:
        best_id = None
        best_distance = float("inf")
        label = _normalize_label(projected.observation.label)
        for instance_id, instance in self._instances.items():
            if _normalize_label(instance.label) != label:
                continue
            distance = float(np.linalg.norm(instance.world_xyz[:2] - projected.world_xyz[:2]))
            if distance > float(self._config.merge_max_centroid_distance_m):
                continue
            iou = _bev_iou(
                instance.world_bounds_min_xyz[:2],
                instance.world_bounds_max_xyz[:2],
                projected.world_bounds_min_xyz[:2],
                projected.world_bounds_max_xyz[:2],
            )
            gap = _bev_gap(
                instance.world_bounds_min_xyz[:2],
                instance.world_bounds_max_xyz[:2],
                projected.world_bounds_min_xyz[:2],
                projected.world_bounds_max_xyz[:2],
            )
            if (
                iou < float(self._config.merge_min_bev_iou)
                and gap > float(self._config.merge_max_bev_gap_m)
            ):
                continue
            if distance < best_distance:
                best_distance = distance
                best_id = instance_id
        return best_id

    def _create_instance(
        self,
        projected: _ProjectedDetection,
        *,
        timestamp_sec: float,
    ) -> None:
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        self._instances[instance_id] = _StoredInstance(
            instance_id=instance_id,
            label=projected.observation.label,
            confidence=float(projected.observation.confidence),
            world_xyz=projected.world_xyz.copy(),
            world_bounds_min_xyz=projected.world_bounds_min_xyz.copy(),
            world_bounds_max_xyz=projected.world_bounds_max_xyz.copy(),
            observation_count=1,
            first_seen_time_sec=float(timestamp_sec),
            last_seen_time_sec=float(timestamp_sec),
            color_rgb=color_for_label(projected.observation.label),
            points_world=projected.points_world.copy(),
        )
        self._logger.warning(
            "[SemanticMap] create id=%s label=%s confidence=%.2f world=[%.2f,%.2f,%.2f]",
            instance_id,
            projected.observation.label,
            projected.observation.confidence,
            *projected.world_xyz,
        )

    def _merge_instance(
        self,
        instance: _StoredInstance,
        projected: _ProjectedDetection,
        *,
        timestamp_sec: float,
    ) -> None:
        previous_count = int(instance.observation_count)
        combined = np.concatenate((instance.points_world, projected.points_world), axis=0)
        instance.points_world = self._downsample_points(combined)
        quantile = float(np.clip(self._config.point_trim_quantile, 0.0, 0.24))
        instance.world_xyz = np.median(instance.points_world, axis=0)
        instance.world_bounds_min_xyz = np.quantile(instance.points_world, quantile, axis=0)
        instance.world_bounds_max_xyz = np.quantile(
            instance.points_world,
            1.0 - quantile,
            axis=0,
        )
        instance.confidence = (
            instance.confidence * previous_count + float(projected.observation.confidence)
        ) / float(previous_count + 1)
        instance.observation_count = previous_count + 1
        instance.last_seen_time_sec = float(timestamp_sec)
        self._logger.info(
            "[SemanticMap] merge id=%s label=%s observations=%s world=[%.2f,%.2f,%.2f]",
            instance.instance_id,
            instance.label,
            instance.observation_count,
            *instance.world_xyz,
        )

    def _downsample_points(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        voxel_size = max(1.0e-3, float(self._config.point_voxel_size_m))
        voxel_keys = np.floor(points / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        points = points[np.sort(unique_indices)]
        max_points = max(32, int(self._config.max_points_per_instance))
        if points.shape[0] > max_points:
            keep = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.intp)
            points = points[keep]
        return points


def _normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def _bev_iou(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    min_a = np.asarray(min_a, dtype=np.float64).reshape(2)
    max_a = np.asarray(max_a, dtype=np.float64).reshape(2)
    min_b = np.asarray(min_b, dtype=np.float64).reshape(2)
    max_b = np.asarray(max_b, dtype=np.float64).reshape(2)
    intersection_extent = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    intersection = float(intersection_extent[0] * intersection_extent[1])
    area_a = float(np.prod(np.maximum(0.02, max_a - min_a)))
    area_b = float(np.prod(np.maximum(0.02, max_b - min_b)))
    return intersection / max(1.0e-9, area_a + area_b - intersection)


def _bev_gap(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    min_a = np.asarray(min_a, dtype=np.float64).reshape(2)
    max_a = np.asarray(max_a, dtype=np.float64).reshape(2)
    min_b = np.asarray(min_b, dtype=np.float64).reshape(2)
    max_b = np.asarray(max_b, dtype=np.float64).reshape(2)
    axis_gap = np.maximum(0.0, np.maximum(min_a, min_b) - np.minimum(max_a, max_b))
    return float(np.linalg.norm(axis_gap))
