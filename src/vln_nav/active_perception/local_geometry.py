from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.camera_geometry import backproject_depth_pixels_to_world


UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)

SOURCE_BITS = {
    "front": np.uint8(1),
    "down": np.uint8(2),
    "arm": np.uint8(4),
}


def _logit(probability: float) -> float:
    return math.log(float(probability) / (1.0 - float(probability)))


@dataclass(frozen=True)
class LocalVoxelMapConfig:
    size_x_m: float = 10.0
    size_y_m: float = 10.0
    min_z_m: float = -0.20
    max_z_m: float = 2.80
    resolution_m: float = 0.05
    slide_threshold_m: float = 0.20
    min_depth_m: float = 0.12
    max_depth_m: float = 5.0
    max_ray_length_m: float = 5.0
    pixel_stride: int = 8
    probability_hit: float = 0.85
    probability_miss: float = 0.30
    probability_min: float = 0.12
    probability_max: float = 0.98
    probability_occupied: float = 0.80
    probability_free: float = 0.45

    def validate(self) -> None:
        positive = (
            self.size_x_m,
            self.size_y_m,
            self.max_z_m - self.min_z_m,
            self.resolution_m,
            self.slide_threshold_m,
            self.min_depth_m,
            self.max_depth_m,
            self.max_ray_length_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("Local voxel dimensions, ranges, and resolution must be positive")
        if self.min_depth_m >= self.max_depth_m or int(self.pixel_stride) < 1:
            raise ValueError("Invalid local voxel depth range or pixel stride")
        probabilities = (
            self.probability_hit,
            self.probability_miss,
            self.probability_min,
            self.probability_max,
            self.probability_occupied,
            self.probability_free,
        )
        if not all(math.isfinite(float(value)) and 0.0 < float(value) < 1.0 for value in probabilities):
            raise ValueError("Occupancy probabilities must be finite and within (0, 1)")
        if not (
            self.probability_min
            < self.probability_miss
            < self.probability_free
            < 0.5
            < self.probability_occupied
            < self.probability_hit
            < self.probability_max
        ):
            raise ValueError("Occupancy probabilities are not consistently ordered")


@dataclass(frozen=True)
class VoxelMapUpdate:
    version: int
    shifted: bool
    source_ray_counts: tuple[tuple[str, int], ...]
    hit_endpoint_count: int
    free_voxel_update_count: int
    occupied_voxel_update_count: int

    @property
    def ray_count(self) -> int:
        return sum(count for _, count in self.source_ray_counts)


@dataclass(frozen=True)
class LocalVoxelSnapshot:
    state: np.ndarray
    evidence_count: np.ndarray
    observed_mask: np.ndarray
    source_mask: np.ndarray
    last_observed_time_sec: np.ndarray
    log_odds: np.ndarray
    origin_world_xyz: np.ndarray
    center_world_xyz: np.ndarray
    resolution_m: float
    version: int
    timestamp_sec: float

    def world_to_indices(self, points_world_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world_xyz, dtype=np.float64).reshape(-1, 3)
        xyz = np.floor((points - self.origin_world_xyz) / self.resolution_m).astype(np.int64)
        zyx = xyz[:, [2, 1, 0]]
        shape = np.asarray(self.state.shape, dtype=np.int64)
        inside = np.all((zyx >= 0) & (zyx < shape), axis=1)
        return zyx, inside

    def query(self, points_world_xyz: np.ndarray) -> np.ndarray:
        zyx, inside = self.world_to_indices(points_world_xyz)
        result = np.full(len(zyx), UNKNOWN, dtype=np.int8)
        valid = zyx[inside]
        result[inside] = self.state[valid[:, 0], valid[:, 1], valid[:, 2]]
        return result

    def source_voxel_count(self, source: str) -> int:
        if source not in SOURCE_BITS:
            raise ValueError(f"Unknown RGB-D source: {source}")
        return int(np.count_nonzero(self.source_mask & SOURCE_BITS[source]))


class LocalVoxelMap:
    """Fixed-memory local map whose axes remain aligned with the world frame.

    The window follows the robot by integer voxel shifts. Overlapping voxels
    retain their evidence; newly exposed cells are reset to unknown. Each RGB-D
    source raycasts from its own timestamped optical pose.
    """

    def __init__(self, config: LocalVoxelMapConfig) -> None:
        config.validate()
        self.config = config
        shape_xyz = np.ceil(
            np.asarray(
                [
                    config.size_x_m,
                    config.size_y_m,
                    config.max_z_m - config.min_z_m,
                ],
                dtype=np.float64,
            )
            / config.resolution_m
        ).astype(np.int64)
        shape_xyz += shape_xyz % 2
        self._shape_xyz = shape_xyz
        shape_zyx = tuple(map(int, shape_xyz[::-1]))
        self._origin = np.zeros(3, dtype=np.float64)
        self._center = np.zeros(3, dtype=np.float64)
        self._log_odds = np.zeros(shape_zyx, dtype=np.float32)
        self._observed = np.zeros(shape_zyx, dtype=bool)
        self._evidence = np.zeros(shape_zyx, dtype=np.uint16)
        self._source_mask = np.zeros(shape_zyx, dtype=np.uint8)
        self._last_observed = np.full(shape_zyx, -np.inf, dtype=np.float32)
        self._version = 0
        self._timestamp = -float("inf")
        self._initialized = False
        self._hit_increment = np.float32(_logit(config.probability_hit))
        self._miss_increment = np.float32(_logit(config.probability_miss))
        self._minimum_log_odds = np.float32(_logit(config.probability_min))
        self._maximum_log_odds = np.float32(_logit(config.probability_max))
        self._occupied_threshold = np.float32(_logit(config.probability_occupied))
        self._free_threshold = np.float32(_logit(config.probability_free))

    @property
    def version(self) -> int:
        return self._version

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self, center_world_xyz: tuple[float, float, float] | np.ndarray) -> None:
        center = self._validate_center(center_world_xyz)
        self._origin = self._desired_origin(center)
        self._center = self._tracking_center_from_origin()
        self._clear_all()
        self._version = 0
        self._timestamp = -float("inf")
        self._initialized = True

    def integrate_batch(
        self,
        observations: tuple[TimedDepthObservation, ...] | list[TimedDepthObservation],
        *,
        center_world_xyz: tuple[float, float, float] | np.ndarray,
    ) -> VoxelMapUpdate:
        if not observations:
            raise ValueError("At least one RGB-D observation is required")
        center = self._validate_center(center_world_xyz)
        if not self._initialized:
            self.reset(center)
        shifted = self._slide_if_needed(center)
        source_counts: list[tuple[str, int]] = []
        hit_count = free_count = occupied_count = 0
        newest_timestamp = self._timestamp
        for observation in observations:
            observation.validate()
            if observation.source not in SOURCE_BITS:
                raise ValueError(
                    f"Unsupported RGB-D source {observation.source!r}; expected {sorted(SOURCE_BITS)}"
                )
            free_indices, occupied_indices, ray_count, endpoint_count = self._observation_indices(
                observation
            )
            source_counts.append((observation.source, ray_count))
            source_bit = SOURCE_BITS[observation.source]
            timestamp = float(observation.timestamp_sec)
            if free_indices.size:
                self._update_indices(
                    free_indices,
                    increment=self._miss_increment,
                    source_bit=source_bit,
                    timestamp_sec=timestamp,
                )
                free_count += int(free_indices.size)
            if occupied_indices.size:
                self._update_indices(
                    occupied_indices,
                    increment=self._hit_increment,
                    source_bit=source_bit,
                    timestamp_sec=timestamp,
                )
                occupied_count += int(occupied_indices.size)
            hit_count += int(endpoint_count)
            newest_timestamp = max(newest_timestamp, timestamp)
        if any(count > 0 for _, count in source_counts):
            self._version += 1
            self._timestamp = newest_timestamp
        return VoxelMapUpdate(
            version=int(self._version),
            shifted=bool(shifted),
            source_ray_counts=tuple(source_counts),
            hit_endpoint_count=int(hit_count),
            free_voxel_update_count=int(free_count),
            occupied_voxel_update_count=int(occupied_count),
        )

    def integrate(self, observation: TimedDepthObservation) -> int:
        """Integrate one frame without moving the current window."""
        if not self._initialized:
            camera_origin = np.asarray(observation.camera_to_world, dtype=np.float64)[:3, 3]
            self.reset(camera_origin)
        update = self.integrate_batch([observation], center_world_xyz=self._center)
        return update.ray_count

    def snapshot(self) -> LocalVoxelSnapshot:
        state = np.full(self._log_odds.shape, UNKNOWN, dtype=np.int8)
        state[self._observed & (self._log_odds <= self._free_threshold)] = FREE
        state[self._observed & (self._log_odds >= self._occupied_threshold)] = OCCUPIED
        return LocalVoxelSnapshot(
            state=state,
            evidence_count=self._evidence.copy(),
            observed_mask=self._observed.copy(),
            source_mask=self._source_mask.copy(),
            last_observed_time_sec=self._last_observed.copy(),
            log_odds=self._log_odds.copy(),
            origin_world_xyz=self._origin.copy(),
            center_world_xyz=self._center.copy(),
            resolution_m=float(self.config.resolution_m),
            version=int(self._version),
            timestamp_sec=float(self._timestamp),
        )

    def _observation_indices(
        self, observation: TimedDepthObservation
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        depth = np.asarray(observation.depth_m, dtype=np.float64)
        stride = int(self.config.pixel_stride)
        rows = np.arange(stride // 2, depth.shape[0], stride, dtype=np.intp)
        columns = np.arange(stride // 2, depth.shape[1], stride, dtype=np.intp)
        if rows.size == 0 or columns.size == 0:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty, 0, 0
        sampled = depth[np.ix_(rows, columns)]
        is_hit = (
            np.isfinite(sampled)
            & (sampled >= self.config.min_depth_m)
            & (sampled <= self.config.max_depth_m)
        )
        is_no_return = np.isposinf(sampled) | (
            np.isfinite(sampled) & (sampled > self.config.max_depth_m)
        )
        usable = is_hit | is_no_return
        if not np.any(usable):
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty, 0, 0
        uu, vv = np.meshgrid(columns.astype(np.float64), rows.astype(np.float64))
        z_depth = np.where(is_hit, sampled, self.config.max_depth_m)
        endpoints = backproject_depth_pixels_to_world(
            pixel_u=uu[usable],
            pixel_v=vv[usable],
            depth_m=z_depth[usable],
            camera_intrinsics=observation.intrinsics,
            camera_to_world=observation.camera_to_world,
        ).reshape(-1, 3)
        endpoint_is_hit = is_hit[usable].reshape(-1)
        origin = np.asarray(observation.camera_to_world, dtype=np.float64)[:3, 3]
        deltas = endpoints - origin
        lengths = np.linalg.norm(deltas, axis=1)
        valid = np.isfinite(endpoints).all(axis=1) & np.isfinite(lengths) & (lengths > 1.0e-6)
        endpoints = endpoints[valid]
        endpoint_is_hit = endpoint_is_hit[valid]
        deltas = deltas[valid]
        lengths = lengths[valid]
        if not lengths.size:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty, 0, 0
        clipped_lengths = np.minimum(lengths, float(self.config.max_ray_length_m))
        directions = deltas / lengths[:, None]
        clipped_endpoints = origin + directions * clipped_lengths[:, None]
        endpoint_is_hit &= lengths <= float(self.config.max_ray_length_m)
        occupied = self._points_to_linear_indices(clipped_endpoints[endpoint_is_hit])
        free_chunks: list[np.ndarray] = []
        resolution = float(self.config.resolution_m)
        chunk_size = 512
        for start in range(0, len(directions), chunk_size):
            stop = min(len(directions), start + chunk_size)
            chunk_directions = directions[start:stop]
            chunk_lengths = clipped_lengths[start:stop]
            chunk_hits = endpoint_is_hit[start:stop]
            maximum_steps = max(1, int(math.ceil(float(np.max(chunk_lengths)) / resolution)))
            sample_distances = (np.arange(maximum_steps, dtype=np.float64) + 0.5) * resolution
            free_limit = chunk_lengths - np.where(chunk_hits, 0.75 * resolution, 0.0)
            within = sample_distances[None, :] < free_limit[:, None]
            if not np.any(within):
                continue
            points = origin + chunk_directions[:, None, :] * sample_distances[None, :, None]
            free_chunks.append(self._points_to_linear_indices(points[within]))
        if free_chunks:
            free = np.unique(np.concatenate(free_chunks))
            if occupied.size:
                free = np.setdiff1d(free, occupied, assume_unique=True)
        else:
            free = np.empty((0,), dtype=np.int64)
        return free, occupied, int(len(directions)), int(np.count_nonzero(endpoint_is_hit))

    def _points_to_linear_indices(self, points_world_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_xyz, dtype=np.float64).reshape(-1, 3)
        if not points.size:
            return np.empty((0,), dtype=np.int64)
        xyz = np.floor((points - self._origin) / self.config.resolution_m).astype(np.int64)
        inside = np.all((xyz >= 0) & (xyz < self._shape_xyz), axis=1)
        xyz = xyz[inside]
        if not xyz.size:
            return np.empty((0,), dtype=np.int64)
        nx, ny, _ = map(int, self._shape_xyz)
        linear = (xyz[:, 2] * ny + xyz[:, 1]) * nx + xyz[:, 0]
        return np.unique(linear.astype(np.int64))

    def _update_indices(
        self,
        linear_indices: np.ndarray,
        *,
        increment: np.float32,
        source_bit: np.uint8,
        timestamp_sec: float,
    ) -> None:
        flat_log_odds = self._log_odds.ravel()
        updated = np.clip(
            flat_log_odds[linear_indices] + increment,
            self._minimum_log_odds,
            self._maximum_log_odds,
        )
        flat_log_odds[linear_indices] = updated
        self._observed.ravel()[linear_indices] = True
        evidence = self._evidence.ravel()
        evidence[linear_indices] = np.minimum(
            evidence[linear_indices].astype(np.uint32) + 1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        self._source_mask.ravel()[linear_indices] |= source_bit
        self._last_observed.ravel()[linear_indices] = np.float32(timestamp_sec)

    def _slide_if_needed(self, center: np.ndarray) -> bool:
        if np.max(np.abs(center - self._center)) < float(self.config.slide_threshold_m):
            return False
        desired_origin = self._desired_origin(center)
        shift_xyz = np.rint(
            (desired_origin - self._origin) / float(self.config.resolution_m)
        ).astype(np.int64)
        if not np.any(shift_xyz):
            return False
        if np.any(np.abs(shift_xyz) >= self._shape_xyz):
            self.reset(center)
            return True
        shift_zyx = shift_xyz[::-1]
        self._log_odds = self._roll_and_clear(self._log_odds, shift_zyx, 0.0)
        self._observed = self._roll_and_clear(self._observed, shift_zyx, False)
        self._evidence = self._roll_and_clear(self._evidence, shift_zyx, 0)
        self._source_mask = self._roll_and_clear(self._source_mask, shift_zyx, 0)
        self._last_observed = self._roll_and_clear(self._last_observed, shift_zyx, -np.inf)
        self._origin = self._origin + shift_xyz.astype(np.float64) * self.config.resolution_m
        self._center = self._tracking_center_from_origin()
        return True

    @staticmethod
    def _roll_and_clear(array: np.ndarray, shift_zyx: np.ndarray, fill_value) -> np.ndarray:
        shifted = np.roll(array, shift=tuple(map(int, -shift_zyx)), axis=(0, 1, 2))
        for axis, shift in enumerate(map(int, shift_zyx)):
            if shift == 0:
                continue
            slices = [slice(None)] * 3
            slices[axis] = slice(-shift, None) if shift > 0 else slice(None, -shift)
            shifted[tuple(slices)] = fill_value
        return shifted

    def _desired_origin(self, center: np.ndarray) -> np.ndarray:
        resolution = float(self.config.resolution_m)
        snapped = np.rint(center / resolution) * resolution
        return np.asarray(
            [
                snapped[0] - 0.5 * self._shape_xyz[0] * resolution,
                snapped[1] - 0.5 * self._shape_xyz[1] * resolution,
                snapped[2] + float(self.config.min_z_m),
            ],
            dtype=np.float64,
        )

    def _tracking_center_from_origin(self) -> np.ndarray:
        resolution = float(self.config.resolution_m)
        return np.asarray(
            [
                self._origin[0] + 0.5 * self._shape_xyz[0] * resolution,
                self._origin[1] + 0.5 * self._shape_xyz[1] * resolution,
                self._origin[2] - float(self.config.min_z_m),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _validate_center(center_world_xyz: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        center = np.asarray(center_world_xyz, dtype=np.float64).reshape(-1)
        if center.size != 3 or not np.isfinite(center).all():
            raise ValueError("Local voxel map center must be a finite XYZ point")
        return center.copy()

    def _clear_all(self) -> None:
        self._log_odds.fill(0.0)
        self._observed.fill(False)
        self._evidence.fill(0)
        self._source_mask.fill(0)
        self._last_observed.fill(-np.inf)
