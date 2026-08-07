from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D
from .spatial_transform import estimate_target_world_xy
from .target_grounding import TargetObservation


@dataclass
class TargetTrackerConfig:
    tracker_type: str = "CSRT"
    max_lost_frames: int = 8
    lost_confirmed_sec: float = 2.0
    min_depth_valid_ratio: float = 0.08
    center_patch_ratio: float = 0.35
    min_depth_m: float = 0.20
    max_depth_m: float = 6.0
    horizontal_fov_deg: float = 90.0
    min_bbox_area_ratio: float = 0.0005
    max_bbox_area_ratio: float = 0.70
    confidence_decay_per_lost_frame: float = 0.06
    min_tracking_confidence: float = 0.35


class TargetTracker:
    """Persistent bbox tracker for a semantically initialized landmark.

    YOLOE/VLM verification establishes identity at low frequency. This tracker only
    propagates the same bbox geometrically between semantic detections.
    """

    def __init__(self, *, config: TargetTrackerConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("target_tracker")
        self._tracker = None
        self._cv2 = None
        self._label = ""
        self._lifecycle = "SEARCHING"
        self._lost_count = 0
        self._last_seen_time_sec = -float("inf")
        self._last_detection_time_sec = -float("inf")
        self._identity_confidence = 0.0
        self._identity_vlm_verified = False
        self._identity_vlm_confidence = 0.0
        self._identity_reason = ""
        self._identity_turn_direction_consistent: bool | None = None
        self._identity_turn_relation_verified = False
        self._identity_turn_relation_reason = ""
        self._last_observation: TargetObservation | None = None
        self._last_log_signature: tuple | None = None
        self._last_log_time_sec = -float("inf")

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def last_observation(self) -> TargetObservation | None:
        return self._last_observation

    def set_horizontal_fov_deg(self, value: float) -> None:
        self._config.horizontal_fov_deg = float(value)

    def set_target_label(self, target_label: str) -> None:
        label = _normalize_label(target_label)
        if label != self._label:
            self.reset(target_label=label)

    def reset(self, target_label: str | None = None) -> None:
        self._tracker = None
        self._label = _normalize_label(target_label or "")
        self._lifecycle = "SEARCHING"
        self._lost_count = 0
        self._last_seen_time_sec = -float("inf")
        self._identity_confidence = 0.0
        self._identity_vlm_verified = False
        self._identity_vlm_confidence = 0.0
        self._identity_reason = ""
        self._identity_turn_direction_consistent = None
        self._identity_turn_relation_verified = False
        self._identity_turn_relation_reason = ""
        self._last_observation = None
        self._last_log_signature = None

    def should_request_detection(self, *, target_label: str, timestamp_sec: float) -> bool:
        # This method is called only at decision keyframes. Re-detect there so
        # geometric tracker drift can never become semantic truth.
        _ = target_label, timestamp_sec
        return True

    def initialize_from_detection(
        self,
        *,
        observation: TargetObservation,
        rgb: np.ndarray | None,
        timestamp_sec: float,
    ) -> bool:
        if rgb is None or np.asarray(rgb).ndim != 3:
            return False
        if not _is_confirmable_detection(observation):
            return False
        bbox = _clip_bbox(observation.bbox, image_shape=np.asarray(rgb).shape)
        if bbox is None:
            return False
        tracker = self._create_tracker()
        if tracker is None:
            return False
        x0, y0, x1, y1 = bbox
        cv_bbox = (int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))
        frame = self._tracker_frame(rgb)
        try:
            init_result = tracker.init(frame, cv_bbox)
            # OpenCV tracker bindings return either True or None on success,
            # depending on the OpenCV build.
            ok = True if init_result is None else bool(init_result)
        except Exception as exc:
            self._logger.warning("[TargetTracker] init_failed target=%s error=%s", observation.label, exc)
            return False
        if not ok:
            self._logger.warning("[TargetTracker] init_rejected target=%s bbox=%s", observation.label, list(bbox))
            return False

        self._tracker = tracker
        self._label = _normalize_label(observation.label)
        self._lifecycle = "TRACKING"
        self._lost_count = 0
        self._last_seen_time_sec = float(timestamp_sec)
        self._last_detection_time_sec = float(timestamp_sec)
        self._identity_confidence = float(observation.confidence)
        self._identity_vlm_verified = bool(getattr(observation, "vlm_verified", False))
        self._identity_vlm_confidence = float(getattr(observation, "vlm_verification_confidence", 0.0))
        self._identity_reason = str(getattr(observation, "vlm_verification_reason", ""))
        self._identity_turn_direction_consistent = getattr(
            observation, "turn_direction_consistent", None
        )
        self._identity_turn_relation_verified = bool(
            getattr(observation, "turn_relation_verified", False)
        )
        self._identity_turn_relation_reason = str(
            getattr(observation, "turn_relation_reason", "")
        )
        self._last_observation = observation
        self._log_state(timestamp_sec=timestamp_sec, reason="initialized", bbox=bbox)
        return True

    def update(
        self,
        *,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        robot_pose: Pose2D,
        timestamp_sec: float,
        frame_index: int,
        camera_intrinsics: np.ndarray | None = None,
        camera_view_matrix_ros: np.ndarray | None = None,
        robot_world_position_xyz: np.ndarray | None = None,
        robot_world_orientation_wxyz: np.ndarray | None = None,
    ) -> TargetObservation:
        label = self._label or "target"
        if self._tracker is None:
            return _empty_track_observation(label=label, timestamp_sec=timestamp_sec, reason="tracker_uninitialized")
        if rgb is None or np.asarray(rgb).ndim != 3:
            return _empty_track_observation(label=label, timestamp_sec=timestamp_sec, reason="tracker_rgb_unavailable")

        try:
            frame = self._tracker_frame(rgb)
            ok, raw_bbox = self._tracker.update(frame)
        except Exception as exc:
            self._logger.warning("[TargetTracker] update_failed target=%s error=%s", label, exc)
            return self._mark_lost(timestamp_sec=timestamp_sec, reason="tracker_update_failed")
        if not ok:
            return self._mark_lost(timestamp_sec=timestamp_sec, reason="tracker_lost")

        bbox = _cv_bbox_to_xyxy(raw_bbox, image_shape=np.asarray(rgb).shape)
        if bbox is None or not self._bbox_area_reasonable(bbox=bbox, image_shape=np.asarray(rgb).shape):
            return self._mark_lost(timestamp_sec=timestamp_sec, reason="tracker_bbox_invalid")

        self._lifecycle = "TRACKING"
        self._lost_count = 0
        self._last_seen_time_sec = float(timestamp_sec)
        observation = self._observation_from_bbox(
            bbox=bbox,
            depth=depth,
            robot_pose=robot_pose,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            camera_intrinsics=camera_intrinsics,
            camera_view_matrix_ros=camera_view_matrix_ros,
            robot_world_position_xyz=robot_world_position_xyz,
            robot_world_orientation_wxyz=robot_world_orientation_wxyz,
        )
        self._last_observation = observation
        self._log_state(timestamp_sec=timestamp_sec, reason=observation.reason, bbox=bbox)
        return observation

    def _mark_lost(self, *, timestamp_sec: float, reason: str) -> TargetObservation:
        self._lost_count += 1
        lost_age_sec = max(0.0, float(timestamp_sec) - float(self._last_seen_time_sec))
        if (
            self._lost_count > int(self._config.max_lost_frames)
            or lost_age_sec >= float(self._config.lost_confirmed_sec)
        ):
            self._lifecycle = "LOST_CONFIRMED"
            self._tracker = None
        else:
            self._lifecycle = "LOST_TEMP"
        label = self._label or "target"
        self._log_state(timestamp_sec=timestamp_sec, reason=reason, bbox=None)
        return _empty_track_observation(
            label=label,
            timestamp_sec=timestamp_sec,
            reason=f"{reason}:{self._lifecycle.lower()}",
        )

    def _observation_from_bbox(
        self,
        *,
        bbox: tuple[int, int, int, int],
        depth: np.ndarray | None,
        robot_pose: Pose2D,
        timestamp_sec: float,
        frame_index: int,
        camera_intrinsics: np.ndarray | None = None,
        camera_view_matrix_ros: np.ndarray | None = None,
        robot_world_position_xyz: np.ndarray | None = None,
        robot_world_orientation_wxyz: np.ndarray | None = None,
    ) -> TargetObservation:
        x0, y0, x1, y1 = bbox
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        center_uv = (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)
        image_h = 0
        image_w = 0
        depth_m = None
        valid_ratio = 0.0
        if depth is not None and np.asarray(depth).ndim == 2:
            depth_np = np.asarray(depth, dtype=np.float32)
            image_h, image_w = depth_np.shape[:2]
            depth_m, valid_ratio = _center_patch_depth(
                depth=depth_np,
                bbox=bbox,
                center_patch_ratio=float(self._config.center_patch_ratio),
                min_depth_m=float(self._config.min_depth_m),
                max_depth_m=float(self._config.max_depth_m),
            )

        confidence = max(
            float(self._config.min_tracking_confidence),
            float(self._identity_confidence) - float(self._lost_count) * float(self._config.confidence_decay_per_lost_frame),
        )
        area_ratio = 0.0
        if image_w > 0 and image_h > 0:
            area_ratio = float(width * height) / max(1.0, float(image_w * image_h))
        if depth_m is None or valid_ratio < float(self._config.min_depth_valid_ratio):
            cx_px = max(1.0, (float(image_w) - 1.0) * 0.5)
            half_fov_rad = float(np.deg2rad(float(self._config.horizontal_fov_deg)) * 0.5)
            bearing_rad = float(
                np.arctan(((float(center_uv[0]) - cx_px) / cx_px) * np.tan(half_fov_rad))
            )
            return TargetObservation(
                label=self._label or "target",
                visible=True,
                bbox=bbox,
                center_uv=center_uv,
                confidence=float(confidence),
                depth_m=None,
                bearing_rad=bearing_rad,
                estimated_world_xy=None,
                depth_valid_ratio=float(valid_ratio),
                area_ratio=float(area_ratio),
                source="opencv_tracker",
                reason="depth_invalid",
                timestamp_sec=float(timestamp_sec),
                request_frame_index=int(frame_index),
                request_sim_time_sec=float(timestamp_sec),
                request_robot_pose=robot_pose,
                result_age_sec=0.0,
                is_current_frame=True,
                evidence_level="potential",
                semantic_source="opencv_tracker",
                can_confirm=False,
                vlm_verified=bool(self._identity_vlm_verified),
                vlm_verification_confidence=float(self._identity_vlm_confidence),
                vlm_verification_reason=self._identity_reason,
                turn_direction_consistent=self._identity_turn_direction_consistent,
                turn_relation_verified=bool(self._identity_turn_relation_verified),
                turn_relation_reason=self._identity_turn_relation_reason,
            )

        geometry = estimate_target_world_xy(
            center_uv=center_uv,
            depth_m=float(depth_m),
            image_width=int(image_w),
            image_height=int(image_h),
            horizontal_fov_deg=float(self._config.horizontal_fov_deg),
            robot_pose=robot_pose,
            camera_intrinsics=camera_intrinsics,
            camera_view_matrix_ros=camera_view_matrix_ros,
            robot_world_position_xyz=robot_world_position_xyz,
            robot_world_orientation_wxyz=robot_world_orientation_wxyz,
        )
        return TargetObservation(
            label=self._label or "target",
            visible=True,
            bbox=bbox,
            center_uv=center_uv,
            confidence=float(confidence),
            depth_m=float(depth_m),
            bearing_rad=float(geometry.bearing_rad),
            estimated_world_xy=list(geometry.world_xy),
            depth_valid_ratio=float(valid_ratio),
            area_ratio=float(area_ratio),
            source="opencv_tracker",
            reason="ok",
            timestamp_sec=float(timestamp_sec),
            request_frame_index=int(frame_index),
            request_sim_time_sec=float(timestamp_sec),
            request_robot_pose=robot_pose,
            result_age_sec=0.0,
            is_current_frame=True,
            evidence_level="tracked_identity",
            semantic_source="opencv_tracker",
            can_confirm=False,
            vlm_verified=bool(self._identity_vlm_verified),
            vlm_verification_confidence=float(self._identity_vlm_confidence),
            vlm_verification_reason=self._identity_reason,
            turn_direction_consistent=self._identity_turn_direction_consistent,
            turn_relation_verified=bool(self._identity_turn_relation_verified),
            turn_relation_reason=self._identity_turn_relation_reason,
        )

    def _bbox_area_reasonable(self, *, bbox: tuple[int, int, int, int], image_shape) -> bool:
        image_h = int(image_shape[0])
        image_w = int(image_shape[1])
        x0, y0, x1, y1 = bbox
        area_ratio = float(max(0, x1 - x0) * max(0, y1 - y0)) / max(1.0, float(image_w * image_h))
        return float(self._config.min_bbox_area_ratio) <= area_ratio <= float(self._config.max_bbox_area_ratio)

    def _create_tracker(self):
        try:
            import cv2
        except Exception as exc:
            self._logger.warning("[TargetTracker] cv2_unavailable error=%s", exc)
            return None
        self._cv2 = cv2
        tracker_type = str(self._config.tracker_type).strip().upper() or "CSRT"
        creators = []
        if tracker_type == "CSRT":
            creators.extend([
                getattr(cv2, "TrackerCSRT_create", None),
                getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
            ])
        elif tracker_type == "KCF":
            creators.extend([
                getattr(cv2, "TrackerKCF_create", None),
                getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None),
            ])
        creators.extend([
            getattr(cv2, "TrackerCSRT_create", None),
            getattr(cv2, "TrackerKCF_create", None),
            getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
            getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None),
            getattr(cv2, "TrackerMIL_create", None),
        ])
        for creator in creators:
            if creator is None:
                continue
            try:
                return creator()
            except Exception:
                continue
        self._logger.warning("[TargetTracker] no_supported_opencv_tracker tracker_type=%s", tracker_type)
        return None

    def _tracker_frame(self, rgb: np.ndarray) -> np.ndarray:
        frame_rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[..., :3])
        if self._cv2 is None:
            return frame_rgb
        try:
            return self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR)
        except Exception:
            return frame_rgb

    def _log_state(self, *, timestamp_sec: float, reason: str, bbox: tuple[int, int, int, int] | None) -> None:
        signature = (
            self._label,
            self._lifecycle,
            int(self._lost_count),
            reason,
            None if bbox is None else tuple(int(round(value / 10.0)) for value in bbox),
        )
        if signature == self._last_log_signature and float(timestamp_sec) - float(self._last_log_time_sec) < 3.0:
            return
        self._last_log_signature = signature
        self._last_log_time_sec = float(timestamp_sec)
        self._logger.info(
            "[TargetTracker] target=%s lifecycle=%s lost=%s reason=%s bbox=%s identity_conf=%.2f vlm_verified=%s",
            self._label or "none",
            self._lifecycle,
            int(self._lost_count),
            reason,
            None if bbox is None else list(bbox),
            float(self._identity_confidence),
            int(bool(self._identity_vlm_verified)),
        )


def _is_confirmable_detection(observation: TargetObservation) -> bool:
    if not bool(observation.visible) or observation.bbox is None:
        return False
    if str(observation.reason).strip().lower() != "ok":
        return False
    if "reused" in str(observation.source).strip().lower():
        return False
    if not bool(getattr(observation, "is_current_frame", True)):
        return False
    return bool(getattr(observation, "can_confirm", False))


def _clip_bbox(bbox, *, image_shape) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0 = max(0, min(image_w - 2, x0))
    y0 = max(0, min(image_h - 2, y0))
    x1 = max(x0 + 1, min(image_w - 1, x1))
    y1 = max(y0 + 1, min(image_h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _cv_bbox_to_xyxy(raw_bbox, *, image_shape) -> tuple[int, int, int, int] | None:
    try:
        x, y, w, h = [float(value) for value in raw_bbox]
    except Exception:
        return None
    return _clip_bbox((x, y, x + w, y + h), image_shape=image_shape)


def _center_patch_depth(
    *,
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    center_patch_ratio: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[float | None, float]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    ratio = float(np.clip(center_patch_ratio, 0.05, 1.0))
    patch_w = max(3, int(round(width * ratio)))
    patch_h = max(3, int(round(height * ratio)))
    cx = int(round((x0 + x1) * 0.5))
    cy = int(round((y0 + y1) * 0.5))
    px0 = max(0, cx - patch_w // 2)
    py0 = max(0, cy - patch_h // 2)
    px1 = min(depth.shape[1], px0 + patch_w)
    py1 = min(depth.shape[0], py0 + patch_h)
    patch = np.asarray(depth[py0:py1, px0:px1], dtype=np.float32)
    if patch.size == 0:
        return None, 0.0
    valid = np.isfinite(patch) & (patch >= float(min_depth_m)) & (patch <= float(max_depth_m))
    valid_ratio = float(np.mean(valid))
    if not np.any(valid):
        return None, valid_ratio
    return float(np.median(patch[valid])), valid_ratio


def _empty_track_observation(*, label: str, timestamp_sec: float, reason: str) -> TargetObservation:
    return TargetObservation(
        label=str(label),
        visible=False,
        bbox=None,
        center_uv=None,
        confidence=0.0,
        depth_m=None,
        bearing_rad=None,
        estimated_world_xy=None,
        depth_valid_ratio=0.0,
        area_ratio=0.0,
        source="opencv_tracker",
        reason=str(reason),
        timestamp_sec=float(timestamp_sec),
        depth_region_source="none",
        evidence_level="none",
        semantic_source="opencv_tracker",
        can_confirm=False,
    )


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()
