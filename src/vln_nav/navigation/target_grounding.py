from __future__ import annotations

import io
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetection,
    GroundingDinoDetector,
)
from .observation_adapter import Pose2D
from .open_nav_client import OpenNavLLMClient
from .sam_segmenter import SamSegmenter, SamSegmenterConfig
from .spatial_transform import estimate_target_world_xy


@dataclass
class TargetObservation:
    label: str
    visible: bool
    bbox: tuple[int, int, int, int] | None
    center_uv: tuple[float, float] | None
    confidence: float
    depth_m: float | None
    bearing_rad: float | None
    estimated_world_xy: list[float] | None
    depth_valid_ratio: float
    area_ratio: float
    source: str
    reason: str
    timestamp_sec: float
    mask_area_ratio: float = 0.0
    coordinate_mode: str = "unknown"
    segmentation_source: str = "none"
    mask: np.ndarray | None = None
    request_frame_index: int = -1
    request_sim_time_sec: float = 0.0
    request_robot_pose: Pose2D | None = None
    result_age_sec: float = 0.0
    is_current_frame: bool = True
    evidence_level: str = "none"  # none / potential / confirmed_candidate
    semantic_source: str = "unknown"
    can_confirm: bool = False
    vlm_verified: bool = False
    vlm_verification_confidence: float = 0.0
    vlm_verification_reason: str = ""
    requires_viewpoint_confirmation: bool = False


@dataclass
class TargetGroundingConfig:
    enabled: bool
    backend: str
    call_period_sec: float
    reuse_last_observation_sec: float
    upload_max_edge_px: int
    min_bbox_area_ratio: float
    max_bbox_area_ratio: float
    min_bbox_width_px: int
    min_bbox_height_px: int
    max_bbox_aspect_ratio: float
    min_depth_valid_ratio: float
    depth_statistic: str
    center_patch_ratio: float
    min_depth_m: float
    max_depth_m: float
    min_confidence: float
    horizontal_fov_deg: float
    coordinate_mode: str = "auto"
    allow_normalized_bbox: bool = True
    normalized_bbox_scale: float = 1000.0
    depth_region: str = "bbox_depth_mask"
    depth_mask_tolerance_ratio: float = 0.35
    depth_mask_min_pixels: int = 16
    grounding_dino_repo_path: str = "../GroundingDINO"
    grounding_dino_config_path: str = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint_path: str = "../GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_device: str = "cuda"
    grounding_dino_box_threshold: float = 0.45
    grounding_dino_text_threshold: float = 0.25
    grounding_dino_text_encoder_path: str = "weights/grounded_sam/bert-base-uncased"
    grounding_dino_max_detections: int = 5
    grounding_dino_nms_iou_threshold: float = 0.50
    sam_checkpoint_path: str = "weights/grounded_sam/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    sam_device: str = "cuda"
    sam_min_mask_area_ratio: float = 0.0005
    sam_enabled: bool = True
    max_result_age_sec: float = 30.0
    max_result_pose_delta_m: float = 0.05
    max_result_yaw_delta_deg: float = 3.0
    semantic_verification_enabled: bool = True
    semantic_verification_min_confidence: float = 0.70
    semantic_verification_max_edge_px: int = 768
    semantic_verification_auto_accept_confidence: float = 0.75


@dataclass
class _GroundingRequest:
    target_label: str
    rgb: np.ndarray
    rgb_shape: tuple[int, int]
    depth: np.ndarray | None
    robot_pose: Pose2D
    timestamp_sec: float
    upload_size: tuple[int, int]
    original_size: tuple[int, int]
    image_png_bytes: bytes
    frame_index: int = -1
    preferred_bbox: tuple[int, int, int, int] | None = None
    preferred_world_xy: list[float] | None = None
    camera_intrinsics: np.ndarray | None = None
    camera_view_matrix_ros: np.ndarray | None = None
    robot_world_position_xyz: np.ndarray | None = None
    robot_world_orientation_wxyz: np.ndarray | None = None


@dataclass
class _MappedBbox:
    bbox: tuple[int, int, int, int]
    coordinate_mode: str


class TargetGrounder:
    def __init__(
        self,
        *,
        config: TargetGroundingConfig,
        llm_client: OpenNavLLMClient,
        logger,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._logger = logger.getChild("target_grounding")
        backend = str(config.backend).strip().lower()
        if backend not in {"grounding_dino", "dino"}:
            raise ValueError(
                f"target grounding backend must be grounding_dino, got {config.backend!r}"
            )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="target_grounding")
        self._future: Future | None = None
        self._future_request: _GroundingRequest | None = None
        self._last_call_started_sec = -float("inf")
        self._last_call_started_wall_sec = -float("inf")
        self._last_observation: TargetObservation | None = None
        self._last_observation_wall_sec = -float("inf")
        self._last_observation_log_signature: tuple | None = None
        self._last_observation_log_wall_sec = -float("inf")
        self._last_verification_signature: tuple | None = None
        self._last_verification_result: tuple[bool, float, str] | None = None
        self._grounding_dino = GroundingDinoDetector(
            config=GroundingDinoConfig(
                enabled=True,
                repo_path=str(config.grounding_dino_repo_path),
                config_path=str(config.grounding_dino_config_path),
                checkpoint_path=str(config.grounding_dino_checkpoint_path),
                device=str(config.grounding_dino_device),
                box_threshold=float(config.grounding_dino_box_threshold),
                text_threshold=float(config.grounding_dino_text_threshold),
                text_encoder_path=str(config.grounding_dino_text_encoder_path),
                max_detections=int(config.grounding_dino_max_detections),
                nms_iou_threshold=float(config.grounding_dino_nms_iou_threshold),
            ),
            logger=logger,
        )
        self._sam = SamSegmenter(
            config=SamSegmenterConfig(
                checkpoint_path=str(config.sam_checkpoint_path),
                model_type=str(config.sam_model_type),
                device=str(config.sam_device),
                min_mask_area_ratio=float(config.sam_min_mask_area_ratio),
                enabled=bool(config.sam_enabled),
            ),
            logger=logger,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def request_in_flight(self) -> bool:
        return self._future is not None and not self._future.done()

    def set_horizontal_fov_deg(self, value: float) -> None:
        self._config.horizontal_fov_deg = float(value)

    def observe(
        self,
        *,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        target_label: str,
        robot_pose: Pose2D,
        timestamp_sec: float,
        frame_index: int = -1,
        allow_new_request: bool = True,
        preferred_world_xy: list[float] | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_view_matrix_ros: np.ndarray | None = None,
        robot_world_position_xyz: np.ndarray | None = None,
        robot_world_orientation_wxyz: np.ndarray | None = None,
    ) -> TargetObservation:
        target_label = str(target_label).strip() or "target"
        if target_label.lower() in {"none", "null", "unknown", "n/a", "na", "target", "object", "thing", "place", "location"}:
            return _empty_observation(target_label=target_label, timestamp_sec=timestamp_sec, reason="invalid_label")
        wall_time_sec = time.monotonic()
        completed_observation = self._collect_finished_request(
            timestamp_sec=timestamp_sec,
            wall_time_sec=wall_time_sec,
            current_robot_pose=robot_pose,
        )
        if completed_observation is not None and completed_observation.label == target_label:
            return completed_observation

        if not bool(self._config.enabled):
            return _empty_observation(target_label=target_label, timestamp_sec=timestamp_sec, reason="disabled")
        if rgb is None or np.asarray(rgb).ndim != 3:
            return self._reuse_or_empty(target_label=target_label, timestamp_sec=timestamp_sec, reason="rgb_unavailable")

        reusable = self._reusable_observation(
            target_label=target_label,
            timestamp_sec=timestamp_sec,
            wall_time_sec=wall_time_sec,
            current_robot_pose=robot_pose,
        )
        if not bool(allow_new_request):
            return reusable or _empty_observation(
                target_label=target_label,
                timestamp_sec=timestamp_sec,
                reason="grounding_paused",
            )
        if self._future is not None:
            return reusable or _empty_observation(
                target_label=target_label,
                timestamp_sec=timestamp_sec,
                reason="grounding_request_in_flight",
            )

        if float(wall_time_sec) - float(self._last_call_started_wall_sec) < max(0.0, float(self._config.call_period_sec)):
            return reusable or _empty_observation(
                target_label=target_label,
                timestamp_sec=timestamp_sec,
                reason="grounding_call_period",
            )

        try:
            request = self._build_request(
                rgb=np.asarray(rgb, dtype=np.uint8),
                depth=None if depth is None else np.asarray(depth, dtype=np.float32).copy(),
                target_label=target_label,
                robot_pose=robot_pose,
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                preferred_world_xy=preferred_world_xy,
                camera_intrinsics=camera_intrinsics,
                camera_view_matrix_ros=camera_view_matrix_ros,
                robot_world_position_xyz=robot_world_position_xyz,
                robot_world_orientation_wxyz=robot_world_orientation_wxyz,
            )
        except Exception as exc:
            self._logger.warning("[GroundingReject] target=%s reason=image_prepare_failed error=%s", target_label, exc)
            return reusable or _empty_observation(
                target_label=target_label,
                timestamp_sec=timestamp_sec,
                reason="image_prepare_failed",
            )

        self._future_request = request
        self._last_call_started_sec = float(timestamp_sec)
        self._last_call_started_wall_sec = float(wall_time_sec)
        self._future = self._executor.submit(self._call_grounding, request)
        return reusable or _empty_observation(
            target_label=target_label,
            timestamp_sec=timestamp_sec,
            reason="grounding_request_started",
        )

    def _reuse_or_empty(self, *, target_label: str, timestamp_sec: float, reason: str) -> TargetObservation:
        reusable = self._reusable_observation(
            target_label=target_label,
            timestamp_sec=timestamp_sec,
            wall_time_sec=time.monotonic(),
        )
        return reusable or _empty_observation(target_label=target_label, timestamp_sec=timestamp_sec, reason=reason)

    def _reusable_observation(
        self,
        *,
        target_label: str,
        timestamp_sec: float,
        wall_time_sec: float,
        current_robot_pose: Pose2D | None = None,
    ) -> TargetObservation | None:
        if self._last_observation is None:
            return None
        if self._last_observation.label != target_label:
            return None
        age_sec = float(wall_time_sec) - float(self._last_observation_wall_sec)
        if age_sec > max(0.0, float(self._config.reuse_last_observation_sec)):
            return None
        result_age_sec = float(timestamp_sec) - float(self._last_observation.request_sim_time_sec) if self._last_observation.request_sim_time_sec > 0.0 else age_sec
        is_current = _is_current_frame(
            result_age_sec=result_age_sec,
            request_pose=self._last_observation.request_robot_pose,
            current_pose=current_robot_pose,
            config=self._config,
        )
        return TargetObservation(
            label=self._last_observation.label,
            visible=self._last_observation.visible,
            bbox=None,  # don't show stale bbox from old frame
            center_uv=self._last_observation.center_uv,
            confidence=self._last_observation.confidence,
            depth_m=self._last_observation.depth_m,
            bearing_rad=self._last_observation.bearing_rad,
            estimated_world_xy=None
            if self._last_observation.estimated_world_xy is None
            else list(self._last_observation.estimated_world_xy),
            depth_valid_ratio=self._last_observation.depth_valid_ratio,
            area_ratio=self._last_observation.area_ratio,
            source=f"{self._last_observation.source}_reused",
            reason="reused_last_observation",
            timestamp_sec=float(timestamp_sec),
            mask_area_ratio=self._last_observation.mask_area_ratio,
            coordinate_mode=self._last_observation.coordinate_mode,
            segmentation_source=self._last_observation.segmentation_source,
            mask=None if self._last_observation.mask is None else np.asarray(self._last_observation.mask, dtype=bool).copy(),
            request_frame_index=self._last_observation.request_frame_index,
            request_sim_time_sec=self._last_observation.request_sim_time_sec,
            request_robot_pose=self._last_observation.request_robot_pose,
            result_age_sec=float(result_age_sec),
            is_current_frame=is_current,
            evidence_level=str(self._last_observation.evidence_level),
            semantic_source=str(self._last_observation.semantic_source),
            can_confirm=bool(self._last_observation.can_confirm),
            vlm_verified=bool(self._last_observation.vlm_verified),
            vlm_verification_confidence=float(self._last_observation.vlm_verification_confidence),
            vlm_verification_reason=str(self._last_observation.vlm_verification_reason),
            requires_viewpoint_confirmation=bool(
                self._last_observation.requires_viewpoint_confirmation
            ),
        )

    def _collect_finished_request(self, *, timestamp_sec: float, wall_time_sec: float, current_robot_pose: Pose2D | None = None) -> TargetObservation | None:
        if self._future is None or not self._future.done():
            return None
        future = self._future
        request = self._future_request
        self._future = None
        self._future_request = None
        if request is None:
            return None
        try:
            content, reason = future.result()
        except Exception as exc:
            self._logger.warning("[GroundingReject] target=%s reason=worker_failed error=%s", request.target_label, exc)
            return None
        if reason != "ok":
            self._logger.warning("[GroundingReject] target=%s reason=%s", request.target_label, reason)
            observation = _copy_observation_with_timestamp(
                _empty_observation(
                    target_label=request.target_label,
                    timestamp_sec=request.timestamp_sec,
                    reason=reason,
                ),
                timestamp_sec=timestamp_sec,
                request_frame_index=request.frame_index,
                request_sim_time_sec=request.timestamp_sec,
                request_robot_pose=request.robot_pose,
                result_age_sec=float(timestamp_sec) - float(request.timestamp_sec),
                is_current_frame=True,
            )
            self._last_observation = observation
            self._last_observation_wall_sec = float(wall_time_sec)
            return observation
        result_age_sec = float(timestamp_sec) - float(request.timestamp_sec)
        is_current = _is_current_frame(
            result_age_sec=result_age_sec,
            request_pose=request.robot_pose,
            current_pose=current_robot_pose,
            config=self._config,
        )
        observation = _copy_observation_with_timestamp(
            self._parse_and_validate(content=content, request=request),
            timestamp_sec=timestamp_sec,
            request_frame_index=request.frame_index,
            request_sim_time_sec=request.timestamp_sec,
            request_robot_pose=request.robot_pose,
            result_age_sec=result_age_sec,
            is_current_frame=is_current,
        )
        self._last_observation = observation
        self._last_observation_wall_sec = float(wall_time_sec)
        if observation.visible and self._should_log_observation(observation=observation, wall_time_sec=wall_time_sec):
            self._logger.info(
                "[Grounding] target=%s visible=1 bbox=%s depth=%.2f valid_ratio=%.2f mask_area=%.3f bearing_deg=%.1f world=%s confidence=%.2f coord=%s seg=%s age=%.2fs current=%s frame=%d evidence=%s semantic_source=%s can_confirm=%s vlm_verified=%s vlm_conf=%.2f vlm_reason=%s",
                observation.label,
                list(observation.bbox or ()),
                float(observation.depth_m or 0.0),
                float(observation.depth_valid_ratio),
                float(observation.mask_area_ratio),
                float(np.degrees(observation.bearing_rad or 0.0)),
                observation.estimated_world_xy,
                float(observation.confidence),
                observation.coordinate_mode,
                observation.segmentation_source,
                float(observation.result_age_sec),
                int(observation.is_current_frame),
                int(observation.request_frame_index),
                observation.evidence_level,
                observation.semantic_source,
                int(observation.can_confirm),
                int(observation.vlm_verified),
                float(observation.vlm_verification_confidence),
                observation.vlm_verification_reason[:80],
            )
        elif not observation.visible and self._should_log_observation(observation=observation, wall_time_sec=wall_time_sec):
            self._logger.info("[GroundingReject] target=%s reason=%s", observation.label, observation.reason)
        return observation

    def _should_log_observation(self, *, observation: TargetObservation, wall_time_sec: float) -> bool:
        bbox = tuple(int(round(float(value) / 12.0)) for value in (observation.bbox or ()))
        depth_bucket = None if observation.depth_m is None else round(float(observation.depth_m), 1)
        bearing_bucket = None if observation.bearing_rad is None else round(float(np.degrees(observation.bearing_rad)) / 10.0)
        signature = (
            str(observation.label),
            bool(observation.visible),
            str(observation.reason),
            str(observation.source),
            bbox,
            depth_bucket,
            bearing_bucket,
        )
        if signature != self._last_observation_log_signature:
            self._last_observation_log_signature = signature
            self._last_observation_log_wall_sec = float(wall_time_sec)
            return True
        if float(wall_time_sec) - float(self._last_observation_log_wall_sec) >= 5.0:
            self._last_observation_log_wall_sec = float(wall_time_sec)
            return True
        return False

    def _call_grounding(self, request: _GroundingRequest) -> tuple[str, str]:
        detections, reason = self._grounding_dino.detect_all(
            rgb=request.rgb,
            target_label=request.target_label,
        )
        if not detections:
            return "", reason
        detection, selected_index, selection_reason = _select_grounding_detection(
            detections=detections,
            request=request,
            config=self._config,
        )
        self._logger.info(
            "[GroundingInstanceSelect] target=%s proposals=%s selected=%s confidence=%.2f bbox=%s reason=%s",
            request.target_label,
            len(detections),
            selected_index,
            float(detection.confidence),
            list(detection.bbox),
            selection_reason,
        )
        payload = {
            "visible": True,
            "label": request.target_label,
            "bbox": list(detection.bbox),
            "confidence": float(detection.confidence),
            "coordinate_mode": "original_pixels",
            "source": "grounding_dino",
            "proposal_count": len(detections),
            "selected_instance_index": selected_index,
        }
        return json.dumps(payload), "ok"

    def _build_request(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        target_label: str,
        robot_pose: Pose2D,
        timestamp_sec: float,
        frame_index: int = -1,
        preferred_world_xy: list[float] | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_view_matrix_ros: np.ndarray | None = None,
        robot_world_position_xyz: np.ndarray | None = None,
        robot_world_orientation_wxyz: np.ndarray | None = None,
    ) -> _GroundingRequest:
        image = Image.fromarray(rgb[:, :, :3].copy()).convert("RGB")
        original_size = (int(image.width), int(image.height))
        upload = image
        max_edge_px = max(1, int(self._config.upload_max_edge_px))
        max_edge = max(upload.size)
        if max_edge > max_edge_px:
            scale = float(max_edge_px) / float(max_edge)
            new_size = (
                max(1, int(round(upload.width * scale))),
                max(1, int(round(upload.height * scale))),
            )
            resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            upload = upload.resize(new_size, resampling)
        buffer = io.BytesIO()
        upload.save(buffer, format="PNG", optimize=True)
        return _GroundingRequest(
            target_label=target_label,
            rgb=rgb[:, :, :3].copy(),
            rgb_shape=(int(original_size[1]), int(original_size[0])),
            depth=depth,
            robot_pose=robot_pose,
            timestamp_sec=float(timestamp_sec),
            upload_size=(int(upload.width), int(upload.height)),
            original_size=original_size,
            image_png_bytes=buffer.getvalue(),
            frame_index=int(frame_index),
            preferred_bbox=(
                tuple(self._last_observation.bbox)
                if self._last_observation is not None
                and self._last_observation.label == target_label
                and self._last_observation.bbox is not None
                and bool(self._last_observation.can_confirm)
                else None
            ),
            preferred_world_xy=(
                None
                if preferred_world_xy is None
                else [float(preferred_world_xy[0]), float(preferred_world_xy[1])]
            ),
            camera_intrinsics=_copy_optional_matrix(camera_intrinsics, shape=(3, 3)),
            camera_view_matrix_ros=_copy_optional_matrix(camera_view_matrix_ros, shape=(4, 4)),
            robot_world_position_xyz=_copy_optional_vector(robot_world_position_xyz, min_length=3),
            robot_world_orientation_wxyz=_copy_optional_vector(
                robot_world_orientation_wxyz, min_length=4
            ),
        )

    def _parse_and_validate(self, *, content: str, request: _GroundingRequest) -> TargetObservation:
        try:
            payload = _parse_json_object(content)
        except Exception:
            return _empty_observation(
                target_label=request.target_label,
                timestamp_sec=request.timestamp_sec,
                reason="invalid_json",
            )
        visible_flag = _parse_bool(payload.get("visible"))
        bbox_upload = _parse_bbox(payload.get("bbox"))
        if not visible_flag and bbox_upload is None:
            return _empty_observation(
                target_label=request.target_label,
                timestamp_sec=request.timestamp_sec,
                reason="target_not_visible",
            )
        confidence = _parse_float(payload.get("confidence"), default=1.0)
        if confidence < float(self._config.min_confidence):
            return _empty_observation(
                target_label=request.target_label,
                timestamp_sec=request.timestamp_sec,
                reason="confidence_too_low",
            )
        if not visible_flag and bbox_upload is not None:
            self._logger.warning(
                "[GroundingRecover] target=%s reason=visible_false_with_bbox raw_bbox=%s",
                request.target_label,
                payload.get("bbox"),
            )
        if bbox_upload is None:
            self._logger.warning(
                "[GroundingRejectDebug] target=%s reject_reason=bbox_missing_or_invalid raw_bbox=%s image_size=%s upload_image_size=%s mapped_bbox=None",
                request.target_label,
                payload.get("bbox"),
                request.original_size,
                request.upload_size,
            )
            return _empty_observation(
                target_label=request.target_label,
                timestamp_sec=request.timestamp_sec,
                reason="bbox_missing_or_invalid",
            )
        requested_mode_value = payload.get("coordinate_mode")
        requested_mode = (
            str(self._config.coordinate_mode)
            if requested_mode_value is None or str(requested_mode_value).strip() == ""
            else str(requested_mode_value)
        )
        mapped_bbox = _map_bbox_to_original(
            bbox=bbox_upload,
            upload_size=request.upload_size,
            original_size=request.original_size,
            requested_mode=requested_mode,
            allow_normalized_bbox=bool(self._config.allow_normalized_bbox),
            normalized_bbox_scale=float(self._config.normalized_bbox_scale),
        )
        if mapped_bbox is None:
            self._logger.warning(
                "[GroundingRejectDebug] target=%s reject_reason=bbox_out_of_range raw_bbox=%s image_size=%s upload_image_size=%s mapped_bbox=None requested_mode=%s normalized_scale=%.1f",
                request.target_label,
                bbox_upload,
                request.original_size,
                request.upload_size,
                requested_mode,
                float(self._config.normalized_bbox_scale),
            )
            return _empty_observation(
                target_label=request.target_label,
                timestamp_sec=request.timestamp_sec,
                reason="bbox_out_of_range",
            )
        source_prefix = str(payload.get("source", "grounding_dino")).strip() or "grounding_dino"
        self._logger.debug(
            "[GroundingBbox] target=%s raw_bbox=%s mapped_bbox=%s image_size=%s upload_image_size=%s coord=%s source=%s",
            request.target_label,
            bbox_upload,
            mapped_bbox.bbox,
            request.original_size,
            request.upload_size,
            mapped_bbox.coordinate_mode,
            source_prefix,
        )
        return self._validate_original_bbox(
            bbox=mapped_bbox.bbox,
            coordinate_mode=mapped_bbox.coordinate_mode,
            confidence=confidence,
            request=request,
            source_prefix=source_prefix,
        )

    def _validate_original_bbox(
        self,
        *,
        bbox: tuple[int, int, int, int],
        coordinate_mode: str,
        confidence: float,
        request: _GroundingRequest,
        source_prefix: str,
    ) -> TargetObservation:
        image_height, image_width = request.rgb_shape
        xmin, ymin, xmax, ymax = bbox
        bbox_width = int(xmax - xmin)
        bbox_height = int(ymax - ymin)
        if bbox_width < int(self._config.min_bbox_width_px) or bbox_height < int(self._config.min_bbox_height_px):
            return _empty_observation(request.target_label, request.timestamp_sec, "bbox_too_small")
        area_ratio = float((bbox_width * bbox_height) / max(1.0, float(image_width * image_height)))
        if area_ratio < float(self._config.min_bbox_area_ratio):
            return _empty_observation(request.target_label, request.timestamp_sec, "bbox_area_too_small")
        if area_ratio > float(self._config.max_bbox_area_ratio):
            return _empty_observation(request.target_label, request.timestamp_sec, "bbox_area_too_large")
        aspect_ratio = max(float(bbox_width) / max(float(bbox_height), 1.0), float(bbox_height) / max(float(bbox_width), 1.0))
        if aspect_ratio > float(self._config.max_bbox_aspect_ratio):
            return _empty_observation(request.target_label, request.timestamp_sec, "bbox_aspect_invalid")
        if request.depth is None or np.asarray(request.depth).ndim != 2:
            return _empty_observation(request.target_label, request.timestamp_sec, "depth_unavailable")

        mask: np.ndarray | None = None
        segmentation_source = str(self._config.depth_region)
        depth_m = None
        valid_ratio = 0.0
        mask_area_ratio = 0.0
        center_uv = (float(xmin + xmax) * 0.5, float(ymin + ymax) * 0.5)

        segmentation = self._sam.segment(rgb=request.rgb, bbox=bbox)
        if segmentation is not None:
            mask = np.asarray(segmentation.mask, dtype=bool)
            depth_m, valid_ratio, mask_area_ratio, center_uv = _mask_depth_statistic(
                depth=np.asarray(request.depth, dtype=np.float32),
                mask=mask,
                min_depth_m=float(self._config.min_depth_m),
                max_depth_m=float(self._config.max_depth_m),
                statistic=str(self._config.depth_statistic),
            )
            segmentation_source = segmentation.source
            self._logger.debug(
                "[SAM] target=%s score=%.3f mask_area=%.3f depth=%s valid_ratio=%.2f",
                request.target_label,
                float(segmentation.score),
                float(mask_area_ratio),
                "none" if depth_m is None else f"{float(depth_m):.2f}",
                float(valid_ratio),
            )

        if depth_m is None or valid_ratio < float(self._config.min_depth_valid_ratio):
            if segmentation is not None:
                self._logger.debug(
                    "[SAMFallback] target=%s reason=mask_depth_invalid depth=%s valid_ratio=%.2f fallback=%s",
                    request.target_label,
                    "none" if depth_m is None else f"{float(depth_m):.2f}",
                    float(valid_ratio),
                    self._config.depth_region,
                )
            mask = None
            depth_m, valid_ratio, mask_area_ratio, center_uv = _bbox_depth_mask_statistic(
                depth=np.asarray(request.depth, dtype=np.float32),
                bbox=bbox,
                min_depth_m=float(self._config.min_depth_m),
                max_depth_m=float(self._config.max_depth_m),
                statistic=str(self._config.depth_statistic),
                depth_region=str(self._config.depth_region),
                center_patch_ratio=float(self._config.center_patch_ratio),
                mask_tolerance_ratio=float(self._config.depth_mask_tolerance_ratio),
                mask_min_pixels=int(self._config.depth_mask_min_pixels),
            )
            segmentation_source = str(self._config.depth_region)
        # Bearing from bbox center (works with or without depth)
        cx_px = (float(image_width) - 1.0) * 0.5
        half_fov_rad = float(np.deg2rad(float(self._config.horizontal_fov_deg)) * 0.5)
        bearing_rad = float(np.arctan(((float(center_uv[0]) - cx_px) / max(cx_px, 1.0)) * np.tan(half_fov_rad)))

        semantic_source = source_prefix.split(":", 1)[0]
        can_confirm = semantic_source.startswith("grounding_dino")
        vlm_verified = False
        vlm_verification_confidence = 0.0
        vlm_verification_reason = "not_run"

        # Depth validation - if failed, flag it but don't reject. Let VLM verify instead.
        depth_failed = valid_ratio < float(self._config.min_depth_valid_ratio) or depth_m is None or not np.isfinite(depth_m)
        if depth_failed:
            self._logger.info(
                "[GroundingNoDepth] target=%s valid_ratio=%.2f depth=%s reason=depth_invalid_but_visible source=%s",
                request.target_label,
                float(valid_ratio),
                "none" if depth_m is None else f"{float(depth_m):.2f}",
                semantic_source,
            )
        if not depth_failed and (depth_m < float(self._config.min_depth_m) or depth_m > float(self._config.max_depth_m)):
            return _empty_observation(request.target_label, request.timestamp_sec, "depth_out_of_range")

        geometry = None
        if not depth_failed:
            geometry = estimate_target_world_xy(
                center_uv=center_uv,
                depth_m=float(depth_m),
                image_width=int(image_width),
                image_height=int(image_height),
                horizontal_fov_deg=float(self._config.horizontal_fov_deg),
                robot_pose=request.robot_pose,
                camera_intrinsics=request.camera_intrinsics,
                camera_view_matrix_ros=request.camera_view_matrix_ros,
                robot_world_position_xyz=request.robot_world_position_xyz,
                robot_world_orientation_wxyz=request.robot_world_orientation_wxyz,
            )
        high_confidence = float(confidence) >= float(
            self._config.semantic_verification_auto_accept_confidence
        )
        requires_viewpoint_confirmation = False
        if can_confirm and bool(self._config.semantic_verification_enabled):
            if high_confidence:
                vlm_verified = True
                vlm_verification_confidence = 1.0
                vlm_verification_reason = "auto_accept_high_detector_confidence"
            else:
                vlm_verified, vlm_verification_confidence, vlm_verification_reason = self._verify_detection_semantics(
                    request=request,
                    bbox=bbox,
                    detector_confidence=float(confidence),
                    semantic_source=semantic_source,
                )
                requires_viewpoint_confirmation = bool(vlm_verified)
            if not vlm_verified or vlm_verification_confidence < float(self._config.semantic_verification_min_confidence):
                can_confirm = False
                self._logger.info(
                    "[GroundingVerifyReject] target=%s source=%s detector_conf=%.2f vlm_conf=%.2f reason=%s",
                    request.target_label,
                    semantic_source,
                    float(confidence),
                    float(vlm_verification_confidence),
                    vlm_verification_reason,
                )

        evidence_level = "confirmed_candidate" if can_confirm else "potential"
        if geometry is not None:
            estimated_xy = list(geometry.world_xy)
        else:
            estimated_xy = None
        depth_out = float(depth_m) if depth_m is not None else None
        bear_out = float(bearing_rad) if bearing_rad is not None else None

        return TargetObservation(
            label=request.target_label,
            visible=True,
            bbox=bbox,
            center_uv=center_uv,
            confidence=float(confidence),
            depth_m=depth_out,
            bearing_rad=bear_out,
            estimated_world_xy=estimated_xy,
            depth_valid_ratio=float(valid_ratio),
            area_ratio=float(area_ratio),
            source=f"{source_prefix}:{coordinate_mode}",
            reason="ok",
            timestamp_sec=float(request.timestamp_sec),
            mask_area_ratio=float(mask_area_ratio),
            coordinate_mode=str(coordinate_mode),
            segmentation_source=str(segmentation_source),
            mask=mask,
            evidence_level=evidence_level,
            semantic_source=semantic_source,
            can_confirm=can_confirm,
            vlm_verified=vlm_verified,
            vlm_verification_confidence=float(vlm_verification_confidence),
            vlm_verification_reason=str(vlm_verification_reason),
            requires_viewpoint_confirmation=bool(requires_viewpoint_confirmation),
        )

    def _verify_detection_semantics(
        self,
        *,
        request: _GroundingRequest,
        bbox: tuple[int, int, int, int],
        detector_confidence: float,
        semantic_source: str,
    ) -> tuple[bool, float, str]:
        verification_signature = _verification_signature(
            target_label=request.target_label,
            bbox=bbox,
            robot_pose=request.robot_pose,
        )
        if (
            verification_signature == self._last_verification_signature
            and self._last_verification_result is not None
        ):
            cached = self._last_verification_result
            self._logger.info(
                "[GroundingVerify] target=%s result=reused_same_observation verified=%s vlm_conf=%.2f",
                request.target_label,
                int(bool(cached[0])),
                float(cached[1]),
            )
            return cached
        image_bytes = _build_verification_image_bytes(
            rgb=request.rgb,
            bbox=bbox,
            max_edge_px=int(self._config.semantic_verification_max_edge_px),
        )
        system_prompt = (
            "You are a strict visual verification module for robot navigation. "
            "You must decide whether the cropped object is actually the requested target. "
            "Return strict JSON only."
        )
        user_prompt = (
            f"Target label: {request.target_label}\n"
            f"Detector source: {semantic_source}\n"
            f"Detector confidence: {float(detector_confidence):.2f}\n"
            "The image contains only the candidate crop. Verify its object category. "
            "Reject shadows, floor texture, reflections, wall patterns, or unrelated objects.\n"
            "Return exactly one JSON object:\n"
            '{"is_target": true, "confidence": 0.0, "reason": "short reason"}'
        )
        result = self._llm.chat(
            request_name="target_semantic_verify",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_png_bytes=image_bytes,
            max_tokens=96,
            temperature=0.0,
        )
        if result.failed or not result.content:
            return False, 0.0, result.reason or "verification_llm_failed"
        try:
            payload = _parse_json_object(result.content)
            verified = _parse_bool(payload.get("is_target"))
            confidence = float(np.clip(_parse_float(payload.get("confidence"), default=0.0), 0.0, 1.0))
            reason = str(payload.get("reason", "")).strip() or "no_reason"
        except Exception as exc:
            return False, 0.0, f"verification_invalid_json:{exc}"
        self._logger.info(
            "[GroundingVerify] target=%s source=%s detector_conf=%.2f verified=%s vlm_conf=%.2f reason=%s",
            request.target_label,
            semantic_source,
            float(detector_confidence),
            int(bool(verified)),
            float(confidence),
            reason[:120],
        )
        verified_result = (bool(verified), float(confidence), reason)
        self._last_verification_signature = verification_signature
        self._last_verification_result = verified_result
        return verified_result


def _empty_observation(target_label: str, timestamp_sec: float, reason: str) -> TargetObservation:
    return TargetObservation(
        label=str(target_label),
        visible=False,
        bbox=None,
        center_uv=None,
        confidence=0.0,
        depth_m=None,
        bearing_rad=None,
        estimated_world_xy=None,
        depth_valid_ratio=0.0,
        area_ratio=0.0,
        source="grounding_dino",
        reason=str(reason),
        timestamp_sec=float(timestamp_sec),
        mask_area_ratio=0.0,
        coordinate_mode="none",
        segmentation_source="none",
        mask=None,
        evidence_level="none",
        semantic_source="unknown",
        can_confirm=False,
        vlm_verified=False,
        vlm_verification_confidence=0.0,
        vlm_verification_reason="",
        requires_viewpoint_confirmation=False,
    )


def _copy_observation_with_timestamp(
    observation: TargetObservation,
    *,
    timestamp_sec: float,
    request_frame_index: int = -1,
    request_sim_time_sec: float = 0.0,
    request_robot_pose: Pose2D | None = None,
    result_age_sec: float = 0.0,
    is_current_frame: bool = True,
) -> TargetObservation:
    return TargetObservation(
        label=observation.label,
        visible=observation.visible,
        bbox=observation.bbox,
        center_uv=observation.center_uv,
        confidence=observation.confidence,
        depth_m=observation.depth_m,
        bearing_rad=observation.bearing_rad,
        estimated_world_xy=None if observation.estimated_world_xy is None else list(observation.estimated_world_xy),
        depth_valid_ratio=observation.depth_valid_ratio,
        area_ratio=observation.area_ratio,
        source=observation.source,
        reason=observation.reason,
        timestamp_sec=float(observation.timestamp_sec) if observation.timestamp_sec > 0.0 else float(timestamp_sec),
        mask_area_ratio=observation.mask_area_ratio,
        coordinate_mode=observation.coordinate_mode,
        segmentation_source=observation.segmentation_source,
        mask=None if observation.mask is None else np.asarray(observation.mask, dtype=bool).copy(),
        request_frame_index=int(request_frame_index) if request_frame_index >= 0 else int(observation.request_frame_index),
        request_sim_time_sec=float(request_sim_time_sec) if request_sim_time_sec > 0.0 else float(observation.request_sim_time_sec),
        request_robot_pose=request_robot_pose if request_robot_pose is not None else observation.request_robot_pose,
        result_age_sec=float(result_age_sec),
        is_current_frame=bool(is_current_frame),
        evidence_level=str(observation.evidence_level),
        semantic_source=str(observation.semantic_source),
        can_confirm=bool(observation.can_confirm),
        vlm_verified=bool(observation.vlm_verified),
        vlm_verification_confidence=float(observation.vlm_verification_confidence),
        vlm_verification_reason=str(observation.vlm_verification_reason),
        requires_viewpoint_confirmation=bool(observation.requires_viewpoint_confirmation),
    )


def _is_current_frame(
    *,
    result_age_sec: float,
    request_pose: Pose2D | None,
    current_pose: Pose2D | None,
    config: TargetGroundingConfig,
) -> bool:
    max_age = float(config.max_result_age_sec)
    if max_age > 0.0 and result_age_sec > max_age:
        return False
    max_pose_delta = float(config.max_result_pose_delta_m)
    max_yaw_delta = float(config.max_result_yaw_delta_deg)
    if request_pose is not None and current_pose is not None:
        pos_delta = float(np.hypot(
            float(current_pose.x) - float(request_pose.x),
            float(current_pose.y) - float(request_pose.y),
        ))
        if max_pose_delta > 0.0 and pos_delta > max_pose_delta:
            return False
        if max_yaw_delta > 0.0:
            yaw_delta = abs(float(np.degrees(float(current_pose.yaw_rad) - float(request_pose.yaw_rad))))
            yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
            if yaw_delta > max_yaw_delta:
                return False
    return True


def _parse_json_object(content: str) -> dict[str, Any]:
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("grounding response is not a JSON object")
    return payload


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "visible"}


def _parse_float(value, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_bbox(value) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = [float(item) for item in value]
    except Exception:
        return None
    if not all(np.isfinite(item) for item in (xmin, ymin, xmax, ymax)):
        return None
    if not (xmin < xmax and ymin < ymax):
        return None
    return xmin, ymin, xmax, ymax


def _map_bbox_to_original(
    *,
    bbox: tuple[float, float, float, float],
    upload_size: tuple[int, int],
    original_size: tuple[int, int],
    requested_mode: str,
    allow_normalized_bbox: bool,
    normalized_bbox_scale: float,
) -> _MappedBbox | None:
    upload_w, upload_h = [int(value) for value in upload_size]
    original_w, original_h = [int(value) for value in original_size]
    xmin, ymin, xmax, ymax = bbox

    requested = str(requested_mode).strip().lower()
    normalized_scale = max(1.0, float(normalized_bbox_scale))
    all_non_negative = min(xmin, ymin, xmax, ymax) >= 0.0
    fits_upload = all_non_negative and xmax <= float(upload_w) and ymax <= float(upload_h)
    fits_original = all_non_negative and xmax <= float(original_w) and ymax <= float(original_h)
    fits_normalized = (
        bool(allow_normalized_bbox)
        and all_non_negative
        and xmax <= normalized_scale * 1.05
        and ymax <= normalized_scale * 1.05
    )

    coordinate_mode = "unknown"
    if requested in {"normalized_1000", "normalized", "qwen_1000"} and fits_normalized:
        sx = float(original_w) / normalized_scale
        sy = float(original_h) / normalized_scale
        coordinate_mode = f"normalized_{int(round(normalized_scale))}"
    elif requested in {"original_pixels", "original", "rgb_pixels"} and fits_original:
        sx = 1.0
        sy = 1.0
        coordinate_mode = "original_pixels"
    elif fits_upload:
        sx = float(original_w) / max(float(upload_w), 1.0)
        sy = float(original_h) / max(float(upload_h), 1.0)
        coordinate_mode = "upload_pixels"
    elif fits_original:
        sx = 1.0
        sy = 1.0
        coordinate_mode = "original_pixels"
    elif fits_normalized:
        sx = float(original_w) / normalized_scale
        sy = float(original_h) / normalized_scale
        coordinate_mode = f"normalized_{int(round(normalized_scale))}"
    else:
        return None

    mapped = (
        int(round(xmin * sx)),
        int(round(ymin * sy)),
        int(round(xmax * sx)),
        int(round(ymax * sy)),
    )
    x0 = max(0, min(original_w - 1, mapped[0]))
    y0 = max(0, min(original_h - 1, mapped[1]))
    x1 = max(0, min(original_w, mapped[2]))
    y1 = max(0, min(original_h, mapped[3]))
    if not (x0 < x1 and y0 < y1):
        return None
    return _MappedBbox(bbox=(x0, y0, x1, y1), coordinate_mode=coordinate_mode)


def _build_verification_image_bytes(
    *,
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_edge_px: int,
) -> bytes:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[:, :, :3].copy()).convert("RGB")
    width, height = image.size
    xmin, ymin, xmax, ymax = [int(value) for value in bbox]
    xmin = max(0, min(width - 1, xmin))
    ymin = max(0, min(height - 1, ymin))
    xmax = max(xmin + 1, min(width, xmax))
    ymax = max(ymin + 1, min(height, ymax))

    margin_x = int(round((xmax - xmin) * 0.35))
    margin_y = int(round((ymax - ymin) * 0.35))
    crop_box = (
        max(0, xmin - margin_x),
        max(0, ymin - margin_y),
        min(width, xmax + margin_x),
        min(height, ymax + margin_y),
    )

    crop = image.crop(crop_box)
    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    max_edge = max(crop.size)
    max_edge_px = max(1, int(max_edge_px))
    if max_edge > max_edge_px:
        scale = float(max_edge_px) / float(max_edge)
        crop = crop.resize(
            (
                max(1, int(round(crop.width * scale))),
                max(1, int(round(crop.height * scale))),
            ),
            resampling,
        )
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _select_grounding_detection(
    *,
    detections: list[GroundingDinoDetection],
    request: _GroundingRequest,
    config: TargetGroundingConfig,
) -> tuple[GroundingDinoDetection, int, str]:
    if len(detections) == 1:
        return detections[0], 0, "single_proposal"

    preferred_world_xy = getattr(request, "preferred_world_xy", None)
    if preferred_world_xy is not None:
        world_matches: list[tuple[float, int, GroundingDinoDetection]] = []
        for index, detection in enumerate(detections):
            world_xy = _rough_detection_world_xy(
                detection=detection,
                request=request,
                config=config,
            )
            if world_xy is None:
                continue
            distance_m = float(np.linalg.norm(
                np.asarray(world_xy, dtype=np.float32)
                - np.asarray(preferred_world_xy, dtype=np.float32)
            ))
            world_matches.append((distance_m, index, detection))
        if world_matches:
            distance_m, best_index, best_detection = min(world_matches, key=lambda item: item[0])
            if distance_m <= 1.5:
                return best_detection, int(best_index), "confirmed_anchor_instance_continuity"

    if request.preferred_bbox is not None:
        continuity_scores: list[tuple[float, int]] = []
        image_h, image_w = request.rgb_shape
        diagonal = max(1.0, float(np.hypot(image_w, image_h)))
        previous_center = _bbox_center(request.preferred_bbox)
        for index, detection in enumerate(detections):
            iou = _bbox_iou(request.preferred_bbox, detection.bbox)
            center = _bbox_center(detection.bbox)
            center_distance = float(np.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )) / diagonal
            score = 2.0 * iou + max(0.0, 1.0 - 4.0 * center_distance) + 0.25 * float(
                detection.confidence
            )
            continuity_scores.append((score, index))
        _, best_index = max(continuity_scores)
        best_detection = detections[best_index]
        best_iou = _bbox_iou(request.preferred_bbox, best_detection.bbox)
        best_center = _bbox_center(best_detection.bbox)
        previous_center = _bbox_center(request.preferred_bbox)
        normalized_center_distance = float(np.hypot(
            best_center[0] - previous_center[0],
            best_center[1] - previous_center[1],
        )) / max(1.0, float(np.hypot(request.rgb_shape[1], request.rgb_shape[0])))
        if best_iou >= 0.10 or normalized_center_distance <= 0.18:
            return best_detection, int(best_index), "temporal_instance_continuity"

    high_confidence = [
        (index, detection)
        for index, detection in enumerate(detections)
        if float(detection.confidence)
        >= float(config.semantic_verification_auto_accept_confidence)
    ]
    credible = high_confidence or [
        (index, detection)
        for index, detection in enumerate(detections)
        if float(detection.confidence) >= float(config.min_confidence)
    ]
    if not credible:
        credible = list(enumerate(detections))

    ranked: list[tuple[float, int, GroundingDinoDetection, float | None]] = []
    for index, detection in credible:
        depth_m = _rough_bbox_depth_m(
            depth=request.depth,
            bbox=detection.bbox,
            config=config,
        )
        confidence_span = max(1e-6, 1.0 - float(config.min_confidence))
        confidence_score = float(np.clip(
            (float(detection.confidence) - float(config.min_confidence)) / confidence_span,
            0.0,
            1.0,
        ))
        depth_score = 0.0 if depth_m is None else 1.0 / (1.0 + max(0.0, float(depth_m)))
        score = 0.55 * confidence_score + 0.45 * depth_score
        ranked.append((score, int(index), detection, depth_m))
    _, best_index, best_detection, best_depth = max(ranked, key=lambda item: item[0])
    reason = "confidence_and_nearest_valid_depth"
    if best_depth is None:
        reason = "highest_confidence_without_valid_depth"
    return best_detection, best_index, reason


def _rough_bbox_depth_m(
    *,
    depth: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    config: TargetGroundingConfig,
) -> float | None:
    if depth is None or np.asarray(depth).ndim != 2:
        return None
    value, valid_ratio, _, _ = _bbox_depth_mask_statistic(
        depth=np.asarray(depth, dtype=np.float32),
        bbox=bbox,
        min_depth_m=float(config.min_depth_m),
        max_depth_m=float(config.max_depth_m),
        statistic=str(config.depth_statistic),
        depth_region="center_patch",
        center_patch_ratio=float(config.center_patch_ratio),
        mask_tolerance_ratio=float(config.depth_mask_tolerance_ratio),
        mask_min_pixels=int(config.depth_mask_min_pixels),
    )
    if value is None or float(valid_ratio) < float(config.min_depth_valid_ratio):
        return None
    return float(value)


def _rough_detection_world_xy(
    *,
    detection: GroundingDinoDetection,
    request: _GroundingRequest,
    config: TargetGroundingConfig,
) -> list[float] | None:
    depth_m = _rough_bbox_depth_m(
        depth=request.depth,
        bbox=detection.bbox,
        config=config,
    )
    if depth_m is None:
        return None
    center_uv = _bbox_center(detection.bbox)
    geometry = estimate_target_world_xy(
        center_uv=center_uv,
        depth_m=float(depth_m),
        image_width=int(request.rgb_shape[1]),
        image_height=int(request.rgb_shape[0]),
        horizontal_fov_deg=float(config.horizontal_fov_deg),
        robot_pose=request.robot_pose,
        camera_intrinsics=request.camera_intrinsics,
        camera_view_matrix_ros=request.camera_view_matrix_ros,
        robot_world_position_xyz=request.robot_world_position_xyz,
        robot_world_orientation_wxyz=request.robot_world_orientation_wxyz,
    )
    return list(geometry.world_xy)


def _copy_optional_matrix(value: np.ndarray | None, *, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Expected finite matrix with shape {shape}, got {matrix.shape}")
    return matrix.copy()


def _copy_optional_vector(value: np.ndarray | None, *, min_length: int) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < int(min_length) or not np.all(np.isfinite(vector[:min_length])):
        raise ValueError(f"Expected a finite vector with at least {min_length} values")
    return vector.copy()


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (
        0.5 * (float(bbox[0]) + float(bbox[2])),
        0.5 * (float(bbox[1]) + float(bbox[3])),
    )


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0 = max(int(a[0]), int(b[0]))
    iy0 = max(int(a[1]), int(b[1]))
    ix1 = min(int(a[2]), int(b[2]))
    iy1 = min(int(a[3]), int(b[3]))
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, int(a[2]) - int(a[0])) * max(0, int(a[3]) - int(a[1]))
    area_b = max(0, int(b[2]) - int(b[0])) * max(0, int(b[3]) - int(b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def _verification_signature(
    *,
    target_label: str,
    bbox: tuple[int, int, int, int],
    robot_pose: Pose2D,
) -> tuple:
    return (
        str(target_label).strip().lower(),
        tuple(int(round(float(value) / 8.0)) for value in bbox),
        int(round(float(robot_pose.x) / 0.05)),
        int(round(float(robot_pose.y) / 0.05)),
        int(round(float(np.degrees(robot_pose.yaw_rad)) / 3.0)),
    )


def _bbox_depth_mask_statistic(
    *,
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    min_depth_m: float,
    max_depth_m: float,
    statistic: str,
    depth_region: str,
    center_patch_ratio: float,
    mask_tolerance_ratio: float,
    mask_min_pixels: int,
) -> tuple[float | None, float, float, tuple[float, float]]:
    height, width = depth.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    bbox_center_uv = (float(xmin + xmax) * 0.5, float(ymin + ymax) * 0.5)
    region = str(depth_region).strip().lower()
    if region in {"center_patch", "bbox_center"}:
        bbox_w = max(1, int(xmax - xmin))
        bbox_h = max(1, int(ymax - ymin))
        patch_w = max(3, int(round(float(bbox_w) * max(0.05, float(center_patch_ratio)))))
        patch_h = max(3, int(round(float(bbox_h) * max(0.05, float(center_patch_ratio)))))
        center_u = int(round(bbox_center_uv[0]))
        center_v = int(round(bbox_center_uv[1]))
        u0 = max(0, min(width - 1, center_u - patch_w // 2))
        u1 = max(u0 + 1, min(width, center_u + patch_w // 2 + 1))
        v0 = max(0, min(height - 1, center_v - patch_h // 2))
        v1 = max(v0 + 1, min(height, center_v + patch_h // 2 + 1))
    else:
        u0 = max(0, min(width - 1, int(xmin)))
        u1 = max(u0 + 1, min(width, int(xmax)))
        v0 = max(0, min(height - 1, int(ymin)))
        v1 = max(v0 + 1, min(height, int(ymax)))

    patch = np.asarray(depth[v0:v1, u0:u1], dtype=np.float32)
    valid_mask = np.isfinite(patch) & (patch >= float(min_depth_m)) & (patch <= float(max_depth_m))
    valid = patch[valid_mask]
    valid_ratio = float(valid.size / max(1, patch.size))
    if valid.size == 0:
        return None, valid_ratio, 0.0, bbox_center_uv

    seed_depth = float(np.median(valid))
    selected_mask = valid_mask
    if region in {"bbox_depth_mask", "mask", "pseudo_mask"} and valid.size >= max(3, int(mask_min_pixels)):
        tolerance = max(0.15, float(seed_depth) * max(0.05, float(mask_tolerance_ratio)))
        near_mask = valid_mask & (np.abs(patch - seed_depth) <= tolerance)
        if int(np.count_nonzero(near_mask)) >= max(3, int(mask_min_pixels)):
            selected_mask = near_mask

    selected = patch[selected_mask]
    mask_area_ratio = float(np.count_nonzero(selected_mask) / max(1, height * width))
    if selected.size == 0:
        selected = valid

    ys, xs = np.nonzero(selected_mask)
    if xs.size > 0:
        center_uv = (float(u0 + np.mean(xs)), float(v0 + np.mean(ys)))
    else:
        center_uv = bbox_center_uv

    if str(statistic).strip().lower() == "trimmed_mean" and selected.size >= 5:
        low = np.percentile(selected, 10.0)
        high = np.percentile(selected, 90.0)
        trimmed = selected[(selected >= low) & (selected <= high)]
        if trimmed.size > 0:
            return float(np.mean(trimmed)), valid_ratio, mask_area_ratio, center_uv
    if str(statistic).strip().lower() in {"near_percentile", "p35", "percentile_35"}:
        return float(np.percentile(selected, 35.0)), valid_ratio, mask_area_ratio, center_uv
    return float(np.median(selected)), valid_ratio, mask_area_ratio, center_uv


def _mask_depth_statistic(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    statistic: str,
) -> tuple[float | None, float, float, tuple[float, float]]:
    depth_np = np.asarray(depth, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=bool)
    if depth_np.ndim != 2 or mask_np.shape != depth_np.shape:
        return None, 0.0, 0.0, (0.0, 0.0)

    mask_count = int(np.count_nonzero(mask_np))
    if mask_count <= 0:
        return None, 0.0, 0.0, (0.0, 0.0)

    valid_mask = (
        mask_np
        & np.isfinite(depth_np)
        & (depth_np >= float(min_depth_m))
        & (depth_np <= float(max_depth_m))
    )
    selected = depth_np[valid_mask]
    valid_ratio = float(selected.size / max(1, mask_count))
    mask_area_ratio = float(mask_count / max(1, depth_np.shape[0] * depth_np.shape[1]))

    ys, xs = np.nonzero(valid_mask if selected.size > 0 else mask_np)
    center_uv = (float(np.mean(xs)), float(np.mean(ys))) if xs.size > 0 else (0.0, 0.0)
    if selected.size == 0:
        return None, valid_ratio, mask_area_ratio, center_uv

    mode = str(statistic).strip().lower()
    if mode == "trimmed_mean" and selected.size >= 5:
        low = np.percentile(selected, 10.0)
        high = np.percentile(selected, 90.0)
        trimmed = selected[(selected >= low) & (selected <= high)]
        if trimmed.size > 0:
            return float(np.mean(trimmed)), valid_ratio, mask_area_ratio, center_uv
    if mode in {"near_percentile", "p35", "percentile_35"}:
        return float(np.percentile(selected, 35.0)), valid_ratio, mask_area_ratio, center_uv
    return float(np.median(selected)), valid_ratio, mask_area_ratio, center_uv
