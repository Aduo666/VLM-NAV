from __future__ import annotations

import io
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .grounding_dino_detector import GroundingDinoConfig, GroundingDinoDetector
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
    grounding_dino_box_threshold: float = 0.30
    grounding_dino_text_threshold: float = 0.25
    sam_checkpoint_path: str = "weights/grounded_sam/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    sam_device: str = "cuda"
    sam_min_mask_area_ratio: float = 0.0005


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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="target_grounding")
        self._future: Future | None = None
        self._future_request: _GroundingRequest | None = None
        self._last_call_started_sec = -float("inf")
        self._last_call_started_wall_sec = -float("inf")
        self._last_observation: TargetObservation | None = None
        self._last_observation_wall_sec = -float("inf")
        self._grounding_dino = GroundingDinoDetector(
            config=GroundingDinoConfig(
                enabled=str(config.backend).strip().lower() in {"auto", "grounding_dino", "dino"},
                repo_path=str(config.grounding_dino_repo_path),
                config_path=str(config.grounding_dino_config_path),
                checkpoint_path=str(config.grounding_dino_checkpoint_path),
                device=str(config.grounding_dino_device),
                box_threshold=float(config.grounding_dino_box_threshold),
                text_threshold=float(config.grounding_dino_text_threshold),
            ),
            logger=logger,
        )
        self._sam = SamSegmenter(
            config=SamSegmenterConfig(
                checkpoint_path=str(config.sam_checkpoint_path),
                model_type=str(config.sam_model_type),
                device=str(config.sam_device),
                min_mask_area_ratio=float(config.sam_min_mask_area_ratio),
            ),
            logger=logger,
        )
        self._dino_fallback_logged = False

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def observe(
        self,
        *,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        target_label: str,
        robot_pose: Pose2D,
        timestamp_sec: float,
    ) -> TargetObservation:
        target_label = str(target_label).strip() or "target"
        wall_time_sec = time.monotonic()
        completed_observation = self._collect_finished_request(
            timestamp_sec=timestamp_sec,
            wall_time_sec=wall_time_sec,
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
    ) -> TargetObservation | None:
        if self._last_observation is None:
            return None
        if self._last_observation.label != target_label:
            return None
        age_sec = float(wall_time_sec) - float(self._last_observation_wall_sec)
        if age_sec > max(0.0, float(self._config.reuse_last_observation_sec)):
            return None
        return TargetObservation(
            label=self._last_observation.label,
            visible=self._last_observation.visible,
            bbox=self._last_observation.bbox,
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
        )

    def _collect_finished_request(self, *, timestamp_sec: float, wall_time_sec: float) -> TargetObservation | None:
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
            return None
        observation = _copy_observation_with_timestamp(
            self._parse_and_validate(content=content, request=request),
            timestamp_sec=timestamp_sec,
        )
        self._last_observation = observation
        self._last_observation_wall_sec = float(wall_time_sec)
        if observation.visible:
            self._logger.info(
                "[Grounding] target=%s visible=1 bbox=%s depth=%.2f valid_ratio=%.2f mask_area=%.3f bearing_deg=%.1f world=%s confidence=%.2f coord=%s seg=%s",
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
            )
        else:
            self._logger.info("[GroundingReject] target=%s reason=%s", observation.label, observation.reason)
        return observation

    def _call_grounding(self, request: _GroundingRequest) -> tuple[str, str]:
        backend = str(self._config.backend).strip().lower() or "auto"
        if backend in {"auto", "grounding_dino", "dino"}:
            detection, reason = self._grounding_dino.detect(rgb=request.rgb, target_label=request.target_label)
            if detection is not None:
                payload = {
                    "visible": True,
                    "label": request.target_label,
                    "bbox": list(detection.bbox),
                    "confidence": float(detection.confidence),
                    "coordinate_mode": "original_pixels",
                    "source": "grounding_dino",
                }
                return json.dumps(payload), "ok"
            if backend in {"grounding_dino", "dino"}:
                return "", reason
            if not self._dino_fallback_logged:
                self._logger.info("[GroundingDINO] unavailable reason=%s; fallback=qwen_vl", reason)
                self._dino_fallback_logged = True
        return self._call_llm(request)

    def _call_llm(self, request: _GroundingRequest) -> tuple[str, str]:
        system_prompt = (
            "You are a strict visual grounding module for a mobile robot. "
            "Return only JSON. Do not include markdown."
        )
        user_prompt = (
            f"Find the current navigation target: {request.target_label}.\n"
            f"The uploaded image size is width={request.upload_size[0]}, height={request.upload_size[1]}.\n"
            "Return exactly one JSON object using this schema:\n"
            '{"visible": true, "label": "target label", "bbox": [xmin, ymin, xmax, ymax], "confidence": 0.0, "coordinate_mode": "upload_pixels"}\n'
            "Preferred bbox coordinates are pixel coordinates in the uploaded image, format [xmin, ymin, xmax, ymax]. "
            "If you internally use Qwen-style normalized coordinates, use 0-1000 coordinates and set coordinate_mode=normalized_1000. "
            "A partially visible but identifiable target counts as visible. Use a tight bbox around the visible part. "
            "If no identifiable part of the target is visible, return visible=false, bbox=null, confidence=0.0."
        )
        result = self._llm.chat(
            request_name="target_grounding",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_png_bytes=request.image_png_bytes,
            max_tokens=160,
            temperature=0.0,
        )
        if result.used_fallback or not result.content:
            return "", result.reason or "llm_failed"
        return result.content, "ok"

    def _build_request(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        target_label: str,
        robot_pose: Pose2D,
        timestamp_sec: float,
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
        source_prefix = str(payload.get("source", "qwen_vl_bbox")).strip() or "qwen_vl_bbox"
        self._logger.info(
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
            self._logger.info(
                "[SAM] target=%s score=%.3f mask_area=%.3f depth=%s valid_ratio=%.2f",
                request.target_label,
                float(segmentation.score),
                float(mask_area_ratio),
                "none" if depth_m is None else f"{float(depth_m):.2f}",
                float(valid_ratio),
            )

        if depth_m is None or valid_ratio < float(self._config.min_depth_valid_ratio):
            if segmentation is not None:
                self._logger.info(
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
        if valid_ratio < float(self._config.min_depth_valid_ratio):
            return _empty_observation(request.target_label, request.timestamp_sec, "invalid_depth_ratio")
        if depth_m is None or not np.isfinite(depth_m):
            return _empty_observation(request.target_label, request.timestamp_sec, "depth_invalid")
        if depth_m < float(self._config.min_depth_m) or depth_m > float(self._config.max_depth_m):
            return _empty_observation(request.target_label, request.timestamp_sec, "depth_out_of_range")

        geometry = estimate_target_world_xy(
            center_uv=center_uv,
            depth_m=float(depth_m),
            image_width=int(image_width),
            image_height=int(image_height),
            horizontal_fov_deg=float(self._config.horizontal_fov_deg),
            robot_pose=request.robot_pose,
        )
        return TargetObservation(
            label=request.target_label,
            visible=True,
            bbox=bbox,
            center_uv=center_uv,
            confidence=float(confidence),
            depth_m=float(depth_m),
            bearing_rad=float(geometry.bearing_rad),
            estimated_world_xy=list(geometry.world_xy),
            depth_valid_ratio=float(valid_ratio),
            area_ratio=float(area_ratio),
            source=f"{source_prefix}:{coordinate_mode}",
            reason="ok",
            timestamp_sec=float(request.timestamp_sec),
            mask_area_ratio=float(mask_area_ratio),
            coordinate_mode=str(coordinate_mode),
            segmentation_source=str(segmentation_source),
            mask=mask,
        )


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
        source="qwen_vl_bbox",
        reason=str(reason),
        timestamp_sec=float(timestamp_sec),
        mask_area_ratio=0.0,
        coordinate_mode="none",
        segmentation_source="none",
        mask=None,
    )


def _copy_observation_with_timestamp(observation: TargetObservation, *, timestamp_sec: float) -> TargetObservation:
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
        timestamp_sec=float(timestamp_sec),
        mask_area_ratio=observation.mask_area_ratio,
        coordinate_mode=observation.coordinate_mode,
        segmentation_source=observation.segmentation_source,
        mask=None if observation.mask is None else np.asarray(observation.mask, dtype=bool).copy(),
    )


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
