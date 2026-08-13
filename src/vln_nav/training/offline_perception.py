from __future__ import annotations

from dataclasses import dataclass
import io
import json

import numpy as np
from PIL import Image

from vln_nav.navigation.grounding_dino_detector import GroundingDinoDetector
from vln_nav.navigation.rgbd_geometry_map import backproject_depth_pixels_to_world
from vln_nav.navigation.vlm_client import VLMClient

from .schemas import ObservedLandmark, PerceptionQuery


@dataclass(frozen=True)
class OfflinePerceptionConfig:
    min_detector_confidence: float = 0.30
    auto_accept_confidence: float = 0.75
    min_depth_valid_ratio: float = 0.12
    min_depth_m: float = 0.20
    max_depth_m: float = 10.0
    center_patch_ratio: float = 0.40
    track_match_radius_m: float = 0.80
    track_memory_sec: float = 5.0


@dataclass
class _Track:
    track_id: str
    label: str
    query_id: str
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    detector_confidence: float
    depth_m: float | None
    estimated_world_xy: tuple[float, float] | None
    observation_count: int
    last_seen_frame_index: int
    last_seen_time_sec: float
    verification_status: str
    source: str


class LowConfidenceVLMVerifier:
    """Verify only ambiguous detector crops. Failure rejects the proposal."""

    def __init__(self, *, client: VLMClient) -> None:
        self._client = client

    def verify(
        self,
        *,
        rgb: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        target_text: str,
    ) -> tuple[bool, str]:
        if not self._client.available:
            return False, "vlm_unavailable"
        crop = _crop_with_context(rgb=rgb, bbox_xyxy=bbox_xyxy, context_ratio=0.20)
        buffer = io.BytesIO()
        Image.fromarray(crop).save(buffer, format="PNG")
        result = self._client.chat(
            request_name="offline_low_confidence_verification",
            system_prompt=(
                "You verify one open-vocabulary detector proposal. The image is only the "
                "candidate crop. Reject visual ambiguity. Return JSON only."
            ),
            user_prompt=(
                f"Does this crop clearly show the target phrase {target_text!r}? "
                'Return {"is_target": true|false, "confidence": 0.0-1.0}.'
            ),
            image_png_bytes=buffer.getvalue(),
            max_tokens=80,
            temperature=0.0,
            json_response=True,
        )
        if result.failed:
            return False, result.reason
        try:
            payload = _extract_json_object(result.content)
            accepted = bool(payload.get("is_target", False)) and float(payload.get("confidence", 0.0)) >= 0.70
        except (TypeError, ValueError, json.JSONDecodeError):
            return False, "vlm_invalid_json"
        return accepted, "vlm_confirmed" if accepted else "vlm_rejected"


class OfflineSemanticTracker:
    def __init__(
        self,
        *,
        detector: GroundingDinoDetector,
        config: OfflinePerceptionConfig | None = None,
        verifier: LowConfidenceVLMVerifier | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or OfflinePerceptionConfig()
        self._verifier = verifier
        self._tracks: dict[str, _Track] = {}
        self._next_track_id = 0

    def observe(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_view_matrix_ros: np.ndarray,
        queries: tuple[PerceptionQuery, ...],
        frame_index: int,
        timestamp_sec: float,
    ) -> tuple[tuple[ObservedLandmark, ...], str]:
        query_by_text = {query.text.lower(): query for query in queries}
        grouped, reason = self._detector.detect_queries(
            rgb=rgb,
            target_labels=tuple(query_by_text),
        )
        for query_text, detections in grouped.items():
            query = query_by_text.get(query_text)
            if query is None:
                continue
            for detection in detections:
                confidence = float(detection.confidence)
                if confidence < float(self._config.min_detector_confidence):
                    continue
                verification_status = "detector_auto_accept"
                if confidence < float(self._config.auto_accept_confidence):
                    if self._verifier is None:
                        continue
                    accepted, verification_status = self._verifier.verify(
                        rgb=rgb,
                        bbox_xyxy=detection.bbox,
                        target_text=query.text,
                    )
                    if not accepted:
                        continue
                depth_m, world_xy = _estimate_detection_world_xy(
                    bbox_xyxy=detection.bbox,
                    depth=depth,
                    camera_intrinsics=camera_intrinsics,
                    camera_view_matrix_ros=camera_view_matrix_ros,
                    config=self._config,
                )
                self._update_track(
                    query=query,
                    bbox_xyxy=detection.bbox,
                    confidence=confidence,
                    depth_m=depth_m,
                    world_xy=world_xy,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    verification_status=verification_status,
                )
        return self.snapshot(timestamp_sec=timestamp_sec), reason

    def snapshot(self, *, timestamp_sec: float) -> tuple[ObservedLandmark, ...]:
        cutoff = float(timestamp_sec) - max(0.0, float(self._config.track_memory_sec))
        stale_ids = [track_id for track_id, track in self._tracks.items() if track.last_seen_time_sec < cutoff]
        for track_id in stale_ids:
            del self._tracks[track_id]
        return tuple(
            ObservedLandmark(
                track_id=track.track_id,
                label=track.label,
                query_id=track.query_id,
                bbox_xyxy=track.bbox_xyxy,
                center_uv=track.center_uv,
                detector_confidence=track.detector_confidence,
                depth_m=track.depth_m,
                estimated_world_xy=track.estimated_world_xy,
                observation_count=track.observation_count,
                last_seen_frame_index=track.last_seen_frame_index,
                verification_status=track.verification_status,
                source=track.source,
            )
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        )

    def _update_track(
        self,
        *,
        query: PerceptionQuery,
        bbox_xyxy: tuple[int, int, int, int],
        confidence: float,
        depth_m: float | None,
        world_xy: tuple[float, float] | None,
        frame_index: int,
        timestamp_sec: float,
        verification_status: str,
    ) -> None:
        matched = self._match_track(label=query.canonical_label, world_xy=world_xy)
        if matched is None:
            track_id = f"landmark_{self._next_track_id:04d}"
            self._next_track_id += 1
            matched = _Track(
                track_id=track_id,
                label=query.canonical_label,
                query_id=query.query_id,
                bbox_xyxy=bbox_xyxy,
                center_uv=_bbox_center(bbox_xyxy),
                detector_confidence=float(confidence),
                depth_m=depth_m,
                estimated_world_xy=world_xy,
                observation_count=0,
                last_seen_frame_index=int(frame_index),
                last_seen_time_sec=float(timestamp_sec),
                verification_status=verification_status,
                source="grounding_dino",
            )
            self._tracks[track_id] = matched
        count = int(matched.observation_count) + 1
        if matched.estimated_world_xy is not None and world_xy is not None:
            alpha = 1.0 / float(min(count, 5))
            previous = np.asarray(matched.estimated_world_xy, dtype=np.float64)
            current = np.asarray(world_xy, dtype=np.float64)
            fused = (1.0 - alpha) * previous + alpha * current
            world_xy = float(fused[0]), float(fused[1])
        matched.bbox_xyxy = bbox_xyxy
        matched.center_uv = _bbox_center(bbox_xyxy)
        matched.detector_confidence = float(confidence)
        matched.depth_m = depth_m
        matched.estimated_world_xy = world_xy if world_xy is not None else matched.estimated_world_xy
        matched.observation_count = count
        matched.last_seen_frame_index = int(frame_index)
        matched.last_seen_time_sec = float(timestamp_sec)
        matched.verification_status = verification_status

    def _match_track(
        self,
        *,
        label: str,
        world_xy: tuple[float, float] | None,
    ) -> _Track | None:
        if world_xy is None:
            return None
        point = np.asarray(world_xy, dtype=np.float64)
        candidates = [
            track
            for track in self._tracks.values()
            if track.label == label and track.estimated_world_xy is not None
        ]
        if not candidates:
            return None
        distances = [
            float(np.linalg.norm(point - np.asarray(track.estimated_world_xy, dtype=np.float64)))
            for track in candidates
        ]
        best_index = int(np.argmin(np.asarray(distances, dtype=np.float64)))
        if distances[best_index] > max(0.05, float(self._config.track_match_radius_m)):
            return None
        return candidates[best_index]


def _estimate_detection_world_xy(
    *,
    bbox_xyxy: tuple[int, int, int, int],
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_view_matrix_ros: np.ndarray,
    config: OfflinePerceptionConfig,
) -> tuple[float | None, tuple[float, float] | None]:
    depth_image = np.asarray(depth, dtype=np.float32)
    x0, y0, x1, y1 = bbox_xyxy
    ratio = float(np.clip(config.center_patch_ratio, 0.10, 1.0))
    center_u, center_v = _bbox_center(bbox_xyxy)
    half_width = max(1, int(round(0.5 * float(x1 - x0) * ratio)))
    half_height = max(1, int(round(0.5 * float(y1 - y0) * ratio)))
    patch_x0 = max(0, int(round(center_u)) - half_width)
    patch_x1 = min(depth_image.shape[1], int(round(center_u)) + half_width + 1)
    patch_y0 = max(0, int(round(center_v)) - half_height)
    patch_y1 = min(depth_image.shape[0], int(round(center_v)) + half_height + 1)
    patch = depth_image[patch_y0:patch_y1, patch_x0:patch_x1]
    valid = (
        np.isfinite(patch)
        & (patch >= float(config.min_depth_m))
        & (patch <= float(config.max_depth_m))
    )
    if patch.size == 0 or float(np.count_nonzero(valid)) / float(patch.size) < float(config.min_depth_valid_ratio):
        return None, None
    depth_m = float(np.median(patch[valid]))
    camera_to_world = np.linalg.inv(np.asarray(camera_view_matrix_ros, dtype=np.float64).reshape(4, 4))
    point = backproject_depth_pixels_to_world(
        pixel_u=np.asarray(center_u, dtype=np.float64),
        pixel_v=np.asarray(center_v, dtype=np.float64),
        depth_m=np.asarray(depth_m, dtype=np.float64),
        camera_intrinsics=camera_intrinsics,
        camera_to_world=camera_to_world,
    )
    point = np.asarray(point, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return depth_m, None
    return depth_m, (float(point[0]), float(point[1]))


def _bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    return (
        0.5 * float(bbox_xyxy[0] + bbox_xyxy[2]),
        0.5 * float(bbox_xyxy[1] + bbox_xyxy[3]),
    )


def _crop_with_context(
    *,
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    context_ratio: float,
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    x0, y0, x1, y1 = bbox_xyxy
    margin_x = int(round((x1 - x0) * float(context_ratio)))
    margin_y = int(round((y1 - y0) * float(context_ratio)))
    x0 = max(0, x0 - margin_x)
    y0 = max(0, y0 - margin_y)
    x1 = min(image.shape[1], x1 + margin_x)
    y1 = min(image.shape[0], y1 + margin_y)
    return image[y0:y1, x0:x1, :3].copy()


def _extract_json_object(content: str) -> dict:
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", text, 0)
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("JSON value is not an object", text, start)
    return payload
