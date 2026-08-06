from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class YOLOEInstanceSegmenterConfig:
    model_path: str
    labels: tuple[str, ...]
    device: str = "cuda:0"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.30
    image_size_px: int = 640
    max_detections: int = 20
    min_mask_pixels: int = 80


@dataclass(frozen=True)
class InstanceMaskObservation:
    label: str
    class_id: int
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    color_rgb: tuple[int, int, int]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Expose the common target-detector bbox contract."""
        return self.bbox_xyxy


def require_yoloe_dependency() -> None:
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError(
            "YOLOE perception requires Ultralytics. Install the project dependencies "
            "with: pip install -r requirements.txt"
        )


class YOLOEInstanceSegmenter:
    """Small, explicit wrapper around Ultralytics YOLOE instance segmentation."""

    def __init__(self, *, config: YOLOEInstanceSegmenterConfig, project_root: Path, logger) -> None:
        self._config = config
        self._project_root = Path(project_root)
        self._logger = logger.getChild("yoloe")
        self._model = None
        self._labels = _normalize_labels(config.labels)
        self._active_labels: tuple[str, ...] = ()
        if not self._labels:
            raise ValueError("YOLOE labels cannot be empty")

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def require_ready(self) -> None:
        if self._model is not None:
            return
        require_yoloe_dependency()
        from ultralytics import YOLOE
        from ultralytics.utils import SETTINGS

        model_reference = self._resolve_model_reference(self._config.model_path)
        try:
            weights_dir = self._project_root / "weights" / "yoloe"
            weights_dir.mkdir(parents=True, exist_ok=True)
            SETTINGS["weights_dir"] = str(weights_dir)
            model = YOLOE(model_reference, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOE model '{model_reference}': {exc}"
            ) from exc
        self._model = model
        self._set_active_labels(self._labels)
        self._logger.warning(
            "[YOLOE] ready model=%s device=%s labels=%s",
            model_reference,
            self._config.device,
            list(self._labels),
        )

    def predict(
        self,
        rgb_image: np.ndarray,
        *,
        labels: tuple[str, ...] | list[str] | None = None,
    ) -> list[InstanceMaskObservation]:
        self.require_ready()
        requested_labels = self._labels if labels is None else _normalize_labels(labels)
        if not requested_labels:
            return []
        self._set_active_labels(requested_labels)
        rgb = _normalize_rgb(rgb_image)
        height, width = rgb.shape[:2]
        try:
            result_list = self._model.predict(
                # Ultralytics treats NumPy arrays as OpenCV/BGR images.  The
                # Isaac camera adapter supplies RGB, so pass a PIL image to
                # preserve channel semantics during open-vocabulary matching.
                source=Image.fromarray(rgb, mode="RGB"),
                conf=float(self._config.confidence_threshold),
                iou=float(self._config.iou_threshold),
                imgsz=max(320, int(self._config.image_size_px)),
                device=str(self._config.device),
                max_det=max(1, int(self._config.max_detections)),
                retina_masks=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLOE inference failed: {exc}") from exc
        if not result_list:
            return []
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or getattr(masks, "data", None) is None:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        confidences = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1)
        mask_data = _to_numpy(masks.data)
        count = min(len(xyxy), len(confidences), len(class_ids), len(mask_data))
        observations: list[InstanceMaskObservation] = []
        for index in range(count):
            class_id = int(class_ids[index])
            label = _class_name(result.names, class_id)
            confidence = float(confidences[index])
            mask = np.asarray(mask_data[index] >= 0.5, dtype=bool)
            if mask.shape != (height, width):
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (width, height),
                        resample=Image.Resampling.NEAREST,
                    )
                ) > 0
            if int(np.count_nonzero(mask)) < max(1, int(self._config.min_mask_pixels)):
                continue
            box = np.asarray(xyxy[index], dtype=np.float64).reshape(-1)
            if box.size < 4 or not np.all(np.isfinite(box[:4])):
                continue
            xmin = int(np.clip(np.floor(box[0]), 0, width - 1))
            ymin = int(np.clip(np.floor(box[1]), 0, height - 1))
            xmax = int(np.clip(np.ceil(box[2]), xmin + 1, width))
            ymax = int(np.clip(np.ceil(box[3]), ymin + 1, height))
            observations.append(
                InstanceMaskObservation(
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox_xyxy=(xmin, ymin, xmax, ymax),
                    mask=mask,
                    color_rgb=color_for_label(label),
                )
            )
        return observations

    def _set_active_labels(self, labels: tuple[str, ...]) -> None:
        if labels == self._active_labels:
            return
        if self._model is None:
            raise RuntimeError("YOLOE model is not initialized")
        try:
            embeddings = self._model.get_text_pe(list(labels))
            self._model.set_classes(list(labels), embeddings)
        except Exception as exc:
            raise RuntimeError(f"Failed to configure YOLOE labels {list(labels)}: {exc}") from exc
        self._active_labels = labels
        self._logger.info("[YOLOE] active_labels=%s", list(labels))

    def render_overlay(
        self,
        *,
        rgb_image: np.ndarray,
        observations: list[InstanceMaskObservation],
        output_size_px: int,
    ) -> Image.Image:
        rgb = _normalize_rgb(rgb_image)
        overlay = rgb.astype(np.float32)
        for observation in observations:
            color = np.asarray(observation.color_rgb, dtype=np.float32)
            mask = np.asarray(observation.mask, dtype=bool)
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
        image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        font = _debug_font(max(14, int(round(min(image.size) / 30.0))))
        for observation in observations:
            xmin, ymin, xmax, ymax = observation.bbox_xyxy
            draw.rectangle(
                (xmin, ymin, max(xmin, xmax - 1), max(ymin, ymax - 1)),
                outline=observation.color_rgb,
                width=3,
            )
            text = f"{observation.label} {observation.confidence:.2f}"
            text_position = (xmin + 2, max(2, ymin - 16))
            text_box = draw.textbbox(text_position, text, font=font)
            draw.rectangle(
                (text_box[0] - 2, text_box[1] - 1, text_box[2] + 2, text_box[3] + 1),
                fill=(10, 10, 10),
            )
            draw.text(text_position, text, fill=(255, 255, 255), font=font)
        return _letterbox(image, max(320, int(output_size_px)))

    def _resolve_model_reference(self, model_path: str) -> str:
        value = str(model_path).strip()
        if not value:
            raise ValueError("YOLOE model_path cannot be empty")
        path = Path(value).expanduser()
        if path.is_absolute() or path.parent != Path("."):
            resolved = path if path.is_absolute() else self._project_root / path
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"YOLOE checkpoint not found: {resolved}. Run scripts/setup_yoloe.sh."
                )
            return str(resolved)
        # A bare official model name is intentionally passed to Ultralytics, which
        # uses a local file when present and otherwise downloads the official weight.
        return value


def color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(label).strip().lower().encode("utf-8")).digest()
    return 55 + digest[0] % 176, 55 + digest[1] % 176, 55 + digest[2] % 176


def _normalize_labels(labels) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in labels:
        label = " ".join(str(value).strip().lower().replace("_", " ").split())
        if not label or label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    return tuple(normalized)


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        value = names.get(int(class_id), str(class_id))
    elif isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
        value = names[int(class_id)]
    else:
        value = str(class_id)
    return str(value).strip().lower().replace("_", " ")


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"YOLOE expects HxWx3/4 RGB input, got shape={array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        array = finite
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _letterbox(image: Image.Image, output_size_px: int) -> Image.Image:
    canvas = Image.new("RGB", (output_size_px, output_size_px), color=(24, 27, 31))
    resized = image.copy()
    resized.thumbnail((output_size_px, output_size_px), resample=Image.Resampling.BILINEAR)
    offset = (
        (output_size_px - resized.width) // 2,
        (output_size_px - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def _debug_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=max(10, int(size)))
    except OSError:
        return ImageFont.load_default()
