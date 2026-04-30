from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SamSegmenterConfig:
    checkpoint_path: str
    model_type: str = "vit_h"
    device: str = "cuda"
    min_mask_area_ratio: float = 0.0005


@dataclass
class SamSegmentation:
    mask: np.ndarray
    score: float
    source: str


class SamSegmenter:
    """Lazy SAM bbox-to-mask wrapper."""

    def __init__(self, *, config: SamSegmenterConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("sam")
        self._predictor = None
        self._load_failed_reason: str | None = None

    def segment(self, *, rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> SamSegmentation | None:
        if self._predictor is None and not self._load():
            return None
        try:
            image = np.asarray(rgb[..., :3], dtype=np.uint8)
            height, width = image.shape[:2]
            x0, y0, x1, y1 = _clip_bbox(bbox=bbox, width=width, height=height)
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                box=np.asarray([x0, y0, x1, y1], dtype=np.float32),
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                return None
            best_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            mask = np.asarray(masks[best_index], dtype=bool)
            mask_area_ratio = float(np.count_nonzero(mask) / max(1, width * height))
            if mask_area_ratio < max(0.0, float(self._config.min_mask_area_ratio)):
                self._logger.info(
                    "[SAMReject] reason=mask_too_small area=%.5f threshold=%.5f score=%.3f",
                    mask_area_ratio,
                    float(self._config.min_mask_area_ratio),
                    float(scores[best_index]),
                )
                return None
            return SamSegmentation(
                mask=mask,
                score=float(scores[best_index]),
                source=f"sam_{self._config.model_type}",
            )
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower() or "cuda oom" in message.lower():
                self._load_failed_reason = "runtime_cuda_out_of_memory"
                self._predictor = None
                self._logger.warning("[SAM] disabled_for_this_run reason=%s error=%s", self._load_failed_reason, exc)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return None
            self._logger.warning("[SAMReject] reason=segment_failed error=%s", exc)
            return None

    def _load(self) -> bool:
        if self._load_failed_reason is not None:
            return False
        checkpoint = Path(self._config.checkpoint_path).expanduser()
        if not checkpoint.exists():
            self._load_failed_reason = f"checkpoint_missing:{checkpoint}"
            self._logger.warning("[SAM] unavailable reason=%s", self._load_failed_reason)
            return False
        try:
            from segment_anything import SamPredictor, sam_model_registry
            import torch
        except Exception as exc:
            self._load_failed_reason = f"import_failed:{exc}"
            self._logger.warning("[SAM] unavailable reason=%s", self._load_failed_reason)
            return False
        try:
            model_type = str(self._config.model_type).strip() or "vit_h"
            if model_type not in sam_model_registry:
                self._load_failed_reason = f"unknown_model_type:{model_type}"
                self._logger.warning("[SAM] unavailable reason=%s", self._load_failed_reason)
                return False
            requested_device = str(self._config.device)
            device = requested_device
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
                self._logger.warning("[SAM] requested cuda but torch.cuda.is_available() is false; using cpu")
            model = sam_model_registry[model_type](checkpoint=str(checkpoint))
            model.to(device=device)
            model.eval()
            self._predictor = SamPredictor(model)
            self._logger.info(
                "[SAM] loaded checkpoint=%s model_type=%s device=%s",
                checkpoint,
                model_type,
                device,
            )
            return True
        except Exception as exc:
            self._load_failed_reason = f"load_failed:{exc}"
            self._logger.warning("[SAM] unavailable reason=%s", self._load_failed_reason)
            return False


def _clip_bbox(*, bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = [int(value) for value in bbox]
    x0 = max(0, min(int(width) - 1, xmin))
    y0 = max(0, min(int(height) - 1, ymin))
    x1 = max(x0 + 1, min(int(width), xmax))
    y1 = max(y0 + 1, min(int(height), ymax))
    return x0, y0, x1, y1
