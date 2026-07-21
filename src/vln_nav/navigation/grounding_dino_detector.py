from __future__ import annotations

import contextlib
import io
import logging
import sys
import types
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GroundingDinoDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    phrase: str


@dataclass
class GroundingDinoConfig:
    enabled: bool
    repo_path: str
    config_path: str
    checkpoint_path: str
    device: str
    box_threshold: float
    text_threshold: float
    text_encoder_path: str = ""


class GroundingDinoDetector:
    """Lazy local GroundingDINO wrapper.

    The dependency is optional. If the repo/checkpoint is unavailable, callers
    should fall back to the VLM grounding path.
    """

    def __init__(self, *, config: GroundingDinoConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("grounding_dino")
        self._model = None
        self._load_failed_reason: str | None = None
        self._runtime_config_path: str | None = None

    @property
    def available(self) -> bool:
        if not bool(self._config.enabled):
            return False
        if self._load_failed_reason is not None:
            return False
        return _path_exists(self._config.repo_path) and _path_exists(self._config.config_path) and _path_exists(
            self._config.checkpoint_path
        )

    def detect(self, *, rgb: np.ndarray, target_label: str) -> tuple[GroundingDinoDetection | None, str]:
        if not bool(self._config.enabled):
            return None, "grounding_dino_disabled"
        if self._load_failed_reason is not None:
            return None, self._load_failed_reason
        if not self.available:
            missing = [
                name
                for name, value in (
                    ("repo", self._config.repo_path),
                    ("config", self._config.config_path),
                    ("checkpoint", self._config.checkpoint_path),
                )
                if not _path_exists(value)
            ]
            return None, f"grounding_dino_missing_{'_'.join(missing) or 'files'}"
        text_encoder_path = Path(self._config.text_encoder_path).expanduser()
        if not text_encoder_path.is_dir() or not (text_encoder_path / "config.json").is_file():
            return None, "grounding_dino_missing_text_encoder"
        model, reason = self._load_model()
        if model is None:
            return None, reason

        image_rgb = np.asarray(rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[-1] < 3:
            return None, "invalid_rgb"
        image_bgr = image_rgb[:, :, :3][:, :, ::-1].copy()
        try:
            with _suppress_grounding_dino_noise():
                detections = model.predict_with_classes(
                    image=image_bgr,
                    classes=[str(target_label).strip().lower()],
                    box_threshold=float(self._config.box_threshold),
                    text_threshold=float(self._config.text_threshold),
                )
        except Exception as exc:
            self._logger.warning("[GroundingDINO] predict_failed target=%s error=%s", target_label, exc)
            return None, "grounding_dino_predict_failed"

        xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        confidence = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
        if xyxy.size == 0 or confidence.size == 0:
            return None, "grounding_dino_no_detection"

        best_idx = int(np.argmax(confidence))
        height, width = image_rgb.shape[:2]
        x0, y0, x1, y1 = xyxy[best_idx].tolist()
        bbox = (
            int(np.clip(round(x0), 0, width - 1)),
            int(np.clip(round(y0), 0, height - 1)),
            int(np.clip(round(x1), 0, width)),
            int(np.clip(round(y1), 0, height)),
        )
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            return None, "grounding_dino_invalid_bbox"
        return (
            GroundingDinoDetection(
                bbox=bbox,
                confidence=float(confidence[best_idx]),
                phrase=str(target_label).strip().lower(),
            ),
            "ok",
        )

    def _load_model(self):
        if self._model is not None:
            return self._model, "ok"
        repo_path = Path(self._config.repo_path).expanduser().resolve()
        try:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("MPLCONFIGDIR", "/tmp")
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            _ensure_supervision_compat()
            _set_noisy_library_log_levels()
            with _suppress_grounding_dino_noise():
                from groundingdino.util.inference import Model

                self._model = Model(
                    model_config_path=self._prepare_runtime_config(),
                    model_checkpoint_path=str(Path(self._config.checkpoint_path).expanduser().resolve()),
                    device=str(self._config.device),
                )
            self._logger.info(
                "[GroundingDINO] loaded config=%s checkpoint=%s device=%s",
                self._config.config_path,
                self._config.checkpoint_path,
                self._config.device,
            )
            return self._model, "ok"
        except Exception as exc:
            self._load_failed_reason = f"grounding_dino_load_failed:{exc}"
            self._logger.warning("[GroundingDINO] load_failed error=%s", exc)
            return None, "grounding_dino_load_failed"

    def _prepare_runtime_config(self) -> str:
        if self._runtime_config_path is not None:
            return self._runtime_config_path
        source_path = Path(self._config.config_path).expanduser().resolve()
        text_encoder_path = Path(self._config.text_encoder_path).expanduser().resolve()
        source = source_path.read_text(encoding="utf-8")
        replacements = (
            ('text_encoder_type = "bert-base-uncased"', f'text_encoder_type = r"{text_encoder_path}"'),
            ("text_encoder_type = 'bert-base-uncased'", f'text_encoder_type = r"{text_encoder_path}"'),
        )
        patched = source
        for old, new in replacements:
            patched = patched.replace(old, new)
        if patched == source:
            raise RuntimeError(
                "GroundingDINO config does not contain the expected bert-base-uncased text_encoder_type"
            )
        fd, path = tempfile.mkstemp(prefix="vlmnav_groundingdino_", suffix=".py", dir="/tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(patched)
        self._runtime_config_path = path
        return path


def _path_exists(value: str) -> bool:
    if not str(value).strip():
        return False
    return Path(value).expanduser().exists()


def _set_noisy_library_log_levels() -> None:
    for name in (
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "transformers",
        "torch",
        "timm",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


@contextlib.contextmanager
def _suppress_grounding_dino_noise():
    """Hide third-party import/inference warnings that otherwise break the interactive prompt."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.filterwarnings("ignore", message=".*torch.meshgrid.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*use_reentrant.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*", category=FutureWarning)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


def _ensure_supervision_compat() -> None:
    """Provide the tiny supervision surface used by GroundingDINO inference.

    CA-Nav pins supervision==0.6.0, but Isaac Sim's Python environment may not
    have it installed. For bbox-only GroundingDINO inference we only need the
    Detections container; annotators are no-ops because this project draws its
    own debug overlays.
    """

    if "supervision" in sys.modules:
        return
    try:
        __import__("supervision")
        return
    except Exception:
        pass

    module = types.ModuleType("supervision")

    class Detections:
        def __init__(
            self,
            *,
            xyxy,
            confidence=None,
            class_id=None,
            mask=None,
            tracker_id=None,
        ) -> None:
            self.xyxy = np.asarray(xyxy, dtype=np.float32)
            self.confidence = (
                np.ones((len(self.xyxy),), dtype=np.float32)
                if confidence is None
                else np.asarray(confidence, dtype=np.float32)
            )
            self.class_id = class_id
            self.mask = mask
            self.tracker_id = tracker_id

        def __len__(self) -> int:
            return int(len(self.xyxy))

        @property
        def box_area(self):
            if len(self.xyxy) == 0:
                return np.asarray([], dtype=np.float32)
            widths = np.maximum(0.0, self.xyxy[:, 2] - self.xyxy[:, 0])
            heights = np.maximum(0.0, self.xyxy[:, 3] - self.xyxy[:, 1])
            return widths * heights

    class _NoOpAnnotator:
        def annotate(self, *, scene, detections, labels=None):
            return scene

    module.Detections = Detections
    module.BoxAnnotator = _NoOpAnnotator
    module.MaskAnnotator = _NoOpAnnotator
    module.__version__ = "compat"
    sys.modules["supervision"] = module
