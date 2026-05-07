from __future__ import annotations

import atexit
import signal
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class DebugVideoRecorder:
    """Low-frequency recorder for the combined navigation debug image."""

    def __init__(self, *, enabled: bool, output_path: Path, fps: float, logger: Any) -> None:
        self._enabled = bool(enabled)
        self._output_path = Path(output_path)
        self._fps = max(0.5, float(fps))
        self._logger = logger
        self._writer = None
        self._cv2 = None
        self._size: tuple[int, int] | None = None
        self._frame_count = 0
        self._closed = False
        atexit.register(self.close)
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_exit_signal)
            except Exception:
                pass

    @property
    def output_path(self) -> Path:
        return self._output_path

    def append(self, image: Image.Image | np.ndarray | None) -> None:
        if not self._enabled or image is None:
            return
        try:
            frame_rgb = self._to_rgb_array(image)
            if frame_rgb is None:
                return
            height, width = frame_rgb.shape[:2]
            if self._writer is None:
                self._open(width=width, height=height)
            if self._writer is None:
                return
            if self._size is not None and (width, height) != self._size:
                frame_rgb = self._cv2.resize(frame_rgb, self._size, interpolation=self._cv2.INTER_AREA)
            frame_bgr = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR)
            self._writer.write(frame_bgr)
            self._frame_count += 1
        except Exception as exc:
            self._enabled = False
            self._logger.warning("Debug video recorder disabled after write failure: %s", exc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        self._logger.info(
            "Debug video saved: %s frames=%s fps=%.1f",
            self._output_path,
            int(self._frame_count),
            float(self._fps),
        )

    def _open(self, *, width: int, height: int) -> None:
        try:
            import cv2
        except Exception as exc:
            self._enabled = False
            self._logger.warning("Debug video disabled because OpenCV import failed: %s", exc)
            return

        self._cv2 = cv2
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self._output_path.suffix.strip().lower()
        if suffix == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._output_path), fourcc, self._fps, (int(width), int(height)))
        if not writer.isOpened():
            self._enabled = False
            self._logger.warning("Debug video disabled because VideoWriter could not open: %s", self._output_path)
            return
        self._writer = writer
        self._size = (int(width), int(height))
        self._logger.info(
            "Debug video output: %s fps=%.1f size=%sx%s",
            self._output_path,
            float(self._fps),
            int(width),
            int(height),
        )

    def _handle_exit_signal(self, signum, frame) -> None:
        _ = frame
        try:
            self._logger.info("Debug video closing on signal=%s", int(signum))
        except Exception:
            pass
        self.close()
        raise SystemExit(128 + int(signum))

    @staticmethod
    def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray | None:
        if isinstance(image, Image.Image):
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[-1] < 3:
            return None
        frame = np.asarray(frame[..., :3], dtype=np.uint8)
        return np.ascontiguousarray(frame)
