from __future__ import annotations

import atexit
import shutil
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class DebugVideoRecorder:
    """Low-frequency recorder for the combined navigation debug image."""

    def __init__(self, *, enabled: bool, output_path: Path, fps: float, logger: Any) -> None:
        self._enabled = bool(enabled)
        self._output_path = _dated_video_path(Path(output_path))
        self._fps = max(0.5, float(fps))
        self._logger = logger
        self._process: subprocess.Popen | None = None
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
        if not self._enabled or self._closed or image is None:
            return
        try:
            frame_rgb = self._to_rgb_array(image)
            if frame_rgb is None:
                return
            height, width = frame_rgb.shape[:2]
            if self._process is None:
                self._open(width=width, height=height)
            if self._process is None or self._process.stdin is None:
                return
            if self._size is not None and (width, height) != self._size:
                frame_rgb = np.asarray(
                    Image.fromarray(frame_rgb, mode="RGB").resize(
                        self._size,
                        resample=Image.Resampling.BILINEAR,
                    ),
                    dtype=np.uint8,
                )
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"FFmpeg exited before frame write with code {self._process.returncode}"
                )
            self._process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
            self._frame_count += 1
        except Exception as exc:
            self._enabled = False
            self._logger.warning("H.264 video recorder disabled after write failure: %s", exc)
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        self._process = None
        if process is None:
            return
        return_code = -1
        stderr_text = ""
        try:
            if process.stdin is not None:
                process.stdin.close()
            return_code = int(process.wait(timeout=30.0))
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = int(process.wait(timeout=5.0))
            stderr_text = "FFmpeg timed out while finalizing the MP4 container"
        finally:
            if process.stderr is not None:
                stderr_bytes = process.stderr.read()
                if stderr_bytes:
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

        if return_code == 0 and self._frame_count > 0:
            self._logger.info(
                "H.264 video saved: %s frames=%s fps=%.1f codec=libx264 pix_fmt=yuv420p",
                self._output_path,
                int(self._frame_count),
                float(self._fps),
            )
            return
        self._logger.warning(
            "H.264 video finalization failed: %s frames=%s return_code=%s error=%s",
            self._output_path,
            int(self._frame_count),
            int(return_code),
            stderr_text or "unknown",
        )

    def _open(self, *, width: int, height: int) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self._enabled = False
            self._logger.warning(
                "H.264 video disabled because FFmpeg is not available on PATH"
            )
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        even_width = max(2, int(width) - int(width) % 2)
        even_height = max(2, int(height) - int(height) % 2)
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{even_width}x{even_height}",
            "-framerate",
            f"{self._fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._output_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            self._enabled = False
            self._logger.warning(
                "H.264 video disabled because FFmpeg could not start: %s",
                exc,
            )
            return
        self._process = process
        self._size = (even_width, even_height)
        self._logger.info(
            "H.264 video output: %s fps=%.1f size=%sx%s codec=libx264",
            self._output_path,
            float(self._fps),
            even_width,
            even_height,
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
        even_height = int(frame.shape[0]) - int(frame.shape[0]) % 2
        even_width = int(frame.shape[1]) - int(frame.shape[1]) % 2
        if even_height < 2 or even_width < 2:
            return None
        return np.ascontiguousarray(frame[:even_height, :even_width])


def _dated_video_path(path: Path) -> Path:
    stem = path.stem or "vln_debug"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{stem}_{timestamp}.mp4")
