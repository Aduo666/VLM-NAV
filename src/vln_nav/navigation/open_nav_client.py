from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class ChatResult:
    content: str
    failed: bool
    reason: str
    finish_reason: str = ""


class OpenNavLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout_sec: int,
        retry_cooldown_sec: float,
        upload_max_edge_px: int,
        image_detail: str,
        logger: logging.Logger,
    ) -> None:
        self._api_key = str(api_key).strip()
        self._model_name = str(model_name).strip()
        self._base_url = str(base_url).rstrip("/")
        self._timeout_sec = int(timeout_sec)
        self._retry_cooldown_sec = max(0.0, float(retry_cooldown_sec))
        self._upload_max_edge_px = max(0, int(upload_max_edge_px))
        self._image_detail = str(image_detail).strip().lower() or "high"
        self._logger = logger.getChild("open_nav_client")
        self._cooldown_until = 0.0
        self._consecutive_failures = 0

        if not self._api_key:
            self._logger.warning("SILICONFLOW_API_KEY/LLM_API_KEY is empty, Open-Nav reasoner will fail closed.")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def chat(
        self,
        *,
        request_name: str = "chat",
        system_prompt: str,
        user_prompt: str,
        image_png_bytes: bytes | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        json_response: bool = False,
    ) -> ChatResult:
        if not self.available:
            return ChatResult(content="", failed=True, reason="llm_unavailable")

        now = time.monotonic()
        if now < self._cooldown_until:
            remaining = max(0.0, self._cooldown_until - now)
            self._logger.info(
                "[LLM] skip name=%s reason=cooldown remaining_sec=%.1f failures=%s",
                request_name,
                remaining,
                self._consecutive_failures,
            )
            return ChatResult(content="", failed=True, reason="llm_cooldown")

        self._logger.info(
            "[LLM] request name=%s timeout_sec=%s image=%s prompt_chars=%s max_tokens=%s",
            request_name,
            self._timeout_sec,
            int(image_png_bytes is not None),
            len(system_prompt) + len(user_prompt),
            int(max_tokens),
        )

        content_items: list[dict[str, Any]] = []
        if image_png_bytes is not None:
            upload_image_bytes = image_png_bytes
            try:
                upload_image_bytes, _, _ = self._prepare_image_upload(image_png_bytes)
            except Exception as exc:
                self._logger.warning("Open-Nav LLM image resize failed name=%s: %s", request_name, exc)
            image_base64 = base64.b64encode(upload_image_bytes).decode("utf-8")
            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": self._image_detail,
                    },
                }
            )
        content_items.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_items},
            ],
            "temperature": float(temperature),
            "stream": False,
            "max_tokens": int(max_tokens),
        }
        if bool(json_response):
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        request_started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self._register_failure()
            self._logger.warning("Open-Nav LLM HTTP name=%s code=%s: %s", request_name, exc.code, body)
            return ChatResult(content="", failed=True, reason="llm_http_error")
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            self._register_failure()
            self._logger.warning("Open-Nav LLM request failed name=%s: %s", request_name, exc)
            return ChatResult(content="", failed=True, reason="llm_request_failed")

        try:
            response_json = json.loads(raw_body)
            choice = response_json["choices"][0]
            content = str(choice["message"]["content"])
            finish_reason = str(choice.get("finish_reason", "")).strip().lower()
        except Exception as exc:
            self._register_failure()
            self._logger.warning("Open-Nav LLM response parse failed name=%s: %s", request_name, exc)
            return ChatResult(content="", failed=True, reason="llm_parse_failed")

        if finish_reason == "length":
            self._register_failure()
            self._logger.error(
                "[LLM] truncated name=%s max_tokens=%s",
                request_name,
                int(max_tokens),
            )
            return ChatResult(
                content=content,
                failed=True,
                reason="llm_output_truncated",
                finish_reason=finish_reason,
            )

        latency_ms = (time.monotonic() - request_started) * 1000.0
        self._cooldown_until = 0.0
        self._consecutive_failures = 0
        self._logger.info(
            "[LLM] success name=%s latency_ms=%.0f image=%s max_tokens=%s",
            request_name,
            latency_ms,
            int(image_png_bytes is not None),
            int(max_tokens),
        )
        return ChatResult(
            content=content,
            failed=False,
            reason="ok",
            finish_reason=finish_reason,
        )

    def _register_failure(self) -> None:
        self._consecutive_failures += 1
        if self._retry_cooldown_sec > 0.0:
            self._cooldown_until = time.monotonic() + self._retry_cooldown_sec

    def _prepare_image_upload(self, image_png_bytes: bytes) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
        with Image.open(io.BytesIO(image_png_bytes)) as source_image:
            source_image.load()
            original_size = (int(source_image.width), int(source_image.height))
            upload_image = source_image.convert("RGB")

        if self._upload_max_edge_px > 0:
            max_edge = max(upload_image.size)
            if max_edge > self._upload_max_edge_px:
                scale = float(self._upload_max_edge_px) / float(max_edge)
                new_size = (
                    max(1, int(round(upload_image.width * scale))),
                    max(1, int(round(upload_image.height * scale))),
                )
                resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                upload_image = upload_image.resize(new_size, resampling)

        upload_buffer = io.BytesIO()
        upload_image.save(upload_buffer, format="PNG", optimize=True)
        return upload_buffer.getvalue(), original_size, (int(upload_image.width), int(upload_image.height))
