from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np


@dataclass(frozen=True)
class InternNavS2Config:
    base_url: str
    endpoint: str
    timeout_sec: float
    jpeg_quality: int
    model_image_width: int
    model_image_height: int

    @classmethod
    def from_mapping(cls, data: dict) -> "InternNavS2Config":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"internnav_s2.server 必须包含这些字段：{sorted(expected)}")
        config = cls(
            base_url=str(data["base_url"]).strip().rstrip("/"),
            endpoint=str(data["endpoint"]).strip(),
            timeout_sec=float(data["timeout_sec"]),
            jpeg_quality=int(data["jpeg_quality"]),
            model_image_width=int(data["model_image_width"]),
            model_image_height=int(data["model_image_height"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("S2 base_url 必须以 http:// 或 https:// 开头")
        if not self.endpoint.startswith("/"):
            raise ValueError("S2 endpoint 必须以 / 开头")
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ValueError("S2 timeout_sec 必须为有限正数")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("S2 jpeg_quality 必须位于 1 至 100")
        if self.model_image_width <= 0 or self.model_image_height <= 0:
            raise ValueError("S2 模型图像尺寸必须为正整数")


@dataclass(frozen=True)
class InternNavS2Result:
    kind: str
    pixel_vu: tuple[float, float] | None
    actions: tuple[int, ...]
    pixel_space_hw: tuple[int, int]
    latency_sec: float
    raw_response: dict


class InternNavS2Client:
    _VALID_ACTIONS = {0, 1, 2, 3, 5}
    _ACTION_ALIASES = {
        "STOP": 0,
        "↑": 1,
        "FORWARD": 1,
        "←": 2,
        "LEFT": 2,
        "→": 3,
        "RIGHT": 3,
        "↓": 5,
        "LOOKDOWN": 5,
        "LOOK_DOWN": 5,
    }

    def __init__(self, config: InternNavS2Config) -> None:
        config.validate()
        self.config = config

    def infer_once(
        self,
        *,
        rgb: np.ndarray,
        instruction: str,
        session_id: str,
        request_id: int,
        view: str,
        reset: bool,
        look_down: bool,
        timestamp_sec: float,
    ) -> InternNavS2Result:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"S2 RGB 必须是 H×W×3，收到 {image.shape}")
        if not str(instruction).strip():
            raise ValueError("S2 instruction 不能为空")
        if view not in {"front", "down"}:
            raise ValueError("S2 view 必须为 front 或 down")
        if not str(session_id).strip() or int(request_id) < 0:
            raise ValueError("S2 session_id 和 request_id 无效")
        if not math.isfinite(float(timestamp_sec)):
            raise ValueError("S2 帧时间戳无效")
        image = np.ascontiguousarray(np.clip(image[:, :, :3], 0, 255).astype(np.uint8))
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("S2 RGB JPEG 编码失败")
        fields = {
            "instruction": str(instruction),
            "session_id": str(session_id),
            "request_id": str(int(request_id)),
            "view": view,
            "reset": json.dumps(bool(reset)),
            "look_down": json.dumps(bool(look_down)),
            "timestamp_sec": repr(float(timestamp_sec)),
            "image_width": str(int(image.shape[1])),
            "image_height": str(int(image.shape[0])),
        }
        start = time.monotonic()
        payload = self._post_multipart(
            fields=fields,
            image_bytes=encoded.tobytes(),
        )
        latency = time.monotonic() - start
        return self._normalize(payload, latency_sec=latency)

    def _post_multipart(self, *, fields: dict[str, str], image_bytes: bytes) -> dict:
        boundary = f"----captainav-s2-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend((
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ))
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ))
        request = urllib.request.Request(
            f"{self.config.base_url}{self.config.endpoint}",
            data=b"".join(chunks),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_sec,
            ) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"S2 HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"S2 请求失败：{exc}") from exc
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"S2 返回了无效 JSON：{response_text[:500]}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("S2 响应必须是 JSON 对象")
        return payload

    def _normalize(self, payload: dict, *, latency_sec: float) -> InternNavS2Result:
        response = payload.get("result", payload)
        if not isinstance(response, dict):
            raise RuntimeError(
                "S2 result 必须是 JSON 对象；"
                f"实际响应={json.dumps(payload, ensure_ascii=False)[:1000]}"
            )
        pixel_key = next(
            (
                key
                for key in ("pixel_goal", "output_pixel", "pixel_vu")
                if key in response and response[key] is not None
            ),
            None,
        )
        action_key = next(
            (
                key
                for key in ("discrete_action", "output_action", "actions")
                if key in response and response[key] is not None
            ),
            None,
        )
        has_pixel = pixel_key is not None
        has_actions = action_key is not None
        if has_pixel == has_actions:
            raise RuntimeError(
                "S2 响应必须且只能包含 pixel_goal/output_pixel 或 "
                "discrete_action/output_action；"
                f"实际响应={json.dumps(payload, ensure_ascii=False)[:1000]}"
            )
        pixel_space = response.get(
            "pixel_space_hw",
            [self.config.model_image_height, self.config.model_image_width],
        )
        if not isinstance(pixel_space, (list, tuple)) or len(pixel_space) != 2:
            raise RuntimeError("S2 pixel_space_hw 必须是 [height,width]")
        pixel_space_hw = (int(pixel_space[0]), int(pixel_space[1]))
        if pixel_space_hw[0] <= 0 or pixel_space_hw[1] <= 0:
            raise RuntimeError("S2 pixel_space_hw 必须为正数")
        if has_pixel:
            pixel = np.asarray(response[pixel_key], dtype=np.float64).reshape(-1)
            if pixel.size != 2 or not np.isfinite(pixel).all():
                raise RuntimeError("S2 pixel_goal 必须包含两个有限数值")
            v, u = float(pixel[0]), float(pixel[1])
            if not (0 <= v < pixel_space_hw[0] and 0 <= u < pixel_space_hw[1]):
                raise RuntimeError(
                    f"S2 pixel_goal {[v, u]} 超出 {pixel_space_hw}"
                )
            return InternNavS2Result(
                kind="pixel_goal",
                pixel_vu=(v, u),
                actions=(),
                pixel_space_hw=pixel_space_hw,
                latency_sec=float(latency_sec),
                raw_response=dict(payload),
            )
        raw_actions = response[action_key]
        if isinstance(raw_actions, (int, float, str)):
            raw_actions = [raw_actions]
        if not isinstance(raw_actions, (list, tuple)) or not raw_actions:
            raise RuntimeError("S2 discrete_action 必须是非空动作列表")
        actions = tuple(self._normalize_action(value) for value in raw_actions)
        if any(action not in self._VALID_ACTIONS for action in actions):
            raise RuntimeError(f"S2 返回了未知动作：{actions}")
        return InternNavS2Result(
            kind="discrete",
            pixel_vu=None,
            actions=actions,
            pixel_space_hw=pixel_space_hw,
            latency_sec=float(latency_sec),
            raw_response=dict(payload),
        )

    def _normalize_action(self, value) -> int:
        if isinstance(value, str):
            token = value.strip().upper()
            if token in self._ACTION_ALIASES:
                return self._ACTION_ALIASES[token]
            try:
                return int(token)
            except ValueError as exc:
                raise RuntimeError(f"S2 返回了未知动作符号：{value!r}") from exc
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"S2 返回了无效动作：{value!r}") from exc
