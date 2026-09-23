from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import re
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_IMAGE_TOKEN = "<image>"
BASE_PROMPT = (
    "You are an autonomous navigation assistant. "
    "Your task is to <instruction>. "
    "Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task."
)

ACTION_TO_CODE = {
    "STOP": 0,
    "↑": 1,
    "←": 2,
    "→": 3,
    "↓": 5,
}


@dataclass
class SessionState:
    instruction: str
    rgb_history: list[Image.Image] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    input_images: list[Image.Image] = field(default_factory=list)
    last_raw_output: str = ""
    step: int = 0
    last_request_id: int = -1
    lookdown_turns: int = 0
    updated_at_sec: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.rgb_history = []
        self.conversation_history = []
        self.input_images = []
        self.last_raw_output = ""
        self.step = 0
        self.last_request_id = -1
        self.lookdown_turns = 0
        self.updated_at_sec = time.monotonic()


class System2Service:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        resize_w: int = 384,
        resize_h: int = 384,
        num_history: int = 8,
        max_new_tokens: int = 128,
        max_lookdown_turns: int = 1,
        session_ttl_sec: float = 1800.0,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.resize_w = int(resize_w)
        self.resize_h = int(resize_h)
        self.num_history = int(num_history)
        self.max_new_tokens = int(max_new_tokens)
        self.max_lookdown_turns = int(max_lookdown_turns)
        self.session_ttl_sec = float(session_ttl_sec)
        if self.resize_w <= 0 or self.resize_h <= 0:
            raise ValueError("resize_w 和 resize_h 必须为正整数")
        if self.num_history <= 0 or self.max_new_tokens <= 0:
            raise ValueError("num_history 和 max_new_tokens 必须为正整数")
        if self.max_lookdown_turns <= 0 or self.session_ttl_sec <= 0:
            raise ValueError("max_lookdown_turns 和 session_ttl_sec 必须为正数")

        print(f"Loading processor: {self.model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "left"

        print(
            f"Loading System 2 model on {self.device}: {self.model_path}",
            flush=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        print("System 2 model loaded.", flush=True)

        self.infer_lock = threading.Lock()
        self.sessions_lock = threading.Lock()
        self.sessions: dict[str, SessionState] = {}

    def reset_session(self, session_id: str, instruction: str) -> SessionState:
        session_id = str(session_id).strip()
        instruction = str(instruction).strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        if not instruction:
            raise ValueError("instruction 不能为空")
        with self.sessions_lock:
            state = self.sessions.get(session_id)
            if state is None:
                state = SessionState(instruction=instruction)
                self.sessions[session_id] = state
            with state.lock:
                state.reset(instruction)
            self._drop_expired_sessions_locked(exclude_session_id=session_id)
            return state

    def infer(
        self,
        *,
        image: Image.Image,
        session_id: str,
        request_id: int,
        view: str,
        look_down: bool,
    ) -> dict:
        session_id = str(session_id).strip()
        view = str(view).strip().lower()
        request_id = int(request_id)
        if not session_id:
            raise ValueError("session_id 不能为空")
        if request_id < 0:
            raise ValueError("request_id 不能为负数")
        if view not in {"front", "down"}:
            raise ValueError("view 必须为 front 或 down")
        if look_down != (view == "down"):
            raise ValueError("look_down 必须与 view=down 保持一致")

        with self.sessions_lock:
            state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError(
                f"session {session_id!r} 不存在，请先调用 POST /reset"
            )

        source_size = [int(image.width), int(image.height)]
        with state.lock:
            if request_id <= state.last_request_id:
                raise RuntimeError(
                    f"request_id 必须递增，上一请求为 {state.last_request_id}"
                )
            if look_down:
                messages, images, step = self._prepare_lookdown_turn(state, image)
            else:
                messages, images, step = self._prepare_front_turn(state, image)

            raw_output, latency_ms = self._generate(messages, images)
            parsed = self._parse_output(raw_output)
            state.last_raw_output = raw_output
            state.last_request_id = request_id
            state.updated_at_sec = time.monotonic()
            if parsed["type"] != "action" or parsed["action_codes"] != [5]:
                state.lookdown_turns = 0

        return self._build_response(
            parsed=parsed,
            raw_output=raw_output,
            source_size=source_size,
            session_id=session_id,
            request_id=request_id,
            view=view,
            look_down=look_down,
            step=step,
            latency_ms=latency_ms,
        )

    def infer_once(self, *, image: Image.Image, instruction: str) -> dict:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction 不能为空")
        source_size = [int(image.width), int(image.height)]
        temporary = SessionState(instruction=instruction)
        messages, images, step = self._prepare_front_turn(temporary, image)
        raw_output, latency_ms = self._generate(messages, images)
        parsed = self._parse_output(raw_output)
        return self._build_response(
            parsed=parsed,
            raw_output=raw_output,
            source_size=source_size,
            session_id=None,
            request_id=None,
            view="front",
            look_down=False,
            step=step,
            latency_ms=latency_ms,
        )

    def _build_response(
        self,
        *,
        parsed: dict,
        raw_output: str,
        source_size: list[int],
        session_id: str | None,
        request_id: int | None,
        view: str,
        look_down: bool,
        step: int,
        latency_ms: float,
    ) -> dict:
        response = {
            "session_id": session_id,
            "request_id": request_id,
            "view": view,
            "look_down": bool(look_down),
            "step": step,
            "raw_output": raw_output,
            "source_image_size_wh": source_size,
            "model_input_size_wh": [self.resize_w, self.resize_h],
            "pixel_space_hw": [self.resize_h, self.resize_w],
            "latency_ms": round(latency_ms, 2),
        }
        if parsed["type"] == "pixel_goal":
            response["pixel_goal"] = parsed["pixel_goal"]
        elif parsed["type"] == "action":
            response["discrete_action"] = parsed["action_codes"]
        else:
            raise RuntimeError(f"无法解析 S2 输出：{raw_output!r}")
        return response

    def health(self) -> dict:
        with self.sessions_lock:
            self._drop_expired_sessions_locked(exclude_session_id=None)
            sessions = {
                session_id: {
                    "instruction": state.instruction,
                    "step": state.step,
                    "last_request_id": state.last_request_id,
                    "lookdown_turns": state.lookdown_turns,
                }
                for session_id, state in self.sessions.items()
            }
        return {
            "status": "ok",
            "model_path": self.model_path,
            "device": self.device,
            "session_count": len(sessions),
            "sessions": sessions,
        }

    def _prepare_front_turn(
        self,
        state: SessionState,
        image: Image.Image,
    ) -> tuple[list[dict], list[Image.Image], int]:
        current_image = self._resize_image(image)
        state.rgb_history.append(current_image)
        current_index = len(state.rgb_history) - 1
        if current_index == 0:
            history_ids: list[int] = []
        else:
            history_ids = np.unique(
                np.linspace(
                    0,
                    current_index - 1,
                    self.num_history,
                    dtype=np.int32,
                )
            ).tolist()
        images = [state.rgb_history[index] for index in history_ids]
        images.append(current_image)
        prompt = BASE_PROMPT.replace("<instruction>.", state.instruction)
        if history_ids:
            prompt += (
                " These are your historical observations: "
                + (DEFAULT_IMAGE_TOKEN + "\n") * len(history_ids)
                + "."
            )
        prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."
        messages = [{"role": "user", "content": self._prompt_content(prompt, images)}]
        state.conversation_history = copy.deepcopy(messages)
        state.input_images = list(images)
        state.last_raw_output = ""
        state.lookdown_turns = 0
        step = state.step
        state.step += 1
        return messages, images, step

    def _prepare_lookdown_turn(
        self,
        state: SessionState,
        image: Image.Image,
    ) -> tuple[list[dict], list[Image.Image], int]:
        if not state.last_raw_output or not state.conversation_history:
            raise RuntimeError("LOOKDOWN 缺少上一轮前视对话")
        if "↓" not in state.last_raw_output:
            raise RuntimeError("上一轮 S2 输出没有请求 LOOKDOWN")
        if state.lookdown_turns >= self.max_lookdown_turns:
            raise RuntimeError(
                f"LOOKDOWN 连续次数超过限制 {self.max_lookdown_turns}"
            )
        down_image = image.convert("RGB")
        state.input_images.append(down_image)
        state.conversation_history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": state.last_raw_output}],
            }
        )
        state.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "you can see"},
                    {"type": "image", "image": down_image},
                ],
            }
        )
        state.lookdown_turns += 1
        return (
            copy.deepcopy(state.conversation_history),
            list(state.input_images),
            state.step,
        )

    def _generate(
        self,
        messages: list[dict],
        images: list[Image.Image],
    ) -> tuple[str, float]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        ).to(self.model.device)
        started = time.perf_counter()
        with self.infer_lock:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        raw_output = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        return raw_output, latency_ms

    def _prompt_content(
        self,
        prompt: str,
        images: list[Image.Image],
    ) -> list[dict]:
        parts = re.split(r"(<image>)", prompt)
        content = []
        image_index = 0
        for part in parts:
            if part == DEFAULT_IMAGE_TOKEN:
                if image_index >= len(images):
                    raise RuntimeError("prompt 中的图像标记数量超过图像数量")
                content.append({"type": "image", "image": images[image_index]})
                image_index += 1
            else:
                clean = part.replace("\n", "").strip()
                if clean:
                    content.append({"type": "text", "text": clean})
        if image_index != len(images):
            raise RuntimeError("图像数量与 prompt 中的图像标记不一致")
        return content

    def _parse_output(self, raw_output: str) -> dict:
        coords = [int(value) for value in re.findall(r"\d+", raw_output)]
        if len(coords) >= 2:
            pixel_goal = [coords[1], coords[0]]
            if not (
                0 <= pixel_goal[0] < self.resize_h
                and 0 <= pixel_goal[1] < self.resize_w
            ):
                raise RuntimeError(
                    f"S2 pixel_goal {pixel_goal} 超出模型图像范围 "
                    f"{self.resize_h}x{self.resize_w}"
                )
            return {
                "type": "pixel_goal",
                "coord_model": [coords[0], coords[1]],
                "pixel_goal": pixel_goal,
            }
        actions = re.findall(r"STOP|↑|←|→|↓", raw_output)
        if actions:
            return {
                "type": "action",
                "actions": actions,
                "action_codes": [ACTION_TO_CODE[action] for action in actions],
            }
        return {"type": "unknown"}

    def _resize_image(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize((self.resize_w, self.resize_h))

    def _drop_expired_sessions_locked(self, exclude_session_id: str | None) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, state in self.sessions.items()
            if session_id != exclude_session_id
            and now - state.updated_at_sec > self.session_ttl_sec
        ]
        for session_id in expired:
            del self.sessions[session_id]


def create_app(service: System2Service) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(service.health())

    @app.post("/infer_once")
    def infer_once():
        if "image" not in request.files:
            return jsonify({"error": "multipart 中缺少 image"}), 400
        instruction = request.form.get("instruction", "").strip()
        try:
            image = Image.open(request.files["image"].stream).convert("RGB")
            result = service.infer_once(image=image, instruction=instruction)
            print(
                f"[S2Once] result={result.get('raw_output')!r}",
                flush=True,
            )
            return jsonify(result)
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/reset")
    def reset():
        data = request.get_json(silent=True) or {}
        session_id = str(data.get("session_id", "legacy")).strip()
        instruction = str(data.get("instruction", "")).strip()
        try:
            service.reset_session(session_id, instruction)
            return jsonify(
                {
                    "status": "ok",
                    "session_id": session_id,
                    "instruction": instruction,
                    "step": 0,
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/infer")
    def infer():
        if "image" not in request.files:
            return jsonify({"error": "multipart 中缺少 image"}), 400
        session_id = request.form.get("session_id", "").strip()
        view = request.form.get("view", "front").strip().lower()
        look_down = _parse_bool(request.form.get("look_down", "false"))
        request_id_text = request.form.get("request_id")
        if request_id_text is None:
            return jsonify({"error": "multipart 中缺少 request_id"}), 400
        try:
            image = Image.open(request.files["image"].stream).convert("RGB")
            result = service.infer(
                image=image,
                session_id=session_id,
                request_id=int(request_id_text),
                view=view,
                look_down=look_down,
            )
            print(
                f"[S2] session={session_id} request={request_id_text} view={view} "
                f"result={result.get('raw_output')!r}",
                flush=True,
            )
            return jsonify(result)
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument("--resize-w", type=int, default=384)
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-lookdown-turns", type=int, default=1)
    parser.add_argument("--session-ttl-sec", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = System2Service(
        model_path=args.model_path,
        device=args.device,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        num_history=args.num_history,
        max_new_tokens=args.max_new_tokens,
        max_lookdown_turns=args.max_lookdown_turns,
        session_ttl_sec=args.session_ttl_sec,
    )
    app = create_app(service)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
