#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vln_nav.navigation.open_nav_client import OpenNavLLMClient
from vln_nav.navigation.vln_reasoner import LAST_CANDIDATE_CHOICE_DUMP_PATH
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, load_config, setup_logging


DEFAULT_SYSTEM_PROMPT = "You are a multimodal navigation assistant. Reply briefly and directly."
DEFAULT_USER_PROMPT = (
    "This is a connectivity test. Reply with a compact JSON object containing "
    '`status` and `summary`. If an image is attached, mention the main scene briefly in `summary`.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SiliconFlow/Open-Nav LLM test")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument(
        "--mode",
        choices=["replay", "manual"],
        default="replay",
        help="replay=use latest navigation candidate_choice prompt; manual=use custom prompt",
    )
    parser.add_argument(
        "--dump-path",
        default=str(LAST_CANDIDATE_CHOICE_DUMP_PATH),
        help="Prompt dump JSON path for replay mode",
    )
    parser.add_argument("--image", default="", help="Optional image path, overrides dumped image in replay mode")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for manual mode")
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT, help="User prompt for manual mode")
    parser.add_argument("--request-name", default="llm_connectivity_test", help="Request name for manual mode")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max response tokens for manual mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for manual mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config.log_level)

    llm_client = OpenNavLLMClient(
        api_key=config.llm_api_key,
        model_name=config.model_name,
        base_url=config.api_base_url,
        timeout_sec=config.llm_timeout_sec,
        retry_cooldown_sec=config.llm_retry_cooldown_sec,
        upload_max_edge_px=config.llm_upload_max_edge_px,
        image_detail=config.camera_image_detail,
        logger=logger,
    )

    if not llm_client.available:
        logger.error("LLM API key is empty. Check llm.api_key, LLM_API_KEY, or SILICONFLOW_API_KEY.")
        return 2

    if args.mode == "replay":
        request_spec = _load_replay_request(args=args, logger=logger)
    else:
        request_spec = {
            "request_name": args.request_name,
            "system_prompt": args.system_prompt,
            "user_prompt": args.prompt,
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "image_path": str(Path(args.image).expanduser().resolve()) if args.image else "",
            "source": "manual",
        }

    image_bytes, image_path = _load_image_bytes(request_spec.get("image_path", ""), logger=logger)

    logger.info(
        "Testing LLM call source=%s model=%s timeout_sec=%s image=%s max_tokens=%s",
        request_spec.get("source", "unknown"),
        config.model_name,
        config.llm_timeout_sec,
        int(image_bytes is not None),
        int(request_spec["max_tokens"]),
    )

    result = llm_client.chat(
        request_name=str(request_spec["request_name"]),
        system_prompt=str(request_spec["system_prompt"]),
        user_prompt=str(request_spec["user_prompt"]),
        image_png_bytes=image_bytes,
        max_tokens=int(request_spec["max_tokens"]),
        temperature=float(request_spec["temperature"]),
    )

    print("\n=== REQUEST ===")
    print(
        json.dumps(
            {
                "source": request_spec.get("source", "unknown"),
                "request_name": request_spec["request_name"],
                "system_prompt": request_spec["system_prompt"],
                "user_prompt": request_spec["user_prompt"],
                "max_tokens": int(request_spec["max_tokens"]),
                "temperature": float(request_spec["temperature"]),
                "image_path": str(image_path) if image_path is not None else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n=== LLM RESPONSE ===")
    print(result.content or "<empty>")
    print("\n=== META ===")
    print(
        json.dumps(
            {
                "used_fallback": bool(result.used_fallback),
                "reason": result.reason,
                "model_name": config.model_name,
                "api_base_url": config.api_base_url,
                "timeout_sec": config.llm_timeout_sec,
                "upload_max_edge_px": config.llm_upload_max_edge_px,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if not result.used_fallback else 1


def _load_replay_request(*, args: argparse.Namespace, logger) -> dict[str, object]:
    dump_path = Path(args.dump_path).expanduser().resolve()
    if not dump_path.exists():
        logger.error("Replay dump does not exist: %s", dump_path)
        logger.error("Run navigation once so it writes %s", dump_path)
        raise SystemExit(2)

    try:
        payload = json.loads(dump_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to parse replay dump %s: %s", dump_path, exc)
        raise SystemExit(2)

    image_path = str(payload.get("image_path", "")).strip()
    if args.image:
        image_path = str(Path(args.image).expanduser().resolve())

    logger.info("Using replay prompt dump: %s", dump_path)
    return {
        "request_name": str(payload.get("request_name", "candidate_choice")).strip() or "candidate_choice",
        "system_prompt": str(payload.get("system_prompt", "")).strip(),
        "user_prompt": str(payload.get("user_prompt", "")).strip(),
        "max_tokens": int(payload.get("max_tokens", 256)),
        "temperature": float(payload.get("temperature", 0.2)),
        "image_path": image_path,
        "source": f"replay:{dump_path}",
    }


def _load_image_bytes(image_path_str: str, logger) -> tuple[bytes | None, Path | None]:
    if not image_path_str:
        return None, None
    image_path = Path(image_path_str).expanduser().resolve()
    if not image_path.exists():
        logger.error("Image does not exist: %s", image_path)
        raise SystemExit(2)
    logger.info("Using image: %s", image_path)
    return image_path.read_bytes(), image_path


if __name__ == "__main__":
    raise SystemExit(main())
