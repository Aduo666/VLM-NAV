from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from vln_nav.navigation.internnav_s2_client import (
    InternNavS2Client,
    InternNavS2Config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试远程 InternVLA-N1 S2 服务")
    parser.add_argument("--server", default="http://127.0.0.1:5801")
    parser.add_argument("--mode", choices=("once", "session"), default="once")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--view", choices=("front", "down"), default="front")
    parser.add_argument("--look-down", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--session-id", default="test-session")
    parser.add_argument("--request-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--model-height", type=int, default=384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    client = InternNavS2Client(
        InternNavS2Config(
            base_url=str(args.server),
            infer_once_endpoint="/infer_once",
            reset_endpoint="/reset",
            infer_endpoint="/infer",
            timeout_sec=float(args.timeout),
            jpeg_quality=92,
            model_image_width=int(args.model_width),
            model_image_height=int(args.model_height),
        )
    )
    if args.mode == "once":
        result = client.infer_once(
            rgb=rgb,
            instruction=str(args.instruction),
            timestamp_sec=time.time(),
        )
    else:
        if args.reset:
            client.reset_session(
                session_id=str(args.session_id),
                instruction=str(args.instruction),
            )
        result = client.infer_session(
            rgb=rgb,
            session_id=str(args.session_id),
            request_id=int(args.request_id),
            view=str(args.view),
            look_down=bool(args.look_down),
            timestamp_sec=time.time(),
        )
    print(json.dumps({
        "kind": result.kind,
        "pixel_vu": result.pixel_vu,
        "actions": result.actions,
        "pixel_space_hw": result.pixel_space_hw,
        "latency_sec": result.latency_sec,
        "raw_response": result.raw_response,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
