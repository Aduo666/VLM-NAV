import argparse
import json
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:5801")
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    with image_path.open("rb") as f:
        response = requests.post(
            f"{args.server.rstrip('/')}/infer_once",
            files={
                "image": (
                    image_path.name,
                    f,
                    "image/jpeg",
                )
            },
            data={
                "instruction": args.instruction,
            },
            timeout=args.timeout,
        )

    print("HTTP status:", response.status_code)

    try:
        data = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)

    response.raise_for_status()


if __name__ == "__main__":
    main()
