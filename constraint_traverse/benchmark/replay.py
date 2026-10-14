from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .geometry import DoorScene, PassageGeometry
from .ui import MapRenderer


def main():
    parser = argparse.ArgumentParser(description="将窄门试验回放保存为 MP4")
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("回放帧率必须为正数")
    directory = args.trial.resolve()
    with (directory / "meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        times, states = archive["time_sec"].copy(), archive["state"].copy()
    if len(times) == 0:
        raise ValueError("该试验没有可回放的状态记录")
    geometry = PassageGeometry(meta["parts"], DoorScene(**meta["scene"]))
    renderer = MapRenderer(geometry, meta["config"]["trial"]["map_size_px"])
    output = args.output or directory / "replay.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (renderer.size, renderer.size + 60))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频输出：{output}")
    plan_paths = sorted((directory / "plans").glob("*.npz"))
    plan_times = []
    for path in plan_paths:
        with np.load(path, allow_pickle=False) as plan:
            plan_times.append(float(plan["timestamp_sec"]))
    current_plan = -1
    prediction = candidates = None
    try:
        for timestamp in np.arange(times[0], times[-1] + 1 / args.fps, 1 / args.fps):
            sample = min(int(np.searchsorted(times, timestamp)), len(times) - 1)
            plan_index = int(np.searchsorted(plan_times, timestamp, side="right")) - 1
            if plan_index >= 0 and plan_index != current_plan:
                with np.load(plan_paths[plan_index], allow_pickle=False) as plan:
                    prediction = plan["prediction"].copy()
                    candidates = plan["candidates"][:24].copy()
                current_plan = plan_index
            rgba = renderer.render(states[sample, :3], meta["goal_world_xy"], prediction, candidates,
                                    states[:sample + 1, :3], meta["reference_path"])
            canvas = Image.new("RGB", (renderer.size, renderer.size + 60), (245, 245, 245))
            canvas.paste(Image.fromarray(rgba).convert("RGB"), (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, renderer.size + 8), f"{meta['method']}  {meta['status']}  t={timestamp:.2f}s", fill=(30, 30, 30))
            draw.text((10, renderer.size + 30), "Blue: executed   Green: predicted   Gray-green: sampled", fill=(30, 30, 30))
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"回放文件：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
