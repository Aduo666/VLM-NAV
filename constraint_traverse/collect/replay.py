from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .recorder import decode_depth_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a collected traversal episode")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
    with np.load(episode / "state.npz", allow_pickle=False) as data:
        count = int(len(data["sample_time_sec"]))
        human = np.asarray(data["human_action_vw"], dtype=np.float32)
        applied = np.asarray(data["applied_action_vw"], dtype=np.float32)
        pose = np.asarray(data["pose_world_xyyaw"], dtype=np.float32)

    delay_ms = max(1, int(round(1000.0 / max(args.fps, 0.1))))
    print(
        f"episode={meta.get('episode_id')} success={meta.get('success')} "
        f"samples={count} duration={meta.get('duration_sec')}"
    )
    start_world = np.asarray(
        meta.get("start_pose_world_xyyaw", pose[0]), dtype=np.float64
    )
    goal_world_value = meta.get("goal_world_xy")
    if goal_world_value is None:
        legacy_goal = meta.get("goal_pose_world_xyyaw")
        goal_world_value = None if legacy_goal is None else legacy_goal[:2]
    goal_world = (
        None
        if goal_world_value is None
        else np.asarray(goal_world_value, dtype=np.float64)
    )
    bev_config = meta.get("bev", {})
    gui_available = _opencv_gui_available()
    output_path = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    elif not gui_available:
        output_path = episode.parent / f"{episode.name}_replay.mp4"
    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"mode=video output={output_path}")
    else:
        print("mode=interactive")

    try:
        for index in range(count):
            stem = f"{index:06d}"
            rgb = cv2.imread(
                str(episode / "front_rgb" / f"{stem}.jpg"),
                cv2.IMREAD_COLOR,
            )
            front_depth = decode_depth_png(
                episode / "front_depth" / f"{stem}.png"
            )
            down_depth = decode_depth_png(
                episode / "down_depth" / f"{stem}.png"
            )
            bev = np.load(
                episode / "bev" / f"{stem}.npy",
                allow_pickle=False,
            )

            if rgb is None:
                raise RuntimeError(f"Missing RGB frame {stem}")
            start_body = _world_xy_to_body(start_world[:2], pose[index])
            trajectory_body = _world_xy_to_body(pose[:, :2], pose[index])
            goal_body = (
                None
                if goal_world is None
                else _world_xy_to_body(goal_world[:2], pose[index])
            )
            canvas = _compose(
                rgb=rgb,
                front_depth=front_depth,
                down_depth=down_depth,
                bev=bev,
                index=index,
                pose=pose[index],
                goal=(
                    np.full(2, np.nan, dtype=np.float32)
                    if goal_body is None
                    else goal_body
                ),
                human=human[index],
                applied=applied[index],
                bev_config=bev_config,
                start_body_xy=start_body,
                goal_body_xy=goal_body,
                trajectory_body_xy=trajectory_body,
            )
            if output_path is not None:
                if writer is None:
                    height, width = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(float(args.fps), 0.1),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Unable to create replay video: {output_path}"
                        )
                writer.write(canvas)
                continue

            cv2.imshow("Constraint Traverse Replay", canvas)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        if writer is not None:
            writer.release()
        if gui_available and output_path is None:
            cv2.destroyAllWindows()

    if output_path is not None:
        print(f"saved={output_path}")
    return 0


def _opencv_gui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            return stripped.split(":", 1)[1].strip().upper() != "NONE"
    return False


def _compose(
    *,
    rgb: np.ndarray,
    front_depth: np.ndarray,
    down_depth: np.ndarray,
    bev: np.ndarray,
    index: int,
    pose: np.ndarray,
    goal: np.ndarray,
    human: np.ndarray,
    applied: np.ndarray,
    bev_config: dict,
    start_body_xy: np.ndarray,
    goal_body_xy: np.ndarray | None,
    trajectory_body_xy: np.ndarray,
) -> np.ndarray:
    target_h = 360
    rgb_panel = cv2.resize(rgb, (480, target_h), interpolation=cv2.INTER_AREA)
    depth_front_panel = _depth_to_bgr(front_depth, (320, 180))
    depth_down_panel = _depth_to_bgr(down_depth, (320, 180))
    depth_panel = np.vstack((depth_front_panel, depth_down_panel))
    bev_panel = _bev_to_bgr(
        bev,
        (360, target_h),
        config=bev_config,
        start_body_xy=start_body_xy,
        goal_body_xy=goal_body_xy,
        trajectory_body_xy=trajectory_body_xy,
    )

    canvas = np.hstack((rgb_panel, depth_panel, bev_panel))
    lines = [
        f"frame={index}",
        f"pose=({pose[0]:.2f}, {pose[1]:.2f}, {np.degrees(pose[2]):.1f}deg)",
        f"goal_body=({goal[0]:.2f}, {goal[1]:.2f})",
        f"human=(v={human[0]:.2f}, w={human[1]:.2f})",
        f"applied=(v={applied[0]:.2f}, w={applied[1]:.2f})",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return canvas


def _depth_to_bgr(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth, 0.0, 5.0)
        normalized[valid] = np.rint((1.0 - clipped[valid] / 5.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)


def _bev_to_bgr(
    bev: np.ndarray,
    size: tuple[int, int],
    *,
    config: dict,
    start_body_xy: np.ndarray,
    goal_body_xy: np.ndarray | None,
    trajectory_body_xy: np.ndarray,
) -> np.ndarray:
    data = np.asarray(bev, dtype=np.uint8)
    occupied = data[0] > 0
    free = data[1] > 0
    bgr = np.full((data.shape[1], data.shape[2], 3), 130, dtype=np.uint8)
    bgr[free] = (238, 238, 238)
    bgr[occupied] = (25, 25, 25)
    h, w = bgr.shape[:2]
    pixels = [
        _body_to_bev_pixel(point, config, width=w, height=h)
        for point in trajectory_body_xy
    ]
    pixels = [point for point in pixels if point is not None]
    if len(pixels) >= 2:
        cv2.polylines(
            bgr,
            [np.asarray(pixels, dtype=np.int32)],
            False,
            (220, 210, 0),
            2,
            cv2.LINE_AA,
        )
    _draw_bev_marker(bgr, start_body_xy, config, (50, 190, 60))
    _draw_bev_marker(bgr, goal_body_xy, config, (50, 50, 230))
    _draw_bev_marker(bgr, np.zeros(2), config, (235, 105, 55))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST)


def _world_xy_to_body(world_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    pose = np.asarray(pose_xyyaw, dtype=np.float64).reshape(3)
    delta = points - pose[:2]
    cosine = np.cos(pose[2])
    sine = np.sin(pose[2])
    result = np.column_stack((
        cosine * delta[:, 0] + sine * delta[:, 1],
        -sine * delta[:, 0] + cosine * delta[:, 1],
    ))
    return result[0] if np.asarray(world_xy).ndim == 1 else result


def _body_to_bev_pixel(
    body_xy: np.ndarray,
    config: dict,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    point = np.asarray(body_xy, dtype=np.float64).reshape(2)
    forward = float(config.get("forward_m", 3.0))
    backward = float(config.get("backward_m", 1.0))
    left = float(config.get("left_m", 2.0))
    right = float(config.get("right_m", 2.0))
    row = int(np.floor((forward - point[0]) / (forward + backward) * height))
    column = int(np.floor((left - point[1]) / (left + right) * width))
    if not (0 <= row < height and 0 <= column < width):
        return None
    return column, row


def _draw_bev_marker(
    image: np.ndarray,
    body_xy: np.ndarray | None,
    config: dict,
    color: tuple[int, int, int],
) -> None:
    if body_xy is None:
        return
    pixel = _body_to_bev_pixel(
        body_xy,
        config,
        width=image.shape[1],
        height=image.shape[0],
    )
    if pixel is not None:
        cv2.circle(image, pixel, 4, color, -1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
