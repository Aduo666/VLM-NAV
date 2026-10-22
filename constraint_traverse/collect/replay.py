from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator

import cv2
import numpy as np

from .recorder import decode_depth_png
from .validate import _find_episode_dirs


@dataclass(frozen=True)
class _ReplayData:
    episode: Path
    meta: dict
    timestamps: np.ndarray
    human: np.ndarray
    applied: np.ndarray
    pose: np.ndarray

    @property
    def count(self) -> int:
        return len(self.timestamps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放采集记录或批量生成 H.264 视频")
    parser.add_argument("episode", type=Path, help="记录目录；批量模式下填写数据总目录或场景目录")
    parser.add_argument("--fps", type=float, help="回放帧率；默认读取记录的时间间隔")
    parser.add_argument("--output", type=Path, help="单条回放的视频输出路径")
    parser.add_argument("--batch", action="store_true", help="批量生成 H.264 回放视频")
    parser.add_argument("--output-dir", type=Path, help="批量视频的输出目录")
    parser.add_argument("--overwrite", action="store_true", help="重新生成已有视频")
    args = parser.parse_args()
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps 必须为有限正数")
    if args.batch:
        if args.output is not None:
            parser.error("批量模式请使用 --output-dir 指定输出目录")
        if args.output_dir is None:
            parser.error("批量模式需要指定 --output-dir")
    elif args.output_dir is not None or args.overwrite:
        parser.error("--output-dir 和 --overwrite 需要与 --batch 一起使用")
    return args


def main() -> int:
    args = parse_args()
    if args.batch:
        try:
            return _export_batch(args.episode, args.output_dir, args.fps, args.overwrite)
        except Exception as exc:
            print(f"批量生成失败：{exc}", flush=True)
            return 1
    return _replay_single(args)


def _load_replay_data(episode: Path) -> _ReplayData:
    meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
    with np.load(episode / "state.npz", allow_pickle=False) as data:
        timestamps = np.asarray(data["sample_time_sec"], dtype=np.float64)
        human = np.asarray(data["human_action_vw"], dtype=np.float32)
        applied = np.asarray(data["applied_action_vw"], dtype=np.float32)
        pose = np.asarray(data["pose_world_xyyaw"], dtype=np.float32)
    count = len(timestamps)
    if count == 0:
        raise ValueError("记录中没有可回放的样本")
    if timestamps.shape != (count,) or human.shape != (count, 2) or applied.shape != (count, 2) or pose.shape != (count, 3):
        raise ValueError("回放状态的帧数或数组形状不一致")
    return _ReplayData(episode, meta, timestamps, human, applied, pose)


def _replay_fps(timestamps: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return requested
    if len(timestamps) == 1:
        return 15.0
    intervals = np.diff(timestamps)
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError("回放时间戳必须有限且严格递增")
    return 1.0 / float(np.median(intervals))


def _replay_single(args: argparse.Namespace) -> int:
    episode = args.episode.expanduser().resolve()
    data = _load_replay_data(episode)
    fps = _replay_fps(data.timestamps, args.fps)
    delay_ms = max(1, int(round(1000.0 / max(fps, 0.1))))
    print(
        f"episode={data.meta.get('episode_id')} success={data.meta.get('success')} "
        f"samples={data.count} duration={data.meta.get('duration_sec')}"
    )
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
        for index, canvas in enumerate(_episode_frames(data)):
            if output_path is not None:
                if writer is None:
                    height, width = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(float(fps), 0.1),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Unable to create replay video: {output_path}"
                        )
                writer.write(canvas)
                continue

            cv2.imshow("Constraint Traverse Replay", canvas)
            frame_delay = delay_ms
            if args.fps is None and index + 1 < data.count:
                frame_delay = max(1, int(round(1000.0 * (data.timestamps[index + 1] - data.timestamps[index]))))
            key = cv2.waitKey(frame_delay) & 0xFF
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


def _episode_frames(data: _ReplayData) -> Iterator[np.ndarray]:
    start_world = np.asarray(data.meta.get("start_pose_world_xyyaw", data.pose[0]), dtype=np.float64)
    goal_world_value = data.meta.get("goal_world_xy")
    if goal_world_value is None:
        legacy_goal = data.meta.get("goal_pose_world_xyyaw")
        goal_world_value = None if legacy_goal is None else legacy_goal[:2]
    goal_world = None if goal_world_value is None else np.asarray(goal_world_value, dtype=np.float64)
    for index in range(data.count):
        stem = f"{index:06d}"
        rgb = cv2.imread(str(data.episode / "front_rgb" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"缺少或无法读取 RGB 图像：{stem}")
        front_depth = decode_depth_png(data.episode / "front_depth" / f"{stem}.png")
        down_depth = decode_depth_png(data.episode / "down_depth" / f"{stem}.png")
        bev = np.load(data.episode / "bev" / f"{stem}.npy", allow_pickle=False)
        goal_body = None if goal_world is None else _world_xy_to_body(goal_world[:2], data.pose[index])
        yield _compose(
            rgb=rgb, front_depth=front_depth, down_depth=down_depth, bev=bev,
            index=index, pose=data.pose[index],
            goal=np.full(2, np.nan, dtype=np.float32) if goal_body is None else goal_body,
            human=data.human[index], applied=data.applied[index],
            bev_config=data.meta.get("bev", {}),
            start_body_xy=_world_xy_to_body(start_world[:2], data.pose[index]),
            goal_body_xy=goal_body,
            trajectory_body_xy=_world_xy_to_body(data.pose[:, :2], data.pose[index]),
        )


def _complete_h264(path: Path, count: int, ffprobe: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or result.stderr.strip():
            return False
        stream = json.loads(result.stdout)["streams"][0]
        return (
            stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and int(stream["width"]) > 0 and int(stream["height"]) > 0
            and int(stream["nb_read_frames"]) == count
        )
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError, TypeError):
        return False


def _stop_encoder(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass


def _export_h264(data: _ReplayData, fps: float, output: Path, ffmpeg: str, ffprobe: str) -> None:
    frames = _episode_frames(data)
    first = next(frames)
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".tmp.mp4", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                 "-framerate", format(fps, ".12g"), "-i", "pipe:0", "-an", "-c:v", "libx264",
                 "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
            )
            try:
                process.stdin.write(np.ascontiguousarray(first).tobytes())
                for canvas in frames:
                    if canvas.shape != first.shape:
                        raise ValueError("同一条记录中的回放画面尺寸发生变化")
                    process.stdin.write(np.ascontiguousarray(canvas).tobytes())
                process.stdin.close()
                returncode = process.wait()
            except BaseException as exc:
                _stop_encoder(process)
                if isinstance(exc, BrokenPipeError):
                    errors.seek(0)
                    raise RuntimeError(f"FFmpeg 编码失败：{errors.read().decode(errors='replace').strip()}") from exc
                raise
            if returncode != 0:
                errors.seek(0)
                raise RuntimeError(f"FFmpeg 编码失败：{errors.read().decode(errors='replace').strip()}")
        if not _complete_h264(temporary, data.count, ffprobe):
            raise RuntimeError("生成的视频未通过 H.264 编码及完整帧数检查")
        # 编码和检查完成后发布正式文件，中断时只清理临时视频。
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _export_batch(root: Path, output_dir: Path, fps: float | None, overwrite: bool) -> int:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not root.is_dir() or root.name.startswith(("episode_", ".episode_")):
        raise ValueError("批量输入需要填写数据总目录或场景目录")
    targets = [path for path in _find_episode_dirs(root) if path.name.startswith("episode_")]
    if not targets:
        raise ValueError(f"没有找到已保存的采集记录：{root}")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("批量生成需要 FFmpeg，请安装能够执行 ffmpeg 和 ffprobe 的版本，并支持 libx264 编码器")
    generated = skipped = failed = 0
    interrupted = False
    try:
        for index, episode in enumerate(targets, 1):
            print(f"[{index}/{len(targets)}] 正在处理：{episode}", flush=True)
            try:
                data = _load_replay_data(episode)
                if data.meta.get("status") == "recording":
                    skipped += 1
                    print("  跳过：该记录仍在录制", flush=True)
                    continue
                scene = Path(root.name) if episode.parent == root else episode.parent.relative_to(root)
                output = output_dir / scene / f"{episode.name}.mp4"
                rate = _replay_fps(data.timestamps, fps)
                if not overwrite and _complete_h264(output, data.count, ffprobe):
                    skipped += 1
                    print(f"  跳过：完整视频已存在：{output}", flush=True)
                    continue
                print(f"  生成：{output}，帧数={data.count}，帧率={rate:.3f}", flush=True)
                _export_h264(data, rate, output, ffmpeg, ffprobe)
                generated += 1
                print(f"  成功：{output}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  失败：{episode}；{exc}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("批量生成已中断，可以重新运行命令继续处理。", flush=True)
    print(f"批量结果：生成={generated}，跳过={skipped}，失败={failed}", flush=True)
    return 130 if interrupted else (1 if failed else 0)


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
