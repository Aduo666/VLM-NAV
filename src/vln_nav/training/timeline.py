from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vln_nav.navigation.occupancy_grid import OccupancyGridSnapshot

from .schemas import FrontierToken, ObservedLandmark, RouteAlignmentRecord


def render_timeline_frame(
    *,
    rgb: np.ndarray,
    snapshot: OccupancyGridSnapshot,
    robot_pose_xyyaw: tuple[float, float, float],
    trajectory_world_xy: tuple[tuple[float, float], ...],
    landmarks: tuple[ObservedLandmark, ...],
    frontiers: tuple[FrontierToken, ...],
    instruction: str,
    active_alignment: RouteAlignmentRecord | None,
    selected_frontier_id: int | None,
    future_reference_path_xy: tuple[tuple[float, float], ...],
    oracle_crossing_world_xy: tuple[float, float] | None,
    oracle_reason: str,
    frame_index: int,
) -> Image.Image:
    panel_size = 600
    first_person = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).convert("RGB")
    first_person.thumbnail((panel_size, panel_size - 90), _resampling())
    rgb_panel = Image.new("RGB", (panel_size, panel_size), (20, 24, 28))
    rgb_panel.paste(
        first_person,
        ((panel_size - first_person.width) // 2, 70 + (panel_size - 70 - first_person.height) // 2),
    )
    rgb_draw = ImageDraw.Draw(rgb_panel)
    _draw_text(rgb_draw, (16, 12), "STUDENT: current RGB", fill=(245, 245, 245), size=24)
    _draw_text(rgb_draw, (16, 42), f"frame={frame_index}", fill=(180, 190, 200), size=18)
    scale_x = first_person.width / max(1.0, float(rgb.shape[1]))
    scale_y = first_person.height / max(1.0, float(rgb.shape[0]))
    offset_x = (panel_size - first_person.width) // 2
    offset_y = 70 + (panel_size - 70 - first_person.height) // 2
    for landmark in landmarks:
        if landmark.last_seen_frame_index != frame_index:
            continue
        x0, y0, x1, y1 = landmark.bbox_xyxy
        box = (
            offset_x + int(round(x0 * scale_x)),
            offset_y + int(round(y0 * scale_y)),
            offset_x + int(round(x1 * scale_x)),
            offset_y + int(round(y1 * scale_y)),
        )
        rgb_draw.rectangle(box, outline=(65, 230, 110), width=3)
        _draw_text(
            rgb_draw,
            (box[0] + 3, max(offset_y, box[1] - 22)),
            f"{landmark.label} {landmark.detector_confidence:.2f}",
            fill=(65, 230, 110),
            size=16,
        )

    causal_map = _render_map(
        snapshot=snapshot,
        robot_pose_xyyaw=robot_pose_xyyaw,
        trajectory_world_xy=trajectory_world_xy,
        landmarks=landmarks,
        frontiers=frontiers,
        selected_frontier_id=None,
        future_reference_path_xy=(),
        oracle_crossing_world_xy=None,
        title="STUDENT: causal partial map",
        panel_size=panel_size,
    )
    oracle_map = _render_map(
        snapshot=snapshot,
        robot_pose_xyyaw=robot_pose_xyyaw,
        trajectory_world_xy=trajectory_world_xy,
        landmarks=landmarks,
        frontiers=frontiers,
        selected_frontier_id=selected_frontier_id,
        future_reference_path_xy=future_reference_path_xy,
        oracle_crossing_world_xy=oracle_crossing_world_xy,
        title="ORACLE AUDIT: future route",
        panel_size=panel_size,
    )
    canvas = Image.new("RGB", (panel_size * 3, panel_size + 130), (12, 15, 18))
    canvas.paste(rgb_panel, (0, 130))
    canvas.paste(causal_map, (panel_size, 130))
    canvas.paste(oracle_map, (panel_size * 2, 130))
    draw = ImageDraw.Draw(canvas)
    _draw_text(draw, (18, 12), instruction[:180], fill=(245, 245, 245), size=22)
    active_text = "alignment=unavailable"
    if active_alignment is not None:
        active_text = (
            f"active={active_alignment.event_type} span={active_alignment.text!r} "
            f"confidence={active_alignment.confidence:.2f}"
        )
    _draw_text(draw, (18, 48), active_text[:190], fill=(255, 210, 80), size=19)
    _draw_text(
        draw,
        (18, 82),
        f"oracle_frontier={selected_frontier_id} reason={oracle_reason} "
        f"landmarks={len(landmarks)} candidates={len(frontiers)}",
        fill=(180, 205, 225),
        size=18,
    )
    return canvas


def write_timeline_html(
    *,
    output_path: Path,
    episode_title: str,
    frame_paths: list[str],
    metadata: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_json = json.dumps(frame_paths, ensure_ascii=False)
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(episode_title)}</title>
<style>
body {{ background:#0d1117; color:#e6edf3; font-family:ui-monospace,monospace; margin:24px; }}
.shell {{ max-width:1600px; margin:auto; }}
img {{ width:100%; border:1px solid #30363d; background:#161b22; }}
input {{ width:100%; }}
pre {{ white-space:pre-wrap; color:#9da7b3; }}
</style>
</head>
<body><div class="shell">
<h2>{html.escape(episode_title)}</h2>
<input id="slider" type="range" min="0" max="{max(0, len(frame_paths) - 1)}" value="0">
<p id="counter"></p><img id="frame" alt="timeline frame"><pre id="meta"></pre>
</div>
<script>
const frames={frames_json}; const metadata={metadata_json};
const slider=document.getElementById('slider');
function render() {{ const i=Number(slider.value); document.getElementById('frame').src=frames[i];
document.getElementById('counter').textContent=`frame ${{i+1}} / ${{frames.length}}`;
document.getElementById('meta').textContent=JSON.stringify(metadata[i], null, 2); }}
slider.addEventListener('input', render); render();
</script></body></html>
"""
    output_path.write_text(document, encoding="utf-8")


def _render_map(
    *,
    snapshot: OccupancyGridSnapshot,
    robot_pose_xyyaw: tuple[float, float, float],
    trajectory_world_xy: tuple[tuple[float, float], ...],
    landmarks: tuple[ObservedLandmark, ...],
    frontiers: tuple[FrontierToken, ...],
    selected_frontier_id: int | None,
    future_reference_path_xy: tuple[tuple[float, float], ...],
    oracle_crossing_world_xy: tuple[float, float] | None,
    title: str,
    panel_size: int,
) -> Image.Image:
    image = Image.new("RGB", (panel_size, panel_size), (78, 82, 88))
    draw = ImageDraw.Draw(image)
    points: list[tuple[float, float]] = list(trajectory_world_xy)
    points.extend(frontier.boundary_world_xy for frontier in frontiers)
    points.extend(frontier.navigation_goal_world_xy for frontier in frontiers)
    points.extend(
        landmark.estimated_world_xy
        for landmark in landmarks
        if landmark.estimated_world_xy is not None
    )
    points.extend(future_reference_path_xy)
    if oracle_crossing_world_xy is not None:
        points.append(oracle_crossing_world_xy)
    points.append((float(robot_pose_xyyaw[0]), float(robot_pose_xyyaw[1])))
    bounds = _map_bounds(snapshot=snapshot, points=points, minimum_size_m=12.0, margin_m=1.5)
    x0, y0, x1, y1 = bounds
    map_top = 54
    map_height = panel_size - map_top

    occupied_rows, occupied_cols = np.nonzero(snapshot.occupied_mask)
    free_rows, free_cols = np.nonzero(snapshot.free_mask)
    _draw_cells(
        draw=draw,
        rows=free_rows,
        cols=free_cols,
        snapshot=snapshot,
        bounds=bounds,
        viewport=(0, map_top, panel_size, map_height),
        fill=(235, 237, 239),
        radius=1,
    )
    _draw_cells(
        draw=draw,
        rows=occupied_rows,
        cols=occupied_cols,
        snapshot=snapshot,
        bounds=bounds,
        viewport=(0, map_top, panel_size, map_height),
        fill=(20, 24, 28),
        radius=2,
    )
    if len(trajectory_world_xy) >= 2:
        draw.line(
            [_world_to_pixel(point, bounds=bounds, viewport=(0, map_top, panel_size, map_height)) for point in trajectory_world_xy],
            fill=(40, 145, 255),
            width=3,
        )
    if len(future_reference_path_xy) >= 2:
        draw.line(
            [_world_to_pixel(point, bounds=bounds, viewport=(0, map_top, panel_size, map_height)) for point in future_reference_path_xy],
            fill=(235, 70, 210),
            width=3,
        )
    for frontier in frontiers:
        boundary_pixel = _world_to_pixel(
            frontier.boundary_world_xy,
            bounds=bounds,
            viewport=(0, map_top, panel_size, map_height),
        )
        goal_pixel = _world_to_pixel(
            frontier.navigation_goal_world_xy,
            bounds=bounds,
            viewport=(0, map_top, panel_size, map_height),
        )
        selected = frontier.frontier_id == selected_frontier_id
        color = (255, 195, 45) if selected else (0, 215, 225)
        draw.line((boundary_pixel, goal_pixel), fill=(0, 150, 165), width=2)
        draw.ellipse(
            (
                boundary_pixel[0] - 4,
                boundary_pixel[1] - 4,
                boundary_pixel[0] + 4,
                boundary_pixel[1] + 4,
            ),
            outline=(0, 215, 225),
            width=2,
        )
        radius = 7 if selected else 4
        draw.ellipse(
            (
                goal_pixel[0] - radius,
                goal_pixel[1] - radius,
                goal_pixel[0] + radius,
                goal_pixel[1] + radius,
            ),
            fill=color,
        )
    if oracle_crossing_world_xy is not None:
        crossing_pixel = _world_to_pixel(
            oracle_crossing_world_xy,
            bounds=bounds,
            viewport=(0, map_top, panel_size, map_height),
        )
        draw.line(
            (
                crossing_pixel[0] - 6,
                crossing_pixel[1] - 6,
                crossing_pixel[0] + 6,
                crossing_pixel[1] + 6,
            ),
            fill=(235, 70, 210),
            width=3,
        )
        draw.line(
            (
                crossing_pixel[0] - 6,
                crossing_pixel[1] + 6,
                crossing_pixel[0] + 6,
                crossing_pixel[1] - 6,
            ),
            fill=(235, 70, 210),
            width=3,
        )
    for landmark in landmarks:
        if landmark.estimated_world_xy is None:
            continue
        pixel = _world_to_pixel(landmark.estimated_world_xy, bounds=bounds, viewport=(0, map_top, panel_size, map_height))
        draw.rectangle((pixel[0] - 6, pixel[1] - 6, pixel[0] + 6, pixel[1] + 6), fill=(65, 230, 110))
        _draw_text(draw, (pixel[0] + 8, pixel[1] - 9), landmark.label, fill=(20, 70, 30), size=14)
    robot_pixel = _world_to_pixel(robot_pose_xyyaw[:2], bounds=bounds, viewport=(0, map_top, panel_size, map_height))
    draw.ellipse((robot_pixel[0] - 7, robot_pixel[1] - 7, robot_pixel[0] + 7, robot_pixel[1] + 7), fill=(235, 60, 55))
    yaw = float(robot_pose_xyyaw[2])
    heading = (robot_pixel[0] + 22 * np.cos(yaw), robot_pixel[1] - 22 * np.sin(yaw))
    draw.line((robot_pixel, heading), fill=(235, 60, 55), width=4)
    _draw_text(draw, (14, 12), title, fill=(245, 245, 245), size=22)
    _draw_text(
        draw,
        (14, 36),
        "ring=boundary dot=safe goal yellow=reliable oracle",
        fill=(185, 205, 215),
        size=13,
    )
    return image


def _map_bounds(
    *,
    snapshot: OccupancyGridSnapshot,
    points: list[tuple[float, float]],
    minimum_size_m: float,
    margin_m: float,
) -> tuple[float, float, float, float]:
    if points:
        array = np.asarray(points, dtype=np.float64)
        center = 0.5 * (np.min(array, axis=0) + np.max(array, axis=0))
        extent = np.max(np.max(array, axis=0) - np.min(array, axis=0)) + 2.0 * margin_m
    else:
        center = np.asarray([0.0, 0.0], dtype=np.float64)
        extent = minimum_size_m
    size = max(float(minimum_size_m), float(extent))
    return center[0] - 0.5 * size, center[1] - 0.5 * size, center[0] + 0.5 * size, center[1] + 0.5 * size


def _draw_cells(
    *,
    draw: ImageDraw.ImageDraw,
    rows: np.ndarray,
    cols: np.ndarray,
    snapshot: OccupancyGridSnapshot,
    bounds: tuple[float, float, float, float],
    viewport: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    radius: int,
) -> None:
    if rows.size == 0:
        return
    world_x = float(snapshot.origin_x) + (cols.astype(np.float64) + 0.5) * float(snapshot.resolution_m)
    world_y = float(snapshot.origin_y) + (rows.astype(np.float64) + 0.5) * float(snapshot.resolution_m)
    inside = (world_x >= bounds[0]) & (world_x <= bounds[2]) & (world_y >= bounds[1]) & (world_y <= bounds[3])
    for point in zip(world_x[inside].tolist(), world_y[inside].tolist()):
        pixel = _world_to_pixel(point, bounds=bounds, viewport=viewport)
        draw.rectangle((pixel[0] - radius, pixel[1] - radius, pixel[0] + radius, pixel[1] + radius), fill=fill)


def _world_to_pixel(
    point_xy,
    *,
    bounds: tuple[float, float, float, float],
    viewport: tuple[int, int, int, int],
) -> tuple[int, int]:
    x0, y0, x1, y1 = bounds
    left, top, width, height = viewport
    x = left + (float(point_xy[0]) - x0) / max(1.0e-8, x1 - x0) * width
    y = top + (y1 - float(point_xy[1])) / max(1.0e-8, y1 - y0) * height
    return int(round(x)), int(round(y))


def _draw_text(draw: ImageDraw.ImageDraw, xy, text: str, *, fill, size: int) -> None:
    draw.text(xy, text, fill=fill, font=_font(size))


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def _resampling():
    return Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
