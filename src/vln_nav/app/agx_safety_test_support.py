"""Sensor and display helpers for the independent AGX trajectory acceptance."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from vln_nav.control.trajectory_safety import TrajectorySafetyConfig, TrajectorySafetySupervisor
from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.occupancy_grid import GlobalOccupancyMap, OccupancyGridConfig
from vln_nav.navigation.rgbd_geometry_map import (
    RGBDGeometryMapConfig, RGBDVirtualRayProjector, RGBDVirtualRayFrame,
    backproject_depth_pixels_to_world,
)


def load_test_settings(project_root):
    with (project_root / "configs/agx_trajectory_test.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if set(data) != {"camera", "near_camera", "geometry", "safety", "debug"}:
        raise ValueError("Invalid AGX acceptance configuration sections")
    TrajectorySafetyConfig.from_mapping(data["safety"])
    for section in ("camera", "geometry"):
        for key, value in data[section].items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"Invalid {section}.{key}")
    if data["geometry"]["min_depth_m"] >= data["geometry"]["max_depth_m"]:
        raise ValueError("Invalid test camera depth range")
    if data["geometry"]["obstacle_min_height_m"] >= data["geometry"]["obstacle_max_height_m"]:
        raise ValueError("Invalid obstacle height range")
    if float(data["debug"]["period_sec"]) <= 0:
        raise ValueError("debug.period_sec must be positive")
    near = data["near_camera"]
    if set(near) != {"prim_path", "position_m", "pitch_down_deg", "horizontal_fov_deg"}:
        raise ValueError("Invalid near_camera fields")
    if not str(near["prim_path"]).strip() or len(near["position_m"]) != 3 or not np.isfinite(near["position_m"]).all():
        raise ValueError("Invalid near_camera path or position")
    if not 10 <= float(near["horizontal_fov_deg"]) <= 150 or not 0 <= float(near["pitch_down_deg"]) <= 90:
        raise ValueError("Invalid near_camera field of view or pitch")
    return data


def mask_robot_depth(depth, segmentation, robot_root):
    """Use rendered prim IDs only to remove the robot itself, never obstacles."""
    if not isinstance(segmentation, dict) or "data" not in segmentation:
        raise ValueError("missing_self_mask")
    ids = np.asarray(segmentation["data"])
    if ids.ndim == 3 and ids.shape[-1] == 1:
        ids = ids[..., 0]
    labels = segmentation.get("info", {}).get("idToLabels")
    if ids.shape != depth.shape or not isinstance(labels, dict):
        raise ValueError("invalid_self_mask")
    own_ids = [int(key) for key, path in labels.items() if isinstance(path, str)
               and (path == robot_root or path.startswith(robot_root.rstrip("/") + "/"))]
    result = depth.copy()
    result[np.isin(ids, own_ids)] = np.nan
    return result


def project_near_depth(depth, intrinsics, view, geometry):
    """Downward camera: bin the whole image by world bearing, not image column.

    A column of a downward image spans both front and rear space. Reducing it
    to one forward ray would discard half of the near-field coverage.
    """
    stride = int(geometry["pixel_stride"])
    vv, uu = np.mgrid[stride // 2:depth.shape[0]:stride, stride // 2:depth.shape[1]:stride]
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z >= geometry["min_depth_m"]) & (z <= geometry["max_depth_m"])
    transform = np.linalg.inv(view)
    origin = transform[:2, 3]
    points = backproject_depth_pixels_to_world(pixel_u=uu[valid], pixel_v=vv[valid], depth_m=z[valid],
                                              camera_intrinsics=intrinsics, camera_to_world=transform)
    delta = points[:, :2] - origin
    distance = np.linalg.norm(delta, axis=1)
    hit = (points[:, 2] >= geometry["obstacle_min_height_m"]) & (points[:, 2] <= geometry["obstacle_max_height_m"])
    floor = (points[:, 2] >= -.2) & (points[:, 2] < geometry["obstacle_min_height_m"])
    valid_range = (distance >= geometry["min_depth_m"]) & (distance <= geometry["max_depth_m"])
    keep = (hit | floor) & valid_range
    points, delta, distance, hit = points[keep], delta[keep], distance[keep], hit[keep]
    if not len(points):
        return RGBDVirtualRayFrame(origin.copy(), np.empty((0, 2)), np.empty(0, dtype=bool), 0)
    bins = np.clip(((np.arctan2(delta[:, 1], delta[:, 0]) + math.pi) / (2 * math.pi) * 720).astype(int), 0, 719)
    # Hit before floor; nearest hit, otherwise farthest visible floor in each bin.
    order = np.lexsort((np.where(hit, distance, -distance), ~hit, bins))
    _, first = np.unique(bins[order], return_index=True)
    selected = order[first]
    return RGBDVirtualRayFrame(origin.copy(), points[selected, :2].copy(), hit[selected].copy(), int(np.count_nonzero(valid)))


class PosedDepthHistory:
    """Match exposure simulation time, never attach a cached depth to today's pose."""
    def __init__(self, tolerance_sec):
        self.tolerance = tolerance_sec
        self.samples = deque(maxlen=256)

    def add(self, timestamp, pose, view_matrix):
        self.samples.append((float(timestamp), tuple(pose), np.asarray(view_matrix, dtype=float).copy()))

    def match(self, timestamp):
        if not self.samples or not math.isfinite(timestamp):
            return None
        sample = min(self.samples, key=lambda x: abs(x[0] - timestamp))
        return sample if abs(sample[0] - timestamp) <= self.tolerance else None


class SafetyTestSensors:
    def __init__(self, settings, logger, dt, *, required_sources=("front",), robot_root=None):
        self.settings, self.logger = settings, logger
        self.config = TrajectorySafetyConfig.from_mapping(settings["safety"])
        self.supervisor = TrajectorySafetySupervisor(self.config)
        self.required_sources = tuple(required_sources)
        self.robot_root = robot_root
        self.histories = {name: PosedDepthHistory(dt * .6 + 1e-5) for name in self.required_sources}
        g = settings["geometry"]
        self.projector = RGBDVirtualRayProjector(config=RGBDGeometryMapConfig(
            min_depth_m=g["min_depth_m"], max_depth_m=g["max_depth_m"], pixel_stride=int(g["pixel_stride"]),
            vertical_min_ratio=0.0, vertical_max_ratio=1.0,
            obstacle_min_height_m=g["obstacle_min_height_m"], obstacle_max_height_m=g["obstacle_max_height_m"],
        ), floor_z_m=0.0)
        self.frame_times = {name: -float("inf") for name in self.required_sources}
        self.min_frame_time = 0.0
        self.depth = None
        self.last_reason = "waiting_rgbd"
        self.frozen = False
        self.reset((0, 0, 0), 0.0)

    def reset(self, pose, timestamp):
        g = self.settings["geometry"]
        self.mapper = GlobalOccupancyMap(config=OccupancyGridConfig(
            size_m=float(g["size_m"]), resolution_m=float(g["resolution_m"]),
            update_period_sec=.02, min_range_m=g["min_depth_m"], max_range_m=g["max_depth_m"],
            occupied_threshold=.5, free_threshold=-.3, occupied_min_neighbors=1,
            inflation_radius_m=0.0, unknown_is_traversable=False,
        ), logger=self.logger)
        self.supervisor.reset(pose)
        self.min_frame_time = float(timestamp)
        self.depth = None
        self.depths = {}
        self.valid_sources = {}
        self.published_times = {name: -float("inf") for name in self.required_sources}
        self.last_reason = "waiting_new_scene_frame"

    def consume(self, camera, *, now, pose, source="front"):
        history = self.histories[source]
        history.add(now, pose, camera.get_view_matrix_ros())
        if self.frozen:
            self.last_reason = "injected_rgbd_freeze"
            return
        frame = camera.get_current_frame()
        stamp = float(frame.get("rendering_time", float("nan")))
        if not math.isfinite(stamp) or stamp <= max(self.frame_times[source], self.min_frame_time):
            return
        self.frame_times[source] = stamp
        matched = history.match(stamp)
        depth = frame.get("distance_to_image_plane")
        if matched is None or depth is None:
            self.last_reason = "rgbd_pose_mismatch" if matched is None else "missing_depth"
            return
        if now - stamp > self.config.sensor_timeout_sec:
            self.last_reason = "stale_capture"
            return
        depth = np.asarray(depth, dtype=np.float32).copy()
        if depth.ndim != 2 or depth.size == 0:
            self.last_reason = "invalid_depth_shape"
            return
        if self.robot_root is not None:
            try:
                depth = mask_robot_depth(depth, frame.get("instance_id_segmentation"), self.robot_root)
            except ValueError as exc:
                self.last_reason = f"{source}:{exc}"
                return
        self.depths[source] = depth
        if source == "front":
            self.depth = depth
        g = self.settings["geometry"]
        valid = np.isfinite(depth) & (depth >= g["min_depth_m"]) & (depth <= g["max_depth_m"])
        if np.mean(valid) < self.config.min_valid_depth_ratio:
            self.last_reason = "insufficient_depth"
            return
        _, capture_pose, view = matched
        if source == "near":
            rays = project_near_depth(depth, camera.get_intrinsics_matrix(), view, g)
        else:
            rays = self.projector.project(depth_image=depth, camera_intrinsics=camera.get_intrinsics_matrix(),
                                          camera_view_matrix_ros=view)
        if rays.ray_count < 8:
            self.last_reason = "insufficient_geometry_rays"
            return
        # Do not discard close returns merely because they overlap the robot's
        # footprint: they could be real intruding obstacles, not self geometry.
        self.valid_sources[source] = (stamp, time.monotonic(), rays, capture_pose)
        if any(name not in self.valid_sources for name in self.required_sources):
            self.last_reason = "waiting_all_cameras"
            return
        # Integrate both cameras in one update. The mapper's time throttle would
        # otherwise silently discard the second camera at the same timestamp.
        if all(self.valid_sources[name][0] > self.published_times[name] for name in self.required_sources):
            for name in self.required_sources:
                source_stamp, _, frame_rays, frame_pose = self.valid_sources[name]
                self.mapper.integrate_world_rays(robot_pose=Pose2D(*frame_pose), sensor_world_xy=frame_rays.sensor_world_xy,
                    beam_endpoints_world_xy=frame_rays.endpoints_world_xy, beam_hit_mask=frame_rays.hit_mask)
                self.published_times[name] = source_stamp
        snapshot = self.mapper.snapshot()
        # A newly visible depth hit must stop motion immediately, even if past
        # free-space evidence has not yet been overcome in the log-odds map.
        hits = np.concatenate([entry[2].endpoints_world_xy[entry[2].hit_mask] for entry in self.valid_sources.values()])
        if len(hits):
            cols = np.floor((hits[:, 0] - snapshot.origin_x) / snapshot.resolution_m).astype(int)
            rows = np.floor((hits[:, 1] - snapshot.origin_y) / snapshot.resolution_m).astype(int)
            inside = ((rows >= 0) & (rows < snapshot.free_mask.shape[0])
                      & (cols >= 0) & (cols < snapshot.free_mask.shape[1]))
            snapshot.occupied_mask[rows[inside], cols[inside]] = True
            snapshot.free_mask[rows[inside], cols[inside]] = False
        self.supervisor.update(snapshot, timestamp=min(entry[0] for entry in self.valid_sources.values()),
                               wall_timestamp=min(entry[1] for entry in self.valid_sources.values()))
        self.last_reason = "ok"


def configure_obstacles(stage, kind, pose):
    """Scene authoring only. Nothing here is passed to perception or safety."""
    from pxr import Gf, UsdGeom, UsdPhysics
    stage.RemovePrim("/World/TestObstacles")
    UsdGeom.Xform.Define(stage, "/World/TestObstacles")
    boxes = []
    if kind == "BLOCKED STRAIGHT":
        boxes = [(2.0, 0.0, .4, 1.4)]
    elif kind == "INNER CURVE":
        # The centerline misses this block; the inner side of the body does not.
        boxes = [(.88, .77, .32, .32)]
    elif kind == "NARROW DOOR":
        gap = .95
        boxes = [(2.0, side * (gap / 2 + .65), .35, 1.3) for side in (-1, 1)]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    for index, (x, y, length, width) in enumerate(boxes):
        cube = UsdGeom.Cube.Define(stage, f"/World/TestObstacles/Block{index}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(pose[0] + c * x - s * y, pose[1] + s * x + c * y, .65))
        xf.AddRotateZOp().Set(math.degrees(pose[2]))
        xf.AddScaleOp().Set(Gf.Vec3f(length, width, 1.3))
        cube.CreateDisplayColorAttr([(0.65, .30, .16)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def save_safety_debug(path, sensors, *, pose, reference_path, decision, status):
    """One overwrite-only PNG: depth + world-aligned raw occupancy and sweep."""
    size, header, footer = 512, 70, 50
    canvas = Image.new("RGB", (2 * size, header + size + footer), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 8), "RGB-D depth: near=black / far=white / invalid=gray", font=font, fill="black")
    draw.text((size + 12, 8), "Raw occupancy: obstacle=black / free=white / unknown=gray", font=font, fill="black")
    draw.text((12, 35), str(status).split("\n")[0][:105], font=font, fill="black")
    for index, source in enumerate(("front", "near")):
        top = header + index * (size // 2)
        entry = sensors.valid_sources.get(source)
        stamp_label = f"frame t={entry[0]:.2f}s" if entry else "waiting for valid depth"
        draw.text((12, top + 2), f"{source.upper()} | {stamp_label}", font=font, fill="black")
        depth = sensors.depths.get(source)
        if depth is None:
            continue
        g = sensors.settings["geometry"]
        valid = np.isfinite(depth) & (depth >= g["min_depth_m"]) & (depth <= g["max_depth_m"])
        gray = np.full(depth.shape, 128, dtype=np.uint8)
        gray[valid] = (255 * np.clip(depth[valid] / g["max_depth_m"], 0, 1)).astype(np.uint8)
        image = Image.fromarray(gray).convert("RGB")
        image.thumbnail((size, size // 2 - 24))
        canvas.paste(image, ((size - image.width) // 2, top + 24))
    snapshot = sensors.supervisor.snapshot
    view_size = 8.0
    map_image = Image.new("RGB", (size, size), (145, 145, 145))
    def pixel(x, y):
        return (int((x - pose[0]) / view_size * size + size / 2),
                int(size / 2 - (y - pose[1]) / view_size * size))
    if snapshot is not None:
        xx = pose[0] + (np.arange(size) + .5 - size / 2) * view_size / size
        yy = pose[1] - (np.arange(size) + .5 - size / 2) * view_size / size
        cols = np.floor((xx - snapshot.origin_x) / snapshot.resolution_m).astype(int)
        rows = np.floor((yy - snapshot.origin_y) / snapshot.resolution_m).astype(int)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        valid = (rr >= 0) & (rr < snapshot.free_mask.shape[0]) & (cc >= 0) & (cc < snapshot.free_mask.shape[1])
        rgb = np.full((size, size, 3), 145, dtype=np.uint8)
        vr, vc = rr[valid], cc[valid]
        colors = np.full((len(vr), 3), 145, dtype=np.uint8)
        colors[snapshot.free_mask[vr, vc]] = 245
        colors[snapshot.occupied_mask[vr, vc]] = 15
        rgb[valid] = colors
        map_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(map_image)
    if reference_path is not None:
        draw.line([pixel(x, y) for x, y in reference_path], fill="green", width=2)
    if decision is not None:
        cfg = sensors.config
        pad = cfg.margin_m + (snapshot.resolution_m / math.sqrt(2) if snapshot is not None else 0.0)
        for sample in decision.sweep[::max(1, len(decision.sweep) // 12)]:
            x, y, yaw = sample
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [(cfg.front_m + pad, cfg.half_width_m + pad),
                       (cfg.front_m + pad, -cfg.half_width_m - pad),
                       (-cfg.rear_m - pad, -cfg.half_width_m - pad),
                       (-cfg.rear_m - pad, cfg.half_width_m + pad)]
            poly = [pixel(x + c * a - s * b, y + s * a + c * b) for a, b in corners]
            draw.line(poly + [poly[0]], fill="orange" if decision.status != "CLEAR" else "cyan", width=1)
        if decision.blocker_world_xy is not None:
            bx, by = pixel(*decision.blocker_world_xy)
            draw.line([(bx - 6, by - 6), (bx + 6, by + 6)], fill="red", width=3)
            draw.line([(bx - 6, by + 6), (bx + 6, by - 6)], fill="red", width=3)
    center = pixel(pose[0], pose[1])
    draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), fill="blue")
    draw.line([center, pixel(pose[0] + .4 * math.cos(pose[2]), pose[1] + .4 * math.sin(pose[2]))], fill="blue", width=3)
    draw.text((10, 10), "+Y up | +X right | 8m view", font=font, fill=(30, 60, 100))
    canvas.paste(map_image, (size, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, header + size + 8), "Green: path | Blue: robot heading | Cyan: clear sweep | Orange: limited/blocked sweep", font=font, fill="black")
    draw.text((12, header + size + 28), "Red cross: first blocking cell (obstacle or unknown) | Camera timestamps: simulation time", font=font, fill="black")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    try:
        canvas.save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
