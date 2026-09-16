"""On-demand P1 measurements; no ground truth or navigation decisions."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .camera_geometry import CameraFrameGeometry
from .local_geometry import FREE, OCCUPIED, UNKNOWN, LocalVoxelSnapshot
from vln_nav.navigation.rgbd_geometry_map import backproject_depth_pixels_to_world


@dataclass(frozen=True)
class MapCheckConfig:
    stable_frames: int = 5
    max_frame_age_sec: float = 0.10
    max_stationary_translation_m: float = 0.01
    max_stationary_rotation_deg: float = 0.5
    min_surface_points: int = 60
    max_reference_sigma_m: float = 0.02
    reference_association_distance_m: float = 0.18
    max_surface_error_m: float = 0.05
    max_map_error_m: float = 0.10
    floor_min_points: int = 80
    floor_min_radius_m: float = 0.30
    floor_max_radius_m: float = 2.50
    floor_search_min_relative_z_m: float = -0.80
    floor_search_max_relative_z_m: float = 0.25
    floor_cluster_half_width_m: float = 0.06
    max_floor_height_span_m: float = 0.05
    max_floor_thickness_m: float = 0.06

    def validate(self) -> None:
        if self.stable_frames < 3 or self.min_surface_points < 10 or self.floor_min_points < 10:
            raise ValueError("Map checks require >=3 frames and >=10 surface points")
        values = (self.max_frame_age_sec, self.max_stationary_translation_m,
                  self.max_stationary_rotation_deg, self.max_reference_sigma_m,
                  self.reference_association_distance_m, self.max_surface_error_m,
                  self.max_map_error_m, self.floor_min_radius_m,
                  self.floor_max_radius_m, self.floor_cluster_half_width_m,
                  self.max_floor_height_span_m, self.max_floor_thickness_m)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Map check thresholds must be finite and positive")
        if self.floor_min_radius_m >= self.floor_max_radius_m:
            raise ValueError("floor_min_radius_m must be smaller than floor_max_radius_m")
        if (not math.isfinite(self.floor_search_min_relative_z_m)
                or not math.isfinite(self.floor_search_max_relative_z_m)
                or self.floor_search_min_relative_z_m >= self.floor_search_max_relative_z_m):
            raise ValueError("Invalid floor relative-height search range")


@dataclass(frozen=True)
class ReferencePlane:
    center: np.ndarray
    normal: np.ndarray
    axes: np.ndarray
    half_size: np.ndarray
    sigma_m: float
    timestamp_sec: float

    def tangential_mask(self, points: np.ndarray) -> np.ndarray:
        delta = points - self.center
        return np.all(np.abs(delta @ self.axes.T) <= self.half_size, axis=1)

    def select(self, points: np.ndarray, max_normal_distance_m: float) -> np.ndarray:
        delta = points - self.center
        return (self.tangential_mask(points)
                & (np.abs(delta @ self.normal) <= max_normal_distance_m))

    def corners(self) -> np.ndarray:
        return np.asarray([self.center + a * self.half_size[0] * self.axes[0]
                           + b * self.half_size[1] * self.axes[1]
                           for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def frame_points(frame: CameraFrameGeometry, min_depth: float, max_depth: float,
                 max_ray: float, stride: int = 4, roi=None) -> np.ndarray:
    depth = frame.depth_m
    x0, y0, x1, y1 = roi or (0, 0, depth.shape[1], depth.shape[0])
    u, v = np.meshgrid(np.arange(x0, x1, stride), np.arange(y0, y1, stride))
    z = depth[v, u]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    points = backproject_depth_pixels_to_world(
        pixel_u=u[valid], pixel_v=v[valid], depth_m=z[valid],
        camera_intrinsics=frame.intrinsics, camera_to_world=frame.world_from_camera,
    ).reshape(-1, 3)
    return points[np.linalg.norm(points - frame.world_from_camera[:3, 3], axis=1) <= max_ray]


def stable_frames_reason(batches: list[dict[str, CameraFrameGeometry]], now: float,
                         config: MapCheckConfig) -> str | None:
    if len(batches) < config.stable_frames:
        return "等待停车后的新相机帧（约 1 秒），然后再次点击。"
    first = batches[0]["front"].world_from_base
    for batch in batches:
        for frame in batch.values():
            pose = frame.world_from_base
            distance = np.linalg.norm(pose[:3, 3] - first[:3, 3])
            cosine = (np.trace(first[:3, :3].T @ pose[:3, :3]) - 1) / 2
            angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            if distance > config.max_stationary_translation_m or angle > config.max_stationary_rotation_deg:
                return "机器人仍在移动；请松开方向键，等稳定后再次点击。"
    if any(not 0 <= now - f.timestamp_sec <= config.max_frame_age_sec
           for f in batches[-1].values()):
        return "最新相机帧已过期；本次不使用旧帧验收。"
    for previous, current in zip(batches, batches[1:]):
        if any(current[s].timestamp_sec <= previous[s].timestamp_sec for s in current):
            return "相机时间戳重复；需要不同时间的新帧。"
    return None


def choose_reference(frame: CameraFrameGeometry, config: MapCheckConfig,
                     min_depth: float, max_depth: float, max_ray: float) -> ReferencePlane:
    h, w = frame.depth_m.shape
    candidates = []
    # Small image patches avoid fitting an entire room as one plane.
    for cy in (0.5, 0.3, 0.7):
        for cx in (0.5, 0.3, 0.7):
            roi = (int((cx - .10) * w), int((cy - .10) * h),
                   int((cx + .10) * w), int((cy + .10) * h))
            points = frame_points(frame, min_depth, max_depth, max_ray, stride=2, roi=roi)
            if len(points) < config.min_surface_points:
                continue
            center = points.mean(axis=0)
            _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
            sigma = float(singular[-1] / math.sqrt(len(points)))
            normal = axes[2]
            if sigma > config.max_reference_sigma_m or abs(normal[2]) > .35:
                continue
            projected = (points - center) @ axes[:2].T
            half_size = np.minimum(np.quantile(np.abs(projected), .90, axis=0), .6)
            if np.min(half_size) < .10:
                continue
            if normal @ (frame.world_from_camera[:3, 3] - center) < 0:
                normal = -normal
            reference = ReferencePlane(center, normal, axes[:2], half_size, sigma, frame.timestamp_sec)
            candidates.append((abs(cx - .5) + abs(cy - .5) + sigma, reference))
    if not candidates:
        raise ValueError("前视相机没有找到足够平整的竖直表面；请靠近一面平墙后重试。")
    return min(candidates, key=lambda item: item[0])[1]


def floor_height_samples(frame: CameraFrameGeometry, config: MapCheckConfig,
                         min_depth: float, max_depth: float, max_ray: float) -> np.ndarray:
    """Extract the dominant nearby horizontal support surface in world Z."""
    points = frame_points(frame, min_depth, max_depth, max_ray)
    if len(points) < config.floor_min_points:
        return np.empty(0, dtype=np.float64)
    base = frame.world_from_base[:3, 3]
    delta = points - base
    radius = np.linalg.norm(delta[:, :2], axis=1)
    relative_z = delta[:, 2]
    keep = ((radius >= config.floor_min_radius_m)
            & (radius <= config.floor_max_radius_m)
            & (relative_z >= config.floor_search_min_relative_z_m)
            & (relative_z <= config.floor_search_max_relative_z_m))
    heights = points[keep, 2]
    if len(heights) < config.floor_min_points:
        return np.empty(0, dtype=np.float64)

    bin_width = max(0.02, config.floor_cluster_half_width_m * 0.5)
    z_min = float(np.min(heights))
    z_max = float(np.max(heights))
    if z_max - z_min < bin_width:
        selected = heights
    else:
        edges = np.arange(z_min, z_max + 2.0 * bin_width, bin_width)
        counts, edges = np.histogram(heights, bins=edges)
        peak = int(np.argmax(counts))
        center = 0.5 * (edges[peak] + edges[peak + 1])
        selected = heights[np.abs(heights - center) <= config.floor_cluster_half_width_m]
    if len(selected) < config.floor_min_points:
        return np.empty(0, dtype=np.float64)
    return selected


@dataclass
class HistoryProbe:
    points: np.ndarray
    state: np.ndarray
    log_odds: np.ndarray
    evidence: np.ndarray
    last_seen: np.ndarray

    @classmethod
    def capture(cls, snapshot: LocalVoxelSnapshot):
        indices = np.argwhere(snapshot.state != UNKNOWN)
        if len(indices) > 4096:
            indices = indices[np.linspace(0, len(indices) - 1, 4096, dtype=int)]
        points = snapshot.origin_world_xyz + (indices[:, [2, 1, 0]] + .5) * snapshot.resolution_m
        key = tuple(indices.T)
        return cls(points, snapshot.state[key].copy(), snapshot.log_odds[key].copy(),
                   snapshot.evidence_count[key].copy(), snapshot.last_observed_time_sec[key].copy())

    def check(self, snapshot: LocalVoxelSnapshot) -> tuple[str, str]:
        indices, inside = snapshot.world_to_indices(self.points)
        original = np.flatnonzero(inside)
        key = tuple(indices[inside].T)
        unchanged_time = snapshot.last_observed_time_sec[key] <= self.last_seen[original]
        original = original[unchanged_time]
        key = tuple(indices[original].T)
        count = len(original)
        if count < 32:
            return "INCONCLUSIVE", "窗口内未重新观测的样本不足 32 个，请移动并让一部分旧表面离开视野后再检查。"
        same = ((snapshot.state[key] == self.state[original])
                & np.isclose(snapshot.log_odds[key], self.log_odds[original], atol=1e-6)
                & (snapshot.evidence_count[key] == self.evidence[original]))
        lost = int(np.count_nonzero(~same))
        return ("PASS" if lost == 0 else "FAIL"), f"未重新观测样本 {count} 个，非预期改变 {lost} 个；移出窗口的样本不计。"


def build_report(reference: ReferencePlane, history: HistoryProbe,
                 batches: list[dict[str, CameraFrameGeometry]], snapshot: LocalVoxelSnapshot,
                 config: MapCheckConfig, min_depth: float, max_depth: float,
                 max_ray: float) -> tuple[str, str]:
    rows = []
    offsets = {}
    statuses = []
    wall_statuses = {}
    for source in ("front", "down", "arm"):
        residuals, medians = [], []
        for batch in batches:
            points = frame_points(batch[source], min_depth, max_depth, max_ray)
            selected = points[reference.select(
                points, config.reference_association_distance_m
            )]
            if len(selected) >= config.min_surface_points:
                distances = (selected - reference.center) @ reference.normal
                residuals.append(distances)
                medians.append(float(np.median(distances)))
        if len(residuals) < config.stable_frames:
            rows.append(
                f"| {source} | INCONCLUSIVE | 有效帧 {len(residuals)}/{config.stable_frames}；"
                "未持续覆盖同一参考区域或样本不足 |"
            )
            continue
        values = np.concatenate(residuals)
        p95 = float(np.quantile(np.abs(values), .95))
        status = "PASS" if p95 <= config.max_surface_error_m else "FAIL"
        statuses.append(status)
        wall_statuses[source] = status
        offsets[source] = float(np.median(values))
        rows.append(f"| {source} | {status} | 偏移 {np.median(values)*100:.1f} cm；厚度 σ {np.std(values)*100:.1f} cm；绝对误差 P95 {p95*100:.1f} cm；帧间偏移跨度 {np.ptp(medians)*100:.1f} cm |")
    pair_rows = []
    for a, b in combinations(offsets, 2):
        error = abs(offsets[a] - offsets[b])
        status = "PASS" if error <= config.max_surface_error_m else "FAIL"
        statuses.append(status)
        pair_rows.append(f"- {a}/{b} 法向偏移差：{error*100:.1f} cm，{status}。")
    if not pair_rows:
        pair_rows.append("- INCONCLUSIVE：不足两台相机覆盖同一参考区域，不能据此宣称多相机一致。")
    zyx = np.argwhere(snapshot.state == OCCUPIED)
    points = snapshot.origin_world_xyz + (zyx[:, [2, 1, 0]] + .5) * snapshot.resolution_m
    selected = points[reference.select(points, config.reference_association_distance_m)]
    if len(selected) >= config.min_surface_points:
        residual = (selected - reference.center) @ reference.normal
        p95 = float(np.quantile(np.abs(residual), .95))
        map_status = "PASS" if p95 <= config.max_map_error_m else "FAIL"
        map_text = f"{map_status}：体素中心绝对误差 P95 {p95*100:.1f} cm，厚度 σ {np.std(residual)*100:.1f} cm，样本 {len(selected)}。"
        statuses.append(map_status)
    else:
        map_text = "INCONCLUSIVE：参考区域内占用体素不足或已移出窗口。"
        map_status = "INCONCLUSIVE"

    floor_batches = []
    for batch in batches:
        samples = floor_height_samples(
            batch["down"], config, min_depth, max_depth, max_ray
        )
        if len(samples) >= config.floor_min_points:
            floor_batches.append(samples)
    floor_status = "INCONCLUSIVE"
    floor_map_status = "INCONCLUSIVE"
    floor_text = (
        f"INCONCLUSIVE：DOWN 相机仅有 {len(floor_batches)}/{config.stable_frames} "
        "帧找到稳定的近场水平支撑面。"
    )
    floor_map_text = "INCONCLUSIVE：没有可靠地面高度，无法检查地面占用体素。"
    if len(floor_batches) == config.stable_frames:
        frame_heights = np.asarray([np.median(values) for values in floor_batches])
        floor_height = float(np.median(frame_heights))
        local_residuals = np.concatenate([
            values - np.median(values) for values in floor_batches
        ])
        thickness_p95 = float(np.quantile(np.abs(local_residuals), 0.95))
        height_span = float(np.ptp(frame_heights))
        floor_status = (
            "PASS"
            if thickness_p95 <= config.max_floor_thickness_m
            and height_span <= config.max_floor_height_span_m
            else "FAIL"
        )
        statuses.append(floor_status)
        floor_text = (
            f"{floor_status}：估计地面世界高度 {floor_height:.3f} m；"
            f"单帧表面厚度 P95 {thickness_p95*100:.1f} cm；"
            f"五帧高度跨度 {height_span*100:.1f} cm。"
        )

        base_xy = batches[-1]["down"].world_from_base[:2, 3]
        radius = np.linalg.norm(points[:, :2] - base_xy, axis=1)
        floor_voxels = points[
            (radius >= config.floor_min_radius_m)
            & (radius <= config.floor_max_radius_m)
            & (np.abs(points[:, 2] - floor_height)
               <= config.reference_association_distance_m)
        ]
        if len(floor_voxels) >= config.floor_min_points:
            map_floor_p95 = float(np.quantile(
                np.abs(floor_voxels[:, 2] - floor_height), 0.95
            ))
            floor_map_status = (
                "PASS" if map_floor_p95 <= config.max_map_error_m else "FAIL"
            )
            statuses.append(floor_map_status)
            floor_map_text = (
                f"{floor_map_status}：地面附近占用体素高度误差 P95 "
                f"{map_floor_p95*100:.1f} cm，样本 {len(floor_voxels)}。"
            )
        else:
            floor_map_text = "INCONCLUSIVE：地面附近占用体素样本不足。"

    history_status, history_text = history.check(snapshot)
    statuses.append(history_status)
    required_pass = (
        wall_statuses.get("front") == "PASS"
        and floor_status == "PASS"
        and history_status == "PASS"
        and (map_status == "PASS" or floor_map_status == "PASS")
    )
    overall = ("FAIL" if "FAIL" in statuses else
               "PASS" if required_pass else "INCONCLUSIVE")
    counts = {name: int(np.count_nonzero(snapshot.state == state))
              for name, state in (("FREE", FREE), ("OCCUPIED", OCCUPIED), ("UNKNOWN", UNKNOWN))}
    text = ["# P1 局部地图检查", "", f"**本次结果：{overall}**", "",
            f"地图版本：{snapshot.version}；仿真时间：{snapshot.timestamp_sec:.2f} s。",
            f"参考面位置：{np.round(reference.center, 3).tolist()} m；初始原始深度厚度 σ：{reference.sigma_m*100:.1f} cm。",
            "青色粗框和圆点是固定在世界坐标中的参考区域，请确认它们落在静止平墙上。", "",
            "## 相机与固定参考面的偏差", "", "| 相机 | 结果 | 测量 |", "|---|---|---|", *rows, "",
            f"只关联参考框内且距参考面不超过 {config.reference_association_distance_m*100:.0f} cm 的点，避免把邻近物体或另一层墙面混入统计。", "",
            "## 多相机重合", "", *pair_rows, "",
            "只检查参考区域的法向一致性，不能替代完整外参标定。", "",
            "## 融合表面", "", map_text,
            f"原始深度 P95 阈值 {config.max_surface_error_m*100:.1f} cm；体素中心 P95 阈值 {config.max_map_error_m*100:.1f} cm。",
            f"体素边长 {snapshot.resolution_m*100:.1f} cm；原始点和体素中心分开测量，不能把体素量化误差当成传感器精度。", "",
            "## 地面支撑面", "", floor_text, floor_map_text,
            "这里检查的是 DOWN 相机看到的近场水平支撑面是否稳定。地面仍保留在 3D 占用图中；本阶段尚未把地面、台阶、落差和障碍物分类成导航语义。", "",
            "## 历史保留", "", f"{history_status}：{history_text}", "",
            "## 三态数量", "", str(counts), "",
            "数量仅说明地图中存在这些状态，不证明真实场景分类正确。遮挡后方可能已被其他相机或历史帧观察，不能一律要求 UNKNOWN。",
            "射线空闲/末端占用/遮挡后未知及滑窗保留的算法正确性由独立合成测试验证。", "",
            "本报告是局部诊断，不是整张地图精度或轨迹可通行性认证。移动后点击 CHECK MAP 仍使用同一参考面；不要重新 SET REFERENCE 掩盖漂移。", ""]
    return overall, "\n".join(text)
