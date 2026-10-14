from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import time
import uuid

import numpy as np
import yaml

from constraint_traverse.collect.bev import local_voxel_config_from_mapping
from constraint_traverse.diagnostics.nearfield import (
    build_stride_comparison,
    newly_covered_straight_cells,
    project_cell_to_depth,
    render_comparison,
)
from vln_nav.active_perception.camera_geometry import (
    capture_camera_frame,
    world_from_pose_wxyz,
)
from vln_nav.navigation.fine_local_planner import FinePlannerConfig
from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    FootprintConfig,
    PlanningGrid,
)
from vln_nav.runtime.agx_geometry import (
    build_agx_footprint,
    collision_bounds_in_frame,
    validate_agx_geometry,
)
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.episode import _parse_episode
from vln_nav.runtime.isaac_runtime_utils import (
    get_world_pose,
    resolve_existing_prim_path,
    resolve_robot_relative_prim_path,
    set_camera_horizontal_fov,
    set_local_xform,
    wait_for_stage_assets,
)
from vln_nav.runtime.scene_lighting import add_indoor_lighting


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    args = _parse_args()
    with (PROJECT_ROOT / "configs/episodes.json").open(encoding="utf-8") as handle:
        episode_rows = json.load(handle)["episodes"]
    selected = next(
        (row for row in episode_rows if row["id"] == args.episode_id),
        None,
    )
    if selected is None:
        raise ValueError(f"任务清单没有 {args.episode_id}")
    episode = _parse_episode(selected, project_root=PROJECT_ROOT)
    with (PROJECT_ROOT / "configs/navigation.yaml").open(
        encoding="utf-8"
    ) as handle:
        navigation = yaml.safe_load(handle)
    profile = load_agx_profile(project_root=PROJECT_ROOT)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = args.output_root.expanduser().resolve() / (
        f"{episode.episode_id}_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("nearfield_diagnostic")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.info("诊断输出：%s", output_dir)

    from isaacsim import SimulationApp

    simulation = navigation["simulation"]
    gpu = simulation["active_gpu"] if args.gpu is None else args.gpu
    app = SimulationApp({
        "headless": not args.show_window,
        "active_gpu": gpu,
        "physics_gpu": gpu,
        "multi_gpu": False,
    })
    try:
        _run(
            app=app,
            episode=episode,
            navigation=navigation,
            profile=profile,
            output_dir=output_dir,
            frames_to_capture=args.frames,
            reference_report=args.reference_report,
            logger=logger,
        )
        return 0
    finally:
        app.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在固定 AGX 位姿检查前视、底盘下视和直行扫掠几何"
    )
    parser.add_argument(
        "--episode-id", default="kujiale_0067_bedroom_001"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "constraint_traverse/data/diagnostics",
    )
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames 必须至少为 1")
    return args


def _run(
    *,
    app,
    episode,
    navigation: dict,
    profile,
    output_dir: Path,
    frames_to_capture: int,
    reference_report: Path | None,
    logger: logging.Logger,
) -> None:
    import carb
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import (
        add_reference_to_stage,
        get_current_stage,
        is_stage_loading,
    )
    from isaacsim.sensors.camera import Camera
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

    scene_cfg = navigation["scene"]
    map_cfg = navigation["mapping"]
    camera_cfg = map_cfg["camera"]
    fine_settings = navigation["fine_planner"]
    fine = FinePlannerConfig.from_mapping({
        field.name: fine_settings[field.name]
        for field in fields(FinePlannerConfig)
    })
    footprint = FootprintConfig.from_mapping(fine_settings["footprint"])
    checker = FootprintCollisionChecker(
        footprint, yaw_resolution_deg=fine.yaw_resolution_deg
    )
    map_values = dict(map_cfg["map"])
    map_values.pop("integration_period_sec")
    voxel_config = local_voxel_config_from_mapping(map_values)

    world = World(
        physics_dt=float(profile.simulation.physics_dt),
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    add_reference_to_stage(str(episode.usd_path), scene_cfg["prim_path"])
    if scene_cfg["add_default_ground_plane"]:
        world.scene.add_default_ground_plane()
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(scene_cfg["stage_load_timeout_sec"]),
        logger=logger,
        label="nearfield diagnostic scene",
    )
    if scene_cfg["add_indoor_lighting"]:
        add_indoor_lighting(
            stage=stage,
            environment_prim_path=scene_cfg["prim_path"],
            logger=logger,
        )
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[
            episode.start_x_m,
            episode.start_y_m,
            float(profile.spawn_z_m),
        ],
        rotation_deg=[0.0, 0.0, episode.start_yaw_deg],
    )
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    validate_agx_geometry(stage, profile, articulation_path)
    robot = world.scene.add(SingleArticulation(
        prim_path=articulation_path,
        name="nearfield_diagnostic_agx",
        reset_xform_properties=False,
    ))
    camera_paths = {
        "front": profile.sensors.camera.front_prim_path,
        "down": profile.sensors.camera.mapping_down_prim_path,
    }
    cameras = {}
    for source, relative_path in camera_paths.items():
        path = resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=relative_path,
            ),
            kind=f"{source} camera",
        )
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
            raise RuntimeError(f"配置的相机路径不是 USD Camera：{path}")
        if source == "front":
            set_camera_horizontal_fov(
                stage=stage,
                prim_path=path,
                fov_deg=float(navigation["camera"]["horizontal_fov_deg"]),
            )
        camera = Camera(
            prim_path=path,
            name=f"nearfield_{source}",
            frequency=int(camera_cfg["frequency_hz"]),
            resolution=(
                int(camera_cfg["width"]),
                int(camera_cfg["height"]),
            ),
        )
        cameras[source] = camera

    world.reset()
    robot.initialize()
    for camera in cameras.values():
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
    warmup = int(navigation["camera"]["warmup_render_frames"])
    for _ in range(warmup):
        world.step(render=True)
    frame_pairs = _capture_pairs(
        world=world,
        robot=robot,
        cameras=cameras,
        count=frames_to_capture,
        integration_period_sec=float(map_cfg["map"]["integration_period_sec"]),
        max_frame_skew_sec=float(map_cfg["max_frame_skew_sec"]),
        logger=logger,
    )
    position, orientation = get_world_pose(robot)
    world_from_base = world_from_pose_wxyz(position, orientation)
    yaw = math.atan2(world_from_base[1, 0], world_from_base[0, 0])
    floor_z = float(position[2]) - float(profile.spawn_z_m)
    logger.info(
        "固定采集位姿 world=[%.4f,%.4f,%.4f] yaw=%.3fdeg",
        *map(float, position[:3]),
        math.degrees(yaw),
    )
    baseline_results, baseline_comparison = build_stride_comparison(
        frame_pairs=frame_pairs,
        map_config=voxel_config,
        fine_config=fine,
        collision_checker=checker,
        robot_world_xyz=position,
        robot_yaw_rad=yaw,
        floor_z_m=floor_z,
        obstacle_min_height_m=float(fine_settings["obstacle_min_height_m"]),
        obstacle_max_height_m=float(fine_settings["obstacle_max_height_m"]),
    )
    reference = _load_reference_cells(
        reference_report,
        checker=checker,
        fine_config=fine,
    )
    footprint, footprint_audit = build_agx_footprint(
        stage, profile, articulation_path, margin_m=footprint.margin_m,
    )
    checker = FootprintCollisionChecker(footprint, yaw_resolution_deg=fine.yaw_resolution_deg)
    results, comparison = build_stride_comparison(
        frame_pairs=frame_pairs,
        map_config=voxel_config,
        fine_config=fine,
        collision_checker=checker,
        robot_world_xyz=position,
        robot_yaw_rad=yaw,
        floor_z_m=floor_z,
        obstacle_min_height_m=float(fine_settings["obstacle_min_height_m"]),
        obstacle_max_height_m=float(fine_settings["obstacle_max_height_m"]),
    )
    reference_cells = _classify_reference_cells(
        reference,
        results=results,
        floor_z_m=floor_z,
        obstacle_min_height_m=float(fine_settings["obstacle_min_height_m"]),
        obstacle_max_height_m=float(fine_settings["obstacle_max_height_m"]),
    )
    for cell in reference_cells:
        cell["camera_projection"] = {
            frame.source: project_cell_to_depth(
                frame,
                world_x=float(cell["world_xy_m"][0]),
                world_y=float(cell["world_xy_m"][1]),
                floor_z_m=floor_z,
            )
            for frame in frame_pairs[-1]
        }

    query = get_physx_scene_query_interface()
    physics = _query_physics(
        stage=stage,
        query=query,
        carb=carb,
        PhysicsSchemaTools=PhysicsSchemaTools,
        Usd=Usd,
        UsdGeom=UsdGeom,
        robot_root_path=profile.asset.prim_path,
        environment_root_path=scene_cfg["prim_path"],
        robot_world_xyz=position,
        robot_yaw_rad=yaw,
        floor_z_m=floor_z,
        distance_m=fine.step_length_m,
        sample_step_m=fine.collision_sample_step_m,
        cell_resolution_m=voxel_config.resolution_m,
        cells=(reference_cells or comparison["cells"]),
        planning_grid=results[10]["planning_grid"],
        collision_checker=checker,
    )
    wheel_geometry = _wheel_geometry(
        stage=stage,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        profile=profile,
        checker=checker,
        world_from_base=world_from_base,
        raw_robot_collider_paths=physics["raw_robot_collider_paths"],
        cells=reference_cells or comparison["cells"],
        cell_resolution_m=voxel_config.resolution_m,
        sweep_distance_m=fine.step_length_m,
    )
    payload = {
        "episode_id": episode.episode_id,
        "scene_path": str(episode.usd_path),
        "robot_world_xyz": list(map(float, position[:3])),
        "robot_yaw_rad": float(yaw),
        "floor_z_m": floor_z,
        "frame_count": len(frame_pairs),
        "frame_timestamps": [
            {frame.source: frame.timestamp_sec for frame in pair}
            for pair in frame_pairs
        ],
        "map_config": {
            "resolution_m": voxel_config.resolution_m,
            "pixel_stride_reference": 10,
            "pixel_stride_comparison": 5,
        },
        "map_updates": {
            str(stride): result["updates"]
            for stride, result in results.items()
        },
        "comparison": comparison,
        "configured_footprint_comparison": baseline_comparison,
        "usd_footprint_audit": footprint_audit,
        "reference_cells": reference_cells,
        "physics": physics,
        "wheel_geometry": wheel_geometry,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    _save_frames_and_maps(output_dir, frame_pairs, results)
    render_comparison(
        results,
        reference_cells or comparison["cells"],
        robot_world_xy=(float(position[0]), float(position[1])),
    ).save(output_dir / "comparison.png")
    render_comparison(
        baseline_results, baseline_comparison["cells"],
        robot_world_xy=(float(position[0]), float(position[1])),
    ).save(output_dir / "configured_footprint_comparison.png")
    logger.info(
        "USD 碰撞体核查=%d，状态=%s；配置轮廓直行未知=%d，USD 轮廓直行未知=%d",
        footprint_audit["collider_count"], footprint_audit["status"],
        len(baseline_comparison["cells"]), len(comparison["cells"]),
    )
    logger.info(
        "stride10 直行未知=%d，物理足迹=%d，余量=%d；stride5 原位置状态=%s",
        len(comparison["cells"]),
        comparison["physical_unknown_count"],
        comparison["margin_only_unknown_count"],
        [cell["stride_5"] for cell in reference_cells or comparison["cells"]],
    )
    logger.info(
        "PhysX 环境碰撞命中=%d，查询错误=%d，报告=%s",
        physics["obstacle_hit_count"],
        len(physics["query_errors"]),
        output_dir / "report.json",
    )


def _capture_pairs(
    *,
    world,
    robot,
    cameras: dict,
    count: int,
    integration_period_sec: float,
    max_frame_skew_sec: float,
    logger: logging.Logger,
) -> list[tuple]:
    pairs = []
    previous_timestamps = {"front": -float("inf"), "down": -float("inf")}
    last_integration_time = -float("inf")
    deadline = time.monotonic() + 90.0
    frame_index = 0
    while len(pairs) < count:
        if time.monotonic() > deadline:
            raise TimeoutError("等待两路同步 RGB-D 帧超时")
        world.step(render=True)
        position, orientation = get_world_pose(robot)
        world_from_base = world_from_pose_wxyz(position, orientation)
        frames = {
            source: capture_camera_frame(
                source=source,
                camera=camera,
                frame_index=frame_index,
                world_from_base=world_from_base,
            )
            for source, camera in cameras.items()
        }
        frame_index += 1
        if any(frame is None for frame in frames.values()):
            continue
        if any(
            frame.timestamp_sec <= previous_timestamps[source] + 1.0e-9
            for source, frame in frames.items()
        ):
            continue
        if abs(
            frames["front"].timestamp_sec - frames["down"].timestamp_sec
        ) > max_frame_skew_sec:
            continue
        if (
            float(world.current_time) - last_integration_time
            < integration_period_sec
        ):
            continue
        pair = (frames["front"], frames["down"])
        for frame in pair:
            frame.validate()
            previous_timestamps[frame.source] = frame.timestamp_sec
        pairs.append(pair)
        last_integration_time = float(world.current_time)
        logger.info(
            "同步 RGB-D 帧 %d/%d：front=%.3fs down=%.3fs",
            len(pairs),
            count,
            pair[0].timestamp_sec,
            pair[1].timestamp_sec,
        )
    return pairs


def _save_frames_and_maps(output_dir: Path, frame_pairs: list, results: dict) -> None:
    arrays = {}
    for index, pair in enumerate(frame_pairs):
        for frame in pair:
            prefix = f"{frame.source}_{index}"
            arrays[f"{prefix}_rgb"] = frame.rgb
            arrays[f"{prefix}_depth_m"] = frame.depth_m
            arrays[f"{prefix}_intrinsics"] = frame.intrinsics
            arrays[f"{prefix}_world_from_camera"] = frame.world_from_camera
            arrays[f"{prefix}_world_from_base"] = frame.world_from_base
    np.savez_compressed(output_dir / "raw_rgbd.npz", **arrays)
    for stride, result in results.items():
        snapshot = result["snapshot"]
        grid = result["planning_grid"]
        np.savez_compressed(
            output_dir / f"map_stride_{stride}.npz",
            state_zyx=snapshot.state,
            source_mask_zyx=snapshot.source_mask,
            evidence_count_zyx=snapshot.evidence_count,
            map_origin_world_xyz=snapshot.origin_world_xyz,
            resolution_m=np.asarray(snapshot.resolution_m),
            map_version=np.asarray(snapshot.version),
            occupied_yx=grid.occupied_mask,
            free_yx=grid.free_mask,
            unknown_yx=grid.unknown_mask,
        )


def _load_reference_cells(
    path: Path | None,
    *,
    checker: FootprintCollisionChecker,
    fine_config: FinePlannerConfig,
) -> list[dict]:
    if path is None:
        return []
    report_path = path.expanduser().resolve()
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    map_path = report_path.with_suffix(".npz")
    with np.load(map_path) as saved:
        shape = saved["unknown_mask"].shape
        origin = saved["map_origin_world_xyz"]
        resolution = float(saved["resolution_m"])
        grid = PlanningGrid(
            occupied_mask=saved["occupied_mask"].copy(),
            free_mask=saved["free_mask"].copy(),
            unknown_mask=saved["unknown_mask"].copy(),
            clearance_m=saved["clearance_m"].copy(),
            origin_x_m=float(origin[0]),
            origin_y_m=float(origin[1]),
            resolution_m=resolution,
            map_version=int(saved["map_version"]),
        )
    action = next(
        item for item in report["action_diagnostics"]
        if item["action_name"] == "forward_2"
    )
    physical_cells = set(map(int, newly_covered_straight_cells(
        grid=grid,
        checker=checker,
        start_pose_world=tuple(map(float, report["start_pose_world"])),
        distance_m=fine_config.step_length_m,
        sample_step_m=fine_config.collision_sample_step_m,
        margin_m=0.0,
    )))
    cells = []
    for linear in action["unknown_linear_indices"]:
        row, column = divmod(int(linear), shape[1])
        cells.append({
            "reference_linear_index": int(linear),
            "footprint_layer": (
                "physical" if int(linear) in physical_cells
                else "safety_margin"
            ),
            "world_xy_m": [
                float(origin[0] + (column + 0.5) * resolution),
                float(origin[1] + (row + 0.5) * resolution),
            ],
        })
    return cells


def _classify_reference_cells(
    reference: list[dict],
    *,
    results: dict,
    floor_z_m: float,
    obstacle_min_height_m: float,
    obstacle_max_height_m: float,
) -> list[dict]:
    from constraint_traverse.diagnostics.nearfield import (
        grid_cell_state,
        source_bits_for_cell,
    )

    classified = []
    for cell in reference:
        row10, column10, inside10 = results[10]["raw_grid"].world_to_cell(
            *cell["world_xy_m"]
        )
        row5, column5, inside5 = results[5]["raw_grid"].world_to_cell(
            *cell["world_xy_m"]
        )
        entry = dict(cell)
        for stride, row, column, inside in (
            (10, row10, column10, inside10),
            (5, row5, column5, inside5),
        ):
            entry[f"stride_{stride}"] = (
                grid_cell_state(results[stride]["raw_grid"], row, column)
                if inside else "OUTSIDE_MAP"
            )
            entry[f"source_bits_stride_{stride}"] = (
                source_bits_for_cell(
                    results[stride]["snapshot"],
                    row,
                    column,
                    floor_z_m=floor_z_m,
                    min_height_m=obstacle_min_height_m,
                    max_height_m=obstacle_max_height_m,
                ) if inside else 0
            )
        classified.append(entry)
    return classified


def _query_physics(
    *,
    stage,
    query,
    carb,
    PhysicsSchemaTools,
    Usd,
    UsdGeom,
    robot_root_path: str,
    environment_root_path: str,
    robot_world_xyz: np.ndarray,
    robot_yaw_rad: float,
    floor_z_m: float,
    distance_m: float,
    sample_step_m: float,
    cell_resolution_m: float,
    cells: list[dict],
    planning_grid: PlanningGrid,
    collision_checker: FootprintCollisionChecker,
) -> dict:
    nearby_paths = set()

    def collect_nearby(hit):
        nearby_paths.add(str(hit.collision))
        return True

    query.overlap_box(
        carb.Float3(3.0, 3.0, 2.0),
        carb.Float3(
            float(robot_world_xyz[0]),
            float(robot_world_xyz[1]),
            float(robot_world_xyz[2] + 0.5),
        ),
        carb.Float4(0.0, 0.0, 0.0, 1.0),
        collect_nearby,
        False,
    )
    robot_paths = sorted(
        path for path in nearby_paths
        if _path_within(path, robot_root_path)
    )
    environment_paths = sorted(
        path for path in nearby_paths
        if _path_within(path, environment_root_path)
    )
    robot_colliders = [
        stage.GetPrimAtPath(path)
        for path in robot_paths
        if stage.GetPrimAtPath(path).IsA(UsdGeom.Gprim)
    ]
    omitted = [
        {
            "path": path,
            "usd_type": stage.GetPrimAtPath(path).GetTypeName(),
            "valid": bool(stage.GetPrimAtPath(path).IsValid()),
        }
        for path in robot_paths
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Gprim)
    ]
    result = {
        "query_type": "PhysX swept USD collision shapes",
        "robot_collider_count": len(robot_colliders),
        "environment_collider_count": len(environment_paths),
        "robot_collider_paths": [
            str(prim.GetPath()) for prim in robot_colliders
        ],
        "raw_robot_collider_paths": robot_paths,
        "robot_colliders_without_shape_query": omitted,
        "environment_collider_paths": environment_paths,
        "sweep_distance_m": float(distance_m),
        "raw_sweep_hits": [],
        "support_like_hits": [],
        "obstacle_hits": [],
        "query_errors": [],
        "cell_overlap": [],
    }
    if not robot_colliders or not environment_paths:
        result["query_errors"].append(
            "机器人或环境附近没有发现 PhysX 场景中的物理碰撞体"
        )
    direction = carb.Float3(
        float(math.cos(robot_yaw_rad)),
        float(math.sin(robot_yaw_rad)),
        0.0,
    )
    for prim in robot_colliders:
        prim_path = str(prim.GetPath())
        hits = []

        def report_hit(hit):
            position = _float3(hit.position)
            normal = _float3(hit.normal)
            collided = str(hit.collision)
            entry = {
                "moving_collider": prim_path,
                "hit_collider": collided,
                "hit_rigid_body": str(hit.rigid_body),
                "distance_m": float(hit.distance),
                "position_world_xyz": position,
                "normal_world_xyz": normal,
            }
            hits.append(entry)
            return True

        try:
            encoded = PhysicsSchemaTools.encodeSdfPath(prim.GetPath())
            query.sweep_shape_all(
                encoded[0], encoded[1], direction, float(distance_m),
                report_hit, False,
            )
        except Exception as exc:
            result["query_errors"].append(f"{prim_path}: {exc}")
            continue
        for hit in hits:
            result["raw_sweep_hits"].append(hit)
            if _path_within(hit["hit_collider"], robot_root_path):
                hit["classification"] = "ROBOT_SELF"
            elif (
                hit["normal_world_xyz"][2] >= 0.9
                and abs(hit["position_world_xyz"][2] - floor_z_m) <= 0.02
            ):
                hit["classification"] = "SUPPORT_LIKE"
                result["support_like_hits"].append(hit)
            else:
                hit["classification"] = "ENVIRONMENT_OBSTACLE"
                result["obstacle_hits"].append(hit)
    result["obstacle_hit_count"] = len(result["obstacle_hits"])
    result["support_like_hit_count"] = len(result["support_like_hits"])
    result["status"] = (
        "INCOMPLETE" if result["query_errors"]
        else "OBSTACLE_HIT" if result["obstacle_hits"]
        else "PARTIAL_ROBOT_SWEEP" if omitted
        else "NO_NON_SUPPORT_HIT"
    )

    for cell in cells:
        world_x, world_y = map(float, cell["world_xy_m"])
        checks = {}
        for label, min_height, max_height in (
            ("wheel_height", 0.05, 0.30),
            ("body_height", 0.05, 1.50),
        ):
            try:
                checks[label] = _overlap_cell(
                    query=query,
                    carb=carb,
                    robot_root_path=robot_root_path,
                    world_x=world_x,
                    world_y=world_y,
                    floor_z_m=floor_z_m,
                    min_height_m=min_height,
                    max_height_m=max_height,
                    cell_resolution_m=cell_resolution_m,
                )
            except Exception as exc:
                checks[label] = {"query_error": str(exc)}
                result["query_errors"].append(
                    f"cell {cell['world_xy_m']} {label}: {exc}"
                )
        result["cell_overlap"].append({
            "world_xy_m": [world_x, world_y],
            "checks": checks,
        })
    initial_cells, _ = collision_checker.covered_linear_indices(
        planning_grid,
        x_m=float(robot_world_xyz[0]),
        y_m=float(robot_world_xyz[1]),
        yaw_rad=float(robot_yaw_rad),
    )
    new_cells = newly_covered_straight_cells(
        grid=planning_grid,
        checker=collision_checker,
        start_pose_world=(
            float(robot_world_xyz[0]),
            float(robot_world_xyz[1]),
            float(robot_yaw_rad),
        ),
        distance_m=distance_m,
        sample_step_m=sample_step_m,
        margin_m=collision_checker.footprint.margin_m,
    )
    all_swept = np.union1d(initial_cells, new_cells)
    envelope_hits = []
    for linear in all_swept:
        row, column = divmod(int(linear), planning_grid.shape[1])
        world_x = planning_grid.origin_x_m + (
            column + 0.5
        ) * planning_grid.resolution_m
        world_y = planning_grid.origin_y_m + (
            row + 0.5
        ) * planning_grid.resolution_m
        try:
            occupancy = _overlap_cell(
                query=query,
                carb=carb,
                robot_root_path=robot_root_path,
                world_x=world_x,
                world_y=world_y,
                floor_z_m=floor_z_m,
                min_height_m=0.05,
                max_height_m=1.50,
                cell_resolution_m=cell_resolution_m,
            )
        except Exception as exc:
            result["query_errors"].append(
                f"扫掠包络栅格 {int(linear)}: {exc}"
            )
            continue
        if occupancy["environment_hits"]:
            envelope_hits.append({
                "linear_index": int(linear),
                "world_xy_m": [world_x, world_y],
                "environment_hits": occupancy["environment_hits"],
            })
    result["planning_swept_envelope"] = {
        "checked_cell_count": len(all_swept),
        "newly_covered_cell_count": len(new_cells),
        "environment_hit_cells": envelope_hits,
        "environment_hit_cell_count": len(envelope_hits),
    }
    if result["query_errors"]:
        result["status"] = "INCOMPLETE"
    return result


def _overlap_cell(
    *,
    query,
    carb,
    robot_root_path: str,
    world_x: float,
    world_y: float,
    floor_z_m: float,
    min_height_m: float,
    max_height_m: float,
    cell_resolution_m: float,
) -> dict:
    hits = []

    def report_hit(hit):
        hits.append({
            "collider": str(hit.collision),
            "rigid_body": str(hit.rigid_body),
        })
        return True

    query.overlap_box(
        carb.Float3(
            cell_resolution_m / 2,
            cell_resolution_m / 2,
            (max_height_m - min_height_m) / 2,
        ),
        carb.Float3(
            world_x,
            world_y,
            floor_z_m + (min_height_m + max_height_m) / 2,
        ),
        carb.Float4(0.0, 0.0, 0.0, 1.0),
        report_hit,
        False,
    )
    return {
        "environment_hits": [
            hit for hit in hits
            if not _path_within(hit["collider"], robot_root_path)
        ],
        "robot_self_hits": sum(
            _path_within(hit["collider"], robot_root_path)
            for hit in hits
        ),
    }


def _path_within(candidate: str, parent: str) -> bool:
    return candidate == parent or candidate.startswith(parent.rstrip("/") + "/")


def _float3(value) -> list[float]:
    return [float(value[index]) for index in range(3)]


def _wheel_geometry(
    *,
    stage,
    Usd,
    UsdGeom,
    UsdPhysics,
    profile,
    checker: FootprintCollisionChecker,
    world_from_base: np.ndarray,
    raw_robot_collider_paths: list[str],
    cells: list[dict],
    cell_resolution_m: float,
    sweep_distance_m: float,
) -> dict:
    robot_root = stage.GetPrimAtPath(profile.asset.prim_path)
    joints = {
        prim.GetName(): UsdPhysics.RevoluteJoint(prim)
        for prim in Usd.PrimRange(robot_root)
        if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    base_from_world = np.linalg.inv(world_from_base)
    result = {
        "wheels": [],
        "configured_parts": {},
        "robot_collider_bounds": [],
        "reference_cell_coverage": [],
        "errors": [],
    }
    for part in checker.footprint.parts:
        result["configured_parts"][part.name] = asdict(part)
    for path in raw_robot_collider_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            result["errors"].append(
                f"PhysX 碰撞体没有有效 USD prim：{path}"
            )
            continue
        try:
            points = collision_bounds_in_frame(
                stage, prim,
                resolve_robot_relative_prim_path(
                    robot_root_path=profile.asset.prim_path,
                    prim_path=profile.asset.articulation_path,
                ),
            )
            bound = {
                "collider_path": path,
                "bound_source": "USD recomputed shape extents",
                "base_min_xyz_m": points.min(axis=0).tolist(),
                "base_max_xyz_m": points.max(axis=0).tolist(),
            }
            result["robot_collider_bounds"].append(bound)
        except ValueError as exc:
            result["errors"].append(str(exc))
    for wheel in profile.agx_controller.continuous.wheels:
        if wheel.drive_joint not in joints:
            result["errors"].append(
                f"找不到驱动关节：{wheel.drive_joint}"
            )
            continue
        targets = joints[wheel.drive_joint].GetBody1Rel().GetTargets()
        if not targets:
            result["errors"].append(
                f"驱动关节没有轮体 Body1：{wheel.drive_joint}"
            )
            continue
        link_path = str(targets[0])
        bounds = [
            item
            for item in result["robot_collider_bounds"]
            if _path_within(item["collider_path"], link_path)
        ]
        if not bounds:
            result["errors"].append(
                f"车轮没有可比较的物理碰撞几何：{link_path}"
            )
            continue
        minimum_xy = np.min(
            [item["base_min_xyz_m"][:2] for item in bounds], axis=0
        )
        maximum_xy = np.max(
            [item["base_max_xyz_m"][:2] for item in bounds], axis=0
        )
        measured_center = 0.5 * (minimum_xy + maximum_xy)
        configured_center = np.asarray(
            [wheel.x_m, wheel.y_m], dtype=np.float64
        )
        result["wheels"].append({
            "drive_joint": wheel.drive_joint,
            "wheel_link": link_path,
            "configured_center_base_xy_m": configured_center.tolist(),
            "collider_center_base_xy_m": measured_center.tolist(),
            "center_error_m": float(np.linalg.norm(
                measured_center - configured_center
            )),
            "collider_min_base_xy_m": minimum_xy.tolist(),
            "collider_max_base_xy_m": maximum_xy.tolist(),
            "colliders": bounds,
        })
    wheel_links = [wheel["wheel_link"] for wheel in result["wheels"]]
    for cell in cells:
        world_x, world_y = map(float, cell["world_xy_m"])
        half = cell_resolution_m / 2
        world_corners = np.asarray([
            [x, y, float(world_from_base[2, 3]), 1.0]
            for x in (world_x - half, world_x + half)
            for y in (world_y - half, world_y + half)
        ])
        cell_base = (base_from_world @ world_corners.T).T[:, :2]
        cell_min = cell_base.min(axis=0)
        cell_max = cell_base.max(axis=0)
        wheel_hits = []
        all_hits = []
        wheel_gap = float("inf")
        for bound in result["robot_collider_bounds"]:
            bound_min = np.asarray(bound["base_min_xyz_m"][:2])
            bound_max = np.asarray(bound["base_max_xyz_m"][:2])
            bound_max[0] += sweep_distance_m
            gap = float(np.linalg.norm(np.maximum(
                np.maximum(bound_min - cell_max, cell_min - bound_max),
                0.0,
            )))
            if gap <= 1.0e-9:
                all_hits.append(bound["collider_path"])
            if any(
                _path_within(bound["collider_path"], link)
                for link in wheel_links
            ):
                wheel_gap = min(wheel_gap, gap)
                if gap <= 1.0e-9:
                    wheel_hits.append(bound["collider_path"])
        result["reference_cell_coverage"].append({
            "world_xy_m": [world_x, world_y],
            "cell_min_base_xy_m": cell_min.tolist(),
            "cell_max_base_xy_m": cell_max.tolist(),
            "wheel_swept_bound_overlaps": wheel_hits,
            "any_robot_swept_bound_overlaps": all_hits,
            "wheel_swept_bound_gap_m": (
                wheel_gap if math.isfinite(wheel_gap) else None
            ),
            "all_robot_bounds_complete": (
                len(result["robot_collider_bounds"])
                == len(raw_robot_collider_paths)
            ),
        })
    result["wheel_status"] = (
        "COMPLETE"
        if len(result["wheels"])
        == len(profile.agx_controller.continuous.wheels)
        else "INCOMPLETE"
    )
    result["status"] = (
        "COMPLETE"
        if result["wheel_status"] == "COMPLETE"
        and len(result["robot_collider_bounds"])
        == len(raw_robot_collider_paths)
        and not result["errors"]
        else "PARTIAL_BOUNDS"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
