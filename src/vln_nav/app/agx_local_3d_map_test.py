"""P1 acceptance for moving three-camera local 3-D occupancy fusion."""
from __future__ import annotations

from dataclasses import fields
import logging
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import yaml

from vln_nav.active_perception.camera_geometry import (
    CameraFrameGeometry,
    capture_camera_frame,
    world_from_pose_wxyz,
)
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import (
    OCCUPIED,
    SOURCE_BITS,
    UNKNOWN,
    LocalVoxelMap,
    LocalVoxelMapConfig,
    LocalVoxelSnapshot,
    VoxelMapUpdate,
)
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from vln_nav.runtime.isaac_runtime_utils import (
    bind_viewport_to_camera,
    get_joint_names,
    get_world_pose,
    resolve_existing_prim_path,
    resolve_robot_relative_prim_path,
    set_local_xform,
    wait_for_stage_assets,
)
from vln_nav.runtime.robot_profile import load_robot_profile


SETTINGS_PATH = PROJECT_ROOT / "configs/agx_local_3d_map_test.yaml"
def _load_settings() -> tuple[dict, LocalVoxelMapConfig]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    if set(settings) != {"sensors", "camera", "map", "validation", "debug"}:
        raise ValueError("P1 config requires exactly sensors, camera, map, validation, and debug")
    sensors = settings["sensors"]
    expected_sensor_keys = {
        "front_prim_path",
        "down_prim_path",
        "arm_prim_path",
        "observer_prim_path",
    }
    if set(sensors) != expected_sensor_keys or any(
        not str(sensors[key]).strip() for key in expected_sensor_keys
    ):
        raise ValueError("P1 requires four non-empty camera prim paths")
    if len({str(value) for value in sensors.values()}) != len(sensors):
        raise ValueError("P1 camera roles must use different prim paths")

    camera = settings["camera"]
    if set(camera) != {"width", "height", "frequency_hz"} or any(
        int(camera[key]) < 1 for key in camera
    ):
        raise ValueError("P1 camera width, height, and frequency_hz must be positive")

    map_settings = settings["map"]
    map_fields = {item.name for item in fields(LocalVoxelMapConfig)}
    if set(map_settings) != map_fields | {"integration_period_sec"}:
        raise ValueError(
            "P1 map fields do not match LocalVoxelMapConfig plus integration_period_sec"
        )
    map_config = LocalVoxelMapConfig(
        **{name: map_settings[name] for name in map_fields}
    )
    map_config.validate()
    if float(map_settings["integration_period_sec"]) <= 0:
        raise ValueError("map.integration_period_sec must be positive")

    validation = settings["validation"]
    validation_keys = {
        "warmup_render_frames",
        "max_frame_skew_sec",
        "required_map_updates",
        "minimum_rays_per_camera",
        "minimum_known_voxels",
        "minimum_occupied_voxels",
    }
    if set(validation) != validation_keys or any(
        float(validation[key]) <= 0 for key in validation
    ):
        raise ValueError("P1 validation fields must be present and positive")

    debug = settings["debug"]
    if set(debug) != {"output_path", "update_period_sec"}:
        raise ValueError("Invalid P1 debug fields")
    if not str(debug["output_path"]).strip() or float(debug["update_period_sec"]) <= 0:
        raise ValueError("Invalid P1 debug output settings")
    return settings, map_config


class _DrivePanel:
    _KEY_ACTIONS = {
        "UP": "FORWARD",
        "DOWN": "BACKWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }

    def __init__(self, *, clear_map_callback) -> None:
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self._carb_input = carb.input
        self._ui = ui
        self._held_actions: dict[str, str] = {}
        self._press_order: list[str] = []
        self._clear_map_callback = clear_map_callback
        self._speed_model = ui.SimpleFloatModel(0.65)
        self._window = ui.Window("AGX Local Map Drive", width=330, height=245, visible=True)
        with self._window.frame:
            with ui.VStack(spacing=7):
                with ui.HStack(height=42):
                    ui.Spacer()
                    self._motion_button("UP", "FORWARD")
                    ui.Spacer()
                with ui.HStack(height=42):
                    self._motion_button("LEFT", "TURN_LEFT")
                    ui.Button("STOP", width=90, clicked_fn=self.clear_motion)
                    self._motion_button("RIGHT", "TURN_RIGHT")
                with ui.HStack(height=42):
                    ui.Spacer()
                    self._motion_button("DOWN", "BACKWARD")
                    ui.Spacer()
                with ui.HStack(height=30):
                    ui.Label("Speed", width=55)
                    ui.FloatSlider(model=self._speed_model, min=0.10, max=1.0, step=0.05)
                ui.Button("CLEAR MAP", height=30, clicked_fn=self._clear_map_callback)

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac Sim app window is unavailable")
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._keyboard_subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

    @property
    def action(self) -> str:
        if not self._press_order:
            return "STOP"
        return self._held_actions.get(self._press_order[-1], "STOP")

    @property
    def speed_scale(self) -> float:
        return float(np.clip(self._speed_model.as_float, 0.10, 1.0))

    def clear_motion(self) -> None:
        self._held_actions.clear()
        self._press_order.clear()

    def shutdown(self) -> None:
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_subscription,
            )
            self._keyboard_subscription = None
        self._window.visible = False

    def _motion_button(self, label: str, action: str):
        return self._ui.Button(
            label,
            width=90,
            mouse_pressed_fn=lambda *_args: self._press(f"button:{action}", action),
            mouse_released_fn=lambda *_args: self._release(f"button:{action}"),
        )

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        key_name = str(event.input.name)
        action = self._KEY_ACTIONS.get(key_name)
        if action is None:
            return True
        source = f"key:{key_name}"
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            self._press(source, action)
        elif (
            event.type == self._carb_input.KeyboardEventType.KEY_REPEAT
            and source not in self._held_actions
        ):
            self._press(source, action)
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._release(source)
        return True

    def _press(self, source: str, action: str) -> None:
        self._held_actions[source] = action
        if source in self._press_order:
            self._press_order.remove(source)
        self._press_order.append(source)

    def _release(self, source: str) -> None:
        self._held_actions.pop(source, None)
        if source in self._press_order:
            self._press_order.remove(source)


def _height_colors(relative_height_m: np.ndarray) -> np.ndarray:
    """Approximate RViz axis coloring without adding a plotting dependency."""
    stops = np.asarray(
        [
            (0.00, 230, 0, 220),
            (0.30, 55, 45, 245),
            (0.62, 0, 225, 235),
            (1.00, 0, 240, 105),
        ],
        dtype=np.float64,
    )
    normalized = np.clip(relative_height_m / 2.4, 0.0, 1.0)
    colors = np.empty((len(normalized), 3), dtype=np.float64)
    for index in range(len(stops) - 1):
        low, high = stops[index], stops[index + 1]
        mask = (normalized >= low[0]) & (normalized <= high[0])
        alpha = (normalized[mask] - low[0]) / (high[0] - low[0])
        colors[mask] = low[1:] + alpha[:, None] * (high[1:] - low[1:])
    return colors.astype(np.uint8)


def _project_local_points(
    points_world: np.ndarray,
    *,
    robot_pose: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world points from a fixed oblique camera behind the robot."""
    width, height = image_size
    robot_x, robot_y, robot_z, robot_yaw = robot_pose
    delta = points_world - np.asarray((robot_x, robot_y, robot_z), dtype=np.float64)
    cosine, sine = math.cos(robot_yaw), math.sin(robot_yaw)
    points_robot = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            delta[:, 2],
        )
    )

    camera_position = np.asarray((-2.4, 0.0, 2.7), dtype=np.float64)
    look_at = np.asarray((2.2, 0.0, 0.65), dtype=np.float64)
    camera_forward = look_at - camera_position
    camera_forward /= np.linalg.norm(camera_forward)
    camera_right = np.cross(camera_forward, np.asarray((0.0, 0.0, 1.0)))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, camera_forward)
    relative = points_robot - camera_position
    depth = relative @ camera_forward
    visible = depth > 0.10
    focal_px = 0.78 * min(width, height)
    u = width * 0.5 + focal_px * (relative @ camera_right) / np.maximum(depth, 0.10)
    v = height * 0.52 - focal_px * (relative @ camera_up) / np.maximum(depth, 0.10)
    visible &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[visible], v[visible], depth[visible], points_robot[visible, 2]


def _render_local_occupancy(
    snapshot: LocalVoxelSnapshot,
    *,
    robot_pose: tuple[float, float, float, float],
    image_size: tuple[int, int] = (1280, 900),
) -> Image.Image:
    occupied_zyx = np.argwhere(snapshot.state == OCCUPIED)
    width, height = image_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if occupied_zyx.size:
        points_world = snapshot.origin_world_xyz + (
            occupied_zyx[:, [2, 1, 0]].astype(np.float64) + 0.5
        ) * snapshot.resolution_m
        u, v, depth, relative_z = _project_local_points(
            points_world,
            robot_pose=robot_pose,
            image_size=image_size,
        )
        colors = _height_colors(relative_z)
        order = np.argsort(depth)[::-1]
        u_int = u[order].astype(np.int32)
        v_int = v[order].astype(np.int32)
        colors = colors[order]
        # Nearer voxels overwrite farther voxels, producing a dense point-cloud surface.
        for offset_y in range(-2, 3):
            for offset_x in range(-2, 3):
                px = u_int + offset_x
                py = v_int + offset_y
                valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                canvas[py[valid], px[valid]] = colors[valid]

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    robot_x, robot_y, robot_z, robot_yaw = robot_pose
    marker_world = np.asarray(
        [
            (robot_x, robot_y, robot_z),
            (
                robot_x + 0.8 * math.cos(robot_yaw),
                robot_y + 0.8 * math.sin(robot_yaw),
                robot_z,
            ),
        ],
        dtype=np.float64,
    )
    marker_u, marker_v, _, _ = _project_local_points(
        marker_world,
        robot_pose=robot_pose,
        image_size=image_size,
    )
    if len(marker_u) == 2:
        robot_pixel = (int(marker_u[0]), int(marker_v[0]))
        heading_pixel = (int(marker_u[1]), int(marker_v[1]))
        draw.ellipse(
            (
                robot_pixel[0] - 7,
                robot_pixel[1] - 7,
                robot_pixel[0] + 7,
                robot_pixel[1] + 7,
            ),
            fill=(255, 205, 35),
        )
        draw.line((*robot_pixel, *heading_pixel), fill=(255, 85, 35), width=4)
    return image


def _readiness_failures(
    *,
    snapshot: LocalVoxelSnapshot,
    update: VoxelMapUpdate,
    settings: dict,
) -> list[str]:
    validation = settings["validation"]
    failures: list[str] = []
    if snapshot.version < int(validation["required_map_updates"]):
        failures.append("Collecting temporal map evidence")
    counts = dict(update.source_ray_counts)
    for source in SOURCE_BITS:
        if counts.get(source, 0) < int(validation["minimum_rays_per_camera"]):
            failures.append(f"{source.upper()} has too few depth rays")
        if snapshot.source_voxel_count(source) == 0:
            failures.append(f"{source.upper()} has not contributed to the map")
    known_count = int(np.count_nonzero(snapshot.state != UNKNOWN))
    occupied_count = int(np.count_nonzero(snapshot.state == OCCUPIED))
    if known_count < int(validation["minimum_known_voxels"]):
        failures.append("Too little space has been observed")
    if occupied_count < int(validation["minimum_occupied_voxels"]):
        failures.append("No stable obstacle surfaces yet")
    return failures


def _save_debug(
    *,
    path: Path,
    snapshot: LocalVoxelSnapshot,
    robot_pose: tuple[float, float, float, float],
) -> None:
    canvas = _render_local_occupancy(snapshot, robot_pose=robot_pose)

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=path.suffix or ".png", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        canvas.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _robot_yaw(world_from_base: np.ndarray) -> float:
    rotation = np.asarray(world_from_base, dtype=np.float64)[:3, :3]
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def main() -> int:
    settings, map_config = _load_settings()
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    logger = logging.getLogger("agx_local_3d_map")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/agx_local_3d_map_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "active_gpu": config.simulation.active_gpu,
            "physics_gpu": config.simulation.physics_gpu,
            "multi_gpu": bool(config.simulation.multi_gpu),
        }
    )
    try:
        try:
            return _run(app, config, profile, settings, map_config, logger)
        except Exception:
            logger.exception("[P1Fatal] local 3D map test crashed")
            return 1
    finally:
        app.close()


def _run(app, config, profile, settings, map_config, logger) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux

    world = World(
        physics_dt=float(profile.simulation.physics_dt),
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    scene_path = Path(str(config.scene.usd_path))
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path
    if not scene_path.is_file():
        raise FileNotFoundError(f"P1 environment USD does not exist: {scene_path}")
    add_reference_to_stage(str(scene_path), config.scene.prim_path)
    if config.scene.add_default_ground_plane:
        world.scene.add_default_ground_plane()
    if config.scene.add_dome_light:
        UsdLux.DomeLight.Define(stage, "/World/P1Light").CreateIntensityAttr(1500.0)
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=config.scene.stage_load_timeout_sec,
        logger=logger,
        label="AGX P1 hospital local 3D map",
    )
    start_position = [
        float(config.scene.robot_start_xy_m[0]),
        float(config.scene.robot_start_xy_m[1]),
        float(profile.spawn_z_m),
    ]
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=start_position,
        rotation_deg=config.scene.robot_start_orientation_deg,
    )
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    robot = world.scene.add(
        SingleArticulation(
            prim_path=articulation_path,
            name="agx_p1_local_3d_map",
            reset_xform_properties=False,
        )
    )

    role_paths = {
        source: resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=settings["sensors"][f"{source}_prim_path"],
            ),
            kind=f"P1 {source} camera",
        )
        for source in SOURCE_BITS
    }
    if len(set(role_paths.values())) != len(role_paths):
        raise RuntimeError(f"P1 camera role collision: {role_paths}")
    for source, path in role_paths.items():
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
            raise RuntimeError(f"P1 {source} prim is not a Camera: {path}")
    observer_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=settings["sensors"]["observer_prim_path"],
        ),
        kind="P1 observer camera",
    )
    if not stage.GetPrimAtPath(observer_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"P1 observer prim is not a Camera: {observer_path}")

    camera_settings = settings["camera"]
    cameras = {
        source: Camera(
            prim_path=path,
            name=f"agx_p1_{source}_camera",
            frequency=int(camera_settings["frequency_hz"]),
            resolution=(int(camera_settings["width"]), int(camera_settings["height"])),
        )
        for source, path in role_paths.items()
    }
    world.reset()
    robot.initialize()
    for camera in cameras.values():
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
    bind_viewport_to_camera(camera_prim_path=observer_path, logger=logger, enabled=True)
    drive = create_locomotion_controller(
        profile=profile,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=get_joint_names(robot),
        logger=logger,
    )
    for source, path in role_paths.items():
        logger.info("[P1Camera] role=%s prim=%s", source, path)
    logger.info("[P1Scene] usd=%s start=%s", scene_path, start_position)
    logger.info("[P1MapConfig] size=%.1fx%.1fx%.1fm resolution=%.2fm", map_config.size_x_m,
                map_config.size_y_m, map_config.max_z_m - map_config.min_z_m,
                map_config.resolution_m)

    output_path = PROJECT_ROOT / settings["debug"]["output_path"]
    for _ in range(int(settings["validation"]["warmup_render_frames"])):
        drive.apply("STOP", speed_scale=0.0)
        world.step(render=True)

    position, orientation = get_world_pose(robot)
    local_map = LocalVoxelMap(map_config)
    local_map.reset(np.asarray(position[:3], dtype=np.float64))
    frame_index = 0
    last_integration_time = -float("inf")
    last_debug_time = -float("inf")
    last_status_signature = None
    latest_update = VoxelMapUpdate(0, False, tuple(), 0, 0, 0)
    integration_period = float(settings["map"]["integration_period_sec"])
    clear_map_requested = [False]
    panel = _DrivePanel(clear_map_callback=lambda: clear_map_requested.__setitem__(0, True))
    last_action = ""

    try:
        while app.is_running():
            if not world.is_playing():
                drive.apply("STOP", speed_scale=0.0)
                panel.clear_motion()
                app.update()
                continue
            action = panel.action
            drive.apply(action, speed_scale=panel.speed_scale)
            world.step(render=True)
            now = float(world.current_time)
            if action != last_action:
                logger.info("[P1Drive] action=%s speed_scale=%.2f", action, panel.speed_scale)
                last_action = action
            position, orientation = get_world_pose(robot)
            if clear_map_requested[0]:
                local_map.reset(np.asarray(position[:3], dtype=np.float64))
                clear_map_requested[0] = False
                logger.info("[P1Map] cleared at robot position")
            if now - last_integration_time < integration_period:
                frame_index += 1
                continue

            world_from_base = world_from_pose_wxyz(position, orientation)
            captured: dict[str, CameraFrameGeometry] = {}
            for source, camera in cameras.items():
                sample = capture_camera_frame(
                    source=source,
                    camera=camera,
                    frame_index=frame_index,
                    world_from_base=world_from_base,
                )
                if sample is not None:
                    captured[source] = sample
            capture_failures: list[str] = []
            if len(captured) != len(SOURCE_BITS):
                missing = sorted(set(SOURCE_BITS) - set(captured))
                capture_failures.append(f"Waiting for cameras: {', '.join(missing)}")
            else:
                timestamps = [sample.timestamp_sec for sample in captured.values()]
                skew = max(timestamps) - min(timestamps)
                if skew > float(settings["validation"]["max_frame_skew_sec"]):
                    capture_failures.append(f"Camera skew is too large: {skew:.3f}s")

            if not capture_failures:
                observations = [
                    TimedDepthObservation(
                        source=source,
                        timestamp_sec=sample.timestamp_sec,
                        depth_m=sample.depth_m,
                        intrinsics=sample.intrinsics,
                        camera_to_world=sample.world_from_camera,
                    )
                    for source, sample in captured.items()
                ]
                latest_update = local_map.integrate_batch(
                    observations,
                    center_world_xyz=np.asarray(position[:3], dtype=np.float64),
                )
                last_integration_time = now

            if now - last_debug_time >= float(settings["debug"]["update_period_sec"]):
                snapshot = local_map.snapshot()
                failures = capture_failures or _readiness_failures(
                    snapshot=snapshot,
                    update=latest_update,
                    settings=settings,
                )
                robot_pose = (
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    _robot_yaw(world_from_base),
                )
                _save_debug(
                    path=output_path,
                    snapshot=snapshot,
                    robot_pose=robot_pose,
                )
                ready = not failures
                signature = (ready, tuple(failures))
                if signature != last_status_signature:
                    logger.info(
                        "[P1Summary] result=%s updates=%d rays=%d known=%d occupied=%d reason=%s",
                        "READY" if ready else "BUILDING",
                        snapshot.version,
                        latest_update.ray_count,
                        int(np.count_nonzero(snapshot.state != UNKNOWN)),
                        int(np.count_nonzero(snapshot.state == OCCUPIED)),
                        "ok" if ready else failures[0],
                    )
                    last_status_signature = signature
                if latest_update.shifted:
                    logger.info("[P1MapShift] map window moved; overlapping voxel history was retained")
                last_debug_time = now
            frame_index += 1
    finally:
        drive.apply("STOP", speed_scale=0.0)
        panel.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
