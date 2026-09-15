"""P0 acceptance for AGX fixed-front, downward, and arm RGB-D geometry."""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from vln_nav.active_perception.camera_geometry import (
    CameraFrameGeometry,
    CameraGeometryDiagnostics,
    capture_camera_frame,
    diagnose_camera_frame,
    world_from_pose_wxyz,
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


SETTINGS_PATH = PROJECT_ROOT / "configs/agx_sensor_calibration_test.yaml"
SOURCE_COLORS = {
    "front": (0, 170, 230),
    "down": (245, 145, 25),
    "arm": (215, 55, 180),
}
CALIBRATION_BOXES = (
    ("front_wall", (3.0, 0.0, 1.15), (0.08, 4.0, 2.3)),
    ("left_marker", (1.65, 1.25, 0.55), (0.42, 0.42, 1.10)),
    ("right_marker", (2.10, -1.10, 0.75), (0.42, 0.42, 1.50)),
)


def _load_settings() -> dict:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if set(data) != {"sensors", "camera", "validation", "debug"}:
        raise ValueError("P0 config requires exactly sensors, camera, validation, and debug")
    sensors = data["sensors"]
    if set(sensors) != {"front", "down", "arm", "observer_prim_path"}:
        raise ValueError("P0 sensors must contain front, down, arm, and observer_prim_path")
    paths = []
    for source in SOURCE_COLORS:
        entry = sensors[source]
        required = {"prim_path", "expected_forward_base_xyz", "check_image_left_is_robot_left"}
        if set(entry) != required or not str(entry["prim_path"]).strip():
            raise ValueError(f"Invalid P0 sensor definition: {source}")
        paths.append(str(entry["prim_path"]).strip())
        expected = entry["expected_forward_base_xyz"]
        if expected is not None:
            values = np.asarray(expected, dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all() or np.linalg.norm(values) < 1.0e-6:
                raise ValueError(f"{source}.expected_forward_base_xyz must be null or a finite XYZ vector")
    if len(set(paths)) != len(paths):
        raise ValueError("P0 sensing camera paths must be unique")
    if not str(sensors["observer_prim_path"]).strip():
        raise ValueError("P0 observer_prim_path cannot be empty")

    camera = data["camera"]
    if set(camera) != {
        "width", "height", "frequency_hz", "valid_depth_range_m", "point_stride"
    }:
        raise ValueError("Invalid P0 camera fields")
    depth_range = tuple(map(float, camera["valid_depth_range_m"]))
    if len(depth_range) != 2 or not 0 < depth_range[0] < depth_range[1]:
        raise ValueError("camera.valid_depth_range_m must be ordered and positive")
    for key in ("width", "height", "frequency_hz", "point_stride"):
        if int(camera[key]) < 1:
            raise ValueError(f"camera.{key} must be positive")

    validation = data["validation"]
    required_validation = {
        "warmup_render_frames", "max_frame_age_sec", "max_frame_skew_sec",
        "min_valid_depth_ratio", "min_axis_alignment_cos",
        "minimum_rigid_determinant", "maximum_rigid_determinant",
    }
    if set(validation) != required_validation:
        raise ValueError("Invalid P0 validation fields")
    if int(validation["warmup_render_frames"]) < 1:
        raise ValueError("validation.warmup_render_frames must be positive")
    if not 0 <= float(validation["min_valid_depth_ratio"]) <= 1:
        raise ValueError("validation.min_valid_depth_ratio must be within [0, 1]")
    if not -1 <= float(validation["min_axis_alignment_cos"]) <= 1:
        raise ValueError("validation.min_axis_alignment_cos must be within [-1, 1]")
    if not 0 < float(validation["minimum_rigid_determinant"]) <= float(
        validation["maximum_rigid_determinant"]
    ):
        raise ValueError("Invalid rigid-transform determinant interval")
    for key in ("max_frame_age_sec", "max_frame_skew_sec"):
        if not math.isfinite(float(validation[key])) or float(validation[key]) <= 0:
            raise ValueError(f"validation.{key} must be positive")

    debug = data["debug"]
    if set(debug) != {
        "output_path", "update_period_sec", "forward_range_m", "left_range_m", "height_range_m"
    }:
        raise ValueError("Invalid P0 debug fields")
    for key in ("forward_range_m", "left_range_m", "height_range_m"):
        interval = tuple(map(float, debug[key]))
        if len(interval) != 2 or not interval[0] < interval[1]:
            raise ValueError(f"debug.{key} must be an ordered pair")
    if float(debug["update_period_sec"]) <= 0 or not str(debug["output_path"]).strip():
        raise ValueError("Invalid P0 debug output settings")
    return data


def _evaluate(
    *,
    sample: CameraFrameGeometry,
    diagnostics: CameraGeometryDiagnostics,
    now_sec: float,
    sensor_settings: dict,
    validation: dict,
) -> list[str]:
    failures: list[str] = []
    age = float(now_sec) - float(sample.timestamp_sec)
    if age < -1.0e-4 or age > float(validation["max_frame_age_sec"]):
        failures.append(f"frame_age={age:.3f}s")
    if diagnostics.valid_depth_ratio < float(validation["min_valid_depth_ratio"]):
        failures.append(f"valid_depth={diagnostics.valid_depth_ratio:.4f}")
    determinant = float(diagnostics.rotation_determinant)
    if not float(validation["minimum_rigid_determinant"]) <= determinant <= float(
        validation["maximum_rigid_determinant"]
    ):
        failures.append(f"rotation_det={determinant:.4f}")
    expected = sensor_settings["expected_forward_base_xyz"]
    if expected is not None:
        expected_vector = np.asarray(expected, dtype=np.float64)
        expected_vector /= np.linalg.norm(expected_vector)
        alignment = float(np.dot(expected_vector, diagnostics.optical_forward_base_xyz))
        if alignment < float(validation["min_axis_alignment_cos"]):
            failures.append(f"axis_alignment={alignment:.3f}")
    if bool(sensor_settings["check_image_left_is_robot_left"]):
        if diagnostics.image_left_delta_base_y_m <= 0:
            failures.append(
                f"image_left_maps_to_base_y={diagnostics.image_left_delta_base_y_m:.3f}"
            )
    return failures


def _create_calibration_scene(stage) -> None:
    from pxr import Gf, UsdGeom

    root_path = "/World/P0CalibrationGeometry"
    stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)
    colors = ((0.35, 0.38, 0.42), (0.20, 0.55, 0.25), (0.70, 0.28, 0.20))
    for (name, center, scale), color in zip(CALIBRATION_BOXES, colors):
        cube = UsdGeom.Cube.Define(stage, f"{root_path}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([color])
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _plain_failure(reason: str) -> str:
    """Translate implementation diagnostics into an actionable operator message."""
    if reason == "waiting_frame":
        return "Waiting for an RGB-D frame"
    if reason == "waiting_synchronized_frames":
        return "Waiting for all three cameras"
    if reason.startswith("frame_age="):
        return "Camera stream is not updating"
    if reason.startswith("frame_skew="):
        return "Camera frames are not synchronized"
    if reason.startswith("valid_depth="):
        return "Too little valid depth data"
    if reason.startswith("rotation_det="):
        return "Camera transform is not rigid"
    if reason.startswith("axis_alignment="):
        return "Camera points in the wrong direction"
    if reason.startswith("image_left_maps_to_base_y="):
        return "Image left/right is mirrored"
    return reason


def _camera_description(source: str, diagnostics: CameraGeometryDiagnostics | None) -> str:
    if source == "front":
        return "Navigation view - faces forward"
    if source == "down":
        return "Near-field safety view - faces forward/down"
    if diagnostics is None:
        return "Active-perception view"
    forward = diagnostics.optical_forward_base_xyz
    horizontal = math.degrees(math.atan2(forward[1], forward[0]))
    vertical = math.degrees(math.atan2(forward[2], math.hypot(forward[0], forward[1])))
    return f"Active-perception view - yaw {horizontal:+.0f} deg, pitch {vertical:+.0f} deg"


def _save_debug(
    *,
    path: Path,
    samples: dict[str, CameraFrameGeometry],
    diagnostics: dict[str, CameraGeometryDiagnostics],
    failures: dict[str, list[str]],
    global_failures: list[str],
    settings: dict,
) -> None:
    width, height = 1500, 1020
    canvas = Image.new("RGB", (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    hero_font, title_font, font, small = _font(32), _font(23), _font(16), _font(13)
    overall_pass = len(samples) == 3 and not global_failures and not any(failures.values())
    hero_color = (18, 145, 75) if overall_pass else (205, 55, 42)
    draw.text(
        (20, 10),
        "P0 AUTOMATIC CHECK: READY" if overall_pass else "P0 AUTOMATIC CHECK: NEEDS ATTENTION",
        fill=hero_color,
        font=hero_font,
    )
    draw.text(
        (20, 52),
        "Three RGB-D cameras must be live, synchronized, and aligned in the robot frame.",
        fill=(55, 65, 80),
        font=font,
    )

    image_top, image_width, image_height = 85, 470, 300
    for index, source in enumerate(SOURCE_COLORS):
        left = 15 + index * 495
        sample = samples.get(source)
        source_failures = failures.get(source, ["waiting_frame"])
        source_ready = sample is not None and not source_failures
        border_color = (20, 155, 78) if source_ready else (210, 65, 45)
        draw.rectangle(
            (left, image_top, left + image_width, image_top + image_height),
            outline=border_color,
            width=4,
        )
        if sample is None:
            draw.text((left + 15, image_top + 130), "Waiting for RGB-D", fill=(35, 35, 35), font=font)
        else:
            image = Image.fromarray(sample.rgb)
            image.thumbnail((image_width, image_height))
            canvas.paste(image, (left + (image_width - image.width) // 2, image_top + (image_height - image.height) // 2))
        draw.rectangle((left, image_top, left + 175, image_top + 30), fill=SOURCE_COLORS[source])
        draw.text((left + 8, image_top + 5), f"{source.upper()} CAMERA", fill="white", font=font)
        status_text = "READY" if source_ready else "CHECK"
        status_width = 72 if source_ready else 68
        draw.rectangle(
            (left + image_width - status_width, image_top, left + image_width, image_top + 30),
            fill=border_color,
        )
        draw.text(
            (left + image_width - status_width + 8, image_top + 5),
            status_text,
            fill="white",
            font=font,
        )
        description = _camera_description(source, diagnostics.get(source))
        draw.rectangle(
            (left, image_top + image_height - 31, left + image_width, image_top + image_height),
            fill=(20, 25, 32),
        )
        draw.text((left + 8, image_top + image_height - 25), description, fill="white", font=small)

    map_left, map_top, map_size = 25, 430, 555
    side_left, side_top, side_width, side_height = 620, 430, 520, 330
    forward_range = tuple(map(float, settings["debug"]["forward_range_m"]))
    left_range = tuple(map(float, settings["debug"]["left_range_m"]))
    height_range = tuple(map(float, settings["debug"]["height_range_m"]))
    draw.rectangle((map_left, map_top, map_left + map_size, map_top + map_size), fill="white", outline=(70, 75, 85), width=2)
    draw.text((map_left + 8, map_top + 8), "FUSED DEPTH - TOP VIEW", fill=(20, 35, 55), font=title_font)
    draw.text((map_left + 8, map_top + 38), "Robot faces UP; image left is robot LEFT", fill=(50, 55, 65), font=small)

    def top_pixel(x_value: float, y_value: float) -> tuple[int, int]:
        px = map_left + int((left_range[1] - y_value) / (left_range[1] - left_range[0]) * map_size)
        py = map_top + int((forward_range[1] - x_value) / (forward_range[1] - forward_range[0]) * map_size)
        return px, py

    for meter in np.arange(math.ceil(forward_range[0]), math.floor(forward_range[1]) + 1):
        _, py = top_pixel(float(meter), 0.0)
        draw.line((map_left, py, map_left + map_size, py), fill=(225, 228, 232))
    for meter in np.arange(math.ceil(left_range[0]), math.floor(left_range[1]) + 1):
        px, _ = top_pixel(0.0, float(meter))
        draw.line((px, map_top, px, map_top + map_size), fill=(225, 228, 232))
    for _, center, scale in CALIBRATION_BOXES:
        x, y, _ = center
        length, box_width, _ = scale
        p0 = top_pixel(x - length / 2, y + box_width / 2)
        p1 = top_pixel(x + length / 2, y - box_width / 2)
        draw.rectangle(
            (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])),
            outline=(20, 20, 20),
            width=2,
        )

    draw.rectangle((side_left, side_top, side_left + side_width, side_top + side_height), fill="white", outline=(70, 75, 85), width=2)
    draw.text((side_left + 8, side_top + 8), "FUSED DEPTH - SIDE VIEW", fill=(20, 35, 55), font=title_font)
    draw.text((side_left + 8, side_top + 38), "Robot faces RIGHT; ground is at the bottom", fill=(50, 55, 65), font=small)

    def side_pixel(x_value: float, z_value: float) -> tuple[int, int]:
        px = side_left + int((x_value - forward_range[0]) / (forward_range[1] - forward_range[0]) * side_width)
        py = side_top + int((height_range[1] - z_value) / (height_range[1] - height_range[0]) * side_height)
        return px, py

    for source, item in diagnostics.items():
        points = item.points_base_xyz
        if len(points) > 12000:
            points = points[:: max(1, len(points) // 12000)]
        color = SOURCE_COLORS[source]
        for x_value, y_value, z_value in points:
            if forward_range[0] <= x_value <= forward_range[1] and left_range[0] <= y_value <= left_range[1]:
                px, py = top_pixel(float(x_value), float(y_value))
                draw.point((px, py), fill=color)
            if forward_range[0] <= x_value <= forward_range[1] and height_range[0] <= z_value <= height_range[1]:
                px, py = side_pixel(float(x_value), float(z_value))
                draw.point((px, py), fill=color)
        origin = item.camera_origin_base_xyz
        px, py = top_pixel(origin[0], origin[1])
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="black")
        forward = item.optical_forward_base_xyz
        end = top_pixel(origin[0] + forward[0] * 0.6, origin[1] + forward[1] * 0.6)
        draw.line((px, py, *end), fill=color, width=4)
        px, py = side_pixel(origin[0], origin[2])
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="black")
        end = side_pixel(origin[0] + forward[0] * 0.6, origin[2] + forward[2] * 0.6)
        draw.line((px, py, *end), fill=color, width=4)

    text_left, line_y = 1170, 430
    draw.text((text_left, line_y), "AUTOMATIC RESULT", fill=(20, 35, 55), font=title_font)
    line_y += 40
    for source in SOURCE_COLORS:
        item = diagnostics.get(source)
        source_failures = failures.get(source, ["waiting_frame"])
        ready = item is not None and not source_failures
        status = "READY" if ready else "CHECK"
        status_color = (20, 145, 65) if ready else (205, 45, 35)
        draw.text((text_left, line_y), f"{source.upper()}: {status}", fill=status_color, font=font)
        line_y += 23
        if item is not None:
            draw.text(
                (text_left, line_y),
                f"Depth coverage: {100 * item.valid_depth_ratio:.0f}%",
                fill=(45, 50, 60),
                font=small,
            )
            line_y += 19
        for reason in source_failures[:2]:
            draw.text((text_left, line_y), _plain_failure(str(reason)), fill=(195, 45, 35), font=small)
            line_y += 19
        line_y += 8
    if global_failures:
        draw.text((text_left, line_y), "SYNC: CHECK", fill=(195, 45, 35), font=font)
        line_y += 23
        for reason in global_failures[:5]:
            draw.text((text_left, line_y), _plain_failure(str(reason)), fill=(195, 45, 35), font=small)
            line_y += 19
    else:
        draw.text((text_left, line_y), "SYNC: READY", fill=(20, 145, 65), font=font)
        line_y += 32
    draw.text((text_left, line_y), "VISUAL CHECK", fill=(20, 35, 55), font=title_font)
    line_y += 36
    for message in (
        "1. Each image matches its label.",
        "2. Colored clouds share wall edges.",
        "3. Left and right are not mirrored.",
    ):
        draw.text((text_left, line_y), message, fill=(40, 45, 55), font=small)
        line_y += 23
    draw.text(
        (620, 785),
        "Acceptance: colored depth from different cameras must overlap on the same wall and boxes.",
        fill=(25, 25, 25),
        font=font,
    )
    legend_x, legend_y = 620, 825
    for source, color in SOURCE_COLORS.items():
        draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 16), fill=color)
        draw.text((legend_x + 30, legend_y - 2), source, fill=(25, 25, 25), font=font)
        legend_x += 165
    draw.text(
        (620, 875),
        "Black outlines are known test objects. This is a sensor check, not a navigation map.",
        fill=(25, 25, 25),
        font=font,
    )
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


def main() -> int:
    settings = _load_settings()
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    logger = logging.getLogger("agx_sensor_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/agx_sensor_calibration_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": False,
        "active_gpu": config.simulation.active_gpu,
        "physics_gpu": config.simulation.physics_gpu,
        "multi_gpu": bool(config.simulation.multi_gpu),
    })
    try:
        try:
            return _run(app, config, profile, settings, logger)
        except Exception:
            logger.exception("[P0Fatal] sensor calibration test crashed")
            return 1
    finally:
        app.close()


def _run(app, config, profile, settings, logger) -> int:
    import omni.ui as ui
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
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/P0Light").CreateIntensityAttr(1500.0)
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=config.scene.stage_load_timeout_sec,
        logger=logger,
        label="AGX P0 sensor calibration",
    )
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[0.0, 0.0, profile.spawn_z_m],
        rotation_deg=[0.0, 0.0, 0.0],
    )
    _create_calibration_scene(stage)
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    robot = world.scene.add(SingleArticulation(
        prim_path=articulation_path,
        name="agx_p0_sensor_calibration",
        reset_xform_properties=False,
    ))

    sensor_paths: dict[str, str] = {}
    for source in SOURCE_COLORS:
        requested = resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=settings["sensors"][source]["prim_path"],
        )
        resolved = resolve_existing_prim_path(stage=stage, prim_path=requested, kind=f"P0 {source} camera")
        if not stage.GetPrimAtPath(resolved).IsA(UsdGeom.Camera):
            raise RuntimeError(f"P0 {source} prim is not a Camera: {resolved}")
        sensor_paths[source] = resolved
    if len(set(sensor_paths.values())) != len(sensor_paths):
        raise RuntimeError(f"P0 camera role collision: {sensor_paths}")
    observer_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=settings["sensors"]["observer_prim_path"],
        ),
        kind="P0 observer camera",
    )
    if not stage.GetPrimAtPath(observer_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"P0 observer prim is not a Camera: {observer_path}")

    camera_settings = settings["camera"]
    cameras = {
        source: Camera(
            prim_path=path,
            name=f"agx_p0_{source}_camera",
            frequency=int(camera_settings["frequency_hz"]),
            resolution=(int(camera_settings["width"]), int(camera_settings["height"])),
        )
        for source, path in sensor_paths.items()
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
    for source, path in sensor_paths.items():
        logger.info("[P0Camera] role=%s prim=%s", source, path)
    logger.info("[P0Camera] role=observer prim=%s perception=false", observer_path)

    window = ui.Window("AGX P0 Sensor Check", width=720, height=210)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Robot stays still while FRONT, DOWN, and ARM RGB-D cameras are checked.")
            status_label = ui.Label("WARMING UP CAMERAS...")
            detail_label = ui.Label("Waiting for all three camera streams")
            ui.Label("Open the image below for the visual alignment check:")
            ui.Label(str(PROJECT_ROOT / settings["debug"]["output_path"]))

    warmup = int(settings["validation"]["warmup_render_frames"])
    for _ in range(warmup):
        drive.apply("STOP", speed_scale=0.0)
        world.step(render=True)

    samples: dict[str, CameraFrameGeometry] = {}
    diagnostics: dict[str, CameraGeometryDiagnostics] = {}
    failures = {source: ["waiting_frame"] for source in SOURCE_COLORS}
    global_failures: list[str] = ["waiting_synchronized_frames"]
    frame_index = 0
    last_debug_time = -float("inf")
    last_status_signature = None
    last_timestamps = dict.fromkeys(SOURCE_COLORS, -float("inf"))
    output_path = PROJECT_ROOT / settings["debug"]["output_path"]
    min_depth, max_depth = map(float, camera_settings["valid_depth_range_m"])

    try:
        while app.is_running():
            if not world.is_playing():
                drive.apply("STOP", speed_scale=0.0)
                app.update()
                continue
            drive.apply("STOP", speed_scale=0.0)
            world.step(render=True)
            now = float(world.current_time)
            position, orientation = get_world_pose(robot)
            world_from_base = world_from_pose_wxyz(position, orientation)
            for source, camera in cameras.items():
                try:
                    sample = capture_camera_frame(
                        source=source,
                        camera=camera,
                        frame_index=frame_index,
                        world_from_base=world_from_base,
                    )
                    if sample is None or sample.timestamp_sec <= last_timestamps[source]:
                        continue
                    diagnostic = diagnose_camera_frame(
                        sample,
                        min_depth_m=min_depth,
                        max_depth_m=max_depth,
                        point_stride=int(camera_settings["point_stride"]),
                    )
                    samples[source] = sample
                    diagnostics[source] = diagnostic
                    last_timestamps[source] = sample.timestamp_sec
                except (ValueError, RuntimeError) as exc:
                    failures[source] = [str(exc)]

            # Re-evaluate cached frames every simulation step so a stalled stream
            # becomes stale instead of retaining an earlier PASS indefinitely.
            for source, sample in samples.items():
                failures[source] = _evaluate(
                    sample=sample,
                    diagnostics=diagnostics[source],
                    now_sec=now,
                    sensor_settings=settings["sensors"][source],
                    validation=settings["validation"],
                )
            global_failures = []
            if len(samples) != len(SOURCE_COLORS):
                global_failures.append("waiting_synchronized_frames")
            else:
                timestamps = [samples[source].timestamp_sec for source in SOURCE_COLORS]
                skew = max(timestamps) - min(timestamps)
                if skew > float(settings["validation"]["max_frame_skew_sec"]):
                    global_failures.append(f"frame_skew={skew:.3f}s")
            passed = len(samples) == 3 and not global_failures and not any(failures.values())
            status_label.text = "READY: automatic checks passed" if passed else "CHECK REQUIRED"
            detail_label.text = " | ".join(
                f"{source.upper()}={'READY' if source in samples and not failures[source] else _plain_failure(failures[source][0])}"
                for source in SOURCE_COLORS
            )

            if now - last_debug_time >= float(settings["debug"]["update_period_sec"]):
                _save_debug(
                    path=output_path,
                    samples=samples,
                    diagnostics=diagnostics,
                    failures=failures,
                    global_failures=global_failures,
                    settings=settings,
                )
                last_debug_time = now
            status_signature = (
                passed,
                tuple((source, tuple(failures[source])) for source in SOURCE_COLORS),
                tuple(global_failures),
            )
            if status_signature != last_status_signature:
                logger.info(
                    "[P0Summary] result=%s front=%s down=%s arm=%s sync=%s",
                    "READY" if passed else "CHECK_REQUIRED",
                    "READY" if not failures["front"] else _plain_failure(failures["front"][0]),
                    "READY" if not failures["down"] else _plain_failure(failures["down"][0]),
                    "READY" if not failures["arm"] else _plain_failure(failures["arm"][0]),
                    "READY" if not global_failures else _plain_failure(global_failures[0]),
                )
                for source, item in diagnostics.items():
                    logger.info(
                        "[P0Geometry] source=%s origin_base=%s forward_base=%s image_left_base_y=%.3f "
                        "valid_depth=%.3f points=%d",
                        source,
                        np.round(item.camera_origin_base_xyz, 3).tolist(),
                        np.round(item.optical_forward_base_xyz, 3).tolist(),
                        item.image_left_delta_base_y_m,
                        item.valid_depth_ratio,
                        len(item.points_base_xyz),
                    )
                last_status_signature = status_signature
            frame_index += 1
    finally:
        drive.apply("STOP", speed_scale=0.0)
        window.visible = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
