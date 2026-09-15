"""Phase-1 acceptance for AGX arm-mounted active-perception viewpoints."""
from __future__ import annotations

import logging
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from vln_nav.active_perception.arm_observer import ArmPoseController, UrdfArmKinematics
from vln_nav.app.nav import (
    _bind_viewport_to_camera,
    _get_joint_names,
    _resolve_robot_relative_prim_path,
    _set_local_xform,
    _wait_for_stage_assets,
)
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import AgxVelocityAdapter
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from vln_nav.runtime.robot_profile import load_robot_profile


SETTINGS_PATH = PROJECT_ROOT / "configs/agx_active_perception_test.yaml"


def _load_settings() -> dict:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if set(data) != {"camera", "arm", "debug"}:
        raise ValueError("Arm pose acceptance config requires exactly camera, arm, and debug")
    arm = data["arm"]
    required = {
        "urdf_path", "base_link", "tip_link", "joint_names",
        "max_speed_deg_per_sec", "position_tolerance_deg", "command_timeout_sec",
        "stable_duration_sec", "required_fresh_depth_frames", "pose_slots", "pose_offsets_deg",
    }
    if set(arm) != required:
        raise ValueError(f"Arm pose acceptance config requires exactly {sorted(required)}")
    if not arm["pose_offsets_deg"] or "HOME" not in arm["pose_offsets_deg"]:
        raise ValueError("Arm pose library requires HOME")
    if len(arm["joint_names"]) < 1:
        raise ValueError("Arm joint_names cannot be empty")
    slots = tuple(map(str, arm["pose_slots"]))
    if not slots or len(slots) != len(set(slots)) or "HOME" in slots:
        raise ValueError("arm.pose_slots must contain unique non-HOME names")
    unknown_poses = set(arm["pose_offsets_deg"]) - {"HOME", *slots}
    if unknown_poses:
        raise ValueError(f"Saved poses are not declared in arm.pose_slots: {sorted(unknown_poses)}")
    for name, offsets in arm["pose_offsets_deg"].items():
        if len(offsets) != len(arm["joint_names"]) or not np.isfinite(offsets).all():
            raise ValueError(f"Pose {name} does not match the configured arm joints")
    positive = (
        data["camera"]["width"], data["camera"]["height"], data["camera"]["frequency_hz"],
        arm["max_speed_deg_per_sec"], arm["position_tolerance_deg"], arm["command_timeout_sec"],
        arm["stable_duration_sec"], arm["required_fresh_depth_frames"],
        data["debug"]["update_period_sec"],
    )
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
        raise ValueError("Arm pose acceptance numeric settings must be positive")
    depth_range = tuple(map(float, data["camera"]["valid_depth_range_m"]))
    if len(depth_range) != 2 or not 0 < depth_range[0] < depth_range[1]:
        raise ValueError("camera.valid_depth_range_m must be [minimum, maximum]")
    return data


def _save_pose_offset(name: str, positions_rad: np.ndarray, home_rad: np.ndarray) -> list[float]:
    payload = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    slots = tuple(map(str, payload["arm"]["pose_slots"]))
    if name not in slots:
        raise ValueError(f"Cannot save undeclared arm pose slot: {name}")
    offsets = np.rad2deg(np.asarray(positions_rad) - np.asarray(home_rad))
    values = [round(float(value), 3) for value in offsets]
    payload["arm"]["pose_offsets_deg"][name] = values
    temporary = SETTINGS_PATH.with_name(f".{SETTINGS_PATH.name}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
    return values


def _to_uint8_rgb(image) -> np.ndarray | None:
    if image is None:
        return None
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255)
    return array.astype(np.uint8)


def _read_arm_camera(camera, settings: dict) -> tuple[float, np.ndarray | None, np.ndarray | None, float]:
    frame = camera.get_current_frame()
    stamp = float(frame.get("rendering_time", float("nan")))
    depth_value = frame.get("distance_to_image_plane")
    depth = None if depth_value is None else np.asarray(depth_value, dtype=np.float32).copy()
    rgb = _to_uint8_rgb(camera.get_rgba())
    valid_ratio = 0.0
    if depth is not None and depth.ndim == 2 and depth.size:
        minimum, maximum = map(float, settings["camera"]["valid_depth_range_m"])
        valid_ratio = float(np.mean(np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)))
    return stamp, rgb, depth, valid_ratio


def _save_debug(
    output_path: Path,
    *,
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    current_deg: np.ndarray,
    target_deg: np.ndarray | None,
    joint_names: tuple[str, ...],
    state: str,
    active_pose: str | None,
    pose_results: dict[str, str],
    valid_depth_ratio: float,
    depth_range_m: tuple[float, float],
) -> None:
    width, image_height, footer = 1280, 480, 270
    canvas = Image.new("RGB", (width, image_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        mono = ImageFont.truetype("DejaVuSansMono.ttf", 15)
    except OSError:
        title_font = font = mono = ImageFont.load_default()

    if rgb is not None:
        rgb_image = Image.fromarray(rgb)
        rgb_image.thumbnail((width // 2, image_height))
        canvas.paste(rgb_image, ((width // 2 - rgb_image.width) // 2, (image_height - rgb_image.height) // 2))
    else:
        draw.text((20, 220), "Waiting for arm-camera RGB", fill="black", font=font)

    if depth is not None and depth.ndim == 2:
        minimum, maximum = depth_range_m
        valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = np.asarray(
            255.0 * (1.0 - np.clip((depth[valid] - minimum) / (maximum - minimum), 0.0, 1.0)),
            dtype=np.uint8,
        )
        depth_image = Image.fromarray(gray).convert("RGB")
        depth_image.thumbnail((width // 2, image_height))
        x = width // 2 + (width // 2 - depth_image.width) // 2
        canvas.paste(depth_image, (x, (image_height - depth_image.height) // 2))
    else:
        draw.text((width // 2 + 20, 220), "Waiting for arm-camera depth", fill="black", font=font)

    draw.rectangle((0, image_height, width, image_height + footer), fill=(245, 247, 249))
    draw.text((18, image_height + 12), "AGX ARM VIEWPOINT ACCEPTANCE", fill=(15, 38, 60), font=title_font)
    draw.text(
        (18, image_height + 48),
        f"State: {state} | Active pose: {active_pose or '-'} | Valid depth: {100.0 * valid_depth_ratio:.1f}%",
        fill=(20, 20, 20), font=font,
    )
    draw.text(
        (18, image_height + 78),
        "Pose results: " + " | ".join(f"{name}={result}" for name, result in pose_results.items()),
        fill=(20, 20, 20), font=font,
    )
    draw.text((18, image_height + 116), "Joint             current(deg)    target(deg)    error(deg)", fill="black", font=mono)
    for index, (name, current) in enumerate(zip(joint_names, current_deg)):
        target = float("nan") if target_deg is None else float(target_deg[index])
        error = float("nan") if target_deg is None else target - float(current)
        target_text = "-" if not math.isfinite(target) else f"{target:10.2f}"
        error_text = "-" if not math.isfinite(error) else f"{error:9.2f}"
        draw.text(
            (18, image_height + 144 + index * 19),
            f"{name:<16}{float(current):10.2f}    {target_text}    {error_text}",
            fill=(30, 30, 30), font=mono,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    canvas.save(temporary)
    temporary.replace(output_path)


def main() -> int:
    settings = _load_settings()
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    logger = logging.getLogger("agx_arm_pose_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/agx_active_perception_test.log"
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
        return _run(app, config, profile, settings, logger)
    finally:
        app.close()


def _run(app, config, profile, settings, logger) -> int:
    import omni.ui as ui
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux

    dt = float(profile.simulation.physics_dt)
    world = World(physics_dt=dt, rendering_dt=profile.simulation.rendering_dt, stage_units_in_meters=1.0)
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/TestLight").CreateIntensityAttr(1500.0)
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    _wait_for_stage_assets(
        simulation_app=app, is_stage_loading=is_stage_loading,
        timeout_sec=config.scene.stage_load_timeout_sec, logger=logger,
        label="AGX arm pose acceptance",
    )
    _set_local_xform(
        stage=stage, prim_path=profile.asset.prim_path,
        translation=[0, 0, profile.spawn_z_m], rotation_deg=[0, 0, 0],
    )
    articulation_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path, prim_path=profile.asset.articulation_path,
    )
    robot = world.scene.add(SingleArticulation(
        prim_path=articulation_path, name="agx_arm_pose_acceptance", reset_xform_properties=False,
    ))
    arm_camera_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path, prim_path=settings["camera"]["arm_prim_path"],
    )
    if not stage.GetPrimAtPath(arm_camera_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"Configured arm camera does not exist: {arm_camera_path}")
    observer_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path, prim_path=profile.sensors.observer_camera_prim_path,
    )
    if not stage.GetPrimAtPath(observer_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"Configured observer camera does not exist: {observer_path}")
    camera = Camera(
        prim_path=arm_camera_path, name="agx_arm_pose_camera",
        frequency=int(settings["camera"]["frequency_hz"]),
        resolution=(int(settings["camera"]["width"]), int(settings["camera"]["height"])),
    )

    world.reset()
    robot.initialize()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    _bind_viewport_to_camera(camera_prim_path=observer_path, logger=logger, enabled=True)

    joint_names = _get_joint_names(robot)
    drive = create_locomotion_controller(
        profile=profile, robot=robot, stage=stage, articulation_path=articulation_path,
        available_joint_names=joint_names, logger=logger,
    )
    adapter = AgxVelocityAdapter(drive, profile.agx_controller.continuous)
    arm_settings = settings["arm"]
    arm_joint_names = tuple(map(str, arm_settings["joint_names"]))
    drive.release_hold_joints(arm_joint_names)
    arm = ArmPoseController(
        articulation_path=articulation_path, available_joint_names=joint_names,
        joint_names=arm_joint_names,
        max_speed_radps=math.radians(float(arm_settings["max_speed_deg_per_sec"])),
        tolerance_rad=math.radians(float(arm_settings["position_tolerance_deg"])),
    )
    home = arm.current()
    kinematics = UrdfArmKinematics(
        PROJECT_ROOT / arm_settings["urdf_path"],
        base_link=str(arm_settings["base_link"]), tip_link=str(arm_settings["tip_link"]),
        joint_names=arm_joint_names,
    )
    pose_targets = {
        name: home + np.deg2rad(np.asarray(offsets, dtype=np.float64))
        for name, offsets in arm_settings["pose_offsets_deg"].items()
    }
    pose_slots = tuple(map(str, arm_settings["pose_slots"]))
    pose_names = ("HOME", *pose_slots)
    for target in pose_targets.values():
        kinematics.validate_positions(target)
    logger.info("[ArmPoseTest] camera=%s", arm_camera_path)
    logger.info("[ArmPoseTest] home_deg=%s", np.round(np.rad2deg(home), 2).tolist())
    for name, target in pose_targets.items():
        logger.info("[ArmPoseTest] pose=%s target_deg=%s", name, np.round(np.rad2deg(target), 2).tolist())

    joint_limits_deg = np.rad2deg(np.asarray(kinematics.joint_limits_rad(), dtype=np.float64))
    editor_models = [ui.SimpleFloatModel(float(value)) for value in np.rad2deg(home)]
    event: list[tuple[str, str | None] | None] = [None]
    window = ui.Window("AGX Arm Viewpoint Authoring", width=920, height=690)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Phase 1: author and validate arm-camera viewpoints. The base always remains stopped.")
            ui.Label("Editing sliders does not move the arm. Press MOVE EDITOR TARGET after setting all joints.")
            ui.Separator()
            for index, name in enumerate(arm_joint_names):
                with ui.HStack(height=30):
                    ui.Label(name, width=75)
                    ui.FloatSlider(
                        model=editor_models[index],
                        min=float(joint_limits_deg[index, 0]),
                        max=float(joint_limits_deg[index, 1]),
                        step=1.0,
                    )
                    ui.FloatField(model=editor_models[index], width=90)
                    ui.Label(
                        f"limit [{joint_limits_deg[index, 0]:.1f}, {joint_limits_deg[index, 1]:.1f}] deg",
                        width=220,
                    )
            with ui.HStack(height=40):
                ui.Button("LOAD CURRENT INTO EDITOR", clicked_fn=lambda: event.__setitem__(0, ("LOAD_EDITOR", None)))
                ui.Button("MOVE EDITOR TARGET", clicked_fn=lambda: event.__setitem__(0, ("EDITOR", None)))
                ui.Button("HOLD CURRENT / STOP", clicked_fn=lambda: event.__setitem__(0, ("STOP", None)))
            ui.Separator()
            with ui.HStack(height=42):
                for name in pose_names:
                    ui.Button(
                        f"MOVE {name}",
                        clicked_fn=lambda value=name: event.__setitem__(0, ("POSE", value)),
                    )
            with ui.HStack(height=42):
                for name in pose_slots:
                    ui.Button(
                        f"SAVE CURRENT AS {name}",
                        clicked_fn=lambda value=name: event.__setitem__(0, ("SAVE", value)),
                    )
            ui.Label("Switch cameras manually from the Isaac Sim Viewport camera menu.")
            state_label = ui.Label("IDLE: MOVE HOME first, then tune an editor target")
            detail_label = ui.Label("Editor values are absolute joint angles in degrees")
            result_label = ui.Label("Saved pose results: all UNTESTED")

    pose_results = dict.fromkeys(pose_names, "UNTESTED")
    state = "IDLE"
    editor_pose_accepted = False
    active_pose: str | None = None
    target: np.ndarray | None = None
    deadline = settle_started = 0.0
    fresh_depth_frames = 0
    settle_frame_stamp = latest_frame_stamp = -float("inf")
    rgb = depth = None
    valid_depth_ratio = 0.0
    progress_log_time = debug_time = -float("inf")
    frame_index = 0
    render_stride = max(1, round(float(profile.simulation.rendering_dt) / dt))
    tolerance_rad = math.radians(float(arm_settings["position_tolerance_deg"]))
    timeout_sec = float(arm_settings["command_timeout_sec"])
    stable_duration_sec = float(arm_settings["stable_duration_sec"])
    required_frames = int(arm_settings["required_fresh_depth_frames"])
    debug_path = PROJECT_ROOT / settings["debug"]["output_path"]

    def editor_target_rad() -> np.ndarray:
        values = np.deg2rad(np.asarray([model.get_value_as_float() for model in editor_models], dtype=np.float64))
        kinematics.validate_positions(values)
        return values

    def set_pose_result(name: str | None, result: str) -> None:
        if name in pose_results:
            pose_results[str(name)] = result

    try:
        while app.is_running():
            if not world.is_playing():
                adapter.stop()
                app.update()
                continue
            now = float(world.current_time)
            if event[0] is not None:
                action, value = event[0]
                event[0] = None
                if action == "STOP":
                    arm.stop_at_current()
                    set_pose_result(active_pose, "STOPPED")
                    editor_pose_accepted = False
                    active_pose = None
                    target = None
                    state = "IDLE"
                    state_label.text = "IDLE: arm is holding its current measured position"
                    logger.info("[ArmPoseTest] operator stop")
                elif action == "LOAD_EDITOR":
                    if state in {"MOVING", "SETTLING"}:
                        state_label.text = "Stop the active motion before changing editor values"
                    else:
                        for model, value in zip(editor_models, np.rad2deg(arm.current())):
                            model.set_value(float(value))
                        editor_pose_accepted = False
                        state_label.text = "Editor loaded from current measured joint positions"
                elif action == "SAVE" and value is not None:
                    if state in {"MOVING", "SETTLING"}:
                        state_label.text = "Wait for the editor pose to PASS before saving it"
                    elif not editor_pose_accepted:
                        state_label.text = "Move the editor target and wait for PASS before saving a viewpoint"
                    else:
                        captured = arm.current()
                        kinematics.validate_positions(captured)
                        pose_targets[value] = captured.copy()
                        offsets = _save_pose_offset(value, captured, home)
                        pose_results[value] = "SAVED"
                        editor_pose_accepted = False
                        state_label.text = f"Saved current measured pose as {value}"
                        logger.info(
                            "[ArmPoseSaved] pose=%s absolute_deg=%s offset_deg=%s",
                            value,
                            np.round(np.rad2deg(captured), 3).tolist(),
                            offsets,
                        )
                elif action == "EDITOR":
                    if state in {"MOVING", "SETTLING"}:
                        state_label.text = "A pose test is active; stop it before moving the editor target"
                    else:
                        try:
                            target = editor_target_rad()
                        except ValueError as exc:
                            state_label.text = f"Invalid editor target: {exc}"
                            logger.error("[ArmPoseEditor] invalid target: %s", exc)
                            target = None
                            continue
                        arm.command(target)
                        editor_pose_accepted = False
                        active_pose = "EDITOR"
                        state = "MOVING"
                        deadline = now + timeout_sec
                        fresh_depth_frames = 0
                        progress_log_time = -float("inf")
                        state_label.text = "MOVING EDITOR: base is locked"
                        logger.info(
                            "[ArmPoseCommand] pose=EDITOR current_deg=%s target_deg=%s",
                            np.round(np.rad2deg(arm.current()), 2).tolist(),
                            np.round(np.rad2deg(target), 2).tolist(),
                        )
                elif action == "POSE" and value is not None:
                    editor_pose_accepted = False
                    if state in {"MOVING", "SETTLING"}:
                        state_label.text = "A pose test is active; press HOLD CURRENT / STOP before changing pose"
                    elif value not in pose_targets:
                        state_label.text = f"{value} is not authored yet; tune the editor and save it first"
                    else:
                        target = pose_targets[value].copy()
                        kinematics.validate_positions(target)
                        arm.command(target)
                        active_pose = value
                        pose_results[value] = "MOVING"
                        state = "MOVING"
                        deadline = now + timeout_sec
                        fresh_depth_frames = 0
                        progress_log_time = -float("inf")
                        state_label.text = f"MOVING {value}: base is locked"
                        logger.info(
                            "[ArmPoseCommand] pose=%s current_deg=%s target_deg=%s",
                            value, np.round(np.rad2deg(arm.current()), 2).tolist(),
                            np.round(np.rad2deg(target), 2).tolist(),
                        )

            adapter.stop()
            if state == "MOVING" and active_pose is not None and target is not None:
                reached = arm.update(dt)
                error_deg = math.degrees(float(np.max(np.abs(target - arm.current()))))
                detail_label.text = f"{active_pose}: maximum joint error {error_deg:.2f} deg"
                if now >= deadline:
                    arm.stop_at_current()
                    set_pose_result(active_pose, "FAIL_TIMEOUT")
                    state_label.text = f"FAIL {active_pose}: motion timeout at {error_deg:.2f} deg error"
                    logger.error("[ArmPoseResult] pose=%s result=FAIL_TIMEOUT max_error_deg=%.2f", active_pose, error_deg)
                    active_pose = None
                    target = None
                    state = "IDLE"
                elif reached:
                    state = "SETTLING"
                    settle_started = now
                    settle_frame_stamp = latest_frame_stamp
                    fresh_depth_frames = 0
                    state_label.text = f"SETTLING {active_pose}: waiting for stable joints and fresh RGB-D"
                    logger.info("[ArmPoseMotion] pose=%s reached_tolerance", active_pose)
                elif now - progress_log_time >= 1.0:
                    logger.info("[ArmPoseMotion] pose=%s max_error_deg=%.2f", active_pose, error_deg)
                    progress_log_time = now
            elif state == "SETTLING" and active_pose is not None and target is not None:
                error = float(np.max(np.abs(target - arm.current())))
                error_deg = math.degrees(error)
                detail_label.text = f"{active_pose}: error={error_deg:.2f} deg, fresh depth={fresh_depth_frames}/{required_frames}"
                if now >= deadline:
                    set_pose_result(active_pose, "FAIL_UNSTABLE_OR_CAMERA")
                    state_label.text = f"FAIL {active_pose}: pose or arm-camera RGB-D did not stabilize"
                    logger.error(
                        "[ArmPoseResult] pose=%s result=FAIL_UNSTABLE_OR_CAMERA error_deg=%.2f frames=%d",
                        active_pose, error_deg, fresh_depth_frames,
                    )
                    active_pose = None
                    target = None
                    state = "IDLE"
                elif error > 1.5 * tolerance_rad:
                    arm.command(target)
                    state = "MOVING"
                    logger.warning("[ArmPoseMotion] pose=%s drifted error_deg=%.2f; correcting", active_pose, error_deg)
                elif now - settle_started >= stable_duration_sec and fresh_depth_frames >= required_frames:
                    set_pose_result(active_pose, "PASS")
                    if active_pose == "EDITOR":
                        editor_pose_accepted = True
                        state_label.text = "PASS EDITOR: inspect the camera, then save the current pose into a slot"
                    else:
                        state_label.text = f"PASS {active_pose}: inspect view, then test another pose or return HOME"
                    logger.info(
                        "[ArmPoseResult] pose=%s result=PASS error_deg=%.2f valid_depth_ratio=%.3f",
                        active_pose, error_deg, valid_depth_ratio,
                    )
                    active_pose = None
                    target = None
                    state = "IDLE"

            world.step(render=frame_index % render_stride == 0)
            if frame_index % render_stride == 0:
                stamp, next_rgb, next_depth, next_ratio = _read_arm_camera(camera, settings)
                if next_rgb is not None:
                    rgb = next_rgb
                if next_depth is not None:
                    depth = next_depth
                valid_depth_ratio = next_ratio
                if math.isfinite(stamp) and stamp > latest_frame_stamp:
                    latest_frame_stamp = stamp
                    if state == "SETTLING" and stamp > settle_frame_stamp and valid_depth_ratio > 0.001:
                        fresh_depth_frames += 1
                        settle_frame_stamp = stamp

            if now - debug_time >= float(settings["debug"]["update_period_sec"]):
                current = arm.current()
                result_label.text = "Pose results: " + " | ".join(f"{name}={result}" for name, result in pose_results.items())
                display_target_deg = (
                    np.rad2deg(target)
                    if target is not None
                    else np.asarray([model.get_value_as_float() for model in editor_models], dtype=np.float64)
                )
                _save_debug(
                    debug_path, rgb=rgb, depth=depth,
                    current_deg=np.rad2deg(current),
                    target_deg=display_target_deg,
                    joint_names=arm_joint_names, state=state, active_pose=active_pose,
                    pose_results=pose_results, valid_depth_ratio=valid_depth_ratio,
                    depth_range_m=tuple(map(float, settings["camera"]["valid_depth_range_m"])),
                )
                debug_time = now
            frame_index += 1
    finally:
        adapter.stop()
        arm.stop_at_current()
        window.visible = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
