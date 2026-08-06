from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from ..control.locomotion import create_locomotion_controller
from ..navigation.rgbd_geometry_map import RGBDGeometryMap, RGBDGeometryMapConfig
from ..navigation.semantic_instance_map import (
    SemanticInstanceMap,
    SemanticInstanceMapConfig,
    SemanticMapUpdate,
)
from ..navigation.yoloe_instance_segmenter import (
    InstanceMaskObservation,
    YOLOEInstanceSegmenter,
    YOLOEInstanceSegmenterConfig,
    require_yoloe_dependency,
)
from ..runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, setup_logging
from ..runtime.isaac_runtime_utils import (
    bind_viewport_to_camera,
    find_articulation_root_path,
    get_joint_names,
    get_world_pose,
    resolve_existing_prim_path,
    resolve_robot_relative_prim_path,
    set_local_xform,
    wait_for_stage_assets,
)


@dataclass(frozen=True)
class _StandaloneMappingSettings:
    geometry: RGBDGeometryMapConfig
    segmenter: YOLOEInstanceSegmenterConfig
    semantic_map: SemanticInstanceMapConfig
    semantic_call_period_sec: float
    stopped_settle_sec: float
    visualization_update_period_sec: float
    visualization_panel_size_px: int
    output_path: Path


class _MappingControlPanel:
    _KEY_ACTIONS = {
        "UP": "FORWARD",
        "DOWN": "BACKWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }

    def __init__(self, *, clear_map_callback, output_path: Path) -> None:
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self._carb_input = carb.input
        self._ui = ui
        self._held_actions: dict[str, str] = {}
        self._press_order: list[str] = []
        self._speed_model = ui.SimpleFloatModel(0.65)
        self._window = ui.Window(
            "AGX RGB-D Semantic Mapping Test",
            width=390,
            height=410,
            visible=True,
            dockPreference=ui.DockPreference.LEFT,
        )
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("RGB-D + YOLOE: LiDAR, VLM and navigation are disabled")
                ui.Label("Hold arrow keys or buttons to drive; release to STOP")
                ui.Separator()
                with ui.HStack(height=42):
                    ui.Spacer()
                    self._motion_button("UP", "FORWARD")
                    ui.Spacer()
                with ui.HStack(height=42):
                    self._motion_button("LEFT", "TURN_LEFT")
                    ui.Spacer()
                    self._motion_button("RIGHT", "TURN_RIGHT")
                with ui.HStack(height=42):
                    ui.Spacer()
                    self._motion_button("DOWN", "BACKWARD")
                    ui.Spacer()
                ui.Button("STOP", height=34, clicked_fn=self.clear_motion)
                ui.Button("CLEAR GEOMETRY + SEMANTICS", height=34, clicked_fn=clear_map_callback)
                with ui.HStack(height=30):
                    ui.Label("Drive speed scale", width=135)
                    ui.FloatSlider(model=self._speed_model, min=0.10, max=1.0, step=0.05)
                ui.Separator()
                self._action_label = ui.Label("Action: STOP")
                self._pose_label = ui.Label("Pose: x=0.00 y=0.00 yaw=0.0 deg")
                self._map_label = ui.Label("Map: waiting for depth")
                self._semantic_label = ui.Label("Semantics: waiting for stopped keyframe")
                self._path_label = ui.Label(f"Output: {output_path}")

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

    def update(
        self,
        *,
        robot_position,
        robot_yaw_rad: float,
        map_update,
        semantic_update: SemanticMapUpdate,
    ) -> None:
        self._action_label.text = f"Action: {self.action} speed_scale={self.speed_scale:.2f}"
        self._pose_label.text = (
            f"Pose: x={float(robot_position[0]):.2f} y={float(robot_position[1]):.2f} "
            f"yaw={float(np.degrees(robot_yaw_rad)):.1f} deg"
        )
        self._map_label.text = (
            f"Map: rays={map_update.ray_count} hits={map_update.hit_ray_count} "
            f"occupied={map_update.occupied_cell_count} free={map_update.free_cell_count}"
        )
        self._semantic_label.text = (
            f"Semantics: detections={semantic_update.input_detection_count} "
            f"projected={semantic_update.projected_detection_count} "
            f"instances={semantic_update.total_instance_count}"
        )

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
        self._window = None

    def _motion_button(self, label: str, action: str):
        return self._ui.Button(
            label,
            width=95,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone AGX RGB-D geometry and YOLOE semantic mapping test"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Base YAML config path")
    parser.add_argument(
        "--mapping-config",
        default=str(PROJECT_ROOT / "configs" / "rgbd_mapping.yaml"),
        help="Standalone RGB-D semantic mapping YAML path",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional override for the combined map + first-person debug PNG",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if config.robot.name != "agx" or config.robot.is_g1:
        raise RuntimeError(
            "The standalone RGB-D mapping test currently supports only robot.profile=agx"
        )
    settings = _load_mapping_settings(
        path=Path(args.mapping_config).expanduser().resolve(),
        output_override=str(args.output),
    )
    require_yoloe_dependency()
    config.simulation.headless = False
    logger = setup_logging(config.logging.level)
    _validate_runtime_dependencies()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(_simulation_app_settings(config))
    try:
        return _run(
            simulation_app=simulation_app,
            config=config,
            settings=settings,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.warning("[RGBDMappingExit] interrupted_by_user=1")
        _flush_logger(logger)
        return 130
    except BaseException:
        # SimulationApp.close() can terminate Kit before Python prints an uncaught
        # traceback, so persist the failure before entering the shutdown path.
        logger.exception("[RGBDMappingFatal] standalone mapping startup/run failed")
        _flush_logger(logger)
        raise
    finally:
        logger.info("[RGBDMappingExit] closing_simulation_app=1")
        _flush_logger(logger)
        simulation_app.close()


def _run(*, simulation_app, config, settings: _StandaloneMappingSettings, logger) -> int:
    try:
        from isaacsim.core.prims import SingleArticulation as RobotArticulation
    except ImportError:
        from isaacsim.core.prims import Articulation as RobotArticulation
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera

    world = World(
        physics_dt=float(config.robot.simulation.physics_dt),
        rendering_dt=float(config.robot.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    environment_path = _resolve_project_path(config.scene.usd_path)
    if environment_path is not None:
        if not environment_path.is_file():
            raise FileNotFoundError(f"Configured environment USD does not exist: {environment_path}")
        add_reference_to_stage(str(environment_path), config.scene.prim_path)
        logger.info("Loaded RGB-D mapping environment: %s", environment_path)
        wait_for_stage_assets(
            simulation_app=simulation_app,
            is_stage_loading=is_stage_loading,
            timeout_sec=float(config.scene.stage_load_timeout_sec),
            logger=logger,
            label="RGB-D mapping environment",
        )
    if config.scene.add_default_ground_plane:
        world.scene.add_default_ground_plane()

    add_reference_to_stage(str(config.robot.asset.usd_path), config.robot.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=simulation_app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(config.scene.stage_load_timeout_sec),
        logger=logger,
        label="AGX mapping robot",
    )
    set_local_xform(
        stage=stage,
        prim_path=config.robot.asset.prim_path,
        translation=config.robot_start_position,
        rotation_deg=config.scene.robot_start_orientation_deg,
    )

    articulation_path = (
        resolve_robot_relative_prim_path(
            robot_root_path=config.robot.asset.prim_path,
            prim_path=config.robot.asset.articulation_path,
        )
        if str(config.robot.asset.articulation_path).strip()
        else find_articulation_root_path(stage=stage, root_path=config.robot.asset.prim_path)
    )
    robot = world.scene.add(
        RobotArticulation(
            prim_path=articulation_path,
            name="rgbd_mapping_agx",
            reset_xform_properties=True,
        )
    )
    front_camera_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=config.robot.asset.prim_path,
            prim_path=config.robot.sensors.camera.front_prim_path,
        ),
        kind="RGB-D mapping front camera",
    )
    observer_camera_path = resolve_robot_relative_prim_path(
        robot_root_path=config.robot.asset.prim_path,
        prim_path=config.robot.sensors.camera.observer_prim_path,
    )
    camera = Camera(
        prim_path=front_camera_path,
        frequency=max(1, int(config.robot.sensors.camera.frequency_hz)),
        resolution=(
            int(config.robot.sensors.camera.width),
            int(config.robot.sensors.camera.height),
        ),
    )
    observer_camera = None
    if config.robot.sensors.camera.observer_enabled:
        observer_camera_path = resolve_existing_prim_path(
            stage=stage,
            prim_path=observer_camera_path,
            kind="RGB-D mapping observer camera",
        )
        observer_camera = Camera(
            prim_path=observer_camera_path,
            frequency=max(1, int(config.robot.sensors.camera.frequency_hz)),
            resolution=(
                int(config.robot.sensors.camera.width),
                int(config.robot.sensors.camera.height),
            ),
        )

    logger.warning("[RGBDStartup] phase=world_reset state=begin")
    world.reset()
    logger.warning("[RGBDStartup] phase=world_reset state=complete")
    robot.initialize()
    logger.warning("[RGBDStartup] phase=robot_initialize state=complete prim=%s", articulation_path)
    camera.initialize(attach_rgb_annotator=True)
    logger.warning("[RGBDStartup] phase=front_camera_rgb state=complete prim=%s", front_camera_path)
    _attach_depth_annotator(camera=camera, logger=logger)
    logger.warning("[RGBDStartup] phase=front_camera_depth state=attached")
    if observer_camera is not None:
        observer_camera.initialize(attach_rgb_annotator=True)
        logger.warning(
            "[RGBDStartup] phase=observer_camera_rgb state=complete prim=%s",
            observer_camera_path,
        )
        bind_viewport_to_camera(
            camera_prim_path=observer_camera_path,
            logger=logger,
            enabled=True,
        )
    else:
        bind_viewport_to_camera(
            camera_prim_path=front_camera_path,
            logger=logger,
            enabled=True,
        )

    # Populate the newly attached render products before controller/model setup.
    world.step(render=True)
    _validate_camera_streams(camera=camera)
    logger.warning("[RGBDStartup] phase=camera_stream_validation state=complete")

    logger.warning("[RGBDStartup] phase=controller_initialize state=begin")
    controller = create_locomotion_controller(
        profile=config.robot,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=get_joint_names(robot),
        logger=logger,
    )
    logger.warning("[RGBDStartup] phase=controller_initialize state=complete")
    for _ in range(max(1, int(config.simulation.warmup_steps))):
        controller.apply("STOP", speed_scale=0.0)
        world.step(render=True)

    start_position, start_orientation = get_world_pose(robot)
    floor_z_m = float(config.robot_start_position[2])
    geometry_map = RGBDGeometryMap(
        config=settings.geometry,
        center_world_xy=(float(start_position[0]), float(start_position[1])),
        floor_z_m=floor_z_m,
        logger=logger,
    )
    logger.warning("[RGBDStartup] phase=yoloe_initialize state=begin")
    segmenter = YOLOEInstanceSegmenter(
        config=settings.segmenter,
        project_root=PROJECT_ROOT,
        logger=logger,
    )
    segmenter.require_ready()
    logger.warning("[RGBDStartup] phase=yoloe_initialize state=complete")
    semantic_map = SemanticInstanceMap(config=settings.semantic_map, logger=logger)
    latest_semantic_rgb: np.ndarray | None = None
    latest_observations: list[InstanceMaskObservation] = []
    latest_semantic_frame_time = -float("inf")
    semantic_update = semantic_map.last_update

    def clear_maps() -> None:
        nonlocal latest_semantic_rgb, latest_observations, latest_semantic_frame_time
        nonlocal semantic_update
        geometry_map.clear()
        semantic_map.clear()
        latest_semantic_rgb = None
        latest_observations = []
        latest_semantic_frame_time = -float("inf")
        semantic_update = semantic_map.last_update

    panel = _MappingControlPanel(
        clear_map_callback=clear_maps,
        output_path=settings.output_path,
    )
    logger.warning(
        "[RGBDMapping] READY source=front_camera_depth+isaac_world_pose lidar=disabled "
        "semantic=yoloe stopped_keyframes=1 output=%s floor_z=%.3f labels=%s",
        settings.output_path,
        floor_z_m,
        list(segmenter.labels),
    )

    sim_time_sec = 0.0
    last_log_time = -float("inf")
    last_motion_time = 0.0
    last_semantic_call_time = -float("inf")
    last_visualization_time = -float("inf")
    last_pose = (float(start_position[0]), float(start_position[1]), _yaw_wxyz(start_orientation))
    latest_rgb: np.ndarray | None = None
    try:
        while simulation_app.is_running():
            current_action = panel.action
            controller.apply(current_action, speed_scale=panel.speed_scale)
            world.step(render=True)
            sim_time_sec += float(config.robot.simulation.physics_dt)
            if current_action != "STOP":
                last_motion_time = sim_time_sec

            position, orientation = get_world_pose(robot)
            yaw_rad = _yaw_wxyz(orientation)
            last_pose = (float(position[0]), float(position[1]), yaw_rad)
            depth = _get_depth_image(camera=camera, logger=logger)
            rgb = _get_rgb_image(camera=camera, logger=logger)
            if rgb is not None:
                latest_rgb = rgb
            intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
            view_matrix = np.asarray(camera.get_view_matrix_ros(), dtype=np.float64)
            if depth is not None:
                map_update = geometry_map.update(
                    depth_image=depth,
                    camera_intrinsics=intrinsics,
                    camera_view_matrix_ros=view_matrix,
                    robot_world_xy=(last_pose[0], last_pose[1]),
                    robot_yaw_rad=last_pose[2],
                    timestamp_sec=sim_time_sec,
                )
            else:
                map_update = geometry_map.last_update

            stopped_keyframe_ready = (
                current_action == "STOP"
                and rgb is not None
                and depth is not None
                and sim_time_sec - last_motion_time >= float(settings.stopped_settle_sec)
                and sim_time_sec - last_semantic_call_time
                >= float(settings.semantic_call_period_sec)
            )
            if stopped_keyframe_ready:
                latest_observations = segmenter.predict(rgb)
                latest_semantic_rgb = rgb.copy()
                latest_semantic_frame_time = sim_time_sec
                semantic_update = semantic_map.update(
                    observations=latest_observations,
                    depth_image=depth,
                    camera_intrinsics=intrinsics,
                    camera_view_matrix_ros=view_matrix,
                    timestamp_sec=sim_time_sec,
                )
                last_semantic_call_time = sim_time_sec
                logger.warning(
                    "[YOLOEKeyframe] detections=%s projected=%s created=%s merged=%s "
                    "instances=%s rejected_depth=%s",
                    semantic_update.input_detection_count,
                    semantic_update.projected_detection_count,
                    semantic_update.created_instance_count,
                    semantic_update.merged_instance_count,
                    semantic_update.total_instance_count,
                    semantic_update.rejected_depth_count,
                )

            if (
                latest_rgb is not None
                and sim_time_sec - last_visualization_time
                >= float(settings.visualization_update_period_sec)
            ):
                _save_combined_debug(
                    geometry_map=geometry_map,
                    segmenter=segmenter,
                    semantic_map=semantic_map,
                    current_rgb=latest_rgb,
                    semantic_rgb=latest_semantic_rgb,
                    observations=latest_observations,
                    semantic_frame_age_sec=(
                        sim_time_sec - latest_semantic_frame_time
                        if np.isfinite(latest_semantic_frame_time)
                        else None
                    ),
                    robot_world_xy=(last_pose[0], last_pose[1]),
                    robot_yaw_rad=last_pose[2],
                    panel_size_px=settings.visualization_panel_size_px,
                    output_path=settings.output_path,
                )
                last_visualization_time = sim_time_sec

            panel.update(
                robot_position=position,
                robot_yaw_rad=yaw_rad,
                map_update=map_update,
                semantic_update=semantic_update,
            )
            if sim_time_sec - last_log_time >= 2.0:
                logger.warning(
                    "[RGBDMapping] action=%s pose=[%.2f,%.2f,%.1fdeg] rays=%s hits=%s "
                    "occupied=%s free=%s depth_ready=%s semantic_instances=%s",
                    current_action,
                    last_pose[0],
                    last_pose[1],
                    float(np.degrees(last_pose[2])),
                    map_update.ray_count,
                    map_update.hit_ray_count,
                    map_update.occupied_cell_count,
                    map_update.free_cell_count,
                    int(depth is not None),
                    semantic_update.total_instance_count,
                )
                last_log_time = sim_time_sec
            time.sleep(0.001)
    finally:
        controller.apply("STOP", speed_scale=0.0)
        if latest_rgb is not None:
            _save_combined_debug(
                geometry_map=geometry_map,
                segmenter=segmenter,
                semantic_map=semantic_map,
                current_rgb=latest_rgb,
                semantic_rgb=latest_semantic_rgb,
                observations=latest_observations,
                semantic_frame_age_sec=(
                    sim_time_sec - latest_semantic_frame_time
                    if np.isfinite(latest_semantic_frame_time)
                    else None
                ),
                robot_world_xy=(last_pose[0], last_pose[1]),
                robot_yaw_rad=last_pose[2],
                panel_size_px=settings.visualization_panel_size_px,
                output_path=settings.output_path,
            )
        panel.shutdown()
    return 0


def _simulation_app_settings(config) -> dict:
    settings = {
        "headless": False,
        "width": int(config.simulation.width),
        "height": int(config.simulation.height),
        "multi_gpu": bool(config.simulation.multi_gpu),
    }
    if config.simulation.active_gpu is not None:
        settings["active_gpu"] = int(config.simulation.active_gpu)
    if config.simulation.physics_gpu is not None:
        settings["physics_gpu"] = int(config.simulation.physics_gpu)
    return settings


def _resolve_project_path(path_value: str) -> Path | None:
    value = str(path_value).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_mapping_settings(
    *,
    path: Path,
    output_override: str,
) -> _StandaloneMappingSettings:
    if not path.is_file():
        raise FileNotFoundError(f"RGB-D mapping config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RGB-D mapping config must be a YAML object: {path}")
    allowed_top_level = {"geometry", "semantics", "visualization"}
    unknown_top_level = set(payload) - allowed_top_level
    if unknown_top_level:
        raise ValueError(f"Unknown RGB-D mapping config sections: {sorted(unknown_top_level)}")

    geometry_values = _require_mapping(payload.get("geometry"), "geometry")
    semantics_values = _require_mapping(payload.get("semantics"), "semantics")
    visualization_values = _require_mapping(payload.get("visualization"), "visualization")

    allowed_visualization = {
        "output_path",
        "update_period_sec",
        "crop_margin_m",
        "min_view_size_m",
        "panel_size_px",
    }
    _reject_unknown(visualization_values, allowed_visualization, "visualization")
    output_value = str(output_override).strip() or str(
        visualization_values.get("output_path", "debug/local/rgbd_semantic_map.png")
    )
    output_path = Path(output_value).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()

    geometry_field_names = {field.name for field in fields(RGBDGeometryMapConfig)}
    allowed_geometry = geometry_field_names - {
        "debug_crop_margin_m",
        "debug_min_view_size_m",
        "debug_output_size_px",
    }
    _reject_unknown(geometry_values, allowed_geometry, "geometry")
    geometry = RGBDGeometryMapConfig(
        **geometry_values,
        debug_crop_margin_m=float(visualization_values.get("crop_margin_m", 2.0)),
        debug_min_view_size_m=float(visualization_values.get("min_view_size_m", 12.0)),
        debug_output_size_px=int(visualization_values.get("panel_size_px", 720)),
    )

    segmenter_field_names = {
        field.name for field in fields(YOLOEInstanceSegmenterConfig)
    }
    semantic_map_field_names = {
        field.name for field in fields(SemanticInstanceMapConfig)
    }
    schedule_fields = {"call_period_sec", "stopped_settle_sec"}
    _reject_unknown(
        semantics_values,
        segmenter_field_names | semantic_map_field_names | schedule_fields,
        "semantics",
    )
    segmenter_values = {
        key: value for key, value in semantics_values.items() if key in segmenter_field_names
    }
    if "labels" in segmenter_values:
        labels = segmenter_values["labels"]
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("semantics.labels must be a YAML list of strings")
        segmenter_values["labels"] = tuple(labels)
    segmenter = YOLOEInstanceSegmenterConfig(**segmenter_values)
    semantic_map = SemanticInstanceMapConfig(
        **{
            key: value
            for key, value in semantics_values.items()
            if key in semantic_map_field_names
        }
    )
    return _StandaloneMappingSettings(
        geometry=geometry,
        segmenter=segmenter,
        semantic_map=semantic_map,
        semantic_call_period_sec=max(
            0.1,
            float(semantics_values.get("call_period_sec", 1.0)),
        ),
        stopped_settle_sec=max(
            0.0,
            float(semantics_values.get("stopped_settle_sec", 0.25)),
        ),
        visualization_update_period_sec=max(
            0.1,
            float(visualization_values.get("update_period_sec", 0.50)),
        ),
        visualization_panel_size_px=max(
            320,
            int(visualization_values.get("panel_size_px", 720)),
        ),
        output_path=output_path,
    )


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"RGB-D mapping config section '{name}' must be a YAML object")
    return dict(value)


def _reject_unknown(values: dict, allowed: set[str], section_name: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unknown RGB-D mapping fields in '{section_name}': {sorted(unknown)}"
        )


def _save_combined_debug(
    *,
    geometry_map: RGBDGeometryMap,
    segmenter: YOLOEInstanceSegmenter,
    semantic_map: SemanticInstanceMap,
    current_rgb: np.ndarray,
    semantic_rgb: np.ndarray | None,
    observations: list[InstanceMaskObservation],
    semantic_frame_age_sec: float | None,
    robot_world_xy: tuple[float, float],
    robot_yaw_rad: float,
    panel_size_px: int,
    output_path: Path,
) -> None:
    snapshots = semantic_map.snapshots()
    map_image = geometry_map.render_debug(
        robot_world_xy=robot_world_xy,
        robot_yaw_rad=robot_yaw_rad,
        semantic_instances=snapshots,
    )
    source_rgb = semantic_rgb if semantic_rgb is not None else current_rgb
    overlay_observations = observations if semantic_rgb is not None else []
    camera_image = segmenter.render_overlay(
        rgb_image=source_rgb,
        observations=overlay_observations,
        output_size_px=panel_size_px,
    )
    panel_size = max(320, int(panel_size_px))
    if map_image.size != (panel_size, panel_size):
        map_image = map_image.resize(
            (panel_size, panel_size),
            resample=Image.Resampling.NEAREST,
        )
    header_height = 52
    combined = Image.new(
        "RGB",
        (panel_size * 2, panel_size + header_height),
        color=(20, 23, 28),
    )
    combined.paste(map_image, (0, header_height))
    combined.paste(camera_image, (panel_size, header_height))
    draw = ImageDraw.Draw(combined)
    header_font = _debug_font(18)
    draw.text(
        (16, 14),
        "GLOBAL GEOMETRY + PERSISTENT SEMANTIC INSTANCES",
        fill=(245, 245, 245),
        font=header_font,
    )
    age_text = "waiting for first stopped keyframe"
    if semantic_frame_age_sec is not None:
        age_text = (
            f"LAST YOLOE STOPPED KEYFRAME | age={max(0.0, semantic_frame_age_sec):.1f}s "
            f"| masks={len(observations)}"
        )
    draw.text(
        (panel_size + 16, 14),
        age_text,
        fill=(245, 245, 245),
        font=header_font,
    )
    _save_image_atomic(combined, output_path)


def _save_image_atomic(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}_",
        suffix=output_path.suffix or ".png",
        dir=str(output_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        image.save(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _debug_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=max(10, int(size)))
    except OSError:
        return ImageFont.load_default()


def _attach_depth_annotator(*, camera, logger) -> None:
    method = getattr(camera, "add_distance_to_image_plane_to_frame", None)
    if not callable(method):
        raise RuntimeError(
            "Isaac Sim Camera does not expose add_distance_to_image_plane_to_frame"
        )
    try:
        method()
    except Exception as exc:
        raise RuntimeError(
            "Failed to attach the Isaac Sim distance_to_image_plane annotator "
            f"to the AGX front camera: {exc}"
        ) from exc
    logger.info("Attached RGB-D mapping depth annotator: distance_to_image_plane")


def _validate_camera_streams(*, camera) -> None:
    frame = camera.get_current_frame()
    if not isinstance(frame, dict):
        raise RuntimeError("Front camera returned a non-dictionary frame after initialization")
    missing = [name for name in ("rgb", "distance_to_image_plane") if name not in frame]
    if missing:
        raise RuntimeError(
            "Front RGB-D camera annotators were not registered after a rendered frame: "
            f"missing={missing} available={sorted(frame)}"
        )


def _flush_logger(logger) -> None:
    for handler in logger.handlers:
        try:
            handler.flush()
        except (AttributeError, OSError):
            continue


def _validate_runtime_dependencies() -> None:
    if np.__version__ != "1.26.0":
        raise RuntimeError(
            "Isaac Sim 5.1 requires numpy==1.26.0, but the active environment has "
            f"numpy=={np.__version__}. Reinstall requirements.txt before startup."
        )
    try:
        import filelock
    except ImportError as exc:
        raise RuntimeError(
            "YOLOE requires filelock with AsyncFileLock support. "
            "Reinstall requirements.txt before startup."
        ) from exc
    if not hasattr(filelock, "AsyncFileLock"):
        raise RuntimeError(
            "YOLOE requires filelock>=3.16.1, but the active environment has "
            f"filelock=={filelock.__version__}. Reinstall requirements.txt "
            "before startup."
        )
    try:
        import clip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "YOLOE requires Ultralytics CLIP. Reinstall requirements.txt "
            "before startup."
        ) from exc


def _get_rgb_image(*, camera, logger) -> np.ndarray | None:
    try:
        value = camera.get_rgba()
    except Exception as exc:
        logger.debug("RGB-D mapping color frame unavailable: %s", exc)
        return None
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        return None
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _get_depth_image(*, camera, logger) -> np.ndarray | None:
    try:
        value = camera.get_depth()
    except Exception as exc:
        logger.debug("RGB-D mapping depth frame unavailable: %s", exc)
        return None
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    depth = np.asarray(value, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        return None
    depth[~np.isfinite(depth)] = np.nan
    return depth


def _yaw_wxyz(quaternion_wxyz) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(quaternion[:4]))
    if norm <= 1.0e-9:
        return 0.0
    w, x, y, z = quaternion[:4] / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


if __name__ == "__main__":
    raise SystemExit(main())
