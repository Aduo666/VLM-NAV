from __future__ import annotations

from dataclasses import dataclass, fields, replace
import logging
import math
from pathlib import Path
import queue
import sys
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from vln_nav.active_perception.bev_goal_map import BevGoalMap, render_bev_goal_map
from vln_nav.active_perception.camera_geometry import (
    capture_camera_frame,
    world_from_pose_wxyz,
)
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import (
    LocalVoxelMap,
    LocalVoxelMapConfig,
    LocalVoxelSnapshot,
)
from vln_nav.app.agx_trajectory_test import _validate_geometry
from vln_nav.app.agx_local_3d_map_test import (
    _project_local_points,
    _render_local_occupancy,
)
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.fine_path_tracker import FinePathTracker, fit_agx_command
from vln_nav.control.trajectory import (
    AgxVelocityAdapter,
    PathTracker,
    VelocityCommand,
    wheel_targets,
)
from vln_nav.navigation.fine_local_planner import (
    FineLocalPlanner,
    FinePlannerConfig,
)
from vln_nav.navigation.navdp_client import (
    NavDPCandidate,
    NavDPClient,
    NavDPClientConfig,
    NavDPResult,
)
from vln_nav.navigation.odometry import OdomState, SimOdometrySensor
from vln_nav.navigation.trajectory_clearance import (
    FootprintConfig,
    FootprintCollisionChecker,
    assess_path_clearance,
    clear_current_footprint,
    planning_grid_from_snapshot,
)
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
from vln_nav.runtime.scene_lighting import add_indoor_lighting


SETTINGS_PATH = PROJECT_ROOT / "configs/agx_navdp_trajectory_test.yaml"


@dataclass(frozen=True)
class _FrozenCandidate:
    source: NavDPCandidate
    odom_path_xy: np.ndarray
    rank: int | None


@dataclass(frozen=True)
class _InferenceJob:
    request_id: int
    intrinsics: np.ndarray
    rgb: np.ndarray
    depth_m: np.ndarray
    goal_forward_m: float
    goal_left_m: float
    odom_state: OdomState


@dataclass(frozen=True)
class _FinePlanningJob:
    request_id: int
    snapshot: LocalVoxelSnapshot
    start_pose_world: tuple[float, float, float]
    goal_world_xy: tuple[float, float]
    lookahead_m: float
    submitted_sim_time_sec: float
    floor_z_m: float


@dataclass(frozen=True)
class _FineFailureInput:
    occupied_mask: np.ndarray
    unknown_mask: np.ndarray
    origin_xy: tuple[float, float]
    start_pose_world: tuple[float, float, float]
    goal_world_xy: tuple[float, float]

    @classmethod
    def from_job(cls, grid, job: _FinePlanningJob) -> "_FineFailureInput":
        return cls(
            occupied_mask=grid.occupied_mask.copy(),
            unknown_mask=grid.unknown_mask.copy(),
            origin_xy=(grid.origin_x_m, grid.origin_y_m),
            start_pose_world=job.start_pose_world,
            goal_world_xy=job.goal_world_xy,
        )

    def matches(self, grid, start_pose_world, goal_world_xy) -> bool:
        yaw_delta = (
            float(start_pose_world[2]) - self.start_pose_world[2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        return bool(
            self.origin_xy == (grid.origin_x_m, grid.origin_y_m)
            and np.array_equal(self.occupied_mask, grid.occupied_mask)
            and np.array_equal(self.unknown_mask, grid.unknown_mask)
            and np.linalg.norm(
                np.asarray(start_pose_world[:2])
                - np.asarray(self.start_pose_world[:2])
            ) < grid.resolution_m
            and abs(yaw_delta) < math.radians(5.0)
            and np.linalg.norm(
                np.asarray(goal_world_xy) - np.asarray(self.goal_world_xy)
            ) < grid.resolution_m
        )


class _NavDPWorker:
    def __init__(self, client: NavDPClient) -> None:
        self._client = client
        self._initialized = False
        self._busy = False
        self._lock = threading.Lock()
        self._results: queue.Queue = queue.Queue()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, job: _InferenceJob) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return True

    def poll(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def _run(self, job: _InferenceJob) -> None:
        try:
            if not self._initialized:
                self._client.reset(job.intrinsics)
                self._initialized = True
            result = self._client.pointgoal_step(
                rgb=job.rgb,
                depth_m=job.depth_m,
                goal_forward_m=job.goal_forward_m,
                goal_left_m=job.goal_left_m,
            )
            self._results.put((job, result, None))
        except Exception as exc:
            self._results.put((job, None, exc))
        finally:
            with self._lock:
                self._busy = False


class _FinePlanningWorker:
    def __init__(
        self,
        *,
        planner: FineLocalPlanner,
        obstacle_min_height_m: float,
        obstacle_max_height_m: float,
    ) -> None:
        self._planner = planner
        self._obstacle_min_height_m = float(obstacle_min_height_m)
        self._obstacle_max_height_m = float(obstacle_max_height_m)
        self._busy = False
        self._lock = threading.Lock()
        self._results: queue.Queue = queue.Queue()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, job: _FinePlanningJob) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return True

    def poll(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def _run(self, job: _FinePlanningJob) -> None:
        grid = None
        try:
            grid = planning_grid_from_snapshot(
                job.snapshot,
                floor_z_m=job.floor_z_m,
                obstacle_min_height_m=self._obstacle_min_height_m,
                obstacle_max_height_m=self._obstacle_max_height_m,
            )
            grid = clear_current_footprint(
                grid,
                x_m=job.start_pose_world[0],
                y_m=job.start_pose_world[1],
                yaw_rad=job.start_pose_world[2],
                collision_checker=self._planner.collision_checker,
            )
            local_goal = self._planner.select_local_goal(
                grid=grid,
                start_pose_world=job.start_pose_world,
                target_world_xy=job.goal_world_xy,
                lookahead_m=job.lookahead_m,
            )
            plan = self._planner.plan(
                grid=grid,
                start_pose_world=job.start_pose_world,
                goal_world_xy=local_goal,
            )
            self._results.put((job, grid, plan, None))
        except Exception as exc:
            self._results.put((job, grid, None, exc))
        finally:
            with self._lock:
                self._busy = False


class _GoalMapPanel:
    _KEY_ACTIONS = {
        "UP": "FORWARD",
        "DOWN": "BACKWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }

    def __init__(self, ui, *, image_size_px: int, command_sink) -> None:
        import carb.input
        import omni.appwindow

        self._command_sink = command_sink
        self._map: BevGoalMap | None = None
        self._carb_input = carb.input
        self._held_actions: dict[str, str] = {}
        self._press_order: list[str] = []
        size = int(image_size_px)
        initial = np.empty((size, size, 4), dtype=np.uint8)
        initial[:, :, :3] = (108, 110, 112)
        initial[:, :, 3] = 255
        self._provider = ui.ByteImageProvider()
        self._provider.set_data_array(initial.reshape(-1), [size, size])
        self.window = ui.Window(
            "AGX Navigation",
            width=290,
            height=105,
            visible=True,
        )
        with self.window.frame:
            with ui.VStack(spacing=5):
                with ui.HStack(height=36):
                    ui.Button(
                        "COARSE",
                        clicked_fn=lambda: command_sink(("SET_MODE", "COARSE")),
                    )
                    ui.Button(
                        "FINE",
                        clicked_fn=lambda: command_sink(("SET_MODE", "FINE")),
                    )
                ui.Button("STOP", height=34, clicked_fn=self._stop_all)
        self.map_window = ui.Window(
            "Goal Map",
            width=max(size + 20, 600),
            height=max(size + 20, 600),
            visible=True,
        )
        with self.map_window.frame:
            self._image = ui.ImageWithProvider(
                self._provider,
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                pixel_aligned=True,
                opaque_for_mouse_events=True,
            )
            self._image.set_mouse_pressed_fn(self._on_map_pressed)

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
    def manual_action(self) -> str:
        if not self._press_order:
            return "STOP"
        return self._held_actions.get(self._press_order[-1], "STOP")

    def update_map(self, goal_map: BevGoalMap) -> None:
        self._map = goal_map
        height, width = goal_map.rgba.shape[:2]
        self._provider.set_data_array(goal_map.rgba.reshape(-1), [width, height])

    def shutdown(self) -> None:
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_subscription,
            )
            self._keyboard_subscription = None
        self.window.visible = False
        self.map_window.visible = False

    def _stop_all(self) -> None:
        self._held_actions.clear()
        self._press_order.clear()
        self._command_sink("STOP")

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

    def _on_map_pressed(self, x: float, y: float, button: int, _modifiers) -> None:
        if button != 0 or self._map is None:
            return
        width = max(float(self._image.computed_width), 1.0)
        height = max(float(self._image.computed_height), 1.0)
        column = int(
            np.clip(
                (float(x) - float(self._image.screen_position_x)) / width
                * self._map.selectable_mask.shape[1],
                0,
                self._map.selectable_mask.shape[1] - 1,
            )
        )
        row = int(
            np.clip(
                (float(y) - float(self._image.screen_position_y)) / height
                * self._map.selectable_mask.shape[0],
                0,
                self._map.selectable_mask.shape[0] - 1,
            )
        )
        if not bool(self._map.selectable_mask[row, column]):
            self._command_sink(("GOAL_ERROR", "Select known free space"))
            return
        forward, left = self._map.pixel_to_local(column, row)
        self._command_sink(("SELECT_GOAL", forward, left))


class _FineGeometryPanel:
    def __init__(self, ui, *, width: int = 800, height: int = 560) -> None:
        initial = np.zeros((height, width, 4), dtype=np.uint8)
        initial[:, :, 3] = 255
        self._provider = ui.ByteImageProvider()
        self._provider.set_data_array(initial.reshape(-1), [width, height])
        self.window = ui.Window(
            "Fine Geometry",
            width=width + 24,
            height=height + 24,
            visible=True,
        )
        with self.window.frame:
            ui.ImageWithProvider(
                self._provider,
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                pixel_aligned=True,
            )

    def update(self, image: Image.Image) -> None:
        rgba = np.ascontiguousarray(np.asarray(image.convert("RGBA"), dtype=np.uint8))
        height, width = rgba.shape[:2]
        self._provider.set_data_array(rgba.reshape(-1), [width, height])

    def shutdown(self) -> None:
        self.window.visible = False


def _load_settings() -> tuple[
    dict,
    NavDPClientConfig,
    dict,
    LocalVoxelMapConfig,
    FinePlannerConfig,
    FootprintConfig,
]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    expected = {
        "scene",
        "server",
        "camera",
        "goal",
        "execution",
        "coarse_tracking",
        "fine_tracking",
        "manual_control",
        "fine_planner",
        "map_selector",
        "debug",
    }
    if set(data) != expected:
        raise ValueError(f"Navigation modes config requires exactly: {sorted(expected)}")
    scene = data["scene"]
    if not isinstance(scene, dict) or set(scene) != {"usd_path", "add_indoor_lighting"}:
        raise ValueError("Navigation modes scene requires usd_path and add_indoor_lighting")
    if not str(scene["usd_path"]).strip():
        raise ValueError("Navigation modes scene path is empty")
    if not isinstance(scene["add_indoor_lighting"], bool):
        raise ValueError("scene.add_indoor_lighting must be true or false")
    server = NavDPClientConfig.from_mapping(data["server"])

    camera = data["camera"]
    if set(camera) != {"width", "height", "frequency_hz", "warmup_render_frames"}:
        raise ValueError("Invalid navigation modes camera fields")
    if any(int(camera[name]) <= 0 for name in camera):
        raise ValueError("Navigation modes camera values must be positive")

    goal = data["goal"]
    if set(goal) != {
        "min_distance_m",
        "max_distance_m",
    }:
        raise ValueError("Invalid navigation modes goal fields")
    if not 0 < float(goal["min_distance_m"]) < float(goal["max_distance_m"]) <= 10.0:
        raise ValueError("NavDP point-goal distance bounds must satisfy 0 < min < max <= 10")

    execution = data["execution"]
    if set(execution) != {
        "default_mode",
        "coarse_planning_period_sec",
        "fine_planning_period_sec",
        "fine_local_goal_lookahead_m",
        "max_result_age_sec",
        "max_path_join_error_m",
        "min_remaining_path_m",
        "goal_reached_tolerance_m",
    }:
        raise ValueError("Invalid navigation modes execution fields")
    if execution["default_mode"] not in {"COARSE", "FINE"}:
        raise ValueError("Invalid default navigation mode")
    if any(
        float(value) <= 0
        for name, value in execution.items()
        if name != "default_mode"
    ):
        raise ValueError("NavDP execution tolerances must be positive")

    coarse_tracking = data["coarse_tracking"]
    coarse_tracking_fields = {"max_linear_mps", "max_angular_radps"}
    if (
        not isinstance(coarse_tracking, dict)
        or set(coarse_tracking) != coarse_tracking_fields
        or any(float(value) <= 0 for value in coarse_tracking.values())
    ):
        raise ValueError(
            f"coarse_tracking requires positive values for: "
            f"{sorted(coarse_tracking_fields)}"
        )

    fine_tracking = data["fine_tracking"]
    fine_tracking_fields = {
        "max_linear_mps",
        "max_angular_radps",
        "lookahead_m",
        "arrival_distance_m",
        "max_path_error_m",
    }
    if not isinstance(fine_tracking, dict) or set(fine_tracking) != fine_tracking_fields:
        raise ValueError(
            f"fine_tracking requires exactly: {sorted(fine_tracking_fields)}"
        )
    if any(float(value) <= 0 for value in fine_tracking.values()):
        raise ValueError("Fine tracking values must be positive")
    if not (
        float(fine_tracking["arrival_distance_m"])
        < float(fine_tracking["lookahead_m"])
        <= float(fine_tracking["max_path_error_m"])
    ):
        raise ValueError(
            "Fine tracking requires arrival_distance < lookahead <= max_path_error"
        )

    manual_control = data["manual_control"]
    manual_control_fields = {"linear_mps", "angular_radps"}
    if (
        not isinstance(manual_control, dict)
        or set(manual_control) != manual_control_fields
        or any(float(value) <= 0 for value in manual_control.values())
    ):
        raise ValueError(
            f"manual_control requires positive values for: "
            f"{sorted(manual_control_fields)}"
        )
    fine = data["fine_planner"]
    fine_fields = {
        "footprint",
        "obstacle_min_height_m",
        "obstacle_max_height_m",
        *{item.name for item in fields(FinePlannerConfig)},
    }
    if set(fine) != fine_fields:
        raise ValueError(f"fine_planner requires exactly: {sorted(fine_fields)}")
    footprint = FootprintConfig.from_mapping(fine["footprint"])
    if not 0 <= float(fine["obstacle_min_height_m"]) < float(
        fine["obstacle_max_height_m"]
    ):
        raise ValueError("Invalid fine planner obstacle height interval")
    fine_config = FinePlannerConfig.from_mapping({
        item.name: fine[item.name] for item in fields(FinePlannerConfig)
    })
    selector = data["map_selector"]
    selector_fields = {
        "source_config_path",
        "sources",
        "image_size_px",
        "view_size_m",
        "rear_view_m",
        "render_period_sec",
        "obstacle_min_relative_z_m",
        "obstacle_max_relative_z_m",
    }
    if set(selector) != selector_fields:
        raise ValueError(f"NavDP map_selector requires exactly: {sorted(selector_fields)}")
    map_sources = tuple(map(str, selector["sources"]))
    if map_sources != ("front", "down"):
        raise ValueError("Navigation modes map sources must be front and down")
    if int(selector["image_size_px"]) < 128:
        raise ValueError("map_selector.image_size_px must be at least 128")
    if float(selector["view_size_m"]) <= 0 or float(selector["render_period_sec"]) <= 0:
        raise ValueError("map_selector view size and render period must be positive")
    if not 0 <= float(selector["rear_view_m"]) < float(selector["view_size_m"]):
        raise ValueError("map_selector.rear_view_m must be within [0, view_size_m)")
    if float(selector["obstacle_min_relative_z_m"]) >= float(
        selector["obstacle_max_relative_z_m"]
    ):
        raise ValueError("map_selector obstacle height interval is invalid")

    map_settings_path = Path(str(selector["source_config_path"]))
    if not map_settings_path.is_absolute():
        map_settings_path = PROJECT_ROOT / map_settings_path
    if not map_settings_path.is_file():
        raise FileNotFoundError(f"map selector source config does not exist: {map_settings_path}")
    with map_settings_path.open(encoding="utf-8") as handle:
        map_settings = yaml.safe_load(handle) or {}
    required_map_sections = {"sensors", "camera", "map", "validation"}
    if not required_map_sections.issubset(map_settings):
        raise ValueError(
            f"map selector source config requires sections: {sorted(required_map_sections)}"
        )
    map_fields = {item.name for item in fields(LocalVoxelMapConfig)}
    map_values = map_settings["map"]
    if set(map_values) != map_fields | {"integration_period_sec"}:
        raise ValueError("map selector source map fields do not match LocalVoxelMapConfig")
    map_config = LocalVoxelMapConfig(**{name: map_values[name] for name in map_fields})
    map_config.validate()
    if float(map_values["integration_period_sec"]) <= 0:
        raise ValueError("map integration period must be positive")

    debug = data["debug"]
    if set(debug) != {
        "candidate_image_path",
        "fine_image_path",
        "fine_update_period_sec",
        "view_size_m",
    }:
        raise ValueError("Invalid navigation modes debug fields")
    if not str(debug["candidate_image_path"]).strip() or not str(
        debug["fine_image_path"]
    ).strip():
        raise ValueError("Navigation debug path is empty")
    if float(debug["view_size_m"]) <= 0 or float(
        debug["fine_update_period_sec"]
    ) <= 0:
        raise ValueError("Invalid NavDP candidate debug view size")
    return (
        data,
        server,
        map_settings,
        map_config,
        fine_config,
        footprint,
    )


def main() -> int:
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    (
        settings,
        client_config,
        map_settings,
        map_config,
        fine_config,
        footprint,
    ) = _load_settings()
    logger = _make_logger()

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
            return _run(
                app,
                config,
                profile,
                settings,
                client_config,
                map_settings,
                map_config,
                fine_config,
                footprint,
                logger,
            )
        except Exception:
            logger.exception("[NavigationModesFatal] test crashed")
            return 1
    finally:
        app.close()


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("agx_navigation_modes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/agx_navigation_modes_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("[NavigationModes] log=%s", log_path)
    return logger


def _run(
    app,
    config,
    profile,
    settings,
    client_config,
    map_settings,
    map_config,
    fine_config,
    footprint,
    logger,
) -> int:
    import omni.ui as ui
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom

    dt = float(profile.simulation.physics_dt)
    world = World(
        physics_dt=dt,
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    scene_path = Path(str(settings["scene"]["usd_path"]))
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path
    if not scene_path.is_file():
        raise FileNotFoundError(f"Navigation modes scene USD does not exist: {scene_path}")
    add_reference_to_stage(str(scene_path), config.scene.prim_path)
    if config.scene.add_default_ground_plane:
        world.scene.add_default_ground_plane()
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=config.scene.stage_load_timeout_sec,
        logger=logger,
        label="AGX navigation modes scene",
    )
    if settings["scene"]["add_indoor_lighting"]:
        try:
            add_indoor_lighting(
                stage=stage,
                environment_prim_path=config.scene.prim_path,
                logger=logger,
            )
        except Exception:
            logger.exception("[InteriorLighting] failed to add scene lights")
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[
            float(config.scene.robot_start_xy_m[0]),
            float(config.scene.robot_start_xy_m[1]),
            float(profile.spawn_z_m),
        ],
        rotation_deg=[
            float(config.scene.robot_start_orientation_deg[0]),
            float(config.scene.robot_start_orientation_deg[1]),
            float(config.scene.robot_start_orientation_deg[2]) - 90.0,
        ],
    )
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    _validate_geometry(stage, profile, articulation_path)
    robot = world.scene.add(
        SingleArticulation(
            prim_path=articulation_path,
            name="agx_navigation_modes",
            reset_xform_properties=False,
        )
    )
    camera_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=profile.sensors.camera.front_prim_path,
        ),
        kind="NavDP front camera",
    )
    observer_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=profile.sensors.observer_camera_prim_path,
        ),
        kind="NavDP observer camera",
    )
    map_sources = tuple(map(str, settings["map_selector"]["sources"]))
    map_role_paths = {
        source: resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=map_settings["sensors"][f"{source}_prim_path"],
            ),
            kind=f"NavDP goal-map {source} camera",
        )
        for source in map_sources
    }
    if map_role_paths["front"] != camera_path:
        raise RuntimeError(
            "NavDP front camera and goal-map front camera must reference the same prim: "
            f"navdp={camera_path} map={map_role_paths['front']}"
        )
    if len(set(map_role_paths.values())) != len(map_role_paths):
        raise RuntimeError(f"NavDP goal-map camera role collision: {map_role_paths}")
    if not stage.GetPrimAtPath(camera_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"NavDP front prim is not a Camera: {camera_path}")
    if not stage.GetPrimAtPath(observer_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"NavDP observer prim is not a Camera: {observer_path}")
    camera_config = settings["camera"]
    camera = Camera(
        prim_path=camera_path,
        name="agx_navdp_rgbd",
        frequency=int(camera_config["frequency_hz"]),
        resolution=(int(camera_config["width"]), int(camera_config["height"])),
    )
    map_camera_config = map_settings["camera"]
    map_cameras = {"front": camera}
    for source in map_sources:
        if source == "front":
            continue
        path = map_role_paths[source]
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
            raise RuntimeError(f"NavDP goal-map {source} prim is not a Camera: {path}")
        map_cameras[source] = Camera(
            prim_path=path,
            name=f"agx_navdp_goal_map_{source}",
            frequency=int(map_camera_config["frequency_hz"]),
            resolution=(
                int(map_camera_config["width"]),
                int(map_camera_config["height"]),
            ),
        )
    world.reset()
    robot.initialize()
    for map_camera in map_cameras.values():
        map_camera.initialize()
        map_camera.add_distance_to_image_plane_to_frame()
    bind_viewport_to_camera(camera_prim_path=observer_path, logger=logger, enabled=True)
    drive = create_locomotion_controller(
        profile=profile,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=get_joint_names(robot),
        logger=logger,
    )
    adapter = AgxVelocityAdapter(drive, profile.agx_controller.continuous)
    odom = SimOdometrySensor(logger=logger)
    worker = _NavDPWorker(NavDPClient(client_config))
    collision_checker = FootprintCollisionChecker(
        footprint,
        yaw_resolution_deg=fine_config.yaw_resolution_deg,
    )
    fine_planner = FineLocalPlanner(fine_config, collision_checker)
    coarse_tracking = settings["coarse_tracking"]
    coarse_tracking_config = replace(
        profile.agx_controller.continuous,
        max_linear_mps=float(coarse_tracking["max_linear_mps"]),
        max_angular_radps=float(coarse_tracking["max_angular_radps"]),
    )
    fine_tracking = settings["fine_tracking"]
    for curvature in fine_config.curvature_samples_per_m:
        wheel_targets(
            VelocityCommand(
                float(fine_tracking["max_linear_mps"]),
                float(fine_tracking["max_linear_mps"]) * float(curvature),
            ),
            profile.agx_controller.continuous,
        )
    fine_worker = _FinePlanningWorker(
        planner=fine_planner,
        obstacle_min_height_m=float(settings["fine_planner"]["obstacle_min_height_m"]),
        obstacle_max_height_m=float(settings["fine_planner"]["obstacle_max_height_m"]),
    )
    fine_tracking_config = replace(
        profile.agx_controller.continuous,
        max_linear_mps=float(fine_tracking["max_linear_mps"]),
        max_angular_radps=float(fine_tracking["max_angular_radps"]),
        lookahead_m=float(fine_tracking["lookahead_m"]),
        arrival_distance_m=float(fine_tracking["arrival_distance_m"]),
        max_path_error_m=float(fine_tracking["max_path_error_m"]),
    )
    logger.info(
        "[NavigationModes] scene=%s camera=%s server=%s frame=base(+X forward,+Y left)",
        scene_path,
        camera_path,
        client_config.base_url,
    )
    logger.info(
        "[NavDPGoalMap] cameras=%s size=%.1fm resolution=%.2fm role=point_goal_selector",
        map_role_paths,
        float(settings["map_selector"]["view_size_m"]),
        float(map_config.resolution_m),
    )

    command = [None]
    panel = _GoalMapPanel(
        ui,
        image_size_px=int(settings["map_selector"]["image_size_px"]),
        command_sink=lambda value: command.__setitem__(0, value),
    )
    fine_panel = _FineGeometryPanel(ui)

    for _ in range(int(camera_config["warmup_render_frames"])):
        adapter.stop()
        world.step(render=True)

    position, orientation = get_world_pose(robot)
    local_map = LocalVoxelMap(map_config)
    local_map.reset(np.asarray(position[:3], dtype=np.float64))

    tracker = None
    tracker_mode: str | None = None
    display_path_odom_xy: np.ndarray | None = None
    frozen_candidates: list[_FrozenCandidate] = []
    selected_index = 0
    request_id = 0
    accepted_request_id = 0
    fine_request_id = 0
    accepted_fine_request_id = 0
    selected_goal_odom_xy: np.ndarray | None = None
    navigation_active = False
    manual_active = False
    mode = str(settings["execution"]["default_mode"])
    generation_index = 0
    idle_status = "Click known free space to select a goal"
    last_reported_idle_status = None
    previous_tracking_status = None
    previous_adapter_status = None
    last_steering_log_time = -float("inf")
    render_stride = max(1, round(float(profile.simulation.rendering_dt) / dt))
    frame_index = 0
    debug_path = PROJECT_ROOT / settings["debug"]["candidate_image_path"]
    fine_debug_path = PROJECT_ROOT / settings["debug"]["fine_image_path"]
    last_map_integration_time = -float("inf")
    last_map_render_time = -float("inf")
    last_coarse_request_time = -float("inf")
    last_fine_request_time = -float("inf")
    last_fine_debug_time = -float("inf")
    map_integration_period = float(map_settings["map"]["integration_period_sec"])
    map_render_period = float(settings["map_selector"]["render_period_sec"])
    latest_snapshot = local_map.snapshot()
    latest_planning_grid = None
    last_failed_fine_input: _FineFailureInput | None = None
    logger.info("[NavigationMode] selected=%s", mode)

    try:
        while app.is_running():
            if not world.is_playing():
                adapter.stop()
                tracker = None
                navigation_active = False
                command[0] = None
                request_id += 1
                accepted_request_id = request_id
                fine_request_id += 1
                accepted_fine_request_id = fine_request_id
                app.update()
                continue

            position, orientation = get_world_pose(robot)
            state = odom.update(
                world_position_xyz=position,
                world_orientation_wxyz=orientation,
                timestamp_sec=float(world.current_time),
            )
            now = float(world.current_time)
            world_from_base = world_from_pose_wxyz(position, orientation)
            robot_yaw_world = math.atan2(
                float(world_from_base[1, 0]),
                float(world_from_base[0, 0]),
            )
            if now - last_map_integration_time >= map_integration_period:
                captured = {
                    source: sample
                    for source, map_camera in map_cameras.items()
                    if (
                        sample := capture_camera_frame(
                            source=source,
                            camera=map_camera,
                            frame_index=frame_index,
                            world_from_base=world_from_base,
                        )
                    )
                    is not None
                }
                if len(captured) == len(map_cameras):
                    timestamps = [sample.timestamp_sec for sample in captured.values()]
                    if max(timestamps) - min(timestamps) <= float(
                        map_settings["validation"]["max_frame_skew_sec"]
                    ):
                        local_map.integrate_batch(
                            [
                                TimedDepthObservation(
                                    source=source,
                                    timestamp_sec=sample.timestamp_sec,
                                    depth_m=sample.depth_m,
                                    intrinsics=sample.intrinsics,
                                    camera_to_world=sample.world_from_camera,
                                )
                                for source, sample in captured.items()
                            ],
                            center_world_xyz=np.asarray(position[:3], dtype=np.float64),
                        )
                        latest_snapshot = local_map.snapshot()
                        latest_planning_grid = planning_grid_from_snapshot(
                            latest_snapshot,
                            floor_z_m=float(position[2]) - float(profile.spawn_z_m),
                            obstacle_min_height_m=float(
                                settings["fine_planner"]["obstacle_min_height_m"]
                            ),
                            obstacle_max_height_m=float(
                                settings["fine_planner"]["obstacle_max_height_m"]
                            ),
                        )
                        latest_planning_grid = clear_current_footprint(
                            latest_planning_grid,
                            x_m=float(position[0]),
                            y_m=float(position[1]),
                            yaw_rad=float(robot_yaw_world),
                            collision_checker=collision_checker,
                        )
                        last_map_integration_time = now

            if now - last_map_render_time >= map_render_period:
                displayed_goal = (
                    None
                    if selected_goal_odom_xy is None
                    else _odom_goal_to_local(selected_goal_odom_xy, state)
                )
                goal_map = render_bev_goal_map(
                    latest_snapshot,
                    robot_world_xyz=position[:3],
                    robot_yaw_rad=robot_yaw_world,
                    view_size_m=float(settings["map_selector"]["view_size_m"]),
                    rear_view_m=float(settings["map_selector"]["rear_view_m"]),
                    image_size_px=int(settings["map_selector"]["image_size_px"]),
                    obstacle_min_relative_z_m=float(
                        settings["map_selector"]["obstacle_min_relative_z_m"]
                    ),
                    obstacle_max_relative_z_m=float(
                        settings["map_selector"]["obstacle_max_relative_z_m"]
                    ),
                    goal_local_xy=displayed_goal,
                )
                if display_path_odom_xy is not None:
                    path_local = _odom_path_to_local(display_path_odom_xy, state)
                    goal_map = _overlay_path_on_goal_map(
                        goal_map,
                        path_local,
                        color=(34, 177, 76, 255)
                        if tracker_mode == "FINE"
                        else (238, 184, 45, 255),
                    )
                panel.update_map(goal_map)
                last_map_render_time = now

            if (
                mode == "FINE"
                and now - last_fine_debug_time
                >= float(settings["debug"]["fine_update_period_sec"])
            ):
                debug_path_world = (
                    None
                    if display_path_odom_xy is None
                    else _odom_path_to_world(display_path_odom_xy, state)
                )
                debug_goal_world = (
                    None
                    if selected_goal_odom_xy is None
                    else _odom_path_to_world(selected_goal_odom_xy[None, :], state)[0]
                )
                geometry_image = _render_fine_geometry(
                    latest_snapshot,
                    robot_pose=(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(robot_yaw_world),
                    ),
                    path_world_xy=debug_path_world,
                    goal_world_xy=debug_goal_world,
                )
                fine_panel.update(geometry_image)
                _save_image(fine_debug_path, geometry_image)
                last_fine_debug_time = now

            completed = worker.poll()
            if completed is not None:
                job, result, error = completed
                if (
                    job.request_id != accepted_request_id
                    or mode != "COARSE"
                    or not navigation_active
                ):
                    logger.info("[NavDPResult] ignored stale request=%d", job.request_id)
                elif error is not None:
                    idle_status = "NavDP inference failed; see log"
                    logger.error("[NavDPResult] request=%d failed: %s", job.request_id, error)
                else:
                    result_age = now - float(job.odom_state.timestamp_sec)
                    frozen_candidates = _freeze_candidates(result, job)
                    selected_index = result.model_choice_index
                    selected = frozen_candidates[selected_index]
                    joined_path, join_error, remaining_length = _join_path_from_current(
                        selected.odom_path_xy,
                        current_xy=(state.pose.x, state.pose.y),
                    )
                    execution = settings["execution"]
                    rejection = _rolling_result_rejection(
                        result_age_sec=result_age,
                        join_error_m=join_error,
                        remaining_length_m=remaining_length,
                        execution=execution,
                    )
                    if rejection is not None:
                        logger.warning(
                            "[NavDPReject] request=%d reason=%s age=%.3fs "
                            "join_error=%.3fm remaining=%.3fm",
                            job.request_id,
                            rejection,
                            result_age,
                            join_error,
                            remaining_length,
                        )
                    else:
                        _save_candidate_debug(
                            debug_path,
                            result,
                            selected_index=selected_index,
                            goal_xy=(job.goal_forward_m, job.goal_left_m),
                            view_size_m=float(settings["debug"]["view_size_m"]),
                        )
                        tracker = PathTracker(
                            joined_path,
                            coarse_tracking_config,
                            initial_speed_mps=state.linear_velocity_mps,
                        )
                        display_path_odom_xy = joined_path.copy()
                        tracker_mode = mode
                        generation_index += 1
                        previous_tracking_status = None
                        previous_adapter_status = None
                        logger.info(
                            "[CoarseSwitch] generation=%d request=%d candidate=%d "
                            "rank=%s value=%s age=%.3fs join_error=%.3fm "
                            "remaining=%.3fm latency=%.3fs",
                            generation_index,
                            job.request_id,
                            selected.source.index,
                            selected.rank if selected.rank is not None else "N/A",
                            _format_value(selected.source.value),
                            result_age,
                            join_error,
                            remaining_length,
                            result.latency_sec,
                        )
            fine_completed = fine_worker.poll()
            if fine_completed is not None:
                job, grid, plan, error = fine_completed
                if (
                    job.request_id != accepted_fine_request_id
                    or mode != "FINE"
                    or not navigation_active
                ):
                    logger.info("[FineResult] ignored stale request=%d", job.request_id)
                elif error is not None:
                    if grid is not None:
                        last_failed_fine_input = _FineFailureInput.from_job(grid, job)
                    idle_status = "Fine planning failed; inspect Fine Geometry and log"
                    logger.error(
                        "[FineResult] request=%d map_version=%d target=[%.2f,%.2f] failed: %s",
                        job.request_id,
                        job.snapshot.version,
                        job.goal_world_xy[0],
                        job.goal_world_xy[1],
                        error,
                    )
                else:
                    result_age = now - job.submitted_sim_time_sec
                    path_odom = _world_path_to_odom(plan.path_world_xy, state)
                    joined_path = np.ascontiguousarray(path_odom)
                    join_error = float(np.linalg.norm(
                        joined_path[0]
                        - np.asarray([state.pose.x, state.pose.y], dtype=np.float64)
                    ))
                    remaining_length = float(np.sum(
                        np.linalg.norm(np.diff(joined_path, axis=0), axis=1)
                    ))
                    path_odom_yaw = _world_yaw_to_odom(
                        plan.path_world_yaw_rad,
                        state,
                    )
                    clearance = assess_path_clearance(
                        grid,
                        plan.path_world_xy,
                        plan.path_world_yaw_rad,
                        collision_checker=collision_checker,
                    )
                    validation_grid = latest_planning_grid or grid
                    latest_clearance = assess_path_clearance(
                        validation_grid,
                        plan.path_world_xy,
                        plan.path_world_yaw_rad,
                        collision_checker=collision_checker,
                    )
                    rejection = _rolling_result_rejection(
                        result_age_sec=result_age,
                        join_error_m=join_error,
                        remaining_length_m=remaining_length,
                        execution=settings["execution"],
                    )
                    if clearance.collision:
                        rejection = "fine_path_collision"
                    elif latest_clearance.collision:
                        rejection = "fine_path_new_collision"
                    elif clearance.outside_ratio > 0 or latest_clearance.outside_ratio > 0:
                        rejection = "fine_path_outside_map"
                    elif (
                        latest_clearance.unknown_ratio
                        > clearance.unknown_ratio + fine_config.max_near_unknown_ratio
                    ):
                        rejection = "fine_path_new_unknown"
                    elif rejection == "result_too_old":
                        yaw_delta = (
                            robot_yaw_world - job.start_pose_world[2] + math.pi
                        ) % (2.0 * math.pi) - math.pi
                        if (
                            join_error <= validation_grid.resolution_m
                            and abs(yaw_delta) <= math.radians(fine_config.yaw_resolution_deg)
                            and remaining_length
                            >= float(settings["execution"]["min_remaining_path_m"])
                        ):
                            rejection = None
                    if rejection is not None:
                        last_failed_fine_input = (
                            _FineFailureInput.from_job(validation_grid, job)
                            if rejection in {
                                "fine_path_collision",
                                "fine_path_new_collision",
                                "fine_path_outside_map",
                                "fine_path_new_unknown",
                            }
                            else None
                        )
                        logger.warning(
                            "[FineReject] request=%d reason=%s age=%.3fs "
                            "join_error=%.3fm remaining=%.3fm",
                            job.request_id,
                            rejection,
                            result_age,
                            join_error,
                            remaining_length,
                        )
                        collision_report = (
                            clearance
                            if clearance.first_collision_world_xyyaw is not None
                            else latest_clearance
                        )
                        if collision_report.first_collision_world_xyyaw is not None:
                            logger.warning(
                                "[FineCollision] request=%d pose_index=%d "
                                "world_xyyaw=%s cell=%s map_version=%d",
                                job.request_id,
                                collision_report.first_collision_pose_index,
                                collision_report.first_collision_world_xyyaw,
                                collision_report.first_collision_cell,
                                validation_grid.map_version,
                            )
                    else:
                        last_failed_fine_input = None
                        tracker = FinePathTracker(
                            joined_path,
                            path_odom_yaw,
                            plan.path_motion_direction,
                            fine_tracking_config,
                            initial_speed_mps=state.linear_velocity_mps,
                        )
                        display_path_odom_xy = joined_path.copy()
                        tracker_mode = mode
                        generation_index += 1
                        previous_tracking_status = None
                        previous_adapter_status = None
                        logger.info(
                            "[FineSwitch] generation=%d request=%d nodes=%d cost=%.3f "
                            "age=%.3fs join_error=%.3fm remaining=%.3fm "
                            "local_goal=[%.2f,%.2f] clearance=%.3fm unknown=%.3f "
                            "rejected=[collision:%d,unknown:%d,outside:%d]",
                            generation_index,
                            job.request_id,
                            plan.expanded_node_count,
                            plan.path_cost,
                            result_age,
                            join_error,
                            remaining_length,
                            plan.local_goal_world_xy[0],
                            plan.local_goal_world_xy[1],
                            clearance.min_clearance_m,
                            clearance.unknown_ratio,
                            plan.rejected_collision_count,
                            plan.rejected_unknown_count,
                            plan.rejected_outside_count,
                        )

            if command[0] is not None:
                action, command[0] = command[0], None
                if isinstance(action, tuple) and action[0] == "SET_MODE":
                    adapter.stop()
                    tracker = None
                    tracker_mode = None
                    display_path_odom_xy = None
                    navigation_active = False
                    request_id += 1
                    accepted_request_id = request_id
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    mode = str(action[1])
                    last_failed_fine_input = None
                    idle_status = f"MODE: {mode}; click a goal"
                    logger.info("[NavigationMode] selected=%s", mode)
                elif isinstance(action, tuple) and action[0] == "GOAL_ERROR":
                    logger.warning("[NavigationGoal] rejected: %s", action[1])
                elif isinstance(action, tuple) and action[0] == "SELECT_GOAL":
                    if tracker is not None or not _is_stopped(state):
                        idle_status = "Stop the robot before selecting a new goal"
                        logger.warning("[NavigationGoal] Stop the robot before selecting a new goal")
                    else:
                        try:
                            forward, left = _validate_goal_values(
                                float(action[1]), float(action[2]), settings
                            )
                            selected_goal_odom_xy = _transform_xy(
                                np.asarray([[forward, left]], dtype=np.float64),
                                state.pose.x,
                                state.pose.y,
                                state.pose.yaw_rad,
                            )[0]
                            request_id += 1
                            accepted_request_id = request_id
                            fine_request_id += 1
                            accepted_fine_request_id = fine_request_id
                            frozen_candidates = []
                            display_path_odom_xy = None
                            selected_index = 0
                            navigation_active = True
                            last_failed_fine_input = None
                            generation_index = 0
                            last_coarse_request_time = -float("inf")
                            last_fine_request_time = -float("inf")
                            idle_status = f"GOAL ACTIVE: {mode}"
                            logger.info(
                                "[NavigationGoal] mode=%s forward=%.2fm left=%+.2fm "
                                "odom=[%.2f,%.2f]",
                                mode,
                                forward,
                                left,
                                float(selected_goal_odom_xy[0]),
                                float(selected_goal_odom_xy[1]),
                            )
                        except ValueError as exc:
                            logger.warning("[NavigationGoal] rejected: %s", exc)
                            idle_status = "Goal rejected; see log"
                elif action == "STOP":
                    request_id += 1
                    accepted_request_id = request_id
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    adapter.stop()
                    tracker = None
                    tracker_mode = None
                    display_path_odom_xy = None
                    navigation_active = False
                    idle_status = "NAVIGATION STOPPED"
                    logger.info("[NavigationControl] STOP")

            goal_distance = float("inf")
            if selected_goal_odom_xy is not None:
                goal_distance = float(np.linalg.norm(
                    selected_goal_odom_xy
                    - np.asarray([state.pose.x, state.pose.y], dtype=np.float64)
                ))
            if navigation_active and goal_distance <= float(
                settings["execution"]["goal_reached_tolerance_m"]
            ):
                navigation_active = False
                tracker = None
                tracker_mode = None
                adapter.stop()
                idle_status = f"GOAL REACHED: distance={goal_distance:.2f}m"
                logger.info(
                    "[NavigationGoalReached] mode=%s generations=%d distance=%.3fm",
                    mode,
                    generation_index,
                    goal_distance,
                )

            manual_action = panel.manual_action
            if manual_action != "STOP":
                if not manual_active:
                    tracker = None
                    tracker_mode = None
                    navigation_active = False
                    display_path_odom_xy = None
                    request_id += 1
                    accepted_request_id = request_id
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    logger.info("[ManualControl] started action=%s", manual_action)
                manual_active = True
                manual_config = settings["manual_control"]
                manual_commands = {
                    "FORWARD": VelocityCommand(
                        float(manual_config["linear_mps"]), 0.0
                    ),
                    "BACKWARD": VelocityCommand(
                        -float(manual_config["linear_mps"]), 0.0
                    ),
                    "TURN_LEFT": VelocityCommand(
                        0.0, float(manual_config["angular_radps"])
                    ),
                    "TURN_RIGHT": VelocityCommand(
                        0.0, -float(manual_config["angular_radps"])
                    ),
                }
                applied = adapter.apply(manual_commands[manual_action])
            elif manual_active:
                adapter.stop()
                manual_active = False
                idle_status = "MANUAL STOPPED"
                logger.info("[ManualControl] stopped")
            elif tracker is not None:
                tracking_config = (
                    fine_tracking_config
                    if tracker_mode == "FINE"
                    else coarse_tracking_config
                )
                tracking = tracker.step(
                    (state.pose.x, state.pose.y, state.pose.yaw_rad), dt
                )
                requested_command = tracking.command
                if tracker_mode == "COARSE":
                    requested_command = fit_agx_command(
                        requested_command,
                        profile.agx_controller.continuous,
                    )
                applied = adapter.apply(requested_command)
                tracker.feedback(applied)
                steering = adapter.steering_diagnostics()
                if (
                    adapter.status != previous_adapter_status
                    or (
                        adapter.status == "STEERING_TRANSIENT"
                        and now - last_steering_log_time >= 1.0
                    )
                ):
                    logger.info(
                        "[NavigationSteering] mode=%s status=%s max_error_deg=%.2f "
                        "target_deg=%s measured_deg=%s",
                        tracker_mode,
                        adapter.status,
                        math.degrees(float(steering["max_error_rad"])),
                        _degree_map(steering["target_rad"]),
                        _degree_map(steering["measured_rad"]),
                    )
                    previous_adapter_status = adapter.status
                    last_steering_log_time = now
                status_text = (
                    f"{tracker_mode} {tracking.status}/{adapter.status} "
                    f"remaining={tracking.remaining_m:.2f}m error={tracking.cross_track_m:.2f}m\n"
                    f"v={applied.linear_mps:.3f}m/s w={applied.angular_radps:.3f}rad/s "
                    f"limit={tracking_config.max_linear_mps:.2f}m/s"
                )
                if tracking.status != previous_tracking_status:
                    logger.info(
                        "[NavigationTracking] %s", status_text.replace("\n", " ")
                    )
                    previous_tracking_status = tracking.status
                if tracking.status != "TRACKING":
                    adapter.stop()
                    tracker = None
                    tracker_mode = None
                    idle_status = (
                        f"TRACKER {tracking.status}; waiting for rolling replanning"
                    )
            else:
                adapter.stop()

            if (
                navigation_active
                and mode == "COARSE"
                and selected_goal_odom_xy is not None
                and not worker.busy
                and now - last_coarse_request_time
                >= float(settings["execution"]["coarse_planning_period_sec"])
            ):
                try:
                    forward, left = _validate_goal_values(
                        *_odom_goal_to_local(selected_goal_odom_xy, state), settings
                    )
                    rgb, depth = _capture_rgbd(camera)
                    request_id += 1
                    accepted_request_id = request_id
                    job = _InferenceJob(
                        request_id=request_id,
                        intrinsics=np.asarray(
                            camera.get_intrinsics_matrix(), dtype=np.float64
                        ).copy(),
                        rgb=rgb.copy(),
                        depth_m=depth.copy(),
                        goal_forward_m=forward,
                        goal_left_m=left,
                        odom_state=state,
                    )
                    if not worker.submit(job):
                        raise RuntimeError("NavDP worker rejected a concurrent request")
                    last_coarse_request_time = now
                    logger.info(
                        "[NavDPRequest] request=%d goal=[%.2f,%.2f] moving=%s",
                        request_id,
                        forward,
                        left,
                        not _is_stopped(state),
                    )
                except Exception as exc:
                    idle_status = "NavDP request failed; see log"
                    logger.error("[NavDPRequest] rejected: %s", exc)

            if (
                navigation_active
                and mode == "FINE"
                and selected_goal_odom_xy is not None
                and latest_snapshot.version > 0
                and latest_planning_grid is not None
                and not fine_worker.busy
                and now - last_fine_request_time
                >= float(settings["execution"]["fine_planning_period_sec"])
            ):
                goal_world = _odom_path_to_world(
                    selected_goal_odom_xy[None, :], state
                )[0]
                start_pose_world = (
                    float(position[0]),
                    float(position[1]),
                    float(robot_yaw_world),
                )
                target_world_xy = (float(goal_world[0]), float(goal_world[1]))
                failed_input_unchanged = (
                    last_failed_fine_input is not None
                    and last_failed_fine_input.matches(
                        latest_planning_grid, start_pose_world, target_world_xy
                    )
                )
                if failed_input_unchanged:
                    idle_status = "Fine planning input unchanged after failure"
                    if idle_status != last_reported_idle_status:
                        logger.warning("[NavigationStatus] %s", idle_status)
                        last_reported_idle_status = idle_status
                else:
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    job = _FinePlanningJob(
                        request_id=fine_request_id,
                        snapshot=latest_snapshot,
                        start_pose_world=start_pose_world,
                        goal_world_xy=target_world_xy,
                        lookahead_m=float(
                            settings["execution"]["fine_local_goal_lookahead_m"]
                        ),
                        submitted_sim_time_sec=now,
                        floor_z_m=float(position[2]) - float(profile.spawn_z_m),
                    )
                    if fine_worker.submit(job):
                        last_fine_request_time = now
                        logger.info(
                            "[FineRequest] request=%d map_version=%d "
                            "target_world=[%.2f,%.2f]",
                            fine_request_id,
                            latest_snapshot.version,
                            target_world_xy[0],
                            target_world_xy[1],
                        )

            if tracker is None and idle_status != last_reported_idle_status:
                logger.info("[NavigationStatus] %s", idle_status)
                last_reported_idle_status = idle_status
            world.step(render=frame_index % render_stride == 0)
            frame_index += 1
    finally:
        adapter.stop()
        panel.shutdown()
        fine_panel.shutdown()
    return 0


def _capture_rgbd(camera) -> tuple[np.ndarray, np.ndarray]:
    rgba = camera.get_rgba()
    if hasattr(rgba, "detach"):
        rgba = rgba.detach()
    if hasattr(rgba, "cpu"):
        rgba = rgba.cpu()
    if hasattr(rgba, "numpy"):
        rgba = rgba.numpy()
    rgb = np.asarray(rgba)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    frame = camera.get_current_frame()
    depth = frame.get("distance_to_image_plane") if isinstance(frame, dict) else None
    if depth is None:
        raise RuntimeError("front camera has no current depth frame")
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"invalid front RGB-D rgb={rgb.shape} depth={depth.shape}")
    return np.ascontiguousarray(rgb[:, :, :3]), np.ascontiguousarray(depth)


def _validate_goal_values(forward: float, left: float, settings) -> tuple[float, float]:
    forward = float(forward)
    left = float(left)
    goal = settings["goal"]
    distance = math.hypot(forward, left)
    if not float(goal["min_distance_m"]) <= distance <= float(goal["max_distance_m"]):
        raise ValueError(
            f"goal distance must be in [{goal['min_distance_m']}, "
            f"{goal['max_distance_m']}]m"
        )
    return forward, left


def _odom_goal_to_local(goal_odom_xy: np.ndarray, state: OdomState) -> tuple[float, float]:
    delta = np.asarray(goal_odom_xy, dtype=np.float64) - np.asarray(
        [state.pose.x, state.pose.y], dtype=np.float64
    )
    cosine, sine = math.cos(state.pose.yaw_rad), math.sin(state.pose.yaw_rad)
    forward = cosine * delta[0] + sine * delta[1]
    left = -sine * delta[0] + cosine * delta[1]
    return float(forward), float(left)


def _odom_path_to_world(path_odom_xy: np.ndarray, state: OdomState) -> np.ndarray:
    return _transform_xy(
        np.asarray(path_odom_xy, dtype=np.float64),
        state.origin_world_xy[0],
        state.origin_world_xy[1],
        state.origin_world_yaw_rad,
    )


def _odom_path_to_local(path_odom_xy: np.ndarray, state: OdomState) -> np.ndarray:
    points = np.asarray(path_odom_xy, dtype=np.float64)
    delta = points - np.asarray([state.pose.x, state.pose.y], dtype=np.float64)
    cosine = math.cos(state.pose.yaw_rad)
    sine = math.sin(state.pose.yaw_rad)
    return delta @ np.asarray([[cosine, -sine], [sine, cosine]])


def _world_path_to_odom(path_world_xy: np.ndarray, state: OdomState) -> np.ndarray:
    points = np.asarray(path_world_xy, dtype=np.float64)
    delta = points - np.asarray(state.origin_world_xy, dtype=np.float64)
    cosine = math.cos(state.origin_world_yaw_rad)
    sine = math.sin(state.origin_world_yaw_rad)
    return delta @ np.asarray([[cosine, -sine], [sine, cosine]])


def _world_yaw_to_odom(path_world_yaw_rad: np.ndarray, state: OdomState) -> np.ndarray:
    yaw = np.asarray(path_world_yaw_rad, dtype=np.float64).reshape(-1)
    converted = yaw - float(state.origin_world_yaw_rad)
    return np.ascontiguousarray((converted + math.pi) % (2.0 * math.pi) - math.pi)


def _overlay_path_on_goal_map(
    goal_map: BevGoalMap,
    path_local_xy: np.ndarray,
    *,
    color: tuple[int, int, int, int],
) -> BevGoalMap:
    points = np.asarray(path_local_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return goal_map
    height, width = goal_map.selectable_mask.shape
    pixels = []
    for forward, left in points:
        column = (0.5 - float(left) / goal_map.view_size_m) * width
        row = (
            1.0
            - (float(forward) + goal_map.rear_view_m) / goal_map.view_size_m
        ) * height
        if 0 <= column < width and 0 <= row < height:
            pixels.append((int(round(column)), int(round(row))))
    if len(pixels) < 2:
        return goal_map
    image = Image.fromarray(goal_map.rgba, mode="RGBA")
    ImageDraw.Draw(image).line(pixels, fill=color, width=4)
    return BevGoalMap(
        rgba=np.ascontiguousarray(np.asarray(image, dtype=np.uint8)),
        selectable_mask=goal_map.selectable_mask,
        view_size_m=goal_map.view_size_m,
        rear_view_m=goal_map.rear_view_m,
    )


def _render_fine_geometry(
    snapshot: LocalVoxelSnapshot,
    *,
    robot_pose: tuple[float, float, float, float],
    path_world_xy: np.ndarray | None,
    goal_world_xy: np.ndarray | None,
    image_size: tuple[int, int] = (800, 560),
) -> Image.Image:
    image = _render_local_occupancy(
        snapshot,
        robot_pose=robot_pose,
        image_size=image_size,
    )
    draw = ImageDraw.Draw(image)
    robot_z = float(robot_pose[2])
    if path_world_xy is not None and len(path_world_xy) >= 2:
        path = np.asarray(path_world_xy, dtype=np.float64)
        path_xyz = np.column_stack((path, np.full(len(path), robot_z)))
        u, v, _, _ = _project_local_points(
            path_xyz,
            robot_pose=robot_pose,
            image_size=image_size,
        )
        if len(u) >= 2:
            pixels = [(int(x), int(y)) for x, y in zip(u, v)]
            draw.line(pixels, fill=(30, 255, 90), width=5)
    if goal_world_xy is not None:
        goal = np.asarray(
            [[float(goal_world_xy[0]), float(goal_world_xy[1]), robot_z]],
            dtype=np.float64,
        )
        u, v, _, _ = _project_local_points(
            goal,
            robot_pose=robot_pose,
            image_size=image_size,
        )
        if len(u) == 1:
            x, y = int(u[0]), int(v[0])
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(255, 45, 35))
    return image


def _save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _join_path_from_current(
    path_xy: np.ndarray,
    *,
    current_xy: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    points = np.asarray(path_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Rolling trajectory requires at least two XY points")
    current = np.asarray(current_xy, dtype=np.float64)
    distances = np.linalg.norm(points - current, axis=1)
    nearest_index = int(np.argmin(distances))
    join_error = float(distances[nearest_index])
    suffix = points[nearest_index:]
    joined = np.vstack((current[None, :], suffix))
    segment_length = np.linalg.norm(np.diff(joined, axis=0), axis=1)
    joined = joined[np.r_[True, segment_length > 1.0e-5]]
    remaining = float(np.sum(np.linalg.norm(np.diff(joined, axis=0), axis=1)))
    return np.ascontiguousarray(joined), join_error, remaining


def _rolling_result_rejection(
    *,
    result_age_sec: float,
    join_error_m: float,
    remaining_length_m: float,
    execution: dict,
) -> str | None:
    if result_age_sec > float(execution["max_result_age_sec"]):
        return "result_too_old"
    if join_error_m > float(execution["max_path_join_error_m"]):
        return "path_join_error"
    if remaining_length_m < float(execution["min_remaining_path_m"]):
        return "path_too_short"
    return None


def _freeze_candidates(result: NavDPResult, job: _InferenceJob) -> list[_FrozenCandidate]:
    scored = [
        (index, candidate.value)
        for index, candidate in enumerate(result.candidates)
        if candidate.value is not None
    ]
    ranked_indices = [index for index, _ in sorted(scored, key=lambda item: item[1], reverse=True)]
    rank_by_index = {index: rank + 1 for rank, index in enumerate(ranked_indices)}
    frozen: list[_FrozenCandidate] = []
    for index, candidate in enumerate(result.candidates):
        local = np.asarray(candidate.trajectory_xy, dtype=np.float64)
        if float(np.linalg.norm(local[0])) > 0.02:
            local = np.vstack((np.zeros((1, 2), dtype=np.float64), local))
        odom_path = _transform_xy(local, job.odom_state.pose.x, job.odom_state.pose.y,
                                  job.odom_state.pose.yaw_rad)
        frozen.append(
            _FrozenCandidate(
                source=candidate,
                odom_path_xy=odom_path,
                rank=rank_by_index.get(index),
            )
        )
    return frozen


def _transform_xy(points_xy, origin_x: float, origin_y: float, yaw_rad: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return points @ np.asarray([[cosine, sine], [-sine, cosine]]) + [origin_x, origin_y]


def _save_candidate_debug(
    path: Path,
    result: NavDPResult,
    *,
    selected_index: int,
    goal_xy: tuple[float, float],
    view_size_m: float,
) -> None:
    size, header = 820, 76
    image = Image.new("RGB", (size, size + header), (20, 23, 29))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()

    def pixel(point):
        forward, left = map(float, point)
        return (
            int(size / 2 - left / view_size_m * size),
            int(header + size - forward / view_size_m * size),
        )

    for metre in np.arange(0.0, view_size_m + 1e-6, 1.0):
        y = int(header + size - metre / view_size_m * size)
        draw.line((0, y, size, y), fill=(45, 49, 58), width=1)
        draw.text((6, y - 17), f"{metre:.0f}m", fill=(145, 151, 162), font=small)
    draw.line((size // 2, header, size // 2, header + size), fill=(80, 86, 98), width=2)

    values = [candidate.value for candidate in result.candidates if candidate.value is not None]
    value_min = min(values) if values else 0.0
    value_max = max(values) if values else 1.0
    for index, candidate in enumerate(result.candidates):
        points = [pixel(point) for point in candidate.trajectory_xy]
        if candidate.value is None or value_max <= value_min:
            shade = 0.5
        else:
            shade = (candidate.value - value_min) / (value_max - value_min)
        color = (int(100 + 120 * shade), int(85 + 100 * shade), 45)
        width = 2
        if candidate.is_model_choice:
            color, width = (0, 205, 255), 4
        if index == selected_index:
            color, width = (20, 255, 80), 6
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
            end = points[-1]
            draw.ellipse((end[0] - 8, end[1] - 8, end[0] + 8, end[1] + 8), fill=color)
            draw.text((end[0] + 9, end[1] - 9), f"#{index}", fill=(245, 245, 245), font=small)
    robot = pixel((0.0, 0.0))
    draw.polygon(
        [(robot[0], robot[1] - 14), (robot[0] - 10, robot[1] + 10),
         (robot[0] + 10, robot[1] + 10)],
        fill=(255, 210, 30),
    )
    goal = pixel(goal_xy)
    draw.ellipse((goal[0] - 10, goal[1] - 10, goal[0] + 10, goal[1] + 10),
                 fill=(255, 45, 35))
    draw.text((14, 10), "NavDP candidate trajectories (robot frame: forward up, left left)",
              fill=(245, 245, 245), font=font)
    draw.text((14, 38),
              f"green=selected  cyan=model choice  red=point goal  candidates={len(result.candidates)}",
              fill=(205, 210, 218), font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _log_candidate_table(logger, candidates: list[_FrozenCandidate]) -> None:
    for candidate in candidates:
        length = float(np.sum(np.linalg.norm(np.diff(candidate.odom_path_xy, axis=0), axis=1)))
        endpoint = candidate.source.trajectory_xy[-1]
        logger.info(
            "[NavDPCandidate] id=%d rank=%s value=%s model_choice=%s selected_only=%s "
            "points=%d length=%.2fm end_local=[%.2f,%.2f]",
            candidate.source.index,
            candidate.rank,
            _format_value(candidate.source.value),
            candidate.source.is_model_choice,
            candidate.source.server_selected_only,
            len(candidate.odom_path_xy),
            length,
            float(endpoint[0]),
            float(endpoint[1]),
        )


def _format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _degree_map(values: dict[str, float]) -> dict[str, float]:
    return {
        name: round(math.degrees(float(value)), 1)
        for name, value in values.items()
    }


def _is_stopped(state: OdomState) -> bool:
    return state.linear_velocity_mps < 0.03 and abs(state.angular_velocity_radps) < 0.05


if __name__ == "__main__":
    raise SystemExit(main())
