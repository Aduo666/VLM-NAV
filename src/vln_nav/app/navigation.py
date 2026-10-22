from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
import argparse
import json
import logging
import math
from pathlib import Path
import queue
import sys
import threading
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from vln_nav.active_perception.bev_goal_map import BevGoalMap, render_planning_goal_map
from vln_nav.active_perception.camera_geometry import (
    CameraFrameGeometry,
    capture_camera_frame,
    world_from_pose_wxyz,
)
from vln_nav.active_perception.contracts import TimedDepthObservation
from constraint_traverse.rgbd_map import RGBDMapConfig, RGBDPlanarMap
from constraint_traverse.reference_path import ReferencePath
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import (
    AgxVelocityAdapter,
    PathTracker,
    VelocityCommand,
    wheel_targets,
)
from vln_nav.navigation.fine_local_planner import (
    FinePlan,
    FineLocalPlanner,
    FinePlanningConstraint,
    FinePlannerConfig,
    FinePlanningFailure,
    FineUnknownBlockedError,
)
from vln_nav.navigation.navdp_client import (
    NavDPCandidate,
    NavDPClient,
    NavDPClientConfig,
    NavDPResult,
)
from vln_nav.navigation.internnav_s2_client import (
    InternNavS2Client,
    InternNavS2Config,
    InternNavS2Result,
)
from vln_nav.navigation.odometry import OdomState, SimOdometrySensor
from vln_nav.navigation.restricted_space import (
    RestrictedSpaceAnalysis,
    RestrictedSpaceAnalyzer,
)
from vln_nav.navigation.route_corridor import RouteCorridorConfig
from vln_nav.navigation.s2_goal_adapter import (
    S2GoalAdapter,
    S2GoalAdapterConfig,
)
from vln_nav.navigation.trajectory_refiner import (
    TrajectoryRefiner,
    TrajectoryRefinerConfig,
)
from vln_nav.navigation.trajectory_clearance import (
    FootprintConfig,
    FootprintCollisionChecker,
    PlanningGrid,
    assess_known_swept_path,
    clear_current_footprint,
)
from vln_nav.navigation.traversability_graph import (
    TraversabilityGraphBuilder,
    TraversabilityGraphConfig,
)
from vln_nav.runtime.isaac_runtime_utils import (
    bind_viewport_to_camera,
    get_joint_names,
    get_world_pose,
    resolve_existing_prim_path,
    resolve_robot_relative_prim_path,
    set_camera_horizontal_fov,
    set_local_xform,
    wait_for_stage_assets,
)
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.episode import NavigationEpisode, load_episodes, select_episode
from vln_nav.runtime.scene_lighting import add_indoor_lighting
from vln_nav.runtime.agx_geometry import build_agx_footprint, validate_agx_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "configs/navigation.yaml"
EPISODES_PATH = PROJECT_ROOT / "configs/episodes.json"
LOW_LEVEL_MODES = ("COARSE", "FINE")
SAMPLING_MODES = ("MPPI", "KMPPI")


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
class _S2InferenceJob:
    request_id: int
    session_id: str
    instruction: str
    view: str
    reset: bool
    look_down: bool
    frame: CameraFrameGeometry
    submitted_sim_time_sec: float


@dataclass(frozen=True)
class _FinePlanningJob:
    request_id: int
    grid: PlanningGrid
    start_pose_world: tuple[float, float, float]
    goal_world_xy: tuple[float, float]
    route_hint_world_xy: np.ndarray | None
    use_restricted_analysis: bool
    lookahead_m: float
    submitted_sim_time_sec: float
    floor_z_m: float
    recent_world_poses: tuple[tuple[float, float, float], ...] = ()
    initial_speed_mps: float = 0.0
    reaction_time_sec: float = 0.25
    braking_deceleration_mps2: float = 0.60


@dataclass(frozen=True)
class _FinePlanningOutcome:
    plan: FinePlan | None
    analysis: RestrictedSpaceAnalysis | None


@dataclass(frozen=True)
class _FineFailureInput:
    occupied_mask: np.ndarray
    unknown_mask: np.ndarray
    origin_xy: tuple[float, float]
    bounds: tuple[int, int, int, int]
    start_pose_world: tuple[float, float, float]
    goal_world_xy: tuple[float, float]

    @classmethod
    def from_job(
        cls,
        grid,
        job: _FinePlanningJob,
        *,
        relevant_radius_m: float,
    ) -> "_FineFailureInput":
        row, column, _ = grid.world_to_cell(
            job.start_pose_world[0], job.start_pose_world[1]
        )
        radius_cells = int(math.ceil(relevant_radius_m / grid.resolution_m))
        row_start = max(0, row - radius_cells)
        row_stop = min(grid.shape[0], row + radius_cells + 1)
        column_start = max(0, column - radius_cells)
        column_stop = min(grid.shape[1], column + radius_cells + 1)
        bounds = (row_start, row_stop, column_start, column_stop)
        region = np.s_[row_start:row_stop, column_start:column_stop]
        return cls(
            occupied_mask=grid.occupied_mask[region].copy(),
            unknown_mask=grid.unknown_mask[region].copy(),
            origin_xy=(grid.origin_x_m, grid.origin_y_m),
            bounds=bounds,
            start_pose_world=job.start_pose_world,
            goal_world_xy=job.goal_world_xy,
        )

    def matches(self, grid, start_pose_world, goal_world_xy) -> bool:
        yaw_delta = (
            float(start_pose_world[2]) - self.start_pose_world[2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        row_start, row_stop, column_start, column_stop = self.bounds
        region = np.s_[row_start:row_stop, column_start:column_stop]
        return bool(
            self.origin_xy == (grid.origin_x_m, grid.origin_y_m)
            and np.array_equal(self.occupied_mask, grid.occupied_mask[region])
            and np.array_equal(self.unknown_mask, grid.unknown_mask[region])
            and np.linalg.norm(
                np.asarray(start_pose_world[:2])
                - np.asarray(self.start_pose_world[:2])
            ) < grid.resolution_m
            and abs(yaw_delta) < math.radians(5.0)
            and np.linalg.norm(
                np.asarray(goal_world_xy) - np.asarray(self.goal_world_xy)
            ) < grid.resolution_m
        )


def _repeated_partial_path(
    plan: FinePlan,
    *,
    current_world_xy: np.ndarray,
    last_switch_world_xy: np.ndarray | None,
    last_endpoint_world_xy: np.ndarray | None,
    resolution_m: float,
) -> bool:
    if (
        not plan.partial
        or last_switch_world_xy is None
        or last_endpoint_world_xy is None
    ):
        return False
    return bool(
        np.linalg.norm(
            np.asarray(current_world_xy, dtype=np.float64)
            - last_switch_world_xy
        ) < resolution_m
        and np.linalg.norm(
            plan.path_world_xy[-1] - last_endpoint_world_xy
        ) < resolution_m
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

    def warm_up(
        self,
        *,
        intrinsics: np.ndarray,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> NavDPResult:
        with self._lock:
            if self._busy:
                raise RuntimeError("NavDP worker 正忙，无法执行预热")
            self._busy = True
        try:
            self._client.reset(intrinsics)
            try:
                return self._client.pointgoal_step(
                    rgb=rgb,
                    depth_m=depth_m,
                    goal_forward_m=1.0,
                    goal_left_m=0.0,
                )
            finally:
                self._client.reset(intrinsics)
                self._initialized = True
        finally:
            with self._lock:
                self._busy = False

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


class _S2Worker:
    def __init__(self, client: InternNavS2Client) -> None:
        self._client = client
        self._busy = False
        self._lock = threading.Lock()
        self._results: queue.Queue = queue.Queue()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, job: _S2InferenceJob) -> bool:
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

    def _run(self, job: _S2InferenceJob) -> None:
        try:
            if job.reset:
                self._client.reset_session(
                    session_id=job.session_id,
                    instruction=job.instruction,
                )
            result = self._client.infer_session(
                rgb=job.frame.rgb,
                session_id=job.session_id,
                request_id=job.request_id,
                view=job.view,
                look_down=job.look_down,
                timestamp_sec=job.frame.timestamp_sec,
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
        restricted_analyzer: RestrictedSpaceAnalyzer,
        trajectory_refiner: TrajectoryRefiner,
    ) -> None:
        self._planner = planner
        self._restricted_analyzer = restricted_analyzer
        self._trajectory_refiner = trajectory_refiner
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
            grid = job.grid
            grid = clear_current_footprint(
                grid,
                x_m=job.start_pose_world[0],
                y_m=job.start_pose_world[1],
                yaw_rad=job.start_pose_world[2],
                collision_checker=self._planner.collision_checker,
            )
            analysis = None
            constraint = None
            if job.use_restricted_analysis:
                analysis = self._restricted_analyzer.analyze(
                    grid=grid,
                    start_pose_world=job.start_pose_world,
                    target_world_xy=job.goal_world_xy,
                    route_hint_world_xy=job.route_hint_world_xy,
                )
                if not analysis.requires_fine_planning:
                    self._results.put((
                        job,
                        grid,
                        _FinePlanningOutcome(plan=None, analysis=analysis),
                        None,
                    ))
                    return
                if analysis.tube is None:
                    self._results.put((
                        job,
                        grid,
                        _FinePlanningOutcome(plan=None, analysis=analysis),
                        None,
                    ))
                    return
                graph = analysis.graph
                robust_indices = np.flatnonzero(analysis.tube.robust_node_mask)
                robust_poses = np.asarray(
                    [graph.nodes[int(index)].world_pose for index in robust_indices],
                    dtype=np.float64,
                )
                robust_layers = np.asarray(
                    [graph.nodes[int(index)].layer_index for index in robust_indices],
                    dtype=np.int64,
                )
                constraint = FinePlanningConstraint(
                    robust_world_poses=robust_poses,
                    robust_layer_indices=robust_layers,
                    restricted_start_layer=analysis.tube.restricted_start_layer,
                    restricted_end_layer=analysis.tube.restricted_end_layer,
                    position_tolerance_m=max(
                        1.6 * graph.lateral_step_m,
                        1.5 * grid.resolution_m,
                    ),
                    heading_tolerance_deg=1.5 * graph.heading_step_deg,
                    corridor=graph.corridor,
                )
                local_goal = analysis.tube.goal_world_pose[:2]
                local_goal_yaw = analysis.tube.goal_world_pose[2]
            else:
                try:
                    local_goal, reference_route = self._planner.select_local_route(
                        grid=grid,
                        start_pose_world=job.start_pose_world,
                        target_world_xy=job.goal_world_xy,
                        lookahead_m=job.lookahead_m,
                    )
                except RuntimeError as exc:
                    raise FinePlanningFailure(
                        str(exc),
                        reason="LOCAL_GOAL_UNREACHABLE",
                        action_diagnostics=self._planner.inspect_initial_actions(
                            grid=grid,
                            start_pose_world=job.start_pose_world,
                        ),
                    ) from exc
                local_goal_yaw = None
            if job.use_restricted_analysis:
                reference_route = np.vstack((
                    np.asarray(job.start_pose_world[:2], dtype=np.float64),
                    np.asarray(local_goal, dtype=np.float64),
                ))
            plan = self._planner.plan(
                grid=grid,
                start_pose_world=job.start_pose_world,
                goal_world_xy=local_goal,
                goal_world_yaw_rad=local_goal_yaw,
                constraint=constraint,
                reference_path_world_xy=reference_route,
                recent_world_poses=np.asarray(
                    job.recent_world_poses, dtype=np.float64
                ).reshape(-1, 3),
                initial_speed_mps=job.initial_speed_mps,
                reaction_time_sec=job.reaction_time_sec,
                braking_deceleration_mps2=job.braking_deceleration_mps2,
            )
            refined = self._trajectory_refiner.refine(
                grid=grid,
                path_world_xy=plan.path_world_xy,
                path_world_yaw_rad=plan.path_world_yaw_rad,
                motion_direction=plan.path_motion_direction,
                pose_allowed=(None if constraint is None else constraint.allows),
            )
            plan = replace(
                plan,
                path_world_xy=refined.path_world_xy,
                path_world_yaw_rad=refined.path_world_yaw_rad,
                refinement_applied=refined.applied,
                refinement_iterations=refined.accepted_iterations,
            )
            outcome = _FinePlanningOutcome(plan=plan, analysis=analysis)
            self._results.put((job, grid, outcome, None))
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

    def __init__(self, ui, *, image_size_px: int, command_sink, mode="COARSE", controller="MPPI", bind_keyboard=True) -> None:
        import carb.input
        import omni.appwindow

        self._command_sink = command_sink
        self._updating_mode = False
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
        self.controls_window = ui.Window("Navigation Controls", width=340, height=225, visible=True)
        with self.controls_window.frame:
            with ui.VStack(spacing=6):
                ui.Label("路径生成", height=24)
                self._mode_combo = ui.ComboBox(LOW_LEVEL_MODES.index(mode), *LOW_LEVEL_MODES, height=28)
                self._mode_combo.model.add_item_changed_fn(self._on_mode_changed)
                ui.Label("运动控制", height=24)
                self._controller_combo = ui.ComboBox(SAMPLING_MODES.index(controller), *SAMPLING_MODES, height=28)
                self._controller_combo.model.add_item_changed_fn(self._on_controller_changed)
                self._status_label = ui.Label(f"{mode} + {controller}", height=70, word_wrap=True)
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

        self._keyboard_subscription = None
        if bind_keyboard:
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

    def update_controls(self, mode, status, controller="MPPI"):
        self._status_label.text = f"{mode} + {controller}\n{status}"
        self._updating_mode = True
        try:
            for combo, choices, selected in ((self._mode_combo, LOW_LEVEL_MODES, mode),
                                              (self._controller_combo, SAMPLING_MODES, controller)):
                value_model = combo.model.get_item_value_model()
                selected_index = choices.index(selected)
                if value_model.as_int != selected_index:
                    value_model.set_value(selected_index)
        finally:
            self._updating_mode = False

    def _on_mode_changed(self, model, _item):
        if not self._updating_mode:
            self._command_sink(("SET_MODE", LOW_LEVEL_MODES[model.get_item_value_model().as_int]))

    def _on_controller_changed(self, model, _item):
        if not self._updating_mode:
            self._command_sink(("SET_CONTROLLER", SAMPLING_MODES[model.get_item_value_model().as_int]))

    def shutdown(self) -> None:
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_subscription,
            )
            self._keyboard_subscription = None
        self.map_window.visible = False
        self.controls_window.visible = False

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


class _S2DecisionPanel:
    def __init__(self, ui, *, width: int = 640, height: int = 480) -> None:
        initial = np.zeros((height, width, 4), dtype=np.uint8)
        initial[:, :, 3] = 255
        self._provider = ui.ByteImageProvider()
        self._provider.set_data_array(initial.reshape(-1), [width, height])
        self.window = ui.Window(
            "S2 Decision",
            width=width + 20,
            height=height + 20,
            visible=True,
        )
        with self.window.frame:
            ui.ImageWithProvider(
                self._provider,
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
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
    InternNavS2Config,
    S2GoalAdapterConfig,
    dict,
    RGBDMapConfig,
    FinePlannerConfig,
    FootprintConfig,
    RouteCorridorConfig,
    TraversabilityGraphConfig,
    TrajectoryRefinerConfig,
]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    expected = {
        "simulation",
        "scene",
        "server",
        "internnav_s2",
        "camera",
        "goal",
        "execution",
        "coarse_tracking",
        "fine_tracking",
        "sampling_planner",
        "manual_control",
        "fine_planner",
        "restricted_space",
        "trajectory_refiner",
        "map_selector",
        "mapping",
        "debug",
    }
    if set(data) != expected:
        raise ValueError(f"Navigation modes config requires exactly: {sorted(expected)}")
    simulation = data["simulation"]
    if not isinstance(simulation, dict) or set(simulation) != {
        "active_gpu", "physics_gpu", "multi_gpu"
    }:
        raise ValueError("Invalid simulation settings")
    if not isinstance(simulation["multi_gpu"], bool):
        raise ValueError("simulation.multi_gpu must be true or false")
    for name in ("active_gpu", "physics_gpu"):
        if simulation[name] is not None and int(simulation[name]) < 0:
            raise ValueError(f"simulation.{name} must be null or nonnegative")

    scene = data["scene"]
    scene_fields = {
        "prim_path", "add_default_ground_plane", "stage_load_timeout_sec",
        "add_indoor_lighting",
    }
    if not isinstance(scene, dict) or set(scene) != scene_fields:
        raise ValueError(f"Navigation scene requires exactly: {sorted(scene_fields)}")
    if not str(scene["prim_path"]).startswith("/"):
        raise ValueError("scene.prim_path must be an absolute USD path")
    if float(scene["stage_load_timeout_sec"]) <= 0:
        raise ValueError("scene.stage_load_timeout_sec must be positive")
    for name in ("add_default_ground_plane", "add_indoor_lighting"):
        if not isinstance(scene[name], bool):
            raise ValueError(f"scene.{name} must be true or false")
    server = NavDPClientConfig.from_mapping(data["server"])
    s2 = data["internnav_s2"]
    s2_fields = {
        "enabled",
        "server",
        "goal_adapter",
        "forward_action_distance_m",
        "goal_reached_tolerance_m",
        "turn_action_deg",
        "turn_redecision_deg",
        "turn_tolerance_deg",
        "turn_gain",
        "turn_max_angular_radps",
        "stop_confirmations",
        "lookdown_settle_frames",
    }
    if not isinstance(s2, dict) or set(s2) != s2_fields:
        raise ValueError(f"internnav_s2 必须包含这些字段：{sorted(s2_fields)}")
    if not isinstance(s2["enabled"], bool):
        raise ValueError("internnav_s2.enabled 必须为布尔值")
    for name in (
        "forward_action_distance_m",
        "goal_reached_tolerance_m",
        "turn_action_deg",
        "turn_redecision_deg",
        "turn_tolerance_deg",
        "turn_gain",
        "turn_max_angular_radps",
    ):
        if not math.isfinite(float(s2[name])) or float(s2[name]) <= 0:
            raise ValueError(f"internnav_s2.{name} 必须为有限正数")
    turn_actions_per_request = (
        float(s2["turn_redecision_deg"]) / float(s2["turn_action_deg"])
    )
    if (
        turn_actions_per_request < 1.0
        or not math.isclose(
            turn_actions_per_request,
            round(turn_actions_per_request),
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("S2 重新请求角度必须是单次转向角度的整数倍")
    if int(s2["stop_confirmations"]) <= 0 or int(s2["lookdown_settle_frames"]) < 0:
        raise ValueError("S2 stop_confirmations/lookdown_settle_frames 无效")
    s2_client_config = InternNavS2Config.from_mapping(s2["server"])
    s2_goal_config = S2GoalAdapterConfig.from_mapping(s2["goal_adapter"])

    camera = data["camera"]
    if set(camera) != {
        "width", "height", "frequency_hz", "warmup_render_frames",
        "horizontal_fov_deg",
    }:
        raise ValueError("Invalid navigation modes camera fields")
    if any(
        int(camera[name]) <= 0
        for name in ("width", "height", "frequency_hz", "warmup_render_frames")
    ):
        raise ValueError("Navigation modes camera values must be positive")
    if not 0.0 < float(camera["horizontal_fov_deg"]) < 180.0:
        raise ValueError("camera.horizontal_fov_deg 必须位于 0 至 180 度之间")

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
        "controller",
        "coarse_planning_period_sec",
        "fine_planning_period_sec",
        "fine_local_goal_lookahead_m",
        "max_result_age_sec",
        "max_path_join_error_m",
        "min_remaining_path_m",
        "goal_reached_tolerance_m",
    }:
        raise ValueError("Invalid navigation modes execution fields")
    if execution["default_mode"] not in LOW_LEVEL_MODES:
        raise ValueError("Invalid default navigation mode")
    if execution["controller"] not in SAMPLING_MODES:
        raise ValueError("连续控制器必须为 MPPI 或 KMPPI")
    sampling = data["sampling_planner"]
    if not isinstance(sampling, dict) or set(sampling) != {"config_path"} or not str(sampling["config_path"]).strip():
        raise ValueError("sampling_planner 必须包含 config_path")
    if any(
        float(value) <= 0
        for name, value in execution.items()
        if name not in {"default_mode", "controller"}
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
        "safety_braking_deceleration_mps2",
        "safety_reaction_time_sec",
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
    restricted = data["restricted_space"]
    if not isinstance(restricted, dict) or set(restricted) != {
        "route_corridor", "traversability_graph"
    }:
        raise ValueError(
            "restricted_space 必须包含 route_corridor 和 traversability_graph"
        )
    corridor_config = RouteCorridorConfig.from_mapping(
        restricted["route_corridor"]
    )
    graph_config = TraversabilityGraphConfig.from_mapping(
        restricted["traversability_graph"]
    )
    refiner_config = TrajectoryRefinerConfig.from_mapping(
        data["trajectory_refiner"]
    )
    selector = data["map_selector"]
    selector_fields = {
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

    map_settings = data["mapping"]
    if not isinstance(map_settings, dict) or set(map_settings) != {
        "camera", "map", "max_frame_skew_sec"
    }:
        raise ValueError("Invalid RGB-D mapping sections")
    if set(map_settings["camera"]) != {"width", "height", "frequency_hz"}:
        raise ValueError("Invalid RGB-D mapping camera settings")
    if float(map_settings["max_frame_skew_sec"]) <= 0:
        raise ValueError("mapping.max_frame_skew_sec must be positive")
    map_values = map_settings["map"]
    map_config = RGBDMapConfig.from_mapping(map_values)
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
        s2_client_config,
        s2_goal_config,
        map_settings,
        map_config,
        fine_config,
        footprint,
        corridor_config,
        graph_config,
        refiner_config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AGX 视觉语言主导航")
    parser.add_argument("--local-planner", choices=LOW_LEVEL_MODES, help="路径生成器")
    parser.add_argument("--controller", choices=SAMPLING_MODES, help="连续运动控制器")
    args = parser.parse_args()
    profile = load_agx_profile(project_root=PROJECT_ROOT)
    (
        settings,
        client_config,
        s2_client_config,
        s2_goal_config,
        map_settings,
        map_config,
        fine_config,
        footprint,
        corridor_config,
        graph_config,
        refiner_config,
    ) = _load_settings()
    if args.local_planner is not None:
        settings["execution"]["default_mode"] = args.local_planner
    if args.controller is not None:
        settings["execution"]["controller"] = args.controller
    logger = _make_logger()
    try:
        episodes = load_episodes(EPISODES_PATH, project_root=PROJECT_ROOT)
        episode = select_episode(episodes)
    except KeyboardInterrupt:
        logger.info("[EpisodeSelection] cancelled")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("[EpisodeSelection] failed: %s", exc)
        return 1
    logger.info(
        "[EpisodeSelected] id=%s name=%s scene=%s start=[%.2f,%.2f,%.1fdeg] "
        "instruction=%s",
        episode.episode_id,
        episode.name,
        episode.usd_path,
        episode.start_x_m,
        episode.start_y_m,
        episode.start_yaw_deg,
        episode.instruction,
    )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "active_gpu": settings["simulation"]["active_gpu"],
            "physics_gpu": settings["simulation"]["physics_gpu"],
            "multi_gpu": settings["simulation"]["multi_gpu"],
        }
    )
    try:
        try:
            return _run(
                app,
                profile,
                episode,
                settings,
                client_config,
                s2_client_config,
                s2_goal_config,
                map_settings,
                map_config,
                fine_config,
                footprint,
                corridor_config,
                graph_config,
                refiner_config,
                logger,
            )
        except Exception:
            logger.exception("[NavigationFatal] navigation crashed")
            return 1
    finally:
        app.close()


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("agx_navigation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/nav_latest.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("[NavigationModes] log=%s", log_path)
    return logger


def _save_fine_failure_diagnostic(
    *,
    job: _FinePlanningJob,
    grid,
    failure: FinePlanningFailure,
    collision_checker: FootprintCollisionChecker,
) -> tuple[Path, Path, Path]:
    output_dir = PROJECT_ROOT / "debug/local"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fine_failure_{job.request_id:04d}_{uuid.uuid4().hex[:8]}"
    map_path = output_dir / f"{stem}.npz"
    report_path = output_dir / f"{stem}.json"
    image_path = output_dir / f"{stem}.png"
    np.savez_compressed(
        map_path,
        occupied_mask=grid.occupied_mask,
        free_mask=grid.free_mask,
        unknown_mask=grid.unknown_mask,
        clearance_m=grid.clearance_m,
        map_origin_world_xy=np.asarray([grid.origin_x_m, grid.origin_y_m]),
        resolution_m=np.asarray(grid.resolution_m),
        map_version=np.asarray(grid.map_version),
        start_pose_world=np.asarray(job.start_pose_world),
        target_world_xy=np.asarray(job.goal_world_xy),
        floor_z_m=np.asarray(job.floor_z_m),
    )
    report = {
        "request_id": job.request_id,
        "map_version": grid.map_version,
        "reason": failure.reason,
        "start_pose_world": job.start_pose_world,
        "target_world_xy": job.goal_world_xy,
        "action_diagnostics": [
            asdict(item) for item in failure.action_diagnostics
        ],
        "footprint": {
            "margin_m": collision_checker.footprint.margin_m,
            "parts": [
                asdict(part) for part in collision_checker.footprint.parts
            ],
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    image = _render_fine_failure_diagnostic(
        grid=grid,
        start_pose_world=job.start_pose_world,
        failure=failure,
        collision_checker=collision_checker,
    )
    image.save(image_path)
    return map_path, report_path, image_path


def _render_fine_failure_diagnostic(
    *,
    grid,
    start_pose_world: tuple[float, float, float],
    failure: FinePlanningFailure,
    collision_checker: FootprintCollisionChecker,
) -> Image.Image:
    rgb = np.full((*grid.shape, 3), (108, 110, 112), dtype=np.uint8)
    rgb[grid.free_mask] = (238, 237, 232)
    rgb[grid.occupied_mask] = (28, 29, 29)
    current_cells, _ = collision_checker.covered_linear_indices(
        grid,
        x_m=start_pose_world[0],
        y_m=start_pose_world[1],
        yaw_rad=start_pose_world[2],
        margin_m=0.0,
    )
    padded_cells, _ = collision_checker.covered_linear_indices(
        grid,
        x_m=start_pose_world[0],
        y_m=start_pose_world[1],
        yaw_rad=start_pose_world[2],
    )
    left = rgb.copy()
    left.reshape(-1, 3)[padded_cells] = (238, 184, 45)
    left.reshape(-1, 3)[current_cells] = (55, 92, 168)
    right = rgb.copy()
    for item in failure.action_diagnostics:
        if item.action_name == "forward_2":
            right.reshape(-1, 3)[list(item.swept_linear_indices)] = (73, 160, 212)
            right.reshape(-1, 3)[list(item.unknown_linear_indices)] = (
                235, 68, 125
            )
        elif item.action_name in {"spin_left", "spin_right"}:
            right.reshape(-1, 3)[list(item.unknown_linear_indices)] = (
                242, 147, 66
            )
    for item in failure.action_diagnostics:
        if item.action_name == "forward_2":
            right.reshape(-1, 3)[list(item.unknown_linear_indices)] = (
                235, 68, 125
            )
            break
    scale = 4
    width = grid.shape[1] * scale
    height = grid.shape[0] * scale
    image = Image.new("RGB", (2 * width, height + 52), (37, 40, 44))
    image.paste(
        Image.fromarray(np.flipud(left), mode="RGB").resize(
            (width, height), Image.NEAREST
        ),
        (0, 32),
    )
    image.paste(
        Image.fromarray(np.flipud(right), mode="RGB").resize(
            (width, height), Image.NEAREST
        ),
        (width, 32),
    )
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "CURRENT BLUE / MARGIN YELLOW", fill=(245, 245, 242))
    draw.text(
        (width + 8, 8),
        "FORWARD 0.1m BLUE / UNKNOWN PINK / SPIN ORANGE",
        fill=(245, 245, 242),
    )
    draw.text(
        (8, height + 35),
        f"map_version={grid.map_version}  resolution={grid.resolution_m:.3f}m",
        fill=(245, 245, 242),
    )
    return image


def _run(
    app,
    profile,
    episode: NavigationEpisode,
    settings,
    client_config,
    s2_client_config,
    s2_goal_config,
    map_settings,
    map_config,
    fine_config,
    footprint,
    corridor_config,
    graph_config,
    refiner_config,
    logger,
    *,
    step_observer=None,
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
    scene_settings = settings["scene"]
    scene_path = episode.usd_path
    add_reference_to_stage(str(scene_path), scene_settings["prim_path"])
    if scene_settings["add_default_ground_plane"]:
        world.scene.add_default_ground_plane()
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(scene_settings["stage_load_timeout_sec"]),
        logger=logger,
        label="AGX navigation modes scene",
    )
    if settings["scene"]["add_indoor_lighting"]:
        try:
            add_indoor_lighting(
                stage=stage,
                environment_prim_path=scene_settings["prim_path"],
                logger=logger,
            )
        except Exception:
            logger.exception("[InteriorLighting] failed to add scene lights")
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[
            episode.start_x_m,
            episode.start_y_m,
            float(profile.spawn_z_m),
        ],
        rotation_deg=[
            0.0,
            0.0,
            episode.start_yaw_deg,
        ],
    )
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    validate_agx_geometry(stage, profile, articulation_path)
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
    camera_roles = {
        "front": profile.sensors.camera.front_prim_path,
        "down": profile.sensors.camera.mapping_down_prim_path,
    }
    map_role_paths = {
        source: resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=camera_roles[source],
            ),
            kind=f"NavDP goal-map {source} camera",
        )
        for source in map_sources
    }
    s2_down_path = resolve_existing_prim_path(
        stage=stage,
        prim_path=resolve_robot_relative_prim_path(
            robot_root_path=profile.asset.prim_path,
            prim_path=profile.sensors.camera.s2_down_prim_path,
        ),
        kind="S2 look-down camera",
    )
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
    if not stage.GetPrimAtPath(s2_down_path).IsA(UsdGeom.Camera):
        raise RuntimeError(f"S2 look-down prim is not a Camera: {s2_down_path}")
    camera_config = settings["camera"]
    focal_mm = set_camera_horizontal_fov(
        stage=stage,
        prim_path=camera_path,
        fov_deg=float(camera_config["horizontal_fov_deg"]),
    )
    logger.info(
        "[FrontCamera] horizontal_fov=%.1fdeg focal_length=%.3fmm path=%s",
        float(camera_config["horizontal_fov_deg"]),
        focal_mm,
        camera_path,
    )
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
    s2_down_camera = Camera(
        prim_path=s2_down_path,
        name="agx_s2_down_rgbd",
        frequency=int(map_camera_config["frequency_hz"]),
        resolution=(
            int(map_camera_config["width"]),
            int(map_camera_config["height"]),
        ),
    )
    world.reset()
    robot.initialize()
    for sensor_camera in (*map_cameras.values(), s2_down_camera):
        sensor_camera.initialize()
        sensor_camera.add_distance_to_image_plane_to_frame()
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
    s2_worker = (
        _S2Worker(InternNavS2Client(s2_client_config))
        if bool(settings["internnav_s2"]["enabled"])
        else None
    )
    s2_goal_adapter = S2GoalAdapter(s2_goal_config)
    footprint, footprint_audit = build_agx_footprint(
        stage, profile, articulation_path, margin_m=footprint.margin_m,
    )
    footprint_audit_path = (
        PROJECT_ROOT / "constraint_traverse/data/diagnostics"
        / f"agx_footprint_{uuid.uuid4().hex[:8]}.json"
    )
    footprint_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with footprint_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(footprint_audit, handle, ensure_ascii=False, indent=2)
    logger.info(
        "[FootprintGeometry] 碰撞体=%d 独立部件=%d 状态=%s 安全余量=%.3fm 核查报告=%s",
        footprint_audit["collider_count"], len(footprint.parts),
        footprint_audit["status"], footprint.margin_m, footprint_audit_path,
    )
    collision_checker = FootprintCollisionChecker(
        footprint,
        yaw_resolution_deg=fine_config.yaw_resolution_deg,
    )
    sampling_controller = None

    def create_sampling_controller(selected_mode):
        from constraint_traverse.navigation import SamplingNavigationController
        config_path = Path(settings["sampling_planner"]["config_path"])
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        return SamplingNavigationController(
            method=selected_mode, hardware=profile.agx_controller.continuous,
            footprint=footprint, physics_dt=world.get_physics_dt(), config_path=config_path, logger=logger,
        )

    control_method = str(settings["execution"]["controller"])
    sampling_controller = create_sampling_controller(control_method)
    fine_planner = FineLocalPlanner(fine_config, collision_checker)
    graph_builder = TraversabilityGraphBuilder(
        config=graph_config,
        planner_config=fine_config,
        collision_checker=collision_checker,
    )
    restricted_analyzer = RestrictedSpaceAnalyzer(
        corridor_config=corridor_config,
        graph_config=graph_config,
        graph_builder=graph_builder,
    )
    trajectory_refiner = TrajectoryRefiner(
        refiner_config,
        collision_checker,
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
        restricted_analyzer=restricted_analyzer,
        trajectory_refiner=trajectory_refiner,
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
    logger.info(
        "[S2Cameras] front=%s down=%s",
        camera_path,
        s2_down_path,
    )

    command = [None]
    panel = _GoalMapPanel(
        ui,
        image_size_px=int(settings["map_selector"]["image_size_px"]),
        command_sink=lambda value: command.__setitem__(0, value),
        mode=str(settings["execution"]["default_mode"]),
        controller=control_method,
    )
    s2_panel = _S2DecisionPanel(ui)

    for _ in range(int(camera_config["warmup_render_frames"])):
        adapter.stop()
        world.step(render=True)

    if str(settings["execution"]["default_mode"]) == "COARSE":
        warmup_rgb, warmup_depth = _capture_rgbd(camera)
        warmup_result = worker.warm_up(
            intrinsics=np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64),
            rgb=warmup_rgb,
            depth_m=warmup_depth,
        )
        logger.info(
            "[NavDPWarmup] complete candidates=%d dropped=%d latency=%.3fs reset=2",
            len(warmup_result.candidates),
            len(warmup_result.dropped_candidates),
            warmup_result.latency_sec,
        )

    position, orientation = get_world_pose(robot)
    local_map = RGBDPlanarMap(
        map_config,
        obstacle_min_height_m=float(settings["fine_planner"]["obstacle_min_height_m"]),
        obstacle_max_height_m=float(settings["fine_planner"]["obstacle_max_height_m"]),
    )
    local_map.reset(np.asarray(position[:2], dtype=np.float64))
    logger.info("[Mapping] type=RGBD_2D sources=front,down resolution=%.3fm size=%.1fx%.1fm",
                map_config.resolution_m, map_config.size_x_m, map_config.size_y_m)

    tracker = None
    tracker_mode: str | None = None
    display_path_odom_xy: np.ndarray | None = None
    reference_path: ReferencePath | None = None
    reference_map_signature = None
    reference_checked_map_version = -1
    reference_map_changed = False
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
    last_map_integration_time = -float("inf")
    last_map_diagnostic_time = -float("inf")
    last_map_render_time = -float("inf")
    last_coarse_request_time = -float("inf")
    last_fine_request_time = -float("inf")
    map_integration_period = float(map_settings["map"]["integration_period_sec"])
    map_render_period = float(settings["map_selector"]["render_period_sec"])
    latest_snapshot = local_map.snapshot()
    latest_planning_grid = None
    last_failed_fine_input: _FineFailureInput | None = None
    fine_diagnostic_saved_for_goal = False
    recent_fine_world_poses: deque[tuple[float, float, float]] = deque(maxlen=30)
    last_partial_switch_world_xy: np.ndarray | None = None
    last_partial_endpoint_world_xy: np.ndarray | None = None
    rejected_unknown_world_xy: np.ndarray | None = None
    latest_camera_frames: dict[str, CameraFrameGeometry] = {}
    s2_settings = settings["internnav_s2"]
    s2_active = s2_worker is not None
    s2_session_id = uuid.uuid4().hex
    s2_request_id = 0
    accepted_s2_request_id = 0
    s2_need_request = s2_active
    s2_next_view = "front"
    s2_reset_pending = s2_active
    s2_min_frame_index = 0
    s2_actions: deque[int] = deque()
    s2_turn_target_yaw_odom: float | None = None
    s2_stop_count = 0
    logger.info("[NavigationMode] path=%s controller=%s", mode, control_method)
    if s2_active:
        idle_status = "S2 waiting for first front frame"
        logger.info(
            "[S2Session] automatic session=%s instruction=%s",
            s2_session_id,
            episode.instruction,
        )

    try:
        while app.is_running():
            if not world.is_playing():
                adapter.stop()
                if sampling_controller is not None:
                    sampling_controller.reset()
                reference_path = None
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
            if step_observer is not None and step_observer(
                world=world, robot=robot, state=state, command=command,
                grid=latest_planning_grid, controller=sampling_controller,
                navigation_active=navigation_active,
            ):
                break
            world_from_base = world_from_pose_wxyz(position, orientation)
            robot_yaw_world = math.atan2(
                float(world_from_base[1, 0]),
                float(world_from_base[0, 0]),
            )
            if navigation_active and mode == "FINE":
                current_pose_world = (
                    float(position[0]), float(position[1]), float(robot_yaw_world)
                )
                if not recent_fine_world_poses:
                    recent_fine_world_poses.append(current_pose_world)
                else:
                    previous_pose = recent_fine_world_poses[-1]
                    yaw_change = (
                        current_pose_world[2] - previous_pose[2] + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    if (
                        math.hypot(
                            current_pose_world[0] - previous_pose[0],
                            current_pose_world[1] - previous_pose[1],
                        ) >= 0.08
                        or abs(yaw_change) >= math.radians(10.0)
                    ):
                        recent_fine_world_poses.append(current_pose_world)
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
                if "front" in captured:
                    latest_camera_frames["front"] = captured["front"]
                s2_down_frame = capture_camera_frame(
                    source="down",
                    camera=s2_down_camera,
                    frame_index=frame_index,
                    world_from_base=world_from_base,
                )
                if s2_down_frame is not None:
                    latest_camera_frames["down"] = s2_down_frame
                if len(captured) == len(map_cameras):
                    timestamps = [sample.timestamp_sec for sample in captured.values()]
                    if max(timestamps) - min(timestamps) <= float(
                        map_settings["max_frame_skew_sec"]
                    ):
                        map_update = local_map.integrate_batch(
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
                            center_world_xy=np.asarray(position[:2], dtype=np.float64),
                            floor_z_m=float(position[2]) - float(profile.spawn_z_m),
                        )
                        latest_snapshot = local_map.snapshot()
                        latest_planning_grid = latest_snapshot.grid
                        latest_planning_grid = clear_current_footprint(
                            latest_planning_grid,
                            x_m=float(position[0]),
                            y_m=float(position[1]),
                            yaw_rad=float(robot_yaw_world),
                            collision_checker=collision_checker,
                        )
                        if now - last_map_diagnostic_time >= 1.0:
                            ray_counts = dict(map_update.source_ray_counts)
                            source_updates = {
                                source: {
                                    "rays": int(ray_counts.get(source, 0)),
                                    "free": int(free_count),
                                    "occupied": int(occupied_count),
                                }
                                for source, free_count, occupied_count
                                in map_update.source_cell_update_counts
                            }
                            cell_count = latest_planning_grid.unknown_mask.size
                            logger.info(
                                "[MapUpdate] version=%d shifted=%s sources=%s "
                                "grid=[free:%.3f occupied:%.3f unknown:%.3f]",
                                map_update.version,
                                map_update.shifted,
                                source_updates,
                                np.count_nonzero(latest_planning_grid.free_mask)
                                / cell_count,
                                np.count_nonzero(latest_planning_grid.occupied_mask)
                                / cell_count,
                                np.count_nonzero(latest_planning_grid.unknown_mask)
                                / cell_count,
                            )
                            last_map_diagnostic_time = now
                        last_map_integration_time = now
                    elif now - last_map_diagnostic_time >= 1.0:
                        logger.warning(
                            "[MapUpdate] skipped reason=frame_skew skew=%.3fs limit=%.3fs",
                            max(timestamps) - min(timestamps),
                            float(map_settings["max_frame_skew_sec"]),
                        )
                        last_map_diagnostic_time = now
                elif now - last_map_diagnostic_time >= 1.0:
                    logger.warning(
                        "[MapUpdate] skipped reason=missing_camera captured=%s expected=%s",
                        sorted(captured),
                        sorted(map_cameras),
                    )
                    last_map_diagnostic_time = now

            if now - last_map_render_time >= map_render_period:
                displayed_goal = (
                    None
                    if selected_goal_odom_xy is None
                    else _odom_goal_to_local(selected_goal_odom_xy, state)
                )
                goal_map = render_planning_goal_map(
                    latest_snapshot.grid if latest_planning_grid is None else latest_planning_grid,
                    source_mask=latest_snapshot.source_mask,
                    robot_world_xyz=position[:3],
                    robot_yaw_rad=robot_yaw_world,
                    view_size_m=float(settings["map_selector"]["view_size_m"]),
                    rear_view_m=float(settings["map_selector"]["rear_view_m"]),
                    image_size_px=int(settings["map_selector"]["image_size_px"]),
                    goal_local_xy=displayed_goal,
                    show_source_coverage=True,
                    rejected_unknown_world_xy=(
                        rejected_unknown_world_xy
                        if mode == "FINE" and navigation_active
                        else None
                    ),
                )
                if display_path_odom_xy is not None:
                    path_local = _odom_path_to_local(display_path_odom_xy, state)
                    goal_map = _overlay_path_on_goal_map(
                        goal_map,
                        path_local,
                        color=(34, 177, 76, 255) if mode == "FINE" else (238, 184, 45, 255),
                    )
                if navigation_active and sampling_controller.last_plan is not None:
                    prediction_odom = _world_path_to_odom(sampling_controller.last_plan.prediction[:, :2], state)
                    goal_map = _overlay_path_on_goal_map(goal_map, _odom_path_to_local(prediction_odom, state),
                                                        color=(51, 139, 230, 255))
                panel.update_map(goal_map)
                last_map_render_time = now

            s2_completed = None if s2_worker is None else s2_worker.poll()
            if s2_completed is not None:
                s2_job, s2_result, s2_error = s2_completed
                if (
                    s2_job.request_id != accepted_s2_request_id
                    or s2_job.session_id != s2_session_id
                    or not s2_active
                ):
                    logger.info(
                        "[S2Result] ignored stale request=%d",
                        s2_job.request_id,
                    )
                elif s2_error is not None:
                    adapter.stop()
                    tracker = None
                    tracker_mode = None
                    navigation_active = False
                    s2_need_request = False
                    idle_status = "S2 request failed; navigation stopped"
                    logger.error(
                        "[S2Result] request=%d view=%s failed: %s",
                        s2_job.request_id,
                        s2_job.view,
                        s2_error,
                    )
                else:
                    logger.info(
                        "[S2Result] request=%d view=%s type=%s latency=%.3fs",
                        s2_job.request_id,
                        s2_job.view,
                        s2_result.kind,
                        s2_result.latency_sec,
                    )
                    logger.info(
                        "[S2Raw] request=%d output=%r",
                        s2_job.request_id,
                        str(s2_result.raw_response.get("raw_output", "")).strip(),
                    )
                    if s2_result.kind == "pixel_goal":
                        original_pixel = _pixel_to_original_image(
                            s2_result.pixel_vu,
                            s2_result.pixel_space_hw,
                            s2_job.frame.rgb.shape,
                        )
                        try:
                            metric_goal = s2_goal_adapter.convert(
                                request_id=s2_job.request_id,
                                frame=s2_job.frame,
                                pixel_vu=s2_result.pixel_vu,
                                pixel_space_hw=s2_result.pixel_space_hw,
                            )
                            selected_goal_odom_xy = _world_path_to_odom(
                                np.asarray([metric_goal.world_xy], dtype=np.float64),
                                state,
                            )[0]
                            reference_path = None
                            sampling_controller.reset()
                            request_id += 1
                            accepted_request_id = request_id
                            fine_request_id += 1
                            accepted_fine_request_id = fine_request_id
                            tracker = None
                            tracker_mode = None
                            display_path_odom_xy = None
                            navigation_active = True
                            last_failed_fine_input = None
                            fine_diagnostic_saved_for_goal = False
                            recent_fine_world_poses.clear()
                            last_partial_switch_world_xy = None
                            last_partial_endpoint_world_xy = None
                            rejected_unknown_world_xy = None
                            last_coarse_request_time = -float("inf")
                            last_fine_request_time = -float("inf")
                            s2_stop_count = 0
                            idle_status = f"S2 GOAL ACTIVE: {mode}"
                            decision_image = _render_s2_decision(
                                s2_job.frame.rgb,
                                label=(
                                    f"{s2_job.view.upper()} PIXEL -> "
                                    f"goal [{metric_goal.body_xy[0]:.2f},"
                                    f"{metric_goal.body_xy[1]:+.2f}] m "
                                    f"depth={metric_goal.depth_m:.2f}m "
                                    f"spread={metric_goal.depth_spread_m:.2f}m"
                                ),
                                pixel_vu=metric_goal.pixel_original_vu,
                            )
                            s2_panel.update(decision_image)
                            _save_image(
                                PROJECT_ROOT / "debug/local/s2_debug.png",
                                decision_image,
                            )
                            logger.info(
                                "[S2Goal] request=%d view=%s pixel_vu=[%.1f,%.1f] "
                                "depth=%.2fm valid=%.3f spread=%.3fm "
                                "body_xy=[%.2f,%+.2f] world_xy=[%.2f,%.2f]",
                                s2_job.request_id,
                                metric_goal.view,
                                metric_goal.pixel_original_vu[0],
                                metric_goal.pixel_original_vu[1],
                                metric_goal.depth_m,
                                metric_goal.depth_valid_fraction,
                                metric_goal.depth_spread_m,
                                metric_goal.body_xy[0],
                                metric_goal.body_xy[1],
                                metric_goal.world_xy[0],
                                metric_goal.world_xy[1],
                            )
                        except Exception as exc:
                            adapter.stop()
                            tracker = None
                            tracker_mode = None
                            navigation_active = False
                            s2_need_request = True
                            s2_next_view = "front"
                            s2_min_frame_index = frame_index + 1
                            idle_status = "S2 pixel goal rejected; requesting a new decision"
                            decision_image = _render_s2_decision(
                                s2_job.frame.rgb,
                                label=f"{s2_job.view.upper()} PIXEL REJECTED",
                                pixel_vu=original_pixel,
                            )
                            s2_panel.update(decision_image)
                            _save_image(
                                PROJECT_ROOT / "debug/local/s2_debug.png",
                                decision_image,
                            )
                            logger.warning(
                                "[S2GoalReject] request=%d reason=%s",
                                s2_job.request_id,
                                exc,
                            )
                    else:
                        actions = tuple(s2_result.actions)
                        repeated_lookdown = s2_job.view == "down" and 5 in actions
                        if repeated_lookdown:
                            actions = tuple(action for action in actions if action != 5)
                            logger.warning(
                                "[S2LookDown] repeated request from down view ignored; "
                                "next_view=front"
                            )
                            if not actions:
                                s2_need_request = True
                                s2_next_view = "front"
                                s2_min_frame_index = frame_index + 1
                                idle_status = "S2 repeated LOOKDOWN; requesting front view"
                        actions = _turn_actions_before_redecision(
                            actions,
                            turn_action_deg=float(s2_settings["turn_action_deg"]),
                            turn_redecision_deg=float(
                                s2_settings["turn_redecision_deg"]
                            ),
                        )
                        s2_actions.extend(actions)
                        s2_stop_count = (
                            s2_stop_count
                            if actions and all(action == 0 for action in actions)
                            else 0
                        )
                        logger.info(
                            "[S2Actions] request=%d actions=%s",
                            s2_job.request_id,
                            list(actions),
                        )
                        action_names = {
                            0: "STOP",
                            1: "FORWARD",
                            2: "LEFT",
                            3: "RIGHT",
                            5: "LOOKDOWN",
                        }
                        label = (
                            "ACTIONS: " + " ".join(
                                action_names[action] for action in actions
                            )
                            if actions
                            else "FRONT REDECISION"
                        )
                        decision_image = _render_s2_decision(
                            s2_job.frame.rgb,
                            label=f"{s2_job.view.upper()} {label}",
                        )
                        s2_panel.update(decision_image)
                        _save_image(
                            PROJECT_ROOT / "debug/local/s2_debug.png",
                            decision_image,
                        )

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
                    logger.error(
                        "[NavDPResult] request=%d failed: %s",
                        job.request_id,
                        error,
                    )
                else:
                    result_age = now - float(job.odom_state.timestamp_sec)
                    model_choice = result.model_choice
                    logger.info(
                        "[NavDPResult] request=%d candidates=%d selected=%d "
                        "value=%s latency=%.3fs",
                        job.request_id,
                        len(result.candidates),
                        result.model_choice_index,
                        _format_value(model_choice.value),
                        result.latency_sec,
                    )
                    if result.dropped_candidates:
                        logger.warning(
                            "[NavDPCandidatesDropped] request=%d indices=%s",
                            job.request_id,
                            [index for index, _ in result.dropped_candidates],
                        )
                    frozen_candidates = _freeze_candidates(result, job)
                    selected_index = result.model_choice_index
                    selected = frozen_candidates[selected_index]
                    joined_path, join_error, remaining_length = _join_path_from_current(
                        selected.odom_path_xy,
                        current_xy=(state.pose.x, state.pose.y),
                    )
                    _save_candidate_debug(
                        debug_path,
                        result,
                        selected_index=selected_index,
                        goal_xy=(job.goal_forward_m, job.goal_left_m),
                        view_size_m=float(settings["debug"]["view_size_m"]),
                    )
                    if remaining_length <= 1.0e-4:
                        reference_path = None
                        sampling_controller.reset()
                        tracker_mode = None
                        display_path_odom_xy = None
                        idle_status = "NavDP zero-motion trajectory; waiting for next inference"
                    else:
                        reference_path = ReferencePath.from_positions(
                            _odom_path_to_world(joined_path, state), initial_yaw_rad=robot_yaw_world,
                            goal_world_xy=_odom_path_to_world(selected_goal_odom_xy[None, :], state)[0],
                            source="COARSE", request_id=job.request_id,
                            created_sim_time_sec=float(job.odom_state.timestamp_sec),
                            map_version=0 if latest_planning_grid is None else latest_planning_grid.map_version,
                            partial=True,
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
                job, grid, outcome, error = fine_completed
                if (
                    job.request_id != accepted_fine_request_id
                    or mode != "FINE"
                    or not navigation_active
                ):
                    logger.info("[FineResult] ignored stale request=%d", job.request_id)
                elif error is not None:
                    if (
                        isinstance(error, FinePlanningFailure)
                        and grid is not None
                        and not fine_diagnostic_saved_for_goal
                    ):
                        try:
                            map_path, report_path, image_path = (
                                _save_fine_failure_diagnostic(
                                    job=job,
                                    grid=grid,
                                    failure=error,
                                    collision_checker=collision_checker,
                                )
                            )
                            fine_diagnostic_saved_for_goal = True
                            logger.info(
                                "[FineDiagnostic] reason=%s map=%s report=%s image=%s",
                                error.reason,
                                map_path,
                                report_path,
                                image_path,
                            )
                            for item in error.action_diagnostics:
                                logger.info(
                                    "[FineAction] action=%s result=%s "
                                    "endpoint=%s unknown=%d area=%.3fm2 "
                                    "parts=%s map_version=%d",
                                    item.action_name,
                                    item.result,
                                    item.endpoint_world_xyyaw,
                                    len(item.unknown_linear_indices),
                                    item.unknown_area_m2,
                                    dict(item.part_unknown_counts),
                                    grid.map_version,
                                )
                        except Exception:
                            logger.exception("[FineDiagnostic] 保存失败")
                    if (
                        isinstance(error, FinePlanningFailure)
                        and error.reason == "SEARCH_BUDGET_EXHAUSTED"
                    ):
                        last_failed_fine_input = None
                    elif grid is not None:
                        last_failed_fine_input = _FineFailureInput.from_job(
                            grid,
                            job,
                            relevant_radius_m=(
                                job.lookahead_m + collision_checker.bounding_radius_m
                            ),
                        )
                    if isinstance(error, FineUnknownBlockedError) and grid is not None:
                        rows, columns = np.unravel_index(
                            error.unknown_linear_indices, grid.shape
                        )
                        rejected_unknown_world_xy = np.column_stack((
                            grid.origin_x_m + (columns + 0.5) * grid.resolution_m,
                            grid.origin_y_m + (rows + 0.5) * grid.resolution_m,
                        ))
                        reference_path = None
                        sampling_controller.reset()
                        adapter.stop()
                        tracker = None
                        tracker_mode = None
                        display_path_odom_xy = None
                        idle_status = "Fine blocked by nearby unknown cells; see Goal Map"
                        logger.warning(
                            "[FineUnknownArea] request=%d cells=%d x=[%.2f,%.2f] "
                            "y=[%.2f,%.2f] map_version=%d",
                            job.request_id,
                            len(rejected_unknown_world_xy),
                            float(np.min(rejected_unknown_world_xy[:, 0])),
                            float(np.max(rejected_unknown_world_xy[:, 0])),
                            float(np.min(rejected_unknown_world_xy[:, 1])),
                            float(np.max(rejected_unknown_world_xy[:, 1])),
                            grid.map_version,
                        )
                    else:
                        rejected_unknown_world_xy = None
                        idle_status = "Fine planning failed; inspect Fine Geometry and log"
                    logger.error(
                        "[FineResult] request=%d map_version=%d "
                        "target=[%.2f,%.2f] status=%s reason=%s failed: %s",
                        job.request_id,
                        job.grid.map_version,
                        job.goal_world_xy[0],
                        job.goal_world_xy[1],
                        (
                            "SEARCH_INCOMPLETE"
                            if isinstance(error, FinePlanningFailure)
                            and error.reason == "SEARCH_BUDGET_EXHAUSTED"
                            else "NO_SAFE_PROGRESS"
                            if isinstance(error, FinePlanningFailure)
                            else "ERROR"
                        ),
                        (
                            error.reason
                            if isinstance(error, FinePlanningFailure)
                            else type(error).__name__
                        ),
                        error,
                    )
                else:
                    if outcome is None or outcome.plan is None:
                        raise RuntimeError("FINE worker 未返回可执行轨迹")
                    plan = outcome.plan
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
                    clearance, planned_unknown = assess_known_swept_path(
                        grid,
                        plan.path_world_xy,
                        plan.path_world_yaw_rad,
                        collision_checker=collision_checker,
                        max_step_m=fine_config.collision_sample_step_m,
                        max_yaw_step_deg=fine_config.yaw_resolution_deg,
                    )
                    validation_grid = latest_planning_grid or grid
                    latest_clearance, current_unknown = assess_known_swept_path(
                        validation_grid,
                        plan.path_world_xy,
                        plan.path_world_yaw_rad,
                        collision_checker=collision_checker,
                        max_step_m=fine_config.collision_sample_step_m,
                        max_yaw_step_deg=fine_config.yaw_resolution_deg,
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
                    elif planned_unknown.cell_count > 0:
                        rejection = "fine_path_unknown"
                    elif current_unknown.cell_count > 0:
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
                            _FineFailureInput.from_job(
                                validation_grid,
                                job,
                                relevant_radius_m=(
                                    job.lookahead_m
                                    + collision_checker.bounding_radius_m
                                ),
                            )
                            if rejection in {
                                "fine_path_collision",
                                "fine_path_new_collision",
                                "fine_path_outside_map",
                                "fine_path_unknown",
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
                        rejected_unknown_world_xy = None
                        last_partial_switch_world_xy = (
                            np.asarray(position[:2], dtype=np.float64).copy()
                            if plan.partial else None
                        )
                        last_partial_endpoint_world_xy = (
                            plan.path_world_xy[-1].copy() if plan.partial else None
                        )
                        reference_path = ReferencePath(
                            plan.path_world_xy, plan.path_world_yaw_rad, plan.path_motion_direction,
                            np.asarray(job.goal_world_xy), "FINE", job.request_id,
                            job.submitted_sim_time_sec, grid.map_version, plan.partial,
                        )
                        reference_map_signature = reference_path.map_signature(validation_grid, radius_m=collision_checker.bounding_radius_m)
                        reference_checked_map_version = validation_grid.map_version
                        reference_map_changed = False
                        display_path_odom_xy = joined_path.copy()
                        tracker_mode = mode
                        generation_index += 1
                        previous_tracking_status = None
                        previous_adapter_status = None
                        logger.info(
                            "[FineSwitch] generation=%d request=%d status=%s partial=%s "
                            "first_direction=%d endpoint=[%.2f,%.2f] "
                            "nodes=%d cost=%.3f "
                            "age=%.3fs join_error=%.3fm remaining=%.3fm "
                            "local_goal=[%.2f,%.2f] clearance=%.3fm unknown=%.3f "
                            "path_unknown=[ratio:%.3f area:%.3fm2] "
                            "rejected=[collision:%d,unknown:%d,outside:%d]",
                            generation_index,
                            job.request_id,
                            plan.status,
                            plan.partial,
                            int(plan.path_motion_direction[1]),
                            plan.path_world_xy[-1, 0],
                            plan.path_world_xy[-1, 1],
                            plan.expanded_node_count,
                            plan.path_cost,
                            result_age,
                            join_error,
                            remaining_length,
                            plan.local_goal_world_xy[0],
                            plan.local_goal_world_xy[1],
                            clearance.min_clearance_m,
                            clearance.unknown_ratio,
                            plan.max_unknown_ratio,
                            plan.max_unknown_area_m2,
                            plan.rejected_collision_count,
                            plan.rejected_unknown_count,
                            plan.rejected_outside_count,
                        )

            if command[0] is not None:
                action, command[0] = command[0], None
                if isinstance(action, tuple) and action[0] in {"SET_MODE", "SET_CONTROLLER"}:
                    selected_mode = str(action[1]) if action[0] == "SET_MODE" else mode
                    selected_controller = str(action[1]) if action[0] == "SET_CONTROLLER" else control_method
                    if (selected_mode in LOW_LEVEL_MODES and selected_controller in SAMPLING_MODES
                            and (selected_mode != mode or selected_controller != control_method)):
                        try:
                            replacement_controller = create_sampling_controller(selected_controller)
                        except Exception as exc:
                            logger.error("[NavigationMode] 无法启用 %s：%s", selected_mode, exc)
                            idle_status = f"Mode unavailable: {selected_mode}; see log"
                            panel.update_controls(mode, idle_status, control_method)
                        else:
                            adapter.stop()
                            if sampling_controller is not None:
                                sampling_controller.close()
                            sampling_controller = replacement_controller
                            mode = selected_mode
                            control_method = selected_controller
                            reference_path = None
                            tracker = tracker_mode = None
                            display_path_odom_xy = None
                            frozen_candidates = []
                            rejected_unknown_world_xy = None
                            last_failed_fine_input = None
                            fine_diagnostic_saved_for_goal = False
                            recent_fine_world_poses.clear()
                            last_partial_switch_world_xy = last_partial_endpoint_world_xy = None
                            last_coarse_request_time = last_fine_request_time = -float("inf")
                            request_id += 1
                            accepted_request_id = request_id
                            fine_request_id += 1
                            accepted_fine_request_id = fine_request_id
                            idle_status = f"{mode} + {control_method}"
                            logger.info("[NavigationMode] path=%s controller=%s goal_retained=%s", mode, control_method, selected_goal_odom_xy is not None)
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
                            reference_path = None
                            sampling_controller.reset()
                            s2_active = False
                            s2_need_request = False
                            s2_actions.clear()
                            s2_turn_target_yaw_odom = None
                            s2_request_id += 1
                            accepted_s2_request_id = s2_request_id
                            request_id += 1
                            accepted_request_id = request_id
                            fine_request_id += 1
                            accepted_fine_request_id = fine_request_id
                            frozen_candidates = []
                            display_path_odom_xy = None
                            selected_index = 0
                            navigation_active = True
                            last_failed_fine_input = None
                            fine_diagnostic_saved_for_goal = False
                            recent_fine_world_poses.clear()
                            last_partial_switch_world_xy = None
                            last_partial_endpoint_world_xy = None
                            rejected_unknown_world_xy = None
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

            goal_distance = float("inf")
            if selected_goal_odom_xy is not None:
                goal_distance = float(np.linalg.norm(
                    selected_goal_odom_xy
                    - np.asarray([state.pose.x, state.pose.y], dtype=np.float64)
                ))
            goal_tolerance = float(
                s2_settings["goal_reached_tolerance_m"]
                if s2_active
                else settings["execution"]["goal_reached_tolerance_m"]
            )
            if navigation_active and goal_distance <= goal_tolerance:
                navigation_active = False
                reference_path = None
                tracker = None
                tracker_mode = None
                adapter.stop()
                if sampling_controller is not None:
                    sampling_controller.reset()
                idle_status = f"GOAL REACHED: distance={goal_distance:.2f}m"
                logger.info(
                    "[NavigationGoalReached] mode=%s generations=%d distance=%.3fm",
                    mode,
                    generation_index,
                    goal_distance,
                )
                selected_goal_odom_xy = None
                fine_diagnostic_saved_for_goal = False
                recent_fine_world_poses.clear()
                last_partial_switch_world_xy = None
                last_partial_endpoint_world_xy = None
                rejected_unknown_world_xy = None
                if s2_active:
                    if not s2_actions:
                        s2_need_request = True
                        s2_next_view = "front"
                        s2_min_frame_index = frame_index + 1
                    idle_status = "S2 local goal reached"

            if (
                s2_active
                and not navigation_active
                and tracker is None
                and s2_turn_target_yaw_odom is None
                and s2_actions
            ):
                action_code = int(s2_actions.popleft())
                if action_code == 0:
                    s2_stop_count += 1
                    s2_actions.clear()
                    adapter.stop()
                    if s2_stop_count >= int(s2_settings["stop_confirmations"]):
                        s2_active = False
                        s2_need_request = False
                        idle_status = "S2 TASK COMPLETE"
                        logger.info(
                            "[S2Stop] task complete confirmations=%d",
                            s2_stop_count,
                        )
                    else:
                        s2_need_request = True
                        s2_next_view = "front"
                        s2_min_frame_index = frame_index + 1
                        idle_status = "S2 STOP awaiting confirmation"
                        logger.info(
                            "[S2Stop] confirmation=%d/%d",
                            s2_stop_count,
                            int(s2_settings["stop_confirmations"]),
                        )
                elif action_code == 1:
                    s2_stop_count = 0
                    distance = float(s2_settings["forward_action_distance_m"])
                    selected_goal_odom_xy = _transform_xy(
                        np.asarray([[distance, 0.0]], dtype=np.float64),
                        state.pose.x,
                        state.pose.y,
                        state.pose.yaw_rad,
                    )[0]
                    reference_path = None
                    sampling_controller.reset()
                    request_id += 1
                    accepted_request_id = request_id
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    navigation_active = True
                    fine_diagnostic_saved_for_goal = False
                    recent_fine_world_poses.clear()
                    last_partial_switch_world_xy = None
                    last_partial_endpoint_world_xy = None
                    rejected_unknown_world_xy = None
                    last_coarse_request_time = -float("inf")
                    last_fine_request_time = -float("inf")
                    idle_status = f"S2 FORWARD: {mode}"
                    logger.info(
                        "[S2Action] FORWARD distance=%.2fm remaining_actions=%s",
                        distance,
                        list(s2_actions),
                    )
                elif action_code in {2, 3}:
                    s2_stop_count = 0
                    turn_count = 1
                    while s2_actions and s2_actions[0] == action_code:
                        s2_actions.popleft()
                        turn_count += 1
                    signed_angle = math.radians(
                        float(s2_settings["turn_action_deg"]) * turn_count
                    )
                    if action_code == 3:
                        signed_angle = -signed_angle
                    s2_turn_target_yaw_odom = _wrap_angle(
                        state.pose.yaw_rad + signed_angle
                    )
                    display_path_odom_xy = None
                    idle_status = "S2 TURNING LEFT" if action_code == 2 else "S2 TURNING RIGHT"
                    logger.info(
                        "[S2Action] %s turn_deg=%.1f target_yaw_deg=%.1f "
                        "remaining_actions=%s",
                        "LEFT" if action_code == 2 else "RIGHT",
                        math.degrees(signed_angle),
                        math.degrees(s2_turn_target_yaw_odom),
                        list(s2_actions),
                    )
                elif action_code == 5:
                    s2_stop_count = 0
                    s2_actions.clear()
                    adapter.stop()
                    s2_need_request = True
                    s2_next_view = "down"
                    s2_min_frame_index = frame_index + int(
                        s2_settings["lookdown_settle_frames"]
                    )
                    idle_status = "S2 LOOKDOWN waiting for down-camera frame"
                    logger.info("[S2LookDown] requested down-camera continuation")

            manual_action = panel.manual_action
            if manual_action != "STOP":
                if not manual_active:
                    if sampling_controller is not None:
                        sampling_controller.reset()
                    reference_path = None
                    tracker = None
                    tracker_mode = None
                    navigation_active = False
                    display_path_odom_xy = None
                    fine_diagnostic_saved_for_goal = False
                    recent_fine_world_poses.clear()
                    last_partial_switch_world_xy = None
                    last_partial_endpoint_world_xy = None
                    rejected_unknown_world_xy = None
                    request_id += 1
                    accepted_request_id = request_id
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    s2_active = False
                    s2_need_request = False
                    s2_actions.clear()
                    s2_turn_target_yaw_odom = None
                    s2_request_id += 1
                    accepted_s2_request_id = s2_request_id
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
            elif s2_turn_target_yaw_odom is not None:
                yaw_error = _wrap_angle(
                    s2_turn_target_yaw_odom - state.pose.yaw_rad
                )
                tolerance = math.radians(float(s2_settings["turn_tolerance_deg"]))
                if abs(yaw_error) <= tolerance:
                    adapter.stop()
                    if abs(state.angular_velocity_radps) < 0.05:
                        completed_yaw = s2_turn_target_yaw_odom
                        s2_turn_target_yaw_odom = None
                        if not s2_actions:
                            s2_need_request = True
                            s2_next_view = "front"
                            s2_min_frame_index = frame_index + 1
                        idle_status = "S2 turn complete"
                        logger.info(
                            "[S2Turn] complete target_deg=%.1f error_deg=%.2f",
                            math.degrees(completed_yaw),
                            math.degrees(yaw_error),
                        )
                else:
                    angular = float(np.clip(
                        float(s2_settings["turn_gain"]) * yaw_error,
                        -float(s2_settings["turn_max_angular_radps"]),
                        float(s2_settings["turn_max_angular_radps"]),
                    ))
                    applied = adapter.apply(VelocityCommand(0.0, angular))
            elif navigation_active and selected_goal_odom_xy is not None:
                goal_world = _odom_path_to_world(selected_goal_odom_xy[None, :], state)[0]
                try:
                    if reference_path is not None and np.linalg.norm(reference_path.goal_world_xy - goal_world) <= 0.01:
                        applied = sampling_controller.step(
                            robot=robot, drive=drive, grid=latest_planning_grid,
                            goal_world_xy=goal_world, now=now, reference_path=reference_path,
                        )
                        idle_status = f"{mode} + {control_method} {sampling_controller.status} v={applied.linear_mps:.2f} w={applied.angular_radps:.2f}"
                        if sampling_controller.status == "REFERENCE_DONE":
                            last_coarse_request_time = last_fine_request_time = -float("inf")
                    else:
                        adapter.stop()
                        idle_status = f"{mode} + {control_method}: waiting for reference path"
                except Exception:
                    adapter.stop()
                    sampling_controller.reset()
                    navigation_active = False
                    idle_status = f"{mode} + {control_method} stopped after error; see log"
                    logger.exception("[SamplingError] mode=%s controller=%s", mode, control_method)
            else:
                adapter.stop()
                if sampling_controller is not None and sampling_controller.goal is not None:
                    sampling_controller.reset()

            if (
                s2_active
                and s2_need_request
                and s2_worker is not None
                and not s2_worker.busy
                and not navigation_active
                and tracker is None
                and s2_turn_target_yaw_odom is None
                and not s2_actions
            ):
                s2_frame = latest_camera_frames.get(s2_next_view)
                if (
                    s2_frame is not None
                    and s2_frame.frame_index >= s2_min_frame_index
                ):
                    s2_request_id += 1
                    accepted_s2_request_id = s2_request_id
                    reset_request = bool(s2_reset_pending)
                    s2_job = _S2InferenceJob(
                        request_id=s2_request_id,
                        session_id=s2_session_id,
                        instruction=episode.instruction,
                        view=s2_next_view,
                        reset=reset_request,
                        look_down=s2_next_view == "down",
                        frame=s2_frame,
                        submitted_sim_time_sec=now,
                    )
                    if s2_worker.submit(s2_job):
                        s2_need_request = False
                        s2_reset_pending = False
                        idle_status = f"S2 inferring from {s2_next_view} view"
                        logger.info(
                            "[S2Request] request=%d session=%s view=%s reset=%s "
                            "frame=%d instruction=%s",
                            s2_request_id,
                            s2_session_id,
                            s2_next_view,
                            reset_request,
                            s2_frame.frame_index,
                            episode.instruction,
                        )

            if (
                navigation_active
                and mode == "COARSE"
                and selected_goal_odom_xy is not None
                and not worker.busy
                and now - last_coarse_request_time
                >= float(settings["execution"]["coarse_planning_period_sec"])
            ):
                last_coarse_request_time = now
                try:
                    forward, left = _odom_goal_to_local(selected_goal_odom_xy, state)
                    rgb, depth = _capture_rgbd(camera)
                    valid_depth = np.isfinite(depth) & (depth > 0.0)
                    depth_valid_fraction = float(
                        np.count_nonzero(valid_depth) / depth.size
                    )
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
                    logger.info(
                        "[NavDPRequest] request=%d goal=[%.2f,%.2f] moving=%s "
                        "image=%dx%d depth_valid=%.3f",
                        request_id,
                        forward,
                        left,
                        not _is_stopped(state),
                        rgb.shape[1],
                        rgb.shape[0],
                        depth_valid_fraction,
                    )
                except Exception as exc:
                    idle_status = "NavDP request failed; see log"
                    logger.error("[NavDPRequest] rejected: %s", exc)

            fine_reference_needs_update = (
                reference_path is None or reference_path.source != "FINE"
                or reference_path.needs_extension((position[0], position[1], robot_yaw_world),
                    lookahead_m=float(settings["execution"]["fine_local_goal_lookahead_m"]))
            ) if mode == "FINE" else False
            if (mode == "FINE" and reference_path is not None and latest_planning_grid is not None
                    and latest_planning_grid.map_version != reference_checked_map_version):
                reference_map_changed = reference_path.map_signature(
                    latest_planning_grid, radius_m=collision_checker.bounding_radius_m) != reference_map_signature
                reference_checked_map_version = latest_planning_grid.map_version
            fine_reference_needs_update |= reference_map_changed if mode == "FINE" else False
            if (
                navigation_active
                and mode == "FINE"
                and fine_reference_needs_update
                and selected_goal_odom_xy is not None
                and latest_snapshot.grid.map_version > 0
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
                    idle_status = (
                        "Fine blocked by nearby unknown cells; see Goal Map"
                        if rejected_unknown_world_xy is not None
                        else "Fine planning input unchanged after failure"
                    )
                    if idle_status != last_reported_idle_status:
                        logger.warning("[NavigationStatus] %s", idle_status)
                        last_reported_idle_status = idle_status
                else:
                    fine_request_id += 1
                    accepted_fine_request_id = fine_request_id
                    job = _FinePlanningJob(
                        request_id=fine_request_id,
                        grid=latest_planning_grid,
                        start_pose_world=start_pose_world,
                        goal_world_xy=target_world_xy,
                        route_hint_world_xy=None,
                        use_restricted_analysis=False,
                        lookahead_m=float(
                            settings["execution"]["fine_local_goal_lookahead_m"]
                        ),
                        submitted_sim_time_sec=now,
                        floor_z_m=float(position[2]) - float(profile.spawn_z_m),
                        recent_world_poses=tuple(recent_fine_world_poses),
                        initial_speed_mps=float(state.linear_velocity_mps),
                        reaction_time_sec=float(
                            fine_tracking["safety_reaction_time_sec"]
                        ),
                        braking_deceleration_mps2=float(
                            fine_tracking["safety_braking_deceleration_mps2"]
                        ),
                    )
                    if fine_worker.submit(job):
                        last_fine_request_time = now
                        logger.info(
                            "[FineRequest] request=%d map_version=%d "
                            "target_world=[%.2f,%.2f]",
                            fine_request_id,
                            latest_snapshot.grid.map_version,
                            target_world_xy[0],
                            target_world_xy[1],
                        )

            if tracker is None and idle_status != last_reported_idle_status:
                logger.info("[NavigationStatus] %s", idle_status)
                last_reported_idle_status = idle_status
            panel.update_controls(mode, idle_status, control_method)
            world.step(render=frame_index % render_stride == 0)
            frame_index += 1
    finally:
        adapter.stop()
        if sampling_controller is not None:
            sampling_controller.close(wait=True)
        panel.shutdown()
        s2_panel.shutdown()
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


def _save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _render_s2_decision(
    rgb: np.ndarray,
    *,
    label: str,
    pixel_vu: tuple[float, float] | None = None,
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[:, :, :3], mode="RGB")
    draw = ImageDraw.Draw(image)
    if pixel_vu is not None:
        v, u = map(float, pixel_vu)
        radius = max(6, int(round(min(image.size) * 0.015)))
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            outline=(255, 45, 35),
            width=max(2, radius // 3),
        )
    text_box_height = 34
    draw.rectangle((0, 0, image.width, text_box_height), fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 255))
    return image


def _pixel_to_original_image(
    pixel_vu: tuple[float, float],
    pixel_space_hw: tuple[int, int],
    image_shape: tuple[int, ...],
) -> tuple[float, float]:
    model_height, model_width = pixel_space_hw
    image_height, image_width = image_shape[:2]
    model_v, model_u = pixel_vu
    return (
        (float(model_v) + 0.5) * image_height / model_height - 0.5,
        (float(model_u) + 0.5) * image_width / model_width - 0.5,
    )


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


def _create_coarse_tracker(path_xy, *, joined_path, config, initial_speed_mps):
    points = np.asarray(path_xy, dtype=np.float64)
    # 静止输出保持静止，局部目标由下一次 NavDP 推理继续处理。
    if len(joined_path) < 2 or not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-6):
        return None
    return PathTracker(joined_path, config, initial_speed_mps=initial_speed_mps)


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


def _turn_actions_before_redecision(
    actions: tuple[int, ...],
    *,
    turn_action_deg: float,
    turn_redecision_deg: float,
) -> tuple[int, ...]:
    maximum_turns = int(round(turn_redecision_deg / turn_action_deg))
    selected: list[int] = []
    direction = None
    turn_count = 0
    for action in actions:
        if action not in {2, 3}:
            if turn_count:
                break
            selected.append(action)
            continue
        if direction is not None and action != direction:
            break
        direction = action
        selected.append(action)
        turn_count += 1
        if turn_count >= maximum_turns:
            break
    return tuple(selected)


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


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    raise SystemExit(main())
