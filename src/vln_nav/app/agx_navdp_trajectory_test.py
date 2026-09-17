"""Phase-1 NavDP integration: click a map goal and navigate to it.

The local map is used only as a human point-goal selector. It does not filter,
score, or certify candidate trajectories; those decisions belong to Phase 2.
NavDP's own selected candidate is followed automatically and regenerated until
the fixed odometry-frame point goal is reached.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
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
    SOURCE_BITS,
    LocalVoxelMap,
    LocalVoxelMapConfig,
)
from vln_nav.app.agx_trajectory_test import _validate_geometry
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import AgxVelocityAdapter, PathTracker
from vln_nav.navigation.navdp_client import (
    NavDPCandidate,
    NavDPClient,
    NavDPClientConfig,
    NavDPResult,
)
from vln_nav.navigation.odometry import OdomState, SimOdometrySensor
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


class _NavDPWorker:
    """Keep network/model latency away from the Isaac simulation thread."""

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


class _GoalMapPanel:
    """Compact control panel whose BEV image is also the point-goal picker."""

    def __init__(self, ui, *, image_size_px: int, command_sink) -> None:
        self._ui = ui
        self._command_sink = command_sink
        self._map: BevGoalMap | None = None
        size = int(image_size_px)
        initial = np.empty((size, size, 4), dtype=np.uint8)
        initial[:, :, :3] = (108, 110, 112)
        initial[:, :, 3] = 255
        self._provider = ui.ByteImageProvider()
        self._provider.set_data_array(initial.reshape(-1), [size, size])
        self.window = ui.Window(
            "AGX NavDP Goal",
            width=size + 34,
            height=size + 150,
            visible=True,
        )
        with self.window.frame:
            with ui.VStack(spacing=7):
                self._image = ui.ImageWithProvider(
                    self._provider,
                    width=size,
                    height=size,
                    fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    pixel_aligned=True,
                    opaque_for_mouse_events=True,
                )
                self._image.set_mouse_pressed_fn(self._on_map_pressed)
                self.goal_label = ui.Label("Click known free space to navigate")
                ui.Button("STOP", height=34, clicked_fn=lambda: command_sink("STOP"))
                self.candidate_label = ui.Label("Candidate: none")
                self.status_label = ui.Label("Building local map", word_wrap=True)

    def update_map(self, goal_map: BevGoalMap) -> None:
        self._map = goal_map
        height, width = goal_map.rgba.shape[:2]
        self._provider.set_data_array(goal_map.rgba.reshape(-1), [width, height])

    def set_goal_text(self, forward_m: float, left_m: float) -> None:
        self.goal_label.text = (
            f"Goal  forward {forward_m:.2f} m   left {left_m:+.2f} m"
        )

    def set_goal_error(self, message: str) -> None:
        self.goal_label.text = message

    def shutdown(self) -> None:
        self.window.visible = False

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
            self.set_goal_error("Select known free space")
            return
        forward, left = self._map.pixel_to_local(column, row)
        self._command_sink(("SELECT_GOAL", forward, left))


def _load_settings() -> tuple[dict, NavDPClientConfig, dict, LocalVoxelMapConfig]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    expected = {"server", "camera", "goal", "execution", "map_selector", "debug"}
    if set(data) != expected:
        raise ValueError(f"NavDP Phase-1 config requires exactly: {sorted(expected)}")
    server = NavDPClientConfig.from_mapping(data["server"])

    camera = data["camera"]
    if set(camera) != {"width", "height", "frequency_hz", "warmup_render_frames"}:
        raise ValueError("Invalid NavDP Phase-1 camera fields")
    if any(int(camera[name]) <= 0 for name in camera):
        raise ValueError("NavDP Phase-1 camera values must be positive")

    goal = data["goal"]
    if set(goal) != {
        "min_distance_m",
        "max_distance_m",
    }:
        raise ValueError("Invalid NavDP Phase-1 goal fields")
    if not 0 < float(goal["min_distance_m"]) < float(goal["max_distance_m"]) <= 10.0:
        raise ValueError("NavDP point-goal distance bounds must satisfy 0 < min < max <= 10")

    execution = data["execution"]
    if set(execution) != {
        "max_result_pose_delta_m",
        "max_result_yaw_delta_deg",
        "goal_reached_tolerance_m",
    }:
        raise ValueError("Invalid NavDP Phase-1 execution fields")
    if any(float(value) <= 0 for value in execution.values()):
        raise ValueError("NavDP execution tolerances must be positive")

    selector = data["map_selector"]
    selector_fields = {
        "source_config_path",
        "image_size_px",
        "view_size_m",
        "rear_view_m",
        "render_period_sec",
        "obstacle_min_relative_z_m",
        "obstacle_max_relative_z_m",
    }
    if set(selector) != selector_fields:
        raise ValueError(f"NavDP map_selector requires exactly: {sorted(selector_fields)}")
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
    if set(debug) != {"candidate_image_path", "view_size_m"}:
        raise ValueError("Invalid NavDP Phase-1 debug fields")
    if not str(debug["candidate_image_path"]).strip():
        raise ValueError("NavDP candidate debug path is empty")
    if float(debug["view_size_m"]) <= 0:
        raise ValueError("Invalid NavDP candidate debug view size")
    return data, server, map_settings, map_config


def main() -> int:
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
    settings, client_config, map_settings, map_config = _load_settings()
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
                logger,
            )
        except Exception:
            logger.exception("[NavDPPhase1Fatal] test crashed")
            return 1
    finally:
        app.close()


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("agx_navdp_phase1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = PROJECT_ROOT / "logs/agx_navdp_trajectory_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("[NavDPPhase1] log=%s", log_path)
    return logger


def _run(
    app,
    config,
    profile,
    settings,
    client_config,
    map_settings,
    map_config,
    logger,
) -> int:
    import omni.ui as ui
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux

    dt = float(profile.simulation.physics_dt)
    world = World(
        physics_dt=dt,
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    scene_path = Path(str(config.scene.usd_path))
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path
    if not scene_path.is_file():
        raise FileNotFoundError(f"NavDP Phase-1 scene USD does not exist: {scene_path}")
    add_reference_to_stage(str(scene_path), config.scene.prim_path)
    if config.scene.add_default_ground_plane:
        world.scene.add_default_ground_plane()
    if config.scene.add_dome_light:
        UsdLux.DomeLight.Define(stage, "/World/NavDPTestLight").CreateIntensityAttr(1500.0)
    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    wait_for_stage_assets(
        simulation_app=app,
        is_stage_loading=is_stage_loading,
        timeout_sec=config.scene.stage_load_timeout_sec,
        logger=logger,
        label="AGX NavDP Phase-1 scene",
    )
    set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[
            float(config.scene.robot_start_xy_m[0]),
            float(config.scene.robot_start_xy_m[1]),
            float(profile.spawn_z_m),
        ],
        rotation_deg=config.scene.robot_start_orientation_deg,
    )
    articulation_path = resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    _validate_geometry(stage, profile, articulation_path)
    robot = world.scene.add(
        SingleArticulation(
            prim_path=articulation_path,
            name="agx_navdp_phase1",
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
    map_role_paths = {
        source: resolve_existing_prim_path(
            stage=stage,
            prim_path=resolve_robot_relative_prim_path(
                robot_root_path=profile.asset.prim_path,
                prim_path=map_settings["sensors"][f"{source}_prim_path"],
            ),
            kind=f"NavDP goal-map {source} camera",
        )
        for source in SOURCE_BITS
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
    for source in ("down", "arm"):
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
    logger.info(
        "[NavDPPhase1] scene=%s camera=%s server=%s frame=base(+X forward,+Y left)",
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

    for _ in range(int(camera_config["warmup_render_frames"])):
        adapter.stop()
        world.step(render=True)

    position, orientation = get_world_pose(robot)
    local_map = LocalVoxelMap(map_config)
    local_map.reset(np.asarray(position[:3], dtype=np.float64))

    tracker = None
    frozen_candidates: list[_FrozenCandidate] = []
    selected_index = 0
    request_id = 0
    accepted_request_id = 0
    active_job: _InferenceJob | None = None
    selected_goal_odom_xy: np.ndarray | None = None
    navigation_active = False
    replan_pending = False
    segment_index = 0
    idle_status = "Click known free space to navigate"
    previous_tracking_status = None
    previous_adapter_status = None
    last_steering_log_time = -float("inf")
    render_stride = max(1, round(float(profile.simulation.rendering_dt) / dt))
    frame_index = 0
    debug_path = PROJECT_ROOT / settings["debug"]["candidate_image_path"]
    last_map_integration_time = -float("inf")
    last_map_render_time = -float("inf")
    map_integration_period = float(map_settings["map"]["integration_period_sec"])
    map_render_period = float(settings["map_selector"]["render_period_sec"])

    try:
        while app.is_running():
            if not world.is_playing():
                adapter.stop()
                tracker = None
                command[0] = None
                panel.status_label.text = "PAUSED: resume, then select a goal"
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
                        last_map_integration_time = now
            if now - last_map_render_time >= map_render_period:
                displayed_goal = (
                    None
                    if selected_goal_odom_xy is None
                    else _odom_goal_to_local(selected_goal_odom_xy, state)
                )
                goal_map = render_bev_goal_map(
                    local_map.snapshot(),
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
                panel.update_map(goal_map)
                if displayed_goal is not None:
                    panel.set_goal_text(*displayed_goal)
                last_map_render_time = now

            completed = worker.poll()
            if completed is not None:
                job, result, error = completed
                if job.request_id != accepted_request_id:
                    logger.info("[NavDPResult] ignored stale request=%d", job.request_id)
                elif error is not None:
                    active_job = None
                    navigation_active = False
                    replan_pending = False
                    idle_status = f"INFERENCE FAILED: {error}"
                    logger.error("[NavDPResult] request=%d failed: %s", job.request_id, error)
                else:
                    active_job = None
                    drift_m = float(np.hypot(
                        state.pose.x - job.odom_state.pose.x,
                        state.pose.y - job.odom_state.pose.y,
                    ))
                    drift_deg = abs(math.degrees(_wrap_angle(
                        state.pose.yaw_rad - job.odom_state.pose.yaw_rad
                    )))
                    execution = settings["execution"]
                    if (
                        drift_m > float(execution["max_result_pose_delta_m"])
                        or drift_deg > float(execution["max_result_yaw_delta_deg"])
                    ):
                        navigation_active = False
                        replan_pending = False
                        idle_status = "RESULT REJECTED: robot moved during inference"
                        logger.error(
                            "[NavDPResult] request=%d rejected pose_delta=%.3fm yaw_delta=%.2fdeg",
                            job.request_id,
                            drift_m,
                            drift_deg,
                        )
                    else:
                        frozen_candidates = _freeze_candidates(result, job)
                        selected_index = result.model_choice_index
                        _save_candidate_debug(
                            debug_path,
                            result,
                            selected_index=selected_index,
                            goal_xy=(job.goal_forward_m, job.goal_left_m),
                            view_size_m=float(settings["debug"]["view_size_m"]),
                        )
                        logger.info(
                            "[NavDPResult] request=%d candidates=%d model_choice=%d "
                            "latency=%.3fs goal=[%.2f,%.2f]",
                            job.request_id,
                            len(frozen_candidates),
                            result.model_choice_index,
                            result.latency_sec,
                            job.goal_forward_m,
                            job.goal_left_m,
                        )
                        _log_candidate_table(logger, frozen_candidates)
                        selected = frozen_candidates[selected_index]
                        tracker = PathTracker(
                            selected.odom_path_xy,
                            profile.agx_controller.continuous,
                        )
                        segment_index += 1
                        previous_tracking_status = None
                        previous_adapter_status = None
                        idle_status = (
                            f"AUTO EXECUTE segment={segment_index} model candidate "
                            f"#{selected.source.index}"
                        )
                        logger.info(
                            "[NavDPAutoExecute] segment=%d candidate=%d rank=%s "
                            "value=%s length=%.2fm",
                            segment_index,
                            selected.source.index,
                            selected.rank,
                            _format_value(selected.source.value),
                            float(tracker.s[-1]),
                        )

            if command[0] is not None:
                action, command[0] = command[0], None
                if isinstance(action, tuple) and action[0] == "SELECT_GOAL":
                    if tracker is not None or worker.busy or not _is_stopped(state):
                        idle_status = "WAIT: stop before selecting a new goal"
                        panel.set_goal_error("Stop before selecting a new goal")
                    else:
                        try:
                            forward, left = _validate_goal_values(
                                float(action[1]),
                                float(action[2]),
                                settings,
                            )
                            selected_goal_odom_xy = _transform_xy(
                                np.asarray([[forward, left]], dtype=np.float64),
                                state.pose.x,
                                state.pose.y,
                                state.pose.yaw_rad,
                            )[0]
                            frozen_candidates = []
                            selected_index = 0
                            navigation_active = True
                            replan_pending = True
                            segment_index = 0
                            panel.set_goal_text(forward, left)
                            idle_status = "Goal selected: waiting to request NavDP"
                            logger.info(
                                "[NavDPGoal] selected forward=%.2fm left=%+.2fm "
                                "odom=[%.2f,%.2f] auto=1",
                                forward,
                                left,
                                float(selected_goal_odom_xy[0]),
                                float(selected_goal_odom_xy[1]),
                            )
                        except ValueError as exc:
                            panel.set_goal_error(str(exc))
                            idle_status = f"GOAL REJECTED: {exc}"
                elif action == "STOP":
                    request_id += 1
                    accepted_request_id = request_id
                    adapter.stop()
                    tracker = None
                    active_job = None
                    navigation_active = False
                    replan_pending = False
                    idle_status = "CANCELLED"
                    logger.info("[NavDPControl] STOP")

            if tracker is not None:
                tracking = tracker.step(
                    (state.pose.x, state.pose.y, state.pose.yaw_rad),
                    dt,
                )
                applied = adapter.apply(tracking.command)
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
                        "[NavDPSteering] status=%s max_error_deg=%.2f "
                        "target_deg=%s measured_deg=%s",
                        adapter.status,
                        math.degrees(float(steering["max_error_rad"])),
                        _degree_map(steering["target_rad"]),
                        _degree_map(steering["measured_rad"]),
                    )
                    previous_adapter_status = adapter.status
                    last_steering_log_time = now
                status_text = (
                    f"{tracking.status}/{adapter.status} candidate={selected_index} "
                    f"remaining={tracking.remaining_m:.2f}m error={tracking.cross_track_m:.2f}m\n"
                    f"v={applied.linear_mps:.3f}m/s w={applied.angular_radps:.3f}rad/s"
                )
                if tracking.status != previous_tracking_status:
                    logger.info("[NavDPTracking] %s", status_text.replace("\n", " "))
                    previous_tracking_status = tracking.status
                if tracking.status != "TRACKING":
                    adapter.stop()
                    if tracking.status == "ARRIVED" and selected_goal_odom_xy is not None:
                        goal_distance = float(np.linalg.norm(
                            selected_goal_odom_xy
                            - np.asarray([state.pose.x, state.pose.y], dtype=np.float64)
                        ))
                        if goal_distance <= float(
                            settings["execution"]["goal_reached_tolerance_m"]
                        ):
                            navigation_active = False
                            replan_pending = False
                            idle_status = f"GOAL REACHED: distance={goal_distance:.2f}m"
                            logger.info(
                                "[NavDPGoalReached] segments=%d distance=%.3fm",
                                segment_index,
                                goal_distance,
                            )
                        else:
                            replan_pending = True
                            idle_status = (
                                f"SEGMENT {segment_index} COMPLETE: goal remains "
                                f"{goal_distance:.2f}m; replanning"
                            )
                            logger.info(
                                "[NavDPSegment] segment=%d complete goal_remaining=%.3fm",
                                segment_index,
                                goal_distance,
                            )
                    else:
                        navigation_active = False
                        replan_pending = False
                        idle_status = status_text
                        logger.error(
                            "[NavDPTrackingFailed] segment=%d status=%s",
                            segment_index,
                            tracking.status,
                        )
                    tracker = None
            else:
                adapter.stop()
                status_text = idle_status

            if (
                navigation_active
                and replan_pending
                and tracker is None
                and not worker.busy
                and selected_goal_odom_xy is not None
            ):
                if not _is_stopped(state):
                    idle_status = "WAITING FOR FULL STOP BEFORE NAVDP"
                else:
                    goal_distance = float(np.linalg.norm(
                        selected_goal_odom_xy
                        - np.asarray([state.pose.x, state.pose.y], dtype=np.float64)
                    ))
                    if goal_distance <= float(
                        settings["execution"]["goal_reached_tolerance_m"]
                    ):
                        navigation_active = False
                        replan_pending = False
                        idle_status = f"GOAL REACHED: distance={goal_distance:.2f}m"
                        logger.info(
                            "[NavDPGoalReached] segments=%d distance=%.3fm",
                            segment_index,
                            goal_distance,
                        )
                    else:
                        try:
                            forward, left = _validate_goal_values(
                                *_odom_goal_to_local(selected_goal_odom_xy, state),
                                settings,
                            )
                            panel.set_goal_text(forward, left)
                            rgb, depth = _capture_rgbd(camera)
                            request_id += 1
                            accepted_request_id = request_id
                            active_job = _InferenceJob(
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
                            if not worker.submit(active_job):
                                raise RuntimeError(
                                    "NavDP worker rejected a concurrent request"
                                )
                            replan_pending = False
                            idle_status = (
                                f"INFERENCE: goal forward={forward:.2f}m "
                                f"left={left:+.2f}m"
                            )
                            logger.info(
                                "[NavDPRequest] request=%d segment=%d "
                                "goal=[%.2f,%.2f] rgb=%s depth=%s",
                                request_id,
                                segment_index + 1,
                                forward,
                                left,
                                rgb.shape,
                                depth.shape,
                            )
                        except Exception as exc:
                            active_job = None
                            navigation_active = False
                            replan_pending = False
                            idle_status = f"REQUEST REJECTED: {exc}"
                            logger.error("[NavDPRequest] rejected: %s", exc)

            if tracker is None:
                status_text = idle_status

            if frozen_candidates:
                selected = frozen_candidates[selected_index]
                panel.candidate_label.text = (
                    f"Candidate #{selected.source.index} | rank={selected.rank or 'N/A'} | "
                    f"value={_format_value(selected.source.value)} | "
                    f"model_choice={selected.source.is_model_choice}"
                )
            else:
                panel.candidate_label.text = "Candidate: none"
            if frame_index % render_stride == 0:
                panel.status_label.text = status_text
            world.step(render=frame_index % render_stride == 0)
            frame_index += 1
    finally:
        adapter.stop()
        panel.shutdown()
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


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


if __name__ == "__main__":
    raise SystemExit(main())
