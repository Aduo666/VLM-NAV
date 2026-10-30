from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import logging
import math
from pathlib import Path
import random
import subprocess

import numpy as np
import yaml

from constraint_traverse.collect.teleop import ContinuousTeleop, TeleopConfig
from constraint_traverse.collect.transforms import relative_goal_xy
from constraint_traverse.rgbd_map import RGBDMapConfig, RGBDPlanarMap
from vln_nav.active_perception.bev_goal_map import render_planning_goal_map
from vln_nav.active_perception.camera_geometry import world_from_pose_wxyz
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.control.trajectory import AgxVelocityAdapter, VelocityCommand
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.isaac_runtime_utils import get_joint_names, resolve_robot_relative_prim_path, set_camera_horizontal_fov, set_local_xform, wait_for_stage_assets

from ..contracts import RobotState
from ..cameras import CAMERA_PATHS
from ..experiment import Experiment
from ..mapping import ObservationMapFusion
from . import snapshot
from .sensors import ObservationSensors


class ExperimentRuntime:
    def __init__(self, app, args, root):
        self.app, self.args, self.root = app, args, Path(root)
        self.logger = logging.getLogger("constraint_traverse.active_passage")
        self.closed = self.placing = False
        self.time_offset = 0.0
        self.goal = None if args.goal is None else np.asarray(args.goal, dtype=float)
        self.requested = np.zeros(2)
        self.applied = np.zeros(2)
        self.observations = ()
        self.last_map_update = None
        self.commands = []
        self.await_key_release = False
        self.message = ""
        self.panel = self.sensors = self.teleop = self.experiment = None
        self.navigator = self.contacts = None
        self.generator = np.random.default_rng(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    @property
    def timestamp(self):
        return float(self.world.current_time) + self.time_offset

    def initialize(self):
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
        from pxr import UsdGeom, UsdLux
        import omni.replicator.core as rep

        self.profile = load_agx_profile(project_root=self.root)
        with (self.root / "configs/navigation.yaml").open(encoding="utf-8") as handle:
            self.navigation = yaml.safe_load(handle)
        with (self.root / "constraint_traverse/configs/collect.yaml").open(encoding="utf-8") as handle:
            collect = yaml.safe_load(handle)
        dt = self.profile.simulation.physics_dt
        self.world = World(physics_dt=dt, rendering_dt=self.profile.simulation.rendering_dt, stage_units_in_meters=1.0)
        self.stage = get_current_stage()
        UsdGeom.Xform.Define(self.stage, "/World/Environment")
        if self.args.usd is None:
            self.world.scene.add_default_ground_plane()
        else:
            add_reference_to_stage(str(self.args.usd), "/World/Environment")
        add_reference_to_stage(str(self.profile.asset.usd_path), self.profile.asset.prim_path)
        wait_for_stage_assets(simulation_app=self.app, is_stage_loading=is_stage_loading, timeout_sec=180,
                              logger=self.logger, label="主动感知实验场景")
        light = UsdLux.DomeLight.Define(self.stage, "/World/ActivePassageLight")
        light.CreateIntensityAttr(700.0)
        start = self.args.start or [0.0, 0.0, self.profile.spawn_z_m, 0.0]
        set_local_xform(stage=self.stage, prim_path=self.profile.asset.prim_path,
                        translation=list(start[:3]), rotation_deg=[0.0, 0.0, start[3]])
        self.articulation_path = resolve_robot_relative_prim_path(robot_root_path=self.profile.asset.prim_path,
                                                                 prim_path=self.profile.asset.articulation_path)
        self.robot = self.world.scene.add(SingleArticulation(prim_path=self.articulation_path,
            name="active_passage_agx", reset_xform_properties=False))
        from .contacts import ContactMonitor
        ContactMonitor.enable(self.stage, self.profile.asset.prim_path)
        if self.args.command == "check-sim":
            from isaacsim.core.api.objects import DynamicCuboid
            self.world.scene.add(DynamicCuboid(prim_path="/World/Environment/ResetProbe", name="reset_probe",
                position=np.asarray(start[:3]) + [-2.0, 1.0, 0.5], scale=np.full(3, 0.2), mass=1.0))
        self.world.reset()
        self.robot.initialize()
        self.joint_names = tuple(get_joint_names(self.robot))
        self.create_drive()
        from vln_nav.runtime.agx_geometry import build_agx_footprint, validate_agx_geometry
        from ..traversal.controller import AutomaticTraversal, load_navigation_config
        settings = load_navigation_config(self.args.navigation_config, self.root)
        validate_agx_geometry(self.stage, self.profile, self.articulation_path)
        footprint, self.geometry_audit = build_agx_footprint(self.stage, self.profile, self.articulation_path,
                                                          margin_m=0.03)
        self.navigator = AutomaticTraversal(footprint=footprint, hardware=self.profile.agx_controller.continuous,
            physics_dt=float(self.world.get_physics_dt()), settings=settings, seed=self.args.seed,
            gpu=self.args.gpu, logger=self.logger)
        self.contacts = ContactMonitor(self)
        self.teleop = ContinuousTeleop(TeleopConfig.from_mapping(collect["teleop"]))
        if not self.args.headless:
            self.teleop.bind_keyboard()
        paths = dict(CAMERA_PATHS)
        front_path = resolve_robot_relative_prim_path(robot_root_path=self.profile.asset.prim_path, prim_path=paths["front"])
        set_camera_horizontal_fov(stage=self.stage, prim_path=front_path,
                                  fov_deg=self.navigation["camera"]["horizontal_fov_deg"])
        camera_cfg = self.navigation["mapping"]["camera"]
        self.sensors = ObservationSensors(self, paths, (int(camera_cfg["width"]), int(camera_cfg["height"])))
        rep.orchestrator.set_capture_on_play(False)
        display = self.navigation["map_selector"]
        self.local_map = RGBDPlanarMap(RGBDMapConfig.from_mapping(self.navigation["mapping"]["map"]),
            obstacle_min_height_m=display["obstacle_min_relative_z_m"],
            obstacle_max_height_m=display["obstacle_max_relative_z_m"])
        self.local_map.reset(self.robot_state().world_from_base[:2, 3])
        self.fusion = ObservationMapFusion(self.local_map, self.args.map_sources)
        self.sensor_period = 1.0 / camera_cfg["frequency_hz"]
        self.last_sensor_time = self.last_map_time = -np.inf
        self.floor_offset = float(self.profile.spawn_z_m)
        # 启动时保持暂停，用户确认放置后才允许物理运动。
        self.world.pause()
        for _ in range(3):
            self.world.render()
        self.observations = self.sensors.capture()
        self.update_map(self.observations)
        self.last_sensor_time = self.timestamp
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
                                capture_output=True, check=True).stdout.strip()
        fingerprint = hashlib.sha256()
        sources = sorted((self.root / "constraint_traverse/active_passage").rglob("*.py"))
        sources += sorted((self.root / "constraint_traverse/active_passage/traversal/native").glob("*"))
        sources += sorted((self.root / "constraint_traverse/benchmark").glob("*.py"))
        sources += [self.root / "constraint_traverse/navigation.py", self.root / "constraint_traverse/reference_path.py",
                    Path(settings["control"]["config_path"]), Path(settings["path"]["primitive_file"])]
        sources += sorted((self.root / "src/vln_nav/control").glob("*.py"))
        sources += [self.root / name for name in ("constraint_traverse/collect/teleop.py",
            "constraint_traverse/collect/transforms.py", "constraint_traverse/rgbd_map.py",
            "src/vln_nav/runtime/agx_profile.py", "src/vln_nav/runtime/isaac_runtime_utils.py",
            "src/vln_nav/active_perception/camera_geometry.py", "src/vln_nav/active_perception/bev_goal_map.py")]
        for source in sources:
            fingerprint.update(str(source.relative_to(self.root) if source.is_relative_to(self.root) else source).encode())
            fingerprint.update(source.read_bytes())
        metadata = dict(code_commit=commit, source_sha256=fingerprint.hexdigest(),
            isaac_sim_version=importlib.metadata.version("isaacsim"),
            pytorch_mppi_version=importlib.metadata.version("pytorch-mppi"),
            seed=self.args.seed, scene=str(self.args.usd), robot_asset=str(self.profile.asset.usd_path),
            physics_dt_sec=float(self.world.get_physics_dt()), robot_profile=asdict(self.profile.agx_controller),
            camera_paths=paths, camera_config=camera_cfg,
            robot_depth_filter=self.sensors.robot_filter.metadata(),
            map_config=self.navigation["mapping"], map_sources=list(self.fusion.sources),
            automatic_navigation=dict(path_planner="NAV2_SMAC_STATE_LATTICE_1.1.20", controller="PYTORCH_MPPI",
                settings=settings, control_config=self.navigator.sampling.config, geometry_audit=self.geometry_audit,
                footprint_polygon=self.navigator.path_planner.polygon.tolist(), unknown_policy="blocked"),
            restoration=dict(protocol="RESET_PHYSICS_AND_RESTORE_NUMERIC_STATE",
                             unavailable=["PhysX 接触求解器内部缓存", "GPU 渲染内部缓存"],
                             sensor_policy="恢复记录缓存，下一次采样同步重新渲染",
                             scene_information="完整场景状态仅由仿真恢复模块持有"))
        self.experiment = Experiment(self, self.args.output, metadata, timeout_sec=self.args.timeout)
        self.placing = True
        self.teleop.set_enabled(False)
        if not self.args.headless:
            import omni.ui as ui
            from .ui import ExperimentPanel
            self.panel = ExperimentPanel(ui, self.commands.append, display["image_size_px"], self.fusion.sources)
            self.select_robot()
        self.message = ""

    def create_drive(self):
        self.drive = create_locomotion_controller(profile=self.profile, robot=self.robot, stage=self.stage,
            articulation_path=self.articulation_path, available_joint_names=list(self.joint_names), logger=self.logger)
        self.adapter = AgxVelocityAdapter(self.drive, self.profile.agx_controller.continuous)

    def robot_state(self):
        position, orientation = self.robot.get_world_pose()
        return RobotState(self.timestamp, world_from_pose_wxyz(position, orientation),
            np.concatenate((self.robot.get_linear_velocity(), self.robot.get_angular_velocity())),
            self.joint_names, self.robot.get_joint_positions(), self.robot.get_joint_velocities())

    def stop_motion(self):
        if self.teleop is not None:
            self.teleop.reset()
        if hasattr(self, "adapter"):
            self.adapter.stop()
        if self.navigator is not None:
            self.navigator.active = False
            self.navigator.epoch += 1
            self.navigator.sampling.reset()
        self.requested[:] = self.applied[:] = 0.0
        self.await_key_release = True

    def capture_snapshot(self):
        if self.placing:
            raise ValueError("Confirm the robot placement first.")
        return snapshot.capture(self)

    def restore_snapshot(self, saved):
        errors = snapshot.restore(self, saved)
        self.world.play()
        return errors

    def select_robot(self):
        import omni.usd
        omni.usd.get_context().get_selection().set_selected_prim_paths([self.articulation_path], True)

    def confirm_placement(self):
        from pxr import UsdGeom
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.articulation_path))
        position, rotation = transform.ExtractTranslation(), transform.ExtractRotationQuat()
        self.robot.set_world_pose(position=np.asarray(position),
            orientation=np.asarray([rotation.GetReal(), *rotation.GetImaginary()]))
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))
        self.stop_motion()
        self.world.physics_sim_view.update_articulations_kinematic()
        self.world.play()
        self.placing = False
        self.teleop.set_enabled(True)
        self.experiment.status = "READY"
        self.message = ""
        if not self.args.headless:
            import omni.usd
            omni.usd.get_context().get_selection().set_selected_prim_paths([], True)

    def update_map(self, observations):
        update = self.fusion.integrate(observations, floor_offset_m=self.floor_offset)
        self.last_map_time = self.timestamp
        if update is not None:
            self.last_map_update = update
            if self.experiment is not None and self.experiment.recorder is not None:
                self.experiment.recorder.event("MAP_UPDATE", self.timestamp, update=update)
        return update

    def set_map_sources(self, sources):
        previous = self.fusion.sources
        if not self.fusion.set_sources(sources, timestamp_sec=self.timestamp):
            return
        self.last_map_time = -np.inf
        if self.experiment is not None:
            self.experiment.metadata["map_sources"] = list(self.fusion.sources)
            if self.experiment.recorder is not None:
                self.experiment.recorder.event("MAP_SOURCES_CHANGED", self.timestamp,
                                               previous=list(previous), selected=list(self.fusion.sources))
        self.logger.info("地图融合相机：%s", self.fusion.sources)

    def reset_map(self):
        self.local_map.reset(self.robot_state().world_from_base[:2, 3])
        self.fusion.reset_history()
        self.last_map_update = None
        self.last_map_time = -np.inf
        if self.navigator is not None:
            self.navigator.reset(self.applied)

    def start_navigation(self):
        if self.placing or not self.world.is_playing():
            raise ValueError("Confirm placement and resume simulation playback.")
        if self.experiment.recorder is not None:
            raise ValueError("Stop the current recording first.")
        if self.goal is None:
            raise ValueError("Select a goal on the map.")
        path = self.experiment.start(execution_mode="autonomous")
        self.contacts.events.clear()
        self.navigator.begin(self.goal[:2])
        self.message = f"Navigation: {path.name}"
        self.logger.info("自动导航记录开始：%s", path)

    def record_navigation(self):
        recorder = self.experiment.recorder
        events, self.navigator.events = self.navigator.events, []
        if recorder is None:
            return
        for name, values in events:
            if name == "REFERENCE_PATH":
                path = values["path"]
                recorder.plan(name, self.timestamp,
                    dict(world_xy=path.world_xy, world_yaw_rad=path.world_yaw_rad,
                         motion_direction=path.motion_direction), request_id=path.request_id,
                    map_version=path.map_version, request_timestamp_sec=path.created_sim_time_sec,
                    diagnostics=values["diagnostics"])
            elif name == "CONTROL_PLAN":
                plan = values.pop("plan")
                recorder.plan(name, self.timestamp, dict(controls=plan.controls, prediction=plan.prediction,
                    candidates=plan.candidates, candidate_costs=plan.candidate_costs),
                    diagnostics=plan.diagnostics, planning_sec=plan.planning_sec, **values)
            else:
                recorder.event(name, self.timestamp, **values)

    def render_map(self):
        robot = self.robot_state()
        position = robot.world_from_base[:3, 3]
        yaw = math.atan2(robot.world_from_base[1, 0], robot.world_from_base[0, 0])
        goal_local = None if self.goal is None else relative_goal_xy([*position[:2], yaw], self.goal[:2])
        snap = self.local_map.snapshot()
        display = self.navigation["map_selector"]
        image = render_planning_goal_map(snap.grid, source_mask=snap.source_mask,
            robot_world_xyz=position, robot_yaw_rad=yaw, view_size_m=display["view_size_m"],
            rear_view_m=display["rear_view_m"], image_size_px=display["image_size_px"], goal_local_xy=goal_local)
        if self.navigator is not None:
            from .overlay import draw_navigation
            image = draw_navigation(image, self.navigator, position, yaw)
        return image, (position.copy(), yaw, float(position[2] - self.floor_offset))

    def step(self, command=None):
        dt = float(self.world.get_physics_dt())
        automatic = self.navigator.active
        if automatic:
            self.teleop.reset()
            self.requested, applied = self.navigator.step(robot=self.robot, drive=self.drive,
                grid=self.local_map.snapshot().grid, now=self.timestamp)
            self.applied = np.asarray([applied.linear_mps, applied.angular_radps])
            self.record_navigation()
            self.message = self.navigator.status
        elif command is None:
            desired = self.teleop.step(dt)
            if self.await_key_release:
                self.await_key_release = desired != VelocityCommand()
                self.teleop.reset()
                desired = VelocityCommand()
        else:
            desired = VelocityCommand(*map(float, command))
        if not automatic:
            self.requested = np.asarray([desired.linear_mps, desired.angular_radps])
            applied = self.navigator.sampling.apply_command(robot=self.robot, drive=self.drive, command=self.requested)
            self.applied = np.asarray([applied.linear_mps, applied.angular_radps])
        self.world.step(render=False)
        observations = ()
        if self.timestamp - self.last_sensor_time >= self.sensor_period - 1e-9:
            observations = self.observations = self.sensors.capture()
            self.last_sensor_time = self.timestamp
            if self.timestamp - self.last_map_time >= self.navigation["mapping"]["map"]["integration_period_sec"] - 1e-9:
                self.update_map(observations)
        self.experiment.sample(self.robot_state(), self.requested, self.applied, observations,
            physical_time_sec=self.world.current_time,
            actuator_targets=dict(wheel_radps=self.navigator.sampling.motor_targets,
                                  steering_rad=self.navigator.sampling.steering_targets))
        contacts, self.contacts.events = self.contacts.events, []
        if self.experiment.recorder is not None:
            for contact in contacts:
                self.experiment.recorder.event("PHYSICAL_COLLISION", self.timestamp, **contact)
            if contacts:
                self.experiment.stop("COLLISION", details=dict(contact_pairs=len(contacts)))
            elif automatic and not self.navigator.active:
                self.experiment.stop(self.navigator.status,
                    details=dict(goal_distance_m=float(np.linalg.norm(self.robot_state().world_from_base[:2, 3] - self.goal[:2]))))

    def handle_commands(self):
        while self.commands:
            command = self.commands.pop(0)
            name = command[0] if isinstance(command, tuple) else command
            try:
                if name == "GOAL":
                    if self.experiment.recorder is not None or self.placing:
                        raise ValueError("Stop recording and confirm placement before selecting a goal.")
                    self.goal = command[1].copy()
                elif name == "PLACE":
                    if self.placing:
                        self.confirm_placement()
                    else:
                        self.stop_motion()
                        self.world.pause()
                        self.teleop.set_enabled(False)
                        self.placing = True
                        self.select_robot()
                elif name == "SAVE_START":
                    self.experiment.save_start()
                    self.message = "Start state saved."
                elif name == "START":
                    if not self.world.is_playing():
                        raise ValueError("Resume simulation playback.")
                    path = self.experiment.start()
                    self.message = f"Recording: {path.name}"
                    self.logger.info("试验记录开始：%s", path)
                elif name == "NAVIGATE":
                    self.start_navigation()
                elif name == "STOP":
                    self.experiment.stop()
                elif name == "RESET_TRIAL":
                    self.experiment.reset()
                    self.message = "Trial state restored."
                    self.logger.info("状态恢复误差：%s", self.experiment.last_reset_errors)
                elif name == "RESET_MAP":
                    if self.experiment.recorder is not None:
                        raise ValueError("Stop recording before resetting the map.")
                    self.reset_map()
                elif name == "MAP_SOURCES":
                    self.set_map_sources(command[1])
                else:
                    raise ValueError(f"Unknown command: {name}")
            except ValueError as error:
                self.message = str(error) if str(error).isascii() else "Operation failed. See terminal for details."
                self.logger.info("操作未执行：%s", error)

    def run(self):
        while self.app.is_running() and not self.closed:
            self.handle_commands()
            if not self.placing and self.world.is_playing():
                self.step()
            else:
                self.teleop.reset()
                self.app.update()
            if self.closed:
                break
            if self.panel is not None:
                self.panel.update(self)
        return 0

    def close(self, wait=False):
        if self.closed:
            return
        self.closed = True
        actions = []
        if self.experiment is not None:
            actions.append(lambda: self.experiment.close())
        if self.navigator is not None:
            actions.append(lambda: self.navigator.close(wait=True))
        for resource in (self.teleop, self.panel, self.sensors, self.contacts):
            if resource is not None:
                actions.append(resource.shutdown if resource is self.teleop else resource.close)
        for action in actions:
            try:
                action()
            except Exception:
                self.logger.exception("实验资源关闭失败")
        if hasattr(self, "world"):
            self.world.stop()
            self.world.clear_instance()
