from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from constraint_traverse.collect.teleop import ContinuousTeleop, TeleopConfig
from constraint_traverse.collect.transforms import pose_xyyaw
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.runtime.agx_geometry import build_agx_footprint, validate_agx_geometry
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.isaac_runtime_utils import get_joint_names, resolve_robot_relative_prim_path, set_local_xform, wait_for_stage_assets

from .geometry import DoorScene, PassageGeometry
from .model import FourWheelModel, MotionLimits, STATE_SIZE, read_steering_targets, read_wheel_targets
from .planner import PassageCost, SamplingPlanner
from .record import TrialRecord, TrialSnapshot
from .ui import BenchmarkPanel, MapRenderer


class PassageRuntime:
    def __init__(self, app, root, config, output, headless=False, goal=None, repeat=1):
        self.app, self.root, self.config = app, Path(root), config
        self.output, self.headless, self.repeat = Path(output), headless, repeat
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64)
        self.logger = logging.getLogger("constraint_traverse.benchmark")
        self.mode, self.result_text = "READY", "-"
        self.record = self.snapshot = self.last_plan = None
        self.planner = self.reference = None
        self.last_command = np.zeros(2, dtype=np.float32)
        self.requested = np.zeros(2, dtype=np.float32)
        self.commands, self.contacts, self.actual = [], [], []
        self.reset_errors = None
        self.completed = 0
        self.map_version = 1
        self.summary = []
        self.motor_targets, self.steering_targets = {}, {}

    def _initialize(self):
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
        from omni.physx import get_physx_simulation_interface

        torch.set_num_threads(1)
        self.profile = load_agx_profile(project_root=self.root)
        self.dt = 1.0 / round(1.0 / self.profile.simulation.physics_dt)
        self.period_steps = round(self.config["planner"]["dt_sec"] / self.dt)
        if not math.isclose(self.period_steps * self.dt, self.config["planner"]["dt_sec"], rel_tol=1.0e-4):
            raise ValueError("规划周期必须是物理周期的整数倍")
        self.world = World(physics_dt=self.dt, rendering_dt=1.0 / round(1.0 / self.profile.simulation.rendering_dt), stage_units_in_meters=1.0)
        self.stage = get_current_stage()
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(str(self.profile.asset.usd_path), self.profile.asset.prim_path)
        wait_for_stage_assets(simulation_app=self.app, is_stage_loading=is_stage_loading, timeout_sec=120,
                              logger=self.logger, label="受限空间实验机器人")
        cfg = self.config["scene"]
        set_local_xform(stage=self.stage, prim_path=self.profile.asset.prim_path,
                        translation=[cfg["start_x_m"], cfg["start_y_m"], self.profile.spawn_z_m],
                        rotation_deg=[0, 0, cfg["start_yaw_deg"]])
        self.articulation_path = resolve_robot_relative_prim_path(robot_root_path=self.profile.asset.prim_path,
                                                                  prim_path=self.profile.asset.articulation_path)
        validate_agx_geometry(self.stage, self.profile, self.articulation_path)
        footprint, self.audit = build_agx_footprint(self.stage, self.profile, self.articulation_path,
                                                   margin_m=self.config["safety"]["margin_m"])
        parts = [[p.min_x_m, p.max_x_m, p.min_y_m, p.max_y_m] for p in footprint.parts]
        width = max(p[3] for p in parts) - min(p[2] for p in parts)
        self.scene = DoorScene(width + cfg["door_clearance_m"], cfg["door_thickness_m"], cfg["room_half_size_m"], cfg["wall_height_m"])
        self.geometry = PassageGeometry(parts, self.scene, **{name: self.config["safety"][name] for name in ("margin_m", "sweep_step_m")})
        for index, bounds in enumerate(self.scene.obstacles):
            cube = UsdGeom.Cube.Define(self.stage, f"/World/Passage/Walls/Wall{index}")
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.65, 0.70)])
            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, self.scene.height_m / 2))
            xform.AddScaleOp().Set(Gf.Vec3d(bounds[1] - bounds[0], bounds[3] - bounds[2], self.scene.height_m))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        light = UsdLux.DomeLight.Define(self.stage, "/World/Passage/Light")
        light.CreateIntensityAttr(700.0)
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.profile.asset.prim_path), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr(0.0)
        self.robot = self.world.scene.add(SingleArticulation(prim_path=self.articulation_path, name="passage_agx", reset_xform_properties=False))
        self.world.reset()
        self.robot.initialize()
        self.drive = self._create_drive()
        limits = MotionLimits(**self.config["motion"])
        self.model = FourWheelModel(self.profile.agx_controller.continuous, limits, self.config["planner"]["dt_sec"],
                                   device=self.config["planner"]["device"], integration_dt_sec=self.dt)
        self.execution_model = FourWheelModel(self.profile.agx_controller.continuous, limits, self.dt, integration_dt_sec=self.dt)
        self.cost = PassageCost(self.geometry, self.config["cost"], device=self.model.device)
        self.renderer = MapRenderer(self.geometry, self.config["trial"]["map_size_px"])
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.teleop = ContinuousTeleop(TeleopConfig(**{name: self.config["motion"][name] for name in TeleopConfig.__dataclass_fields__}))
        self.panel = None
        if not self.headless:
            import omni.ui as ui
            from isaacsim.core.utils.viewports import set_camera_view
            self.teleop.bind_keyboard()
            self.panel = BenchmarkPanel(ui, self.renderer, self.commands.append, self.config["planner"]["method"],
                                        self.config["trial"]["navigation_condition"])
            set_camera_view(eye=np.asarray([-3, 4, 3]), target=np.asarray([0, 0, 0.5]))
        for _ in range(int(self.config["trial"]["settle_frames"])):
            self._apply(np.zeros(2))
            self.world.step(render=not self.headless)
        self.logger.info("[Geometry] colliders=%d body_width=%.4f door_width=%.4f clearance=%.3f",
                         self.audit["collider_count"], self.geometry.body_width_m, self.scene.width_m, cfg["door_clearance_m"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.world.render()

    def _create_drive(self):
        return create_locomotion_controller(profile=self.profile, robot=self.robot, stage=self.stage,
                                            articulation_path=self.articulation_path, available_joint_names=get_joint_names(self.robot), logger=self.logger)

    def _state(self):
        position, orientation = self.robot.get_world_pose()
        pose = pose_xyyaw(position, orientation)
        linear = np.asarray(self.robot.get_linear_velocity())
        angular = np.asarray(self.robot.get_angular_velocity())
        c, s = math.cos(pose[2]), math.sin(pose[2])
        readback = self.drive.readback()
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        state[:3] = pose
        state[3:6] = [c * linear[0] + s * linear[1], -s * linear[0] + c * linear[1], angular[2]]
        for index, wheel in enumerate(self.profile.agx_controller.continuous.wheels):
            state[6 + index] = readback["steering_position"][wheel.steering_joint] * wheel.steering_sign
            state[10 + index] = readback["wheel_velocity"][wheel.drive_joint] * wheel.radius_m * wheel.drive_sign
        state[14:16] = self.last_command
        state[17:21] = read_steering_targets(self.drive, self.profile.agx_controller.continuous, state[6:10])
        state[21:25] = read_wheel_targets(self.drive, self.profile.agx_controller.continuous, state[10:14])
        return state

    def _apply(self, command):
        with torch.no_grad():
            next_state = self.execution_model.step(self._state(), command).numpy()
        motors, steering = {}, {}
        for index, wheel in enumerate(self.profile.agx_controller.continuous.wheels):
            motors[wheel.drive_joint] = float(next_state[21 + index] / wheel.radius_m / wheel.drive_sign)
            steering[wheel.steering_joint] = float(next_state[17 + index] / wheel.steering_sign)
        self.drive.apply_continuous_joint_targets(motors, steering)
        self.motor_targets, self.steering_targets = motors, steering
        self.last_command = next_state[14:16].copy()
        return self.last_command

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        for header in headers:
            if header.num_contact_data == 0:
                continue
            paths = [str(PhysicsSchemaTools.intToSdfPath(getattr(header, field))) for field in ("actor0", "actor1", "collider0", "collider1")]
            if any(p.startswith(self.profile.asset.prim_path + "/") for p in paths) and any(p.startswith("/World/Passage/Walls/") for p in paths):
                if self.mode == "RUNNING":
                    self.contacts.append({"time_sec": float(self.world.current_time - self.start_time), "paths": paths,
                                          "contact_count": int(header.num_contact_data)})

    def _reference_path(self, state):
        anchors = np.asarray([state[:2], [-0.85, 0], [0, 0], [0.85, 0], self.goal])
        chunks = [np.linspace(a, b, max(2, math.ceil(np.linalg.norm(b - a) / 0.05))) for a, b in zip(anchors[:-1], anchors[1:])]
        return np.concatenate(chunks)

    def _start(self, method, condition):
        self.world.physics_sim_view.update_articulations_kinematic()
        state = self._state()
        cfg = self.config["trial"]
        if self.goal is None:
            self.result_text = "SELECT GOAL"
            return False
        if (np.linalg.norm(state[3:5]) > cfg["stopped_linear_mps"]
                or abs(state[5]) > cfg["stopped_angular_radps"]
                or abs(state[14]) > cfg["stopped_linear_mps"]
                or abs(state[15]) > cfg["stopped_angular_radps"]
                or np.max(np.abs(state[10:14])) > cfg["stopped_linear_mps"]):
            self.result_text = "ROBOT MOVING"
            return False
        if not self.geometry.validate_poses([state[:3]]).safe:
            self.result_text = "UNSAFE START"
            return False
        if self.geometry.completely_passed(state[:3]):
            self.result_text = "START ALREADY PASSED"
            return False
        if self.goal[0] <= self.scene.exit_x_m or not self.geometry.validate_poses([[*self.goal, 0]]).safe:
            self.result_text = "INVALID GOAL"
            return False
        self.teleop.reset()
        self.snapshot = TrialSnapshot.capture(self.robot, self.last_command, self.goal, self.config["planner"]["seed"])
        self.saved_link_states = self._link_states()
        self.saved_controller = copy.deepcopy({"motors": self.motor_targets, "steering": self.steering_targets,
                                               "hold": self.drive.hold_joint_targets})
        self.saved_map_version = self.map_version
        # 所有试验使用相同的接触缓存重建过程，包括第一次 START。
        self._reset()
        state = self._state()
        self.reference = self._reference_path(state) if condition == "reference" else None
        if self.reference is not None and not self.geometry.validate_poses(np.column_stack((self.reference, np.zeros(len(self.reference))))).safe:
            self.result_text = "INVALID REFERENCE"
            return False
        self.cost.set_target(self.goal, self.reference)
        self.planner = SamplingPlanner(method, self.model, self.cost, self.config["planner"], self.config["safety"])
        trial_config = copy.deepcopy(self.config)
        trial_config["planner"]["method"] = method
        trial_config["trial"]["navigation_condition"] = condition
        metadata = {
            "config": trial_config, "method": method, "navigation_condition": condition,
            "map_condition": "GROUND_TRUTH", "map_version": self.map_version,
            "door_width_m": self.scene.width_m, "body_width_m": self.geometry.body_width_m,
            "geometry_audit": self.audit, "parts": self.geometry.parts.tolist(),
            "scene": self.scene.__dict__, "initial_state": self.snapshot.as_dict(), "goal_world_xy": self.goal.tolist(),
            "initial_rigid_body_states": self.saved_link_states, "initial_controller_state": self.saved_controller,
            "reference_path": None if self.reference is None else self.reference.tolist(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("pytorch-mppi", "torch", "shapely")},
            "robot_usd": str(self.profile.asset.usd_path), "code": self._code_metadata(),
            "robot_motion_config": self._hardware_metadata(),
            "physics_dt_sec": self.dt, "reset_protocol": "STATIC_ARTICULATION_AND_CONTROLLER_REBUILD",
            "simulation_schedule": "SYNCHRONOUS_PLANNING",
        }
        self.record = TrialRecord(self.output, metadata)
        self.contacts, self.actual = [], []
        self.start_time = float(self.world.current_time)
        self.step_index, self.stopped_since = 0, None
        self.mode, self.result_text = "RUNNING", "-"
        self.logger.info("[TrialStart] method=%s condition=%s goal=%s output=%s", method, condition, self.goal.tolist(), self.record.path)
        return True

    def _code_metadata(self):
        def git(*args):
            return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=True).stdout
        sources = sorted((self.root / "constraint_traverse/benchmark").glob("*.py"))
        fingerprint = hashlib.sha256()
        for source in sources:
            fingerprint.update(source.name.encode())
            fingerprint.update(source.read_bytes())
        return {"commit": git("rev-parse", "HEAD").strip(), "worktree": git("status", "--short"),
                "benchmark_sha256": fingerprint.hexdigest(), "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()}

    def _hardware_metadata(self):
        from dataclasses import asdict
        return asdict(self.profile.agx_controller.continuous)

    def _finish(self, reason):
        state = self._state()
        reached = self.goal is not None and np.linalg.norm(state[:2] - self.goal) <= self.config["trial"]["goal_tolerance_m"]
        metrics = self.record.finish(reason, passed=self.geometry.completely_passed(state[:3]), goal_reached=reached,
                                     contacts=self.contacts, reset_errors=self.reset_errors)
        self.logger.info("[TrialEnd] %s path=%s", json.dumps(metrics, ensure_ascii=False), self.record.path)
        self.summary.append({"path": str(self.record.path), **metrics})
        self.record = None
        self.completed += 1
        self.mode, self.result_text = "READY", reason
        self.requested[:] = 0
        self.teleop.reset()

    def _reset(self):
        if self.mode == "RUNNING":
            self._finish("RESET_BY_USER")
        if self.snapshot is None:
            self.result_text = "NO SAVED TRIAL"
            return
        self.world.reset()
        self.robot.initialize()
        self.snapshot.restore(self.robot)
        self.drive = self._create_drive()
        self.drive.hold_joint_targets.update(self.saved_controller["hold"])
        self.last_command = self.snapshot.last_command.copy()
        motors, steering = self.saved_controller["motors"].copy(), self.saved_controller["steering"].copy()
        self.drive._apply_targets(wheel_targets=motors, steering_targets=steering, force_wheel_state=False)
        # PhysX 根据底盘和全部关节状态重建非根刚体的运动学缓存。
        self.world.physics_sim_view.update_articulations_kinematic()
        self.motor_targets, self.steering_targets = motors, steering
        self.goal = self.snapshot.goal.copy()
        self.config["planner"]["seed"] = self.snapshot.seed
        self.teleop.reset()
        self.planner = self.last_plan = self.reference = None
        self.contacts, self.actual = [], []
        self.reset_errors = self.snapshot.errors(self.robot)
        restored_links = self._link_states()
        if len(restored_links) != len(self.saved_link_states):
            raise RuntimeError("重置后的刚体数量与保存状态不一致")
        self.reset_errors["rigid_body_states"] = max(
            float(np.max(np.abs(np.asarray(restored[name]) - np.asarray(saved[name]))))
            for restored, saved in zip(restored_links, self.saved_link_states)
            for name in ("position", "orientation_xyzw", "linear_velocity", "angular_velocity")
        )
        if max(self.reset_errors.values()) > 1.0e-5:
            raise RuntimeError(f"试验状态恢复检查失败：{self.reset_errors}")
        self.mode, self.result_text = "READY", "RESET COMPLETE"
        self.map_version = self.saved_map_version
        self.requested[:] = 0
        self.logger.info("[TrialReset] errors=%s", self.reset_errors)

    def _link_states(self):
        dc, articulation = self.drive._dc, self.drive._articulation
        result = []
        for index in range(dc.get_articulation_body_count(articulation)):
            body = dc.get_articulation_body(articulation, index)
            pose = dc.get_rigid_body_pose(body)
            linear, angular = dc.get_rigid_body_linear_velocity(body), dc.get_rigid_body_angular_velocity(body)
            result.append({"index": index, "position": [pose.p.x, pose.p.y, pose.p.z],
                           "orientation_xyzw": [pose.r.x, pose.r.y, pose.r.z, pose.r.w],
                           "linear_velocity": [linear.x, linear.y, linear.z],
                           "angular_velocity": [angular.x, angular.y, angular.z]})
        return result

    def _handle_commands(self):
        from pxr import UsdGeom
        import omni.usd
        while self.commands:
            command = self.commands.pop(0)
            if isinstance(command, tuple):
                if command[0] == "GOAL" and self.mode == "READY":
                    if self.geometry.validate_poses([[*command[1], 0]]).safe:
                        self.goal = command[1].copy()
                elif command[0] == "START" and self.mode == "READY":
                    self._start(command[1], command[2])
            elif command == "STOP":
                if self.mode == "RUNNING":
                    self._finish("STOP_BY_USER")
            elif command == "RESET_TRIAL":
                self._reset()
            elif command == "RESET_MAP" and self.mode == "READY":
                self.map_version += 1
                self.renderer = MapRenderer(self.geometry, self.config["trial"]["map_size_px"])
                self.result_text = "MAP REBUILT"
            elif command == "BORN":
                if self.mode == "READY":
                    self.teleop.reset()
                    self.world.pause()
                    omni.usd.get_context().get_selection().set_selected_prim_paths([self.articulation_path], True)
                    self.mode = "PLACING"
                elif self.mode == "PLACING":
                    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.articulation_path))
                    quaternion = matrix.ExtractRotationQuat()
                    position = np.asarray(matrix.ExtractTranslation(), dtype=np.float32)
                    orientation = np.asarray([quaternion.GetReal(), *quaternion.GetImaginary()], dtype=np.float32)
                    self.robot.set_world_pose(position=position, orientation=orientation)
                    self.robot.set_linear_velocity(np.zeros(3))
                    self.robot.set_angular_velocity(np.zeros(3))
                    self.last_command[:] = 0
                    omni.usd.get_context().get_selection().set_selected_prim_paths([], True)
                    self.world.play()
                    self.mode = "READY"

    def _running_step(self):
        state = self._state()
        elapsed = float(self.world.current_time - self.start_time)
        cfg = self.config["trial"]
        self.actual.append(state[:3].copy())
        clearance = self.geometry.clearance(state[:3])
        self.record.add_sample(elapsed, state, self.requested, self.last_command, clearance)
        if self.contacts:
            self._finish("COLLISION")
            return
        reached = np.linalg.norm(state[:2] - self.goal) <= cfg["goal_tolerance_m"]
        passed = self.geometry.completely_passed(state[:3])
        if reached and passed:
            self.requested[:] = 0
            stopped = np.linalg.norm(state[3:5]) <= cfg["stopped_linear_mps"] and abs(state[5]) <= cfg["stopped_angular_radps"]
            self.stopped_since = elapsed if stopped and self.stopped_since is None else self.stopped_since
            if not stopped:
                self.stopped_since = None
            if self.stopped_since is not None and elapsed - self.stopped_since >= cfg["stopped_hold_sec"]:
                self._finish("SUCCESS")
                return
        elif self.step_index % self.period_steps == 0:
            self.last_plan = self.planner.plan(state)
            self.record.add_plan(self.last_plan, elapsed)
            self.requested = self.last_plan.command.copy()
            if not self.last_plan.valid:
                self._finish("INVALID_OUTPUT")
                return
        if elapsed >= cfg["timeout_sec"]:
            self._finish("TIMEOUT")
            return
        self._apply(self.requested)
        self.step_index += 1

    def run(self):
        self._initialize()
        try:
            if self.headless and not self._start(self.config["planner"]["method"], self.config["trial"]["navigation_condition"]):
                raise RuntimeError(f"试验无法开始：{self.result_text}")
            while self.app.is_running():
                self._handle_commands()
                if self.mode == "PLACING":
                    self.app.update()
                    continue
                if not self.world.is_playing():
                    self.requested[:] = 0
                    self.app.update()
                    continue
                if self.mode == "RUNNING":
                    self._running_step()
                    if self.headless and self.mode == "READY":
                        continue
                else:
                    if self.headless and self.completed >= self.repeat:
                        break
                    if self.headless:
                        self._reset()
                        if not self._start(self.config["planner"]["method"], self.config["trial"]["navigation_condition"]):
                            raise RuntimeError(f"重置后无法开始试验：{self.result_text}")
                        continue
                    else:
                        manual = self.teleop.step(self.dt)
                        self._apply([manual.linear_mps, manual.angular_radps])
                if self.panel is not None and self.world.current_time_step_index % self.period_steps == 0:
                    state = self._state()
                    plan = self.last_plan
                    rgba = self.renderer.render(state[:3], self.goal, None if plan is None else plan.prediction,
                                                None if plan is None else plan.candidates[:self.config["planner"]["candidate_display_count"]],
                                                self.actual, self.reference)
                    self.panel.update(rgba, self.mode, self.result_text)
                self.world.step(render=not self.headless)
        finally:
            if self.record is not None:
                self._finish("APPLICATION_CLOSED")
            self.teleop.shutdown()
            self.subscription = None
            if self.panel is not None:
                self.panel.close()
        return 0
