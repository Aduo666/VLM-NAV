from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

import numpy as np
import torch
import carb
import flatdict
import toml
import yaml
from pxr import Usd, UsdPhysics, UsdGeom
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaacsim.sensors.camera import Camera

from constraint_traverse.collect.bev import BevConfig, local_voxel_config_from_mapping, snapshot_to_robot_bev
from constraint_traverse.collect.sampling import PhysicsState
from vln_nav.active_perception.camera_geometry import world_from_pose_wxyz
from vln_nav.active_perception.local_geometry import LocalVoxelMap
from vln_nav.control.fine_path_tracker import fit_agx_command
from vln_nav.control.trajectory import TrajectoryConfig, VelocityCommand, wheel_targets
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.agx_geometry import build_agx_footprint, validate_agx_geometry
from vln_nav.runtime.isaac_runtime_utils import bind_viewport_to_camera
from vln_nav.runtime.scene_lighting import add_indoor_lighting
from .data import ACTION_SCALE, OBS_SIZE, Episode, map_age, state_vector
from .geometry import footprint_points
from .task_state import TaskState


def _collector_render_settings():
    directory = Path(sim_utils.__file__).resolve().parents[4] / "apps/rendering_modes"
    keys = set()
    for path in directory.glob("*.kit"):
        values = flatdict.FlatDict(toml.load(path), delimiter=".")
        keys.update("/" + key.replace(".", "/") for key in values)
    settings = carb.settings.get_settings()
    values = {key: settings.get(key) for key in sorted(keys)}
    missing = [key for key, value in values.items() if value is None]
    if not keys or missing:
        raise RuntimeError(f"无法读取采集程序的标准渲染设置：{missing}")
    return values


class CollectionCamera(Camera):
    def post_reset(self):
        super().post_reset()
        self._current_frame["rendering_time"] = -float("inf")

    def _data_acquisition_callback(self, event):
        # Isaac Lab 手动物理步进通过同步渲染接口取得图像与采集时间。
        return None

    def capture(self, timestamp):
        self._current_frame["rendering_time"] = float(timestamp)
        for name, annotator in self._custom_annotators.items():
            self._current_frame[name] = annotator.get_data()

    def get_intrinsics_matrix(self):
        matrix = super().get_intrinsics_matrix()
        return matrix.detach().cpu().numpy() if isinstance(matrix, torch.Tensor) else np.asarray(matrix)

class TraverseEnv(DirectRLEnv):
    @torch.inference_mode()
    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    @torch.inference_mode()
    def step(self, action):
        return super().step(action)

    def __init__(self, episodes: list[Episode], *, num_envs=1, device="cuda:0", seed=0,
                 timeout_sec=60.0, collision_force_threshold_n=1.0, render_mode=None):
        self.collision_force_threshold_n = float(collision_force_threshold_n)
        if not np.isfinite(self.collision_force_threshold_n) or self.collision_force_threshold_n <= 0:
            raise ValueError("底盘碰撞力阈值必须是大于零的有限数值")
        self.episodes = episodes
        self.meta = episodes[0].meta
        if any(e.meta["usd_path"] != self.meta["usd_path"] for e in episodes):
            raise ValueError("同一次环境创建使用同一个房间的 Episode")
        if any(e.meta["controller"] != self.meta["controller"] or e.dt != episodes[0].dt for e in episodes):
            raise ValueError("同一环境的底盘或物理配置不同")
        self.controller = TrajectoryConfig.from_mapping(self.meta["controller"])
        self.profile = load_agx_profile(project_root=Path(__file__).resolve().parents[2])
        collect_config_path = Path(__file__).resolve().parents[1] / "configs/collect.yaml"
        with collect_config_path.open(encoding="utf-8") as handle:
            self.collect_scene_config = yaml.safe_load(handle)["scene"]
        self.collect_render_settings = _collector_render_settings()
        self.selection = np.zeros(num_envs, dtype=np.int64)
        self.reset_sequence = np.zeros(num_envs, dtype=np.int64)
        self.maps = [LocalVoxelMap(local_voxel_config_from_mapping(self.meta["voxel_map"])) for _ in range(num_envs)]
        self.raw_collision = np.zeros(num_envs, dtype=bool)
        self.contact_count = np.zeros(num_envs, dtype=np.int64)
        self.collision_contacts = [None for _ in range(num_envs)]
        self.last_contact_time = np.full(num_envs, -float("inf"))
        self.camera_valid_ratio = np.zeros((num_envs, 2))
        self.map_updates = np.zeros(num_envs, dtype=np.int64)
        self._cached_obs = None
        self._cached_step = -1
        self.contact_views = []
        self.reset_time = np.zeros(num_envs)
        self.cameras = []
        self.samplers = []
        self.map_panel = None
        self.last_sensor_time = -float("inf")
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        cfg = DirectRLEnvCfg(decimation=4, episode_length_s=timeout_sec, is_finite_horizon=False,
            action_space=2, observation_space=OBS_SIZE, state_space=0, seed=seed,
            sim=sim_utils.SimulationCfg(dt=episodes[0].dt, device=device, render_interval=1, enable_scene_query_support=True,
                render=sim_utils.RenderCfg(carb_settings=self.collect_render_settings)),
            scene=InteractiveSceneCfg(num_envs=num_envs, env_spacing=40.0, replicate_physics=False),
            num_rerenders_on_reset=2)
        super().__init__(cfg, render_mode=render_mode)
        from isaacsim.core.simulation_manager import SimulationManager
        room_paths = []
        for prim in self.sim.stage.Traverse():
            if "/Room/" not in str(prim.GetPath()) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                parent = prim.GetParent()
                nested = False
                while parent and "/Room" in str(parent.GetPath()):
                    if parent.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(parent).GetRigidBodyEnabledAttr().Get():
                        nested = True
                        break
                    parent = parent.GetParent()
                if not nested:
                    room_paths.append(str(prim.GetPath()))
        self.room_view = (SimulationManager.get_physics_sim_view().create_rigid_body_view(room_paths)
                          if room_paths else None)
        if self.room_view is not None:
            self.room_initial_transforms = self.room_view.get_transforms().clone()
            self.room_initial_velocities = self.room_view.get_velocities().clone()
            if set(self.room_view.prim_paths) != set(room_paths):
                raise RuntimeError("房间刚体恢复视图未覆盖全部启用对象")
            self.room_env_indices = [torch.tensor([j for j, path in enumerate(self.room_view.prim_paths)
                if path.startswith(f"/World/envs/env_{i}/")], device=self.device, dtype=torch.int32)
                for i in range(self.num_envs)]
        names = self.robot.joint_names
        self.steering_ids = [names.index(n) for n in self.meta["steering_joint_order"]]
        self.wheel_ids = [names.index(n) for n in self.meta["wheel_joint_order"]]
        self.steering_effort = self.robot.data.joint_effort_limits[:, self.steering_ids].clone()
        self.target_position = self.robot.data.default_joint_pos.clone()
        self.target_velocity = torch.zeros_like(self.target_position)
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.applied_command = torch.zeros_like(self.command)
        self.previous_command = torch.zeros_like(self.command)
        self.task = TaskState(self.num_envs, self.device, timeout_sec=timeout_sec)
        self.start_pose = torch.zeros(self.num_envs, 3, device=self.device)
        self.reset_errors = torch.zeros(self.num_envs, 6, device=self.device)
        physics = SimulationManager.get_physics_sim_view()
        if self.robot.body_names.count("base_link") != 1:
            raise RuntimeError("AGX 模型必须包含唯一的底盘刚体 base_link")
        chassis_index = self.robot.body_names.index("base_link")
        room_body_paths = set(room_paths)
        for i in range(self.num_envs):
            bodies = [self.robot.root_physx_view.link_paths[i][chassis_index]]
            filters = set()
            room = self.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Room")
            for prim in Usd.PrimRange(room, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    continue
                owner = prim
                while owner and str(owner.GetPath()) not in room_body_paths:
                    owner = owner.GetParent()
                filters.add(str(owner.GetPath()) if owner else str(prim.GetPath()))
            filters = sorted(filters)
            if not filters:
                raise RuntimeError("房间没有可以检测底盘接触的碰撞体")
            view = physics.create_rigid_contact_view(bodies, filter_patterns=[filters] * len(bodies),
                                                     max_contact_data_count=4096)
            if view.sensor_count != len(bodies) or view.filter_count != len(filters):
                raise RuntimeError("接触视图与车辆、场景对象数量不同")
            self.contact_views.append((view, bodies, filters))
        from .temporal import MapSampler
        for i in range(self.num_envs):
            pair = {}
            for source, key in (("front", "front_prim_path"), ("down", "down_prim_path")):
                suffix = self.meta["cameras"][key].removeprefix("/World/Robot")
                path = f"/World/envs/env_{i}/Robot" + suffix
                # 同步取图的周期由 _apply_action 按仿真时间控制。
                camera = CollectionCamera(prim_path=path, name=f"traverse_{source}_{i}",
                    resolution=(self.meta["cameras"]["width"], self.meta["cameras"]["height"]))
                intrinsics = self.episodes[0].frames[f"{source}_intrinsics"]
                usd_camera = UsdGeom.Camera(self.sim.stage.GetPrimAtPath(path))
                aperture = float(usd_camera.GetHorizontalApertureAttr().Get())
                usd_camera.GetFocalLengthAttr().Set(aperture * float(intrinsics[0, 0]) / self.meta["cameras"]["width"])
                camera.initialize()
                camera.add_distance_to_image_plane_to_frame()
                np.testing.assert_allclose(camera.get_intrinsics_matrix(), intrinsics, atol=1e-4)
                pair[source] = camera
            self.cameras.append(pair)
            self.samplers.append(MapSampler(self.maps[i], max_skew_sec=self.meta["cameras"]["max_frame_skew_sec"],
                min_valid_ratio=self.meta["collection"]["min_depth_valid_ratio"],
                integration_period_sec=self.meta["map_integration_period_sec"]))
        self.sim.render()
        carb.settings.get_settings().set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)
        rep.orchestrator.set_capture_on_play(False)
        self._set_initial_view()
        if self.sim.has_gui():
            import omni.ui as ui
            from .ui import EvaluationMapPanel
            display_config_path = Path(__file__).resolve().parents[2] / "configs/navigation.yaml"
            with display_config_path.open(encoding="utf-8") as handle:
                display_config = yaml.safe_load(handle)["map_selector"]
            self.map_panel = EvaluationMapPanel(ui, display_config=display_config)

    def _setup_scene(self):
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        room = sim_utils.UsdFileCfg(usd_path=self.meta["usd_path"])
        room.func("/World/envs/env_0/Room", room)
        if self.collect_scene_config["add_indoor_lighting"]:
            add_indoor_lighting(stage=self.sim.stage, environment_prim_path="/World/envs/env_0/Room",
                lights_prim_path="/World/envs/env_0/InteriorLights", logger=logging.getLogger(__name__))
        robot_path = "/World/envs/env_0/Robot"
        spawn = sim_utils.UsdFileCfg(usd_path=str(self.profile.asset.usd_path), activate_contact_sensors=True)
        spawn.func(robot_path, spawn)
        scoped_profile = replace(self.profile, asset=replace(self.profile.asset, prim_path=robot_path))
        articulation_path = robot_path + "/" + self.profile.asset.articulation_path
        validate_agx_geometry(self.sim.stage, scoped_profile, articulation_path)
        _, geometry = build_agx_footprint(self.sim.stage, scoped_profile, articulation_path, margin_m=0.0)
        self.collision_geometry = geometry["colliders"]
        self.body_points = footprint_points(self.collision_geometry)
        root = self.sim.stage.GetPrimAtPath(robot_path)
        joint_prims = [p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
        joint_names = [p.GetName() for p in joint_prims]
        joint_positions = {}
        for prim in joint_prims:
            angular = prim.IsA(UsdPhysics.RevoluteJoint)
            instance = "angular" if angular else "linear"
            value = prim.GetAttribute(f"state:{instance}:physics:position").Get()
            if value is None:
                drive = UsdPhysics.DriveAPI.Get(prim, instance)
                value = drive.GetTargetPositionAttr().Get() if drive else None
            value = 0.0 if value is None else float(value)
            if not np.isfinite(value):
                raise ValueError(f"USD 关节初始位置无效：{prim.GetPath()}")
            # USD 旋转关节使用角度，Isaac Lab 使用弧度。
            joint_positions[prim.GetName()] = float(np.deg2rad(value)) if angular else value
        wheel_names = self.meta["wheel_joint_order"]
        steering_names = self.meta["steering_joint_order"]
        hold_names = [n for n in joint_names if n not in wheel_names + steering_names]
        agx = self.profile.agx_controller
        actuators = dict(
            wheels=ImplicitActuatorCfg(joint_names_expr=wheel_names, stiffness=0.0, damping=agx.wheel_drive_damping),
            steering=ImplicitActuatorCfg(joint_names_expr=steering_names, stiffness=agx.steering_stiffness, damping=agx.steering_damping))
        if hold_names:
            actuators["hold"] = ImplicitActuatorCfg(joint_names_expr=hold_names,
                stiffness=agx.hold_joint_stiffness, damping=agx.hold_joint_damping)
        self.robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(self.meta["start_position_world_xyz"]),
                joint_pos=joint_positions), actuators=actuators))
        self.top_camera_path = robot_path + "/" + self.profile.asset.articulation_path + "/TopCamera"
        if not self.sim.stage.GetPrimAtPath(self.top_camera_path).IsA(UsdGeom.Camera):
            raise RuntimeError(f"机器人模型缺少 TopCamera：{self.top_camera_path}")
        transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.camera_mounts = {}
        optical_from_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        for source, key in (("front", "front_prim_path"), ("down", "down_prim_path")):
            path = robot_path + self.meta["cameras"][key].removeprefix("/World/Robot")
            camera_prim = self.sim.stage.GetPrimAtPath(path)
            body = camera_prim.GetParent()
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise RuntimeError(f"相机没有所属的物理刚体：{path}")
            world_from_body = np.asarray(transforms.GetLocalToWorldTransform(body)).T
            world_from_camera = np.asarray(transforms.GetLocalToWorldTransform(camera_prim)).T
            self.camera_mounts[source] = (body.GetName(), np.linalg.inv(world_from_body) @ world_from_camera @ optical_from_usd)
        self.scene.clone_environments(copy_from_source=True)
        self.scene.articulations["robot"] = self.robot
        self.scene.filter_collisions(global_prim_paths=[])

    def _set_initial_view(self):
        self.cfg.viewer.cam_prim_path = self.top_camera_path
        if self.sim.has_gui() and not bind_viewport_to_camera(camera_prim_path=self.top_camera_path,
                logger=logging.getLogger(__name__), enabled=True):
            raise RuntimeError(f"无法切换评估窗口至 TopCamera：{self.top_camera_path}")

    def _check_contacts(self):
        now = float(self.sim.current_time)
        for i, (view, bodies, filters) in enumerate(self.contact_views):
            if now <= self.reset_time[i] + 1e-7 or now <= self.last_contact_time[i] + 1e-7 or self.raw_collision[i]:
                continue
            self.last_contact_time[i] = now
            net_force = view.get_net_contact_forces(dt=self.physics_dt)[0].cpu().numpy()
            pair_forces = view.get_contact_force_matrix(dt=self.physics_dt)[0].cpu().numpy()
            if not np.isfinite(net_force).all() or not np.isfinite(pair_forces).all():
                raise RuntimeError("底盘接触力包含非有限数值")
            forces, points, normals, separations, counts, starts = view.get_contact_data(dt=self.physics_dt)
            counts, starts = counts.cpu().numpy(), starts.cpu().numpy()
            if counts.sum() >= view.max_contact_data_count:
                raise RuntimeError("底盘接触记录超出缓存容量")
            self.contact_count[i] += int(counts.sum())
            pair_magnitudes = np.linalg.norm(pair_forces, axis=-1)
            # 同一对象上的相反接触力可能抵消，单独检查各个接触点。
            for other in np.flatnonzero(counts[0]):
                start, count = int(starts[0, other]), int(counts[0, other])
                contact_forces = forces[start:start + count, 0].cpu().numpy()
                if not np.isfinite(contact_forces).all():
                    raise RuntimeError("底盘接触点的法向力包含非有限数值")
                pair_magnitudes[other] = max(pair_magnitudes[other], float(np.abs(contact_forces).max()))
            other = int(pair_magnitudes.argmax())
            force_n = max(float(np.linalg.norm(net_force)), float(pair_magnitudes[other]))
            if force_n <= self.collision_force_threshold_n:
                continue
            record = dict(body_path=bodies[0], scene_body_path=None,
                timestamp_sec=now, physics_step=int(round(now / self.physics_dt)),
                force_n=force_n, threshold_n=self.collision_force_threshold_n,
                net_force_world_xyz_n=net_force.tolist(), pair_force_world_xyz_n=None,
                position_world_xyz=None, normal_world_xyz=None, normal_force_n=None,
                normal_impulse_ns=None, separation_m=None)
            count = int(counts[0, other])
            if float(pair_magnitudes[other]) > 0 and count:
                start = int(starts[0, other])
                patch_forces = forces[start:start + count, 0].cpu().numpy()
                index = start + int(np.abs(patch_forces).argmax())
                normal_force = float(forces[index, 0])
                record.update(scene_body_path=filters[other], pair_force_world_xyz_n=pair_forces[other].tolist(),
                    position_world_xyz=points[index].cpu().tolist(), normal_world_xyz=normals[index].cpu().tolist(),
                    normal_force_n=normal_force, normal_impulse_ns=normal_force * self.physics_dt,
                    separation_m=float(separations[index, 0]))
            self.collision_contacts[i] = record
            self.raw_collision[i] = True

    def _stop_collided_envs(self):
        ids = torch.as_tensor(np.flatnonzero(self.raw_collision), dtype=torch.long, device=self.device)
        if not len(ids):
            return
        self.target_velocity[ids] = 0
        self.target_position[ids[:, None], self.steering_ids] = self.robot.data.joint_pos[ids[:, None], self.steering_ids]
        self.applied_command[ids] = 0
        self._stopped[ids] = True
        self.robot.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        joint_ids = self.wheel_ids + self.steering_ids
        self.robot.write_joint_velocity_to_sim(torch.zeros(len(ids), len(joint_ids), device=self.device),
                                               joint_ids=joint_ids, env_ids=ids)

    def _pose_velocity(self):
        root = self.robot.data.root_link_state_w
        q = root[:, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        pose = torch.cat((root[:, :2], yaw[:, None]), dim=-1)
        velocity = torch.stack((root[:, 7] * yaw.cos() + root[:, 8] * yaw.sin(), root[:, 12]), dim=-1)
        return pose, velocity

    def _pre_physics_step(self, actions):
        self.extras = {}
        self.command = actions.clone() * torch.as_tensor(ACTION_SCALE, device=self.device)
        self.previous_command.copy_(self.command)
        self.target_position.copy_(self.robot.data.default_joint_pos)
        self.target_velocity.zero_()
        for i, values in enumerate(self.command.detach().cpu().numpy()):
            fitted = fit_agx_command(VelocityCommand(float(values[0]), float(values[1])), self.controller)
            drives, steering, applied = wheel_targets(fitted, self.controller)
            self.target_position[i, self.steering_ids] = torch.tensor([steering[n] for n in self.meta["steering_joint_order"]], device=self.device)
            self.target_velocity[i, self.wheel_ids] = torch.tensor([drives[n] for n in self.meta["wheel_joint_order"]], device=self.device)
            self.applied_command[i] = torch.tensor([applied.linear_mps, applied.angular_radps], device=self.device)
        stopped = self.target_velocity[:, self.wheel_ids].abs().amax(dim=-1) < 1e-9
        centering = float(self.meta["collection"]["steering_centering_max_effort_nm"])
        efforts = torch.where(stopped[:, None], self.steering_effort.clamp_min(centering), self.steering_effort)
        self.robot.write_joint_effort_limit_to_sim(efforts, joint_ids=self.steering_ids)
        self._stopped = stopped

    def _apply_action(self):
        self._check_contacts()
        self._stop_collided_envs()
        now = float(self.sim.current_time)
        if now - self.last_sensor_time >= 1.0 / self.meta["cameras"]["frequency_hz"] - 1e-9:
            self.sim.forward()
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False, wait_for_render=True)
            if abs(float(self.sim.current_time) - now) > 1e-7:
                raise RuntimeError("同步相机渲染意外推进了物理时间")
            for pair in self.cameras:
                for camera in pair.values():
                    camera.capture(now)
            self.last_sensor_time = now
        self._record_physics_state()
        for i, sampler in enumerate(self.samplers):
            if self.raw_collision[i]:
                continue
            sampler.poll(self.cameras[i]["front"], self.cameras[i]["down"], now=self.sim.current_time)
            self.camera_valid_ratio[i] = sampler.valid_ratios
            self.map_updates[i] = sampler.fusion_count
        self.robot.set_joint_position_target(self.target_position)
        self.robot.set_joint_velocity_target(self.target_velocity)
        if self._stopped.any():
            ids = torch.nonzero(self._stopped).flatten()
            positions = self.robot.data.joint_pos[ids].clone()
            velocities = self.robot.data.joint_vel[ids].clone()
            velocities[:, self.wheel_ids] = 0
            self.robot.write_joint_state_to_sim(positions, velocities, env_ids=ids)

    def _get_dones(self):
        self._check_contacts()
        self._stop_collided_envs()
        self._record_physics_state()
        pose, velocity = self._pose_velocity()
        collision = torch.tensor(self.raw_collision, device=self.device)
        normalized = self.command / torch.as_tensor(ACTION_SCALE, device=self.device)
        self.task.action_variation += torch.linalg.vector_norm(normalized - self.task.previous_action, dim=-1)
        terminated, truncated = self.task.update(pose[:, :2], velocity, collision, self.step_dt, yaw=pose[:, 2])
        ids = torch.nonzero(terminated | truncated).flatten()
        if len(ids):
            self.extras["terminal"] = self.task.terminal(ids, pose)
            self.extras["terminal"]["contact_count"] = torch.tensor(self.contact_count[ids.cpu().numpy()].copy(), device=self.device)
            self.extras["collision_contacts"] = [self.collision_contacts[i] for i in ids.tolist()]
        return terminated, truncated

    def _get_rewards(self):
        action = self.command / torch.as_tensor(ACTION_SCALE, device=self.device)
        change = torch.square(action - self.task.previous_action).sum(dim=-1)
        reward = self.task.previous_distance - self.task.distance + 10 * self.task.success.float()
        reward -= 10 * self.task.collision.float() + 0.05 * self.step_dt + 0.01 * change
        self.task.previous_action.copy_(action)
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.room_view is not None:
            room_ids = torch.cat([self.room_env_indices[i] for i in ids.tolist()])
            if len(room_ids):
                self.room_view.set_transforms(self.room_initial_transforms, room_ids)
                self.room_view.set_velocities(self.room_initial_velocities, room_ids)
        root = self.robot.data.default_root_state[ids].clone()
        joints = self.robot.data.default_joint_pos[ids].clone()
        joint_vel = torch.zeros_like(joints)
        origins = self.scene.env_origins.detach().cpu().numpy()
        now = float(self.sim.current_time)
        for j, env_id in enumerate(ids.tolist()):
            chosen = int(self.reset_sequence[env_id] % len(self.episodes))
            self.selection[env_id] = chosen
            self.reset_sequence[env_id] += 1
            episode = self.episodes[chosen]
            initial = episode.controls
            position = initial["world_position_xyz"][0].astype(np.float64) + origins[env_id]
            quaternion = initial["world_orientation_wxyz"][0].copy()
            root[j, :3] = torch.tensor(position, device=self.device)
            root[j, 3:7] = torch.tensor(quaternion, device=self.device)
            root[j, 7:] = 0
            qw, qx, qy, qz = quaternion
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            v, w = initial["measured_velocity_vw"][0]
            root[j, 7:9] = torch.tensor([v * np.cos(yaw), v * np.sin(yaw)], device=self.device)
            root[j, 12] = float(w)
            joints[j, self.steering_ids] = torch.tensor(initial["steering_position"][0], device=self.device)
            joint_vel[j, self.steering_ids] = torch.tensor(initial["steering_velocity"][0], device=self.device)
            joint_vel[j, self.wheel_ids] = torch.tensor(initial["wheel_velocity"][0], device=self.device)
            snapshot = episode.initial_map
            shift = now - float(episode.meta["start_time_sec"])
            observed_times = snapshot.last_observed_time_sec.copy()
            observed_times[snapshot.observed_mask] += shift
            shifted = replace(snapshot, origin_world_xyz=snapshot.origin_world_xyz + origins[env_id],
                center_world_xyz=snapshot.center_world_xyz + origins[env_id],
                timestamp_sec=snapshot.timestamp_sec + shift, last_observed_time_sec=observed_times)
            self.maps[env_id].restore(shifted)
            self.samplers[env_id].reset(start_time_sec=now, map_timestamp_sec=shifted.timestamp_sec)
            for camera in self.cameras[env_id].values():
                camera.post_reset()
            goal = np.asarray(episode.meta["goal_world_xy"]) + origins[env_id, :2]
            self.task.reset(ids[j:j + 1], torch.tensor(goal[None], dtype=torch.float32, device=self.device),
                            root[j:j + 1, :2], yaw=torch.tensor([yaw], dtype=torch.float32, device=self.device))
            self.reset_time[env_id] = now
            self.last_contact_time[env_id] = now
            self.raw_collision[env_id] = False
            self.contact_count[env_id] = 0
            self.collision_contacts[env_id] = None
            self.map_updates[env_id] = 0
        self.robot.write_root_pose_to_sim(root[:, :7], ids)
        self.robot.write_root_link_velocity_to_sim(root[:, 7:], ids)
        self.robot.write_joint_state_to_sim(joints, joint_vel, env_ids=ids)
        self.target_position[ids] = joints
        self.target_velocity[ids] = 0
        self.robot.set_joint_position_target(joints, env_ids=ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(joints), env_ids=ids)
        self.previous_command[ids] = 0
        self.command[ids] = 0
        self.applied_command[ids] = 0
        self._cached_obs = None
        pose, _ = self._pose_velocity()
        self.start_pose[ids] = pose[ids]
        self.reset_errors[ids, 0] = (self.robot.data.root_link_state_w[ids, :3] - root[:, :3]).abs().amax(dim=-1)
        self.reset_errors[ids, 1] = (self.robot.data.joint_pos[ids][:, self.steering_ids] - joints[:, self.steering_ids]).abs().amax(dim=-1)
        self.reset_errors[ids, 2] = (self.robot.data.joint_vel[ids][:, self.steering_ids] - joint_vel[:, self.steering_ids]).abs().amax(dim=-1)
        self.reset_errors[ids, 3] = (self.robot.data.joint_vel[ids][:, self.wheel_ids] - joint_vel[:, self.wheel_ids]).abs().amax(dim=-1)
        self.reset_errors[ids, 4] = (self.robot.data.root_link_state_w[ids, 7:] - root[:, 7:]).abs().amax(dim=-1)
        self.reset_errors[ids, 5] = (self.robot.data.root_link_state_w[ids, 3:7] - root[:, 3:7]).abs().amax(dim=-1)
        self._record_physics_state()

    def _record_physics_state(self):
        if not self.samplers:
            return
        root = self.robot.data.root_link_state_w.detach().cpu().numpy()
        pose, velocity = self._pose_velocity()
        positions = self.robot.data.joint_pos.detach().cpu().numpy()
        velocities = self.robot.data.joint_vel.detach().cpu().numpy()
        now = float(self.sim.current_time)
        for i, sampler in enumerate(self.samplers):
            sampler.append_state(PhysicsState(timestamp_sec=now, physics_step=int(round(now / self.physics_dt)),
                position=root[i, :3].copy(), orientation=root[i, 3:7].copy(), pose=pose[i].cpu().numpy(),
                velocity=velocity[i].cpu().numpy(), steering_position=positions[i, self.steering_ids].copy(),
                steering_velocity=velocities[i, self.steering_ids].copy(), wheel_velocity=velocities[i, self.wheel_ids].copy(),
                front_world_from_camera=self.camera_world_transform(i, "front"),
                down_world_from_camera=self.camera_world_transform(i, "down")))

    def camera_world_transform(self, env_id, source):
        name, mount = self.camera_mounts[source]
        link = self.robot.data.body_link_pose_w[env_id, self.robot.body_names.index(name)].detach().cpu().numpy()
        return world_from_pose_wxyz(link[:3], link[3:7]) @ mount

    def _get_observations(self):
        if self._cached_obs is not None and self._cached_step == self._sim_step_counter:
            return {"policy": self._cached_obs}
        self._record_physics_state()
        pose, velocity = self._pose_velocity()
        poses = pose.detach().cpu().numpy()
        velocities = velocity.detach().cpu().numpy()
        position = self.robot.data.root_link_state_w[:, :3].detach().cpu().numpy()
        steering = self.robot.data.joint_pos[:, self.steering_ids].detach().cpu().numpy()
        wheels = self.robot.data.joint_vel[:, self.wheel_ids].detach().cpu().numpy()
        previous = self.previous_command.detach().cpu().numpy()
        goals = self.task.goal.detach().cpu().numpy()
        now = float(self.sim.current_time)
        rows = []
        for i in range(self.num_envs):
            snapshot = self.maps[i].snapshot()
            floor_z = position[i, 2] - self.episodes[self.selection[i]].floor_offset
            bev_config = BevConfig.from_mapping(self.meta["bev"])
            bev = snapshot_to_robot_bev(snapshot, robot_world_xyyaw=poses[i], floor_z_m=float(floor_z),
                                       config=bev_config)
            state = state_vector(goals[i], poses[i], velocities[i], steering[i], wheels[i], previous[i], map_age(now, snapshot.timestamp_sec))
            rows.append(np.concatenate((bev.reshape(-1).astype(np.float32), state)))
            if i == 0 and self.map_panel is not None:
                self.map_panel.update(snapshot=snapshot, bev_config=bev_config,
                    robot_world_xyz=position[i], robot_yaw_rad=float(poses[i, 2]), floor_z_m=float(floor_z),
                    goal_body_xy=state[:2],
                    episode=self.episodes[self.selection[i]].path, elapsed_sec=float(self.task.elapsed[i]),
                    timeout_sec=self.task.timeout_sec, map_age_sec=float(state[-1]))
        self._cached_obs = torch.tensor(np.stack(rows), device=self.device)
        self._cached_step = self._sim_step_counter
        return {"policy": self._cached_obs}

    def close(self):
        if getattr(self, "_is_closed", False):
            return
        if self.map_panel is not None:
            self.map_panel.shutdown()
            self.map_panel = None
        self.contact_views = []
        for pair in self.cameras:
            for camera in pair.values():
                camera.pause()
        if hasattr(self, "sim"):
            self.sim.clear_instance()
        super().close()
