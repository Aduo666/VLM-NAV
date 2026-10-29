from __future__ import annotations

from copy import deepcopy

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ..state import RandomState, TrialSnapshot, robot_errors


def scene_transforms(runtime):
    root = runtime.stage.GetPrimAtPath("/World/Environment")
    return {str(prim.GetPath()): np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).copy()
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Xformable) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)}


def rigid_states(runtime):
    dc = runtime.drive._dc
    states = {}
    for prim in runtime.stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(runtime.profile.asset.prim_path + "/") or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        handle = dc.get_rigid_body(path)
        pose = dc.get_rigid_body_pose(handle)
        linear, angular = dc.get_rigid_body_linear_velocity(handle), dc.get_rigid_body_angular_velocity(handle)
        states[path] = np.asarray([pose.p.x, pose.p.y, pose.p.z, pose.r.x, pose.r.y, pose.r.z, pose.r.w,
                                   linear.x, linear.y, linear.z, angular.x, angular.y, angular.z])
    return states


def controller_state(runtime):
    drive, dc = runtime.drive, runtime.drive._dc
    targets = {}
    for name in runtime.joint_names:
        handle = dc.find_articulation_dof(drive._articulation, name)
        props = dc.get_dof_properties(handle)
        targets[name] = [dc.get_dof_position_target(handle), dc.get_dof_velocity_target(handle),
                         props.stiffness, props.damping, props.max_effort]
    return dict(targets=targets, hold=deepcopy(drive.hold_joint_targets),
                reset_pending=drive._steering_reset_pending,
                centering_active=drive._steering_centering_active,
                centering_efforts=deepcopy(drive._steering_centering_original_efforts),
                adapter=dict(status=runtime.adapter.status,
                             target=deepcopy(runtime.adapter._steering_target_rad),
                             measured=deepcopy(runtime.adapter._steering_measured_rad),
                             error=runtime.adapter._max_steering_error_rad),
                requested=runtime.requested.copy(), applied=runtime.applied.copy(),
                await_key_release=runtime.await_key_release)


def capture(runtime):
    return TrialSnapshot(runtime.robot_state(), scene_transforms(runtime), rigid_states(runtime),
        controller_state(runtime),
        dict(linear=runtime.teleop._linear, angular=runtime.teleop._angular,
             pressed=set(runtime.teleop._pressed_keys)),
        {name: deepcopy(getattr(runtime.local_map, name)) for name in ("origin", "log_odds", "source_mask", "version")},
        deepcopy(runtime.observations), None if runtime.goal is None else runtime.goal.copy(),
        RandomState.capture(runtime.generator), runtime.sensors.sequence,
        runtime.last_sensor_time, runtime.last_map_time, runtime.fusion.snapshot())


def restore(runtime, saved):
    from omni.isaac.dynamic_control import _dynamic_control

    for path, matrix in saved.scene_transforms.items():
        prim = runtime.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"场景节点已经缺失：{path}")
        transform = UsdGeom.Xformable(prim)
        if not np.allclose(np.asarray(transform.GetLocalTransformation()), matrix, atol=1e-12, rtol=0):
            transform.MakeMatrixXform().Set(Gf.Matrix4d(matrix.tolist()))
    runtime.world.reset()
    runtime.robot.initialize()
    robot = saved.robot
    rotation = Gf.Matrix3d(robot.world_from_base[:3, :3].T.tolist()).ExtractRotation().GetQuat()
    imaginary = rotation.GetImaginary()
    quaternion = np.asarray([rotation.GetReal(), *imaginary], dtype=np.float32)
    runtime.robot.set_world_pose(position=robot.world_from_base[:3, 3], orientation=quaternion)
    runtime.robot.set_joint_positions(robot.joint_positions)
    runtime.robot.set_joint_velocities(robot.joint_velocities)
    runtime.robot.set_linear_velocity(robot.base_twist[:3])
    runtime.robot.set_angular_velocity(robot.base_twist[3:])
    runtime.create_drive()
    drive, dc = runtime.drive, runtime.drive._dc
    for path, state in saved.rigid_bodies.items():
        handle = dc.get_rigid_body(path)
        pose = _dynamic_control.Transform()
        pose.p, pose.r = tuple(state[:3]), tuple(state[3:7])
        dc.set_rigid_body_pose(handle, pose)
        dc.set_rigid_body_linear_velocity(handle, tuple(state[7:10]))
        dc.set_rigid_body_angular_velocity(handle, tuple(state[10:13]))
    for name, values in saved.controller["targets"].items():
        handle = dc.find_articulation_dof(drive._articulation, name)
        properties = dc.get_dof_properties(handle)
        properties.stiffness, properties.damping, properties.max_effort = values[2:]
        dc.set_dof_properties(handle, properties)
        dc.set_dof_position_target(handle, values[0])
        dc.set_dof_velocity_target(handle, values[1])
    drive.hold_joint_targets = deepcopy(saved.controller["hold"])
    drive._steering_reset_pending = saved.controller["reset_pending"]
    drive._steering_centering_active = saved.controller["centering_active"]
    drive._steering_centering_original_efforts = deepcopy(saved.controller["centering_efforts"])
    runtime.adapter.status = saved.controller["adapter"]["status"]
    runtime.adapter._steering_target_rad = deepcopy(saved.controller["adapter"]["target"])
    runtime.adapter._steering_measured_rad = deepcopy(saved.controller["adapter"]["measured"])
    runtime.adapter._max_steering_error_rad = saved.controller["adapter"]["error"]
    runtime.requested, runtime.applied = saved.controller["requested"].copy(), saved.controller["applied"].copy()
    runtime.await_key_release = saved.controller["await_key_release"]
    runtime.teleop._linear, runtime.teleop._angular = saved.teleop["linear"], saved.teleop["angular"]
    runtime.teleop._pressed_keys = set(saved.teleop["pressed"])
    runtime.world.physics_sim_view.update_articulations_kinematic()
    runtime.time_offset = robot.timestamp_sec - float(runtime.world.current_time)
    for name, value in saved.map_state.items():
        setattr(runtime.local_map, name, deepcopy(value))
    runtime.fusion.restore(saved.fusion_state)
    runtime.last_map_update = None
    runtime.experiment.metadata["map_sources"] = list(runtime.fusion.sources)
    runtime.goal = None if saved.goal_world_xyz is None else saved.goal_world_xyz.copy()
    runtime.observations = deepcopy(saved.observations)
    runtime.sensors.sequence = saved.sensor_sequence
    runtime.sensors.epoch += 1
    runtime.last_sensor_time, runtime.last_map_time = saved.sensor_time_sec, saved.map_time_sec
    saved.random.restore(runtime.generator)
    errors = robot_errors(saved.robot, runtime.robot_state())
    actual_rigid = rigid_states(runtime)
    if actual_rigid.keys() != saved.rigid_bodies.keys():
        raise RuntimeError("恢复后的场景刚体集合发生变化")
    errors["rigid_body_error"] = max((float(np.max(np.abs(actual_rigid[path] - value)))
                                       for path, value in saved.rigid_bodies.items()), default=0.0)
    actual_transforms = scene_transforms(runtime)
    if actual_transforms.keys() != saved.scene_transforms.keys():
        raise RuntimeError("恢复后的场景节点集合发生变化")
    errors["scene_transform_error"] = max((float(np.max(np.abs(actual_transforms[path] - value)))
                                           for path, value in saved.scene_transforms.items()), default=0.0)
    actual_controller = controller_state(runtime)
    errors["controller_error"] = max(float(np.max(np.abs(np.asarray(actual_controller["targets"][name]) - values)))
                                       for name, values in saved.controller["targets"].items())
    errors["map_error"] = max(float(np.max(np.abs(np.asarray(getattr(runtime.local_map, name), dtype=float)
                                                   - np.asarray(value, dtype=float))))
                              for name, value in saved.map_state.items())
    if runtime.fusion.snapshot() != saved.fusion_state:
        errors["map_error"] = float("inf")
    return errors

