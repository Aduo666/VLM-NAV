from __future__ import annotations

import numpy as np

from isaacsim.sensors.camera import Camera
from pxr import UsdGeom, UsdPhysics

from vln_nav.active_perception.camera_geometry import camera_rgb_to_uint8, world_from_pose_wxyz
from vln_nav.runtime.isaac_runtime_utils import resolve_existing_prim_path, resolve_robot_relative_prim_path

from ..contracts import CameraFrame, RealObservation
from .robot_filter import RobotDepthFilter


def numpy_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class SynchronousCamera(Camera):
    def _data_acquisition_callback(self, event):
        # 图像由同步 Replicator 渲染取得，物理时间在渲染期间保持不变。
        return None


class ObservationSensors:
    def __init__(self, runtime, paths, resolution):
        self.runtime = runtime
        self.cameras = {}
        self.mounts = {}
        self.sequence = 0
        self.epoch = 0
        transforms = UsdGeom.XformCache()
        optical_from_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        for source, relative in paths.items():
            path = resolve_existing_prim_path(stage=runtime.stage,
                prim_path=resolve_robot_relative_prim_path(robot_root_path=runtime.profile.asset.prim_path,
                                                           prim_path=relative), kind=source)
            prim = runtime.stage.GetPrimAtPath(path)
            if not prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"相机节点无效：{path}")
            body = prim.GetParent()
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise RuntimeError(f"相机缺少所属刚体：{path}")
            world_from_body = np.asarray(transforms.GetLocalToWorldTransform(body)).T
            world_from_camera = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
            self.mounts[source] = (str(body.GetPath()), np.linalg.inv(world_from_body) @ world_from_camera @ optical_from_usd)
            camera = SynchronousCamera(prim_path=path, name="active_passage_" + source, resolution=resolution)
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
            self.cameras[source] = camera
        self.robot_filter = RobotDepthFilter(runtime)

    def camera_transform(self, source):
        path, mount = self.mounts[source]
        dc = self.runtime.drive._dc
        handle = dc.get_rigid_body(path)
        pose = dc.get_rigid_body_pose(handle)
        return world_from_pose_wxyz([pose.p.x, pose.p.y, pose.p.z],
                                   [pose.r.w, pose.r.x, pose.r.y, pose.r.z]) @ mount

    def capture(self):
        import omni.replicator.core as rep

        world = self.runtime.world
        before = float(world.current_time)
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False, wait_for_render=True)
        if abs(float(world.current_time) - before) > 1e-7:
            raise RuntimeError("传感器同步渲染改变了物理时间")
        robot = self.runtime.robot_state()
        self.robot_filter.update()
        results = []
        for source, camera in self.cameras.items():
            rgb = camera_rgb_to_uint8(camera.get_rgb())
            depth = camera.get_depth()
            if rgb is None or depth is None:
                raise RuntimeError(f"相机尚未产生完整 RGB-D：{source}")
            depth = numpy_value(depth)
            intrinsics, transform = numpy_value(camera.get_intrinsics_matrix()), self.camera_transform(source)
            mask = self.robot_filter.mask(depth, intrinsics, transform)
            frame = CameraFrame(source, robot.timestamp_sec, rgb, depth, intrinsics, transform, mask)
            results.append(RealObservation(f"{self.epoch}:{self.sequence}:{source}", frame, robot))
        self.sequence += 1
        return tuple(results)

    def close(self):
        for camera in self.cameras.values():
            camera.pause()
            for name in tuple(camera._custom_annotators):
                camera.detach_annotator(name)
        self.cameras.clear()
        self.robot_filter = None

