from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics

from vln_nav.active_perception.camera_geometry import world_from_pose_wxyz


@wp.kernel
def transform_vertices(local: wp.array(dtype=wp.vec3), bodies: wp.array(dtype=int),
                       transforms: wp.array(dtype=wp.mat44), world: wp.array(dtype=wp.vec3)):
    index = wp.tid()
    point, matrix = local[index], transforms[bodies[index]]
    world[index] = wp.vec3(matrix[0, 0] * point[0] + matrix[0, 1] * point[1] + matrix[0, 2] * point[2] + matrix[0, 3],
                           matrix[1, 0] * point[0] + matrix[1, 1] * point[1] + matrix[1, 2] * point[2] + matrix[1, 3],
                           matrix[2, 0] * point[0] + matrix[2, 1] * point[1] + matrix[2, 2] * point[2] + matrix[2, 3])


@wp.kernel
def match_robot_surface(mesh: wp.uint64, depth: wp.array2d(dtype=float), camera: wp.mat44,
                        fx: float, fy: float, cx: float, cy: float, tolerance: float,
                        mask: wp.array2d(dtype=wp.uint8)):
    row, column = wp.tid()
    z = depth[row, column]
    if wp.isfinite(z) and z > 0.0:
        x, y = (float(column) - cx) * z / fx, (float(row) - cy) * z / fy
        point = wp.vec3(camera[0, 0] * x + camera[0, 1] * y + camera[0, 2] * z + camera[0, 3],
                        camera[1, 0] * x + camera[1, 1] * y + camera[1, 2] * z + camera[1, 3],
                        camera[2, 0] * x + camera[2, 1] * y + camera[2, 2] * z + camera[2, 3])
        query = wp.mesh_query_point_no_sign(mesh, point, tolerance)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            if wp.length(point - closest) <= tolerance:
                mask[row, column] = wp.uint8(1)


class RobotDepthFilter:
    def __init__(self, runtime, tolerance_m=0.003):
        self.runtime, self.tolerance_m = runtime, float(tolerance_m)
        self.device = wp.get_device(f"cuda:{runtime.args.gpu}")
        self.body_paths, points, triangles, body_indices = [], [], [], []
        cache = UsdGeom.XformCache()
        count = 0
        root = runtime.stage.GetPrimAtPath(runtime.profile.asset.prim_path)
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or imageable.ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
            if not len(vertices) or not len(faces):
                continue
            if not (counts == 3).all():
                raise ValueError(f"机器人表面模型包含尚未支持的非三角 Mesh：{prim.GetPath()}")
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise ValueError(f"机器人表面没有所属刚体：{prim.GetPath()}")
            path = str(body.GetPath())
            if path not in self.body_paths:
                self.body_paths.append(path)
            index = self.body_paths.index(path)
            relative = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse())
            local = vertices @ relative[:3, :3] + relative[3, :3]
            points.append(local.astype(np.float32))
            triangles.append(faces + count)
            body_indices.append(np.full(len(vertices), index, dtype=np.int32))
            count += len(vertices)
        if not points:
            raise ValueError("机器人没有可用于深度过滤的可见表面")
        self.local = wp.array(np.concatenate(points), dtype=wp.vec3, device=self.device)
        self.bodies = wp.array(np.concatenate(body_indices), dtype=int, device=self.device)
        self.world = wp.empty(count, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(np.concatenate(triangles), dtype=int, device=self.device)
        self._update_vertices()
        self.mesh = wp.Mesh(points=self.world, indices=self.indices)

    def _update_vertices(self):
        transforms = []
        dc = self.runtime.drive._dc
        for path in self.body_paths:
            pose = dc.get_rigid_body_pose(dc.get_rigid_body(path))
            transforms.append(world_from_pose_wxyz([pose.p.x, pose.p.y, pose.p.z],
                               [pose.r.w, pose.r.x, pose.r.y, pose.r.z]))
        self.transforms = wp.array(np.asarray(transforms, dtype=np.float32), dtype=wp.mat44, device=self.device)
        wp.launch(transform_vertices, dim=len(self.local),
                  inputs=[self.local, self.bodies, self.transforms, self.world], device=self.device)

    def update(self):
        self._update_vertices()
        self.mesh.refit()

    def mask(self, depth, intrinsics, world_from_camera):
        depth = np.asarray(depth, dtype=np.float32)
        source = wp.array(depth, dtype=float, device=self.device)
        result = wp.zeros(depth.shape, dtype=wp.uint8, device=self.device)
        matrix = wp.mat44(np.asarray(world_from_camera, dtype=np.float32))
        wp.launch(match_robot_surface, dim=depth.shape,
                  inputs=[self.mesh.id, source, matrix, float(intrinsics[0, 0]), float(intrinsics[1, 1]),
                          float(intrinsics[0, 2]), float(intrinsics[1, 2]), self.tolerance_m, result], device=self.device)
        return result.numpy().astype(bool)

    def metadata(self):
        return dict(method="KNOWN_ROBOT_VISIBLE_TRIANGLES", tolerance_m=self.tolerance_m,
                    body_count=len(self.body_paths), triangle_count=len(self.indices) // 3)
