from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig


def collision_bounds_in_frame(stage, prim, frame_path, *, include_details=False):
    from pxr import Usd, UsdGeom

    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    frame = stage.GetPrimAtPath(frame_path)
    if not frame.IsValid():
        raise ValueError(f"车体坐标系路径无效：{frame_path}")
    frame_inverse = transforms.GetLocalToWorldTransform(frame).GetInverse()
    chunks = []
    details = []
    for geometry in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not geometry.IsA(UsdGeom.Boundable):
            continue
        boundable = UsdGeom.Boundable(geometry)
        # 从实际形状参数和 Mesh 顶点重新计算，USD 中已有的 extent 可能过期。
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, Usd.TimeCode.Default())
        if extent is None or len(extent) != 2:
            raise ValueError(f"无法重新计算碰撞几何范围：{geometry.GetPath()}")
        minimum, maximum = np.asarray(extent, dtype=np.float64)
        if not np.isfinite([minimum, maximum]).all() or np.any(minimum >= maximum):
            raise ValueError(f"碰撞几何范围无效：{geometry.GetPath()}")
        corners = np.asarray([
            [x, y, z, 1.0]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ])
        relative = transforms.GetLocalToWorldTransform(geometry) * frame_inverse
        chunks.append((corners @ np.asarray(relative, dtype=np.float64))[:, :3])
        authored = boundable.GetExtentAttr().Get()
        details.append({
            "prim_path": str(geometry.GetPath()),
            "usd_type": geometry.GetTypeName(),
            "extent_source": "USD_SHAPE_COMPUTED_EXTENT",
            "authored_extent_matches_shape": (
                bool(np.allclose(np.asarray(authored), extent, atol=1.0e-6, rtol=1.0e-6))
                if authored is not None and len(authored) == 2 else None
            ),
        })
    if not chunks:
        raise ValueError(f"碰撞体没有可以核查的几何：{prim.GetPath()}")
    points = np.concatenate(chunks)
    return (points, details) if include_details else points


def rotation_envelope(points, pivot, axis):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if not np.isfinite(points).all() or not np.isfinite(pivot).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("转向包络需要有效的几何、转轴和转向中心")
    axis = axis / norm
    vectors = points - pivot
    parallel = np.outer(vectors @ axis, axis)
    cosine = vectors - parallel
    sine = np.cross(axis, vectors)
    radius = np.hypot(cosine, sine)
    # 完整旋转的解析极值同时覆盖连续转向和轮胎转向过渡。
    return (pivot + parallel - radius).min(axis=0), (pivot + parallel + radius).max(axis=0)


def build_agx_footprint(stage, profile, articulation_path, *, margin_m, expected_collider_count=23):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(profile.asset.prim_path)
    if not root.IsValid():
        raise ValueError(f"机器人根路径无效：{profile.asset.prim_path}")
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if expected_collider_count is not None and len(colliders) != expected_collider_count:
        raise ValueError(f"USD 碰撞体核查数量为 {len(colliders)}，预期为 {expected_collider_count}")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_inverse = transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(articulation_path)).GetInverse()
    joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim) for prim in prims if prim.IsA(UsdPhysics.RevoluteJoint)}
    steering_links = {}
    wheel_names = {"fl_wheel": "front_left_wheel", "fr_wheel": "front_right_wheel", "rl_wheel": "rear_left_wheel", "rr_wheel": "rear_right_wheel"}
    for wheel in profile.agx_controller.continuous.wheels:
        steer = joints[wheel.steering_joint]
        drive = joints[wheel.drive_joint]
        body0 = stage.GetPrimAtPath(steer.GetBody0Rel().GetTargets()[0])
        body_transform = transforms.GetLocalToWorldTransform(body0) * base_inverse
        pivot = np.asarray(body_transform.Transform(Gf.Vec3d(*steer.GetLocalPos0Attr().Get())), dtype=float)
        quat = steer.GetLocalRot0Attr().Get()
        rotation = Gf.Rotation(Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(*quat.GetImaginary())))
        basis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[str(steer.GetAxisAttr().Get())]
        axis = np.asarray(body_transform.TransformDir(rotation.TransformDir(Gf.Vec3d(*basis))), dtype=float)
        for target, is_tire in ((steer.GetBody1Rel().GetTargets()[0], False), (drive.GetBody1Rel().GetTargets()[0], True)):
            steering_links[str(target)] = (wheel, pivot, axis, is_tire, steer)
    parts = []
    records = []
    name_counts = {}
    for prim in colliders:
        path = str(prim.GetPath())
        points, geometry_details = collision_bounds_in_frame(stage, prim, articulation_path, include_details=True)
        current_min, current_max = points.min(axis=0), points.max(axis=0)
        minimum, maximum = current_min.copy(), current_max.copy()
        relative_path = path[len(profile.asset.prim_path.rstrip("/")) + 1:]
        link_name = relative_path.split("/")[0]
        name = {"base_link": "chassis", "box_link": "body_shell"}.get(link_name, link_name)
        clearing = None
        steering_record = None
        for link_path, (wheel, pivot, axis, is_tire, steer) in steering_links.items():
            if path == link_path or path.startswith(link_path + "/"):
                minimum, maximum = rotation_envelope(points, pivot, axis)
                name = wheel_names.get(wheel.drive_joint, wheel.drive_joint)
                if not is_tire:
                    name += "_steering_support"
                # 只有轮胎内部始终被占据的中心区域提供当前占用证据。
                core_radius = min(*(pivot[:2] - current_min[:2]), *(current_max[:2] - pivot[:2]))
                if core_radius <= 0 or np.linalg.norm(axis[:2]) > 1.0e-5:
                    raise ValueError(f"无法验证转向部件的当前占用中心：{path}")
                half = 0.95 * core_radius / math.sqrt(2.0)
                clearing = (float(pivot[0] - half), float(pivot[0] + half), float(pivot[1] - half), float(pivot[1] + half))
                steering_record = {
                    "joint": wheel.steering_joint,
                    "pivot_base_xyz_m": pivot.tolist(),
                    "axis_base_xyz": axis.tolist(),
                    "controller_max_steering_rad": profile.agx_controller.continuous.max_steering_rad,
                    "usd_limits_deg": [float(steer.GetLowerLimitAttr().Get()), float(steer.GetUpperLimitAttr().Get())],
                    "envelope": "FULL_ROTATION_CONSERVATIVE",
                }
                break
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name += f"_{name_counts[name]}"
        enabled = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        part = FootprintPartConfig(name, float(minimum[0]), float(maximum[0]), float(minimum[1]), float(maximum[1]), clearing)
        part.validate()
        if enabled:
            parts.append(part)
        records.append({
            "collision_prim_path": path,
            "usd_type": prim.GetTypeName(),
            "purpose": str(UsdGeom.Imageable(prim).ComputePurpose()),
            "enabled": enabled,
            "geometry_status": "CONSERVATIVE_USD_BOUND_VERIFIED",
            "geometry_descendants": geometry_details,
            "current_min_base_xyz_m": current_min.tolist(),
            "current_max_base_xyz_m": current_max.tolist(),
            "envelope_min_base_xyz_m": minimum.tolist(),
            "envelope_max_base_xyz_m": maximum.tolist(),
            "part": asdict(part),
            "steering": steering_record,
        })
    footprint = FootprintConfig(tuple(parts), float(margin_m))
    footprint.validate()
    return footprint, {
        "asset_path": str(profile.asset.usd_path),
        "base_frame_path": articulation_path,
        "collider_count": len(colliders),
        "enabled_collider_count": len(parts),
        "status": "COMPLETE_CONSERVATIVE_BOUNDS",
        "safety_margin_m": float(margin_m),
        "colliders": records,
    }


def validate_agx_geometry(stage, profile, articulation_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(profile.asset.prim_path)
    by_name = {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            if prim.GetName() in by_name:
                raise ValueError(f"Ambiguous joint name: {prim.GetName()}")
            by_name[prim.GetName()] = prim
    chassis_inverse = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(articulation_path)
    ).GetInverse()
    for wheel in profile.agx_controller.continuous.wheels:
        for name, expected_axis in (
            (wheel.steering_joint, (0, 0, wheel.steering_sign)),
            (wheel.drive_joint, (0, wheel.drive_sign, 0)),
        ):
            if name not in by_name:
                raise ValueError(f"Configured AGX joint is missing from USD: {name}")
            joint = UsdPhysics.RevoluteJoint(by_name[name])
            body = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            body_to_world = cache.GetLocalToWorldTransform(body)
            basis = {
                "X": (1, 0, 0),
                "Y": (0, 1, 0),
                "Z": (0, 0, 1),
            }[str(joint.GetAxisAttr().Get())]
            local_rotation = joint.GetLocalRot0Attr().Get()
            rotation = Gf.Rotation(
                Gf.Quatd(
                    float(local_rotation.GetReal()),
                    Gf.Vec3d(*local_rotation.GetImaginary()),
                )
            )
            axis = chassis_inverse.TransformDir(
                body_to_world.TransformDir(rotation.TransformDir(Gf.Vec3d(*basis)))
            )
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            if np.linalg.norm(axis - expected_axis) > 0.05:
                raise ValueError(
                    f"USD joint {name} axis={axis}, expected={expected_axis}"
                )
            if name == wheel.steering_joint:
                position = chassis_inverse.Transform(
                    body_to_world.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get()))
                )
                if np.linalg.norm(
                    np.asarray(position)[:2] - [wheel.x_m, wheel.y_m]
                ) > 0.04:
                    raise ValueError(
                        f"USD joint {name} XY={tuple(position)[:2]} disagrees "
                        "with continuous wheel geometry"
                    )
                lower = float(joint.GetLowerLimitAttr().Get())
                upper = float(joint.GetUpperLimitAttr().Get())
                required = math.degrees(
                    float(profile.agx_controller.continuous.max_steering_rad)
                )
                if lower > -required or upper < required:
                    raise ValueError(
                        f"USD steering limits for {name} do not support "
                        f"+/-{required:.1f} degrees"
                    )
