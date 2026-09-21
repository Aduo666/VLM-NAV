from __future__ import annotations

import math

import numpy as np


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
