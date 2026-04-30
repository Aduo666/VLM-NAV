from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def wait_for_stage_assets(*, simulation_app, is_stage_loading, timeout_sec: float, logger, label: str) -> None:
    start_time = time.monotonic()
    last_log_time = start_time - 10.0
    while is_stage_loading():
        simulation_app.update()
        now = time.monotonic()
        if now - last_log_time >= 5.0:
            logger.info("Waiting for %s assets to finish loading...", label)
            last_log_time = now
        if now - start_time > timeout_sec:
            raise TimeoutError(f"Timed out while waiting for {label} assets to load.")


def bind_viewport_to_camera(*, camera_prim_path: str, logger, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except ImportError:
        logger.warning("Viewport API binding is unavailable; keeping perspective camera.")
        return False
    viewport_api = get_active_viewport()
    if viewport_api is None:
        logger.warning("Active viewport not found; keeping perspective camera.")
        return False
    try:
        viewport_api.camera_path = camera_prim_path
    except Exception as exc:
        logger.warning("Failed to bind active viewport to %s: %s", camera_prim_path, exc)
        return False
    logger.info("Bound active viewport to camera: %s", camera_prim_path)
    return True


def resolve_robot_relative_prim_path(*, robot_root_path: str, prim_path: str) -> str:
    candidate = str(prim_path).strip()
    if not candidate:
        return str(robot_root_path)
    if candidate.startswith("/"):
        return candidate
    robot_name = Path(str(robot_root_path).rstrip("/")).name
    if candidate == robot_name:
        return str(robot_root_path)
    if candidate.startswith(f"{robot_name}/"):
        candidate = candidate[len(robot_name) + 1 :]
    return f"{str(robot_root_path).rstrip('/')}/{candidate.lstrip('/')}"


def resolve_existing_prim_path(*, stage, prim_path: str, kind: str) -> str:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        return prim_path

    head, slash, tail = prim_path.rpartition("/")
    if slash and tail:
        fallback_names = []
        if tail == "camera":
            fallback_names.append("Camera")
        elif tail == "Camera":
            fallback_names.append("camera")

        for fallback_name in fallback_names:
            fallback_path = f"{head}/{fallback_name}"
            fallback_prim = stage.GetPrimAtPath(fallback_path)
            if fallback_prim and fallback_prim.IsValid():
                return fallback_path

    raise RuntimeError(f"Configured {kind} prim does not exist: {prim_path}")


def find_articulation_root_path(*, stage, root_path: str) -> str:
    from pxr import UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Robot prim path is invalid: {root_path}")

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(root_path):
            continue
        if UsdPhysics.ArticulationRootAPI(prim):
            return prim_path
    return root_path


def set_local_xform(*, stage, prim_path: str, translation: list[float], rotation_deg: list[float]) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for transform setup: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()
    translate_op.Set(tuple(float(value) for value in translation))
    rotate_op.Set(tuple(float(value) for value in rotation_deg))


def build_hold_joint_targets(*, robot, available_joint_names: list[str], hold_joint_names: list[str]) -> dict[str, float]:
    joint_positions = get_joint_positions(robot)
    return {
        name: float(joint_positions[available_joint_names.index(name)])
        for name in hold_joint_names
        if name in available_joint_names
    }


def get_joint_names(robot) -> list[str]:
    try:
        names = robot.dof_names
    except Exception as exc:
        raise RuntimeError(f"Unable to query articulation DOF names: {exc}") from exc
    return [str(name) for name in names]


def get_joint_positions(robot) -> np.ndarray:
    positions = robot.get_joint_positions()
    if positions is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(positions, "detach"):
        positions = positions.detach()
    if hasattr(positions, "cpu"):
        positions = positions.cpu()
    if hasattr(positions, "numpy"):
        positions = positions.numpy()
    return np.asarray(positions, dtype=np.float32).reshape(-1)


def get_world_pose(robot) -> tuple[np.ndarray, np.ndarray]:
    pose = robot.get_world_pose()
    if isinstance(pose, tuple) and len(pose) == 2:
        position, orientation = pose
    else:
        raise RuntimeError("Unexpected world pose return value from articulation.")
    return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)
