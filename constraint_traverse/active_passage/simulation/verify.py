from __future__ import annotations

from copy import deepcopy
import random
import json

import numpy as np

from ..recording import inspect_record
from ..cameras import CAMERA_PATHS
from ..state import require_reset_accuracy
from constraint_traverse.rgbd_map import SOURCE_BITS
from vln_nav.runtime.isaac_runtime_utils import resolve_robot_relative_prim_path
from vln_nav.active_perception.camera_geometry import backproject_depth_pixels_to_world


def verify_robot_depth_filter(runtime):
    saved, previous_start = runtime.capture_snapshot(), runtime.experiment.saved
    try:
        for camera in runtime.sensors.cameras.values():
            camera.add_instance_id_segmentation_to_frame()
        runtime.observations = runtime.sensors.capture()
        details = {}
        for observation in runtime.observations:
            frame, robot = observation.frame, observation.robot
            data = runtime.sensors.cameras[frame.frame_id]._custom_annotators["instance_id_segmentation"].get_data()
            labels = data["info"]["idToLabels"]
            robot_ids = [int(key) for key, value in labels.items()
                         if str(value).startswith(runtime.profile.asset.prim_path + "/")]
            stride = runtime.local_map.config.pixel_stride
            u, v = np.meshgrid(np.arange(stride // 2, frame.depth_m.shape[1], stride),
                               np.arange(stride // 2, frame.depth_m.shape[0], stride))
            depth = frame.depth_m[v, u]
            valid = np.isfinite(depth) & (depth >= runtime.local_map.config.min_depth_m) & (depth <= runtime.local_map.config.max_depth_m)
            points = backproject_depth_pixels_to_world(pixel_u=u, pixel_v=v, depth_m=np.where(valid, depth, 0),
                camera_intrinsics=frame.intrinsics, camera_to_world=frame.world_from_camera)
            local = (points - robot.world_from_base[:3, 3]) @ robot.world_from_base[:3, :3]
            floor_z = robot.world_from_base[2, 3] - runtime.floor_offset
            hit = valid & (points[..., 2] - floor_z >= runtime.local_map.min_height) & (points[..., 2] - floor_z <= runtime.local_map.max_height)
            near = hit & (local[..., 0] > 0) & (local[..., 0] < 1.5) & (np.abs(local[..., 1]) < 0.8)
            ids = np.asarray(data["data"]).reshape(frame.depth_m.shape)[v, u]
            self_hit = near & np.isin(ids, robot_ids)
            remaining = int(np.count_nonzero(self_hit & ~frame.robot_mask[v, u]))
            details[frame.frame_id] = dict(raw_robot_obstacle_rays=int(self_hit.sum()), remaining_robot_obstacle_rays=remaining)
            if remaining:
                raise AssertionError(f"机器人自身障碍回波仍有遗漏：{frame.frame_id}，{remaining}")
        if runtime.args.usd is None:
            grid = runtime.local_map.snapshot().grid
            rows, columns = np.indices(grid.shape)
            base = runtime.robot_state().world_from_base
            position = base[:3, 3]
            dx = grid.origin_x_m + (columns + 0.5) * grid.resolution_m - position[0]
            dy = grid.origin_y_m + (rows + 0.5) * grid.resolution_m - position[1]
            x, y = base[0, 0] * dx + base[1, 0] * dy, base[0, 1] * dx + base[1, 1] * dy
            count = int(np.count_nonzero(grid.occupied_mask & (x > 0) & (x < 1.5) & (np.abs(y) < 0.8)))
            if count:
                raise AssertionError(f"空白地面前方仍有历史障碍栅格：{count}")
            dc = runtime.drive._dc
            handle = dc.get_rigid_body("/World/Environment/ResetProbe")
            pose = dc.get_rigid_body_pose(handle)
            pose.p = tuple(position + base[:3, :3] @ [1.2, 0.6, 0.8])
            dc.set_rigid_body_pose(handle, pose)
            dc.set_rigid_body_linear_velocity(handle, (0, 0, 0))
            runtime.step([0, 0])
            runtime.observations = runtime.sensors.capture()
            visible, removed = 0, 0
            for observation in runtime.observations:
                frame = observation.frame
                data = runtime.sensors.cameras[frame.frame_id]._custom_annotators["instance_id_segmentation"].get_data()
                ids = [int(key) for key, value in data["info"]["idToLabels"].items()
                       if str(value).startswith("/World/Environment/ResetProbe")]
                probe = np.isin(np.asarray(data["data"]).reshape(frame.depth_m.shape), ids)
                visible += int(probe.sum())
                removed += int(np.count_nonzero(probe & frame.robot_mask))
            if not visible or removed:
                raise AssertionError(f"真实测试障碍的过滤检查失败：可见像素 {visible}，被移除像素 {removed}")
            details["environment_probe"] = dict(visible_pixels=visible, removed_pixels=removed, initial_front_occupied_cells=count)
        return details
    finally:
        for camera in runtime.sensors.cameras.values():
            camera.remove_instance_id_segmentation_from_frame()
        runtime.experiment.saved = saved
        runtime.experiment.reset()
        runtime.experiment.saved = previous_start


def verify_map_fusion(runtime):
    saved, previous_start = runtime.capture_snapshot(), runtime.experiment.saved
    try:
        runtime.set_map_sources(("front", "down"))
        runtime.reset_map()
        runtime.observations = runtime.sensors.capture()
        runtime.update_map(runtime.observations)
        baseline = runtime.local_map.source_mask.copy()
        path = runtime.experiment.start()
        if runtime.panel is not None:
            runtime.panel.update(runtime)
            runtime.panel.source_models["arm"].set_value(True)
            runtime.handle_commands()
        else:
            runtime.set_map_sources(tuple(CAMERA_PATHS))
        if runtime.fusion.sources != tuple(CAMERA_PATHS):
            raise AssertionError("机械臂融合开关没有启用对应来源")
        runtime.observations = runtime.sensors.capture()
        arm = tuple(value for value in runtime.observations if value.frame.frame_id == "arm")
        update = runtime.update_map(arm)
        if update is None or {row["source"] for row in update["observations"]} != {"arm"}:
            raise AssertionError("机械臂真实观测没有提交地图融合")
        runtime.experiment.sample(runtime.robot_state(), runtime.requested, runtime.applied,
                                   runtime.observations, physical_time_sec=runtime.world.current_time)
        arm_cells = runtime.fusion.source_cell_counts()["arm"]
        new_cells = int(np.count_nonzero((baseline == 0) & ((runtime.local_map.source_mask & SOURCE_BITS["arm"]) != 0)))
        if arm_cells == 0:
            raise AssertionError("机械臂没有提供有效地图证据")
        runtime.experiment.stop("STOPPED")
        record = inspect_record(path)
        with (path / "records.jsonl").open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle]
        if not any(row.get("name") == "MAP_SOURCES_CHANGED" for row in events):
            raise AssertionError("融合来源切换没有写入试验记录")
        if record["map_sources"] != ["front", "down"]:
            raise AssertionError("试验初始融合条件被后续来源切换覆盖")
        runtime.set_map_sources(("arm",))
        runtime.reset_map()
        runtime.observations = runtime.sensors.capture()
        runtime.update_map(runtime.observations)
        if runtime.fusion.source_cell_counts()["arm"] == 0 or (runtime.local_map.source_mask & 3).any():
            raise AssertionError("机械臂独立融合没有通过来源隔离检查")
        version, mask = runtime.local_map.version, runtime.local_map.source_mask.copy()
        runtime.set_map_sources(())
        if runtime.update_map(runtime.sensors.capture()) is not None or runtime.local_map.version != version:
            raise AssertionError("关闭全部融合后仍然更新了地图")
        np.testing.assert_array_equal(runtime.local_map.source_mask, mask)
        return dict(arm_observed_cells=arm_cells, arm_newly_observed_cells=new_cells,
                    arm_only_fusion=True, disabled_sources_preserve_history=True,
                    provenance_record_verified=True, ui_switch_verified=runtime.panel is not None)
    finally:
        runtime.experiment.saved = saved
        runtime.experiment.reset()
        require_reset_accuracy(runtime.experiment.last_reset_errors)
        runtime.experiment.saved = previous_start


def verify(runtime, repeats):
    active_paths = {source: camera.prim_path for source, camera in runtime.sensors.cameras.items()}
    expected_paths = {source: resolve_robot_relative_prim_path(robot_root_path=runtime.profile.asset.prim_path,
                                                               prim_path=path)
                      for source, path in CAMERA_PATHS.items()}
    if active_paths != expected_paths:
        raise AssertionError(f"实际启动的相机与指定分工不同：{active_paths}")
    runtime.confirm_placement()
    for _ in range(60):
        runtime.step([0, 0])
    runtime.observations = runtime.sensors.capture()
    ratios = {o.frame.frame_id: float(np.mean(np.isfinite(o.frame.depth_m) & (o.frame.depth_m > 0)))
              for o in runtime.observations}
    if any(value < 0.02 for value in ratios.values()):
        raise AssertionError(f"相机有效深度不足：{ratios}")
    runtime.goal = runtime.robot_state().world_from_base[:3, 3].copy() + [0.5, 0, 0]
    depth_filter_check = verify_robot_depth_filter(runtime)
    fusion_check = verify_map_fusion(runtime)
    ui_goal_verified = False
    ui_english_verified = False
    if runtime.panel is not None:
        runtime.panel.update(runtime)
        for _ in range(5):
            runtime.app.update()
        panel = runtime.panel
        image = panel.image
        old_goal = runtime.goal.copy()
        panel._click(image.screen_position_x + image.computed_width * 0.5,
                     image.screen_position_y + image.computed_height * 0.5, 0, 0)
        runtime.handle_commands()
        if runtime.goal is None or not np.isfinite(runtime.goal).all() or np.linalg.norm(runtime.goal - old_goal) < 0.1:
            raise AssertionError("地图点击没有生成有效世界坐标目标")
        ui_goal_verified = True
        import omni.ui as ui
        pending = [runtime.panel.controls.frame, runtime.panel.map_window.frame]
        while pending:
            widget = pending.pop()
            text = getattr(widget, "text", None)
            if isinstance(text, str) and not text.isascii():
                raise AssertionError(f"面板文字包含非英文字符：{text!r}")
            pending.extend(ui.Inspector.get_children(widget))
        ui_english_verified = True
    initial = runtime.experiment.save_start()
    expected_random = None
    trajectories, errors, records = [], [], []
    for index in range(repeats):
        from pxr import Gf, UsdGeom
        root = UsdGeom.Xformable(runtime.stage.GetPrimAtPath("/World/Environment"))
        changed = Gf.Matrix4d(root.GetLocalTransformation())
        changed.SetTranslateOnly(Gf.Vec3d(0.15, 0.0, 0.0))
        root.MakeMatrixXform().Set(changed)
        dc = runtime.drive._dc
        handle = dc.get_rigid_body("/World/Environment/ResetProbe")
        dc.set_rigid_body_linear_velocity(handle, (0.3, 0.2, 0.1))
        runtime.experiment.saved = initial
        runtime.experiment.reset()
        errors.append(runtime.experiment.last_reset_errors)
        values = [random.random(), float(np.random.random()), float(runtime.generator.random())]
        if expected_random is None:
            expected_random = values
        np.testing.assert_array_equal(values, expected_random)
        runtime.experiment.start()
        trajectory = []
        for step in range(12):
            runtime.step([0.2, 0.0])
            trajectory.append(runtime.robot_state().world_from_base[:3, 3].copy())
        trajectories.append(np.asarray(trajectory))
        runtime.experiment.stop("STOPPED")
        records.append(inspect_record(runtime.experiment.last_record))
        if runtime.panel is not None:
            runtime.panel.update(runtime)
            runtime.app.update()
        runtime.logger.info("物理恢复验收完成：%s/%s", index + 1, repeats)
    maximum = max(float(np.max(np.linalg.norm(value - trajectories[0], axis=1))) for value in trajectories)
    displacement = float(np.linalg.norm(trajectories[0][-1] - initial.robot.world_from_base[:3, 3]))
    if displacement < 0.001:
        raise AssertionError(f"底盘运动未通过验收，实际位移仅为 {displacement} m")
    if maximum > 0.02:
        raise AssertionError(f"相同控制重复执行的轨迹偏差超过 2 cm：{maximum}")
    runtime.experiment.saved = initial
    runtime.experiment.reset()
    runtime.experiment.start()
    runtime.experiment.timeout_sec = runtime.world.get_physics_dt() * 2
    for _ in range(3):
        runtime.step([0, 0])
    timeout_record = inspect_record(runtime.experiment.last_record)
    if timeout_record["status"] != "TIMEOUT":
        raise AssertionError("超时试验没有保存正确结果")
    map_before = deepcopy(initial.map_state)
    runtime.local_map.reset(runtime.robot_state().world_from_base[:2, 3])
    runtime.experiment.saved = initial
    runtime.experiment.reset()
    np.testing.assert_array_equal(runtime.local_map.log_odds, map_before["log_odds"])
    np.testing.assert_array_equal(runtime.local_map.source_mask, map_before["source_mask"])
    return dict(reset_repetitions=repeats, reset_errors=errors, repeat_max_position_error_m=maximum,
                actual_motion_displacement_m=displacement, scene_rigid_bodies=len(initial.rigid_bodies),
                camera_paths=active_paths, camera_valid_ratios=ratios, synchronized_observations=True,
                map_restored=True, random_state_restored=True, records_verified=len(records) + 2,
                map_fusion=fusion_check,
                robot_depth_filter=depth_filter_check, ui_english_verified=ui_english_verified,
                timeout_saved=True, ui_goal_verified=ui_goal_verified,
                execution_mode="manual", automatic_navigation_tested=False)
