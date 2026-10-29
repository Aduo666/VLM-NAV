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
    fusion_check = verify_map_fusion(runtime)
    ui_goal_verified = False
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
                timeout_saved=True, ui_goal_verified=ui_goal_verified,
                execution_mode="manual", automatic_navigation_tested=False)
