from __future__ import annotations

from datetime import datetime
import json
import logging

import numpy as np
import torch
import yaml

from .run import ROOT, load_config


def main():
    config = load_config(ROOT / "constraint_traverse/configs/benchmark.yaml")
    logging.basicConfig(level=logging.INFO)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, **config["simulation"]})
    try:
        from .runtime import PassageRuntime
        output = ROOT / "constraint_traverse/data/prediction_validation"
        runtime = PassageRuntime(app, ROOT, config, output, True, [1.6, 0])
        runtime._initialize()
        if not runtime._start("MPPI", "goal"):
            raise RuntimeError(runtime.result_text)
        runtime._finish("VALIDATION_SETUP")
        reports = {}
        for name, action in (("straight", [0.3, 0]), ("turn", [0.25, 0.15]), ("brake", [0, 0])):
            runtime._reset()
            if name == "brake":
                for _ in range(60):
                    runtime._apply([0.3, 0])
                    runtime.world.step(render=False)
            initial = runtime._state()
            predicted, legacy = torch.as_tensor(initial), torch.as_tensor(initial)
            requested = torch.tensor(action, dtype=torch.float32)
            samples = {}
            for step in range(1, 241):
                runtime._apply(action)
                runtime.world.step(render=False)
                if step % runtime.period_steps == 0:
                    predicted = runtime.model.step(predicted, requested)
                    legacy = runtime.model._physical_step(legacy, requested, runtime.model.dt_sec)
                if step in (12, 60, 240):
                    actual = runtime._state()
                    samples[str(round(step * runtime.dt, 3))] = {
                        "position_error_m": float(np.linalg.norm(actual[:2] - predicted[:2].numpy())),
                        "legacy_position_error_m": float(np.linalg.norm(actual[:2] - legacy[:2].numpy())),
                        "yaw_error_deg": float(np.rad2deg(abs(actual[2] - float(predicted[2])))),
                        "legacy_yaw_error_deg": float(np.rad2deg(abs(actual[2] - float(legacy[2])))),
                    }
            reports[name] = samples
            print(json.dumps({"case": name, "measurements": samples}, ensure_ascii=False), flush=True)
        from vln_nav.control.trajectory import PathTracker, AgxVelocityAdapter
        from vln_nav.control.fine_path_tracker import fit_agx_command
        reports["coarse_tracking"] = {}
        for name, distance in (("reverse", -0.5), ("short_forward", 0.04)):
            runtime._reset()
            initial = runtime._state()
            path_xy = np.asarray([initial[:2], initial[:2] + [distance, 0]])
            tracker = PathTracker(path_xy, runtime.profile.agx_controller.continuous)
            adapter = AgxVelocityAdapter(runtime.drive, runtime.profile.agx_controller.continuous)
            for _ in range(600):
                tracking = tracker.step(runtime._state()[:3], runtime.dt)
                adapter.apply(fit_agx_command(tracking.command, runtime.profile.agx_controller.continuous))
                if tracking.status == "ARRIVED":
                    break
                runtime.world.step(render=False)
            reports["coarse_tracking"][name] = {"status": tracking.status,
                                                 "displacement_x_m": float(runtime._state()[0] - initial[0])}
        print(json.dumps({"coarse_tracking": reports["coarse_tracking"]}, ensure_ascii=False), flush=True)
        runtime._reset()
        from isaacsim.sensors.camera import Camera
        from constraint_traverse.rgbd_map import RGBDPlanarMap, RGBDMapConfig
        from vln_nav.active_perception.contracts import TimedDepthObservation
        from vln_nav.active_perception.camera_geometry import capture_camera_frame, world_from_pose_wxyz
        from vln_nav.runtime.isaac_runtime_utils import resolve_robot_relative_prim_path, set_camera_horizontal_fov
        with (ROOT / "configs/navigation.yaml").open(encoding="utf-8") as handle:
            navigation = yaml.safe_load(handle)
        cameras = {}
        for source, relative_path in (("front", runtime.profile.sensors.camera.front_prim_path),
                                      ("down", runtime.profile.sensors.camera.mapping_down_prim_path)):
            prim_path = resolve_robot_relative_prim_path(robot_root_path=runtime.profile.asset.prim_path, prim_path=relative_path)
            if source == "front":
                set_camera_horizontal_fov(stage=runtime.stage, prim_path=prim_path,
                                          fov_deg=navigation["camera"]["horizontal_fov_deg"])
            camera = Camera(prim_path=prim_path, name=f"validation_{source}", frequency=15, resolution=(640, 480))
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
            cameras[source] = camera
        mapper = RGBDPlanarMap(RGBDMapConfig.from_mapping(navigation["mapping"]["map"]),
                               obstacle_min_height_m=navigation["fine_planner"]["obstacle_min_height_m"],
                               obstacle_max_height_m=navigation["fine_planner"]["obstacle_max_height_m"])
        updates = []
        for frame in range(60):
            runtime._apply([0, 0])
            runtime.world.step(render=True)
            if frame % 15:
                continue
            position, orientation = runtime.robot.get_world_pose()
            samples = [capture_camera_frame(source=source, camera=camera, frame_index=frame,
                                             world_from_base=world_from_pose_wxyz(position, orientation))
                       for source, camera in cameras.items()]
            if any(sample is None for sample in samples):
                continue
            update = mapper.integrate_batch(
                [TimedDepthObservation(sample.source, sample.timestamp_sec, sample.depth_m,
                                       sample.intrinsics, sample.world_from_camera) for sample in samples],
                center_world_xy=position[:2], floor_z_m=float(position[2]) - runtime.profile.spawn_z_m,
            )
            updates.append({"sources": update.source_cell_update_counts})
        grid = mapper.snapshot().grid
        reports["rgbd_map"] = {"updates": updates, "free_cells": int(grid.free_mask.sum()),
                                "occupied_cells": int(grid.occupied_mask.sum()), "unknown_cells": int(grid.unknown_mask.sum())}
        if not updates or not grid.free_mask.any():
            raise RuntimeError("真实 RGB-D 相机未生成二维空闲区域")
        print(json.dumps({"rgbd_map": reports["rgbd_map"]}, ensure_ascii=False), flush=True)
        output.mkdir(parents=True, exist_ok=True)
        path = output / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_report.json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(reports, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"report": str(path)}, ensure_ascii=False), flush=True)
        runtime.subscription = None
    finally:
        app.close()


if __name__ == "__main__":
    main()
