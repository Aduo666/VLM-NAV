from __future__ import annotations

from datetime import datetime
import json
import logging

import numpy as np
import shapely

from constraint_traverse.benchmark.run import ROOT, load_config


def main():
    config = load_config(ROOT / "constraint_traverse/configs/benchmark.yaml")
    logging.basicConfig(level=logging.INFO)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, **config["simulation"]})
    try:
        from constraint_traverse.benchmark.runtime import PassageRuntime
        from constraint_traverse.navigation import SamplingNavigationController
        from vln_nav.app.navigation import _GoalMapPanel, LOW_LEVEL_MODES
        from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig, PlanningGrid
        import omni.ui as ui

        output = ROOT / "constraint_traverse/data/navigation_validation"
        runtime = PassageRuntime(app, ROOT, config, output, True, [1.6, 0])
        runtime._initialize()
        runtime._start("MPPI", "goal")
        runtime._finish("VALIDATION_SETUP")
        size, resolution = 120, 0.05
        edges = -3 + np.arange(size + 1) * resolution
        rows, columns = np.indices((size, size))
        cells = shapely.box(edges[columns], edges[rows], edges[columns + 1], edges[rows + 1])
        occupied = shapely.intersects(cells, runtime.geometry.obstacle_geometry)
        grid = PlanningGrid(occupied, ~occupied, np.zeros_like(occupied), np.zeros_like(occupied, dtype=np.float32), -3, -3, resolution, 1)
        footprint = FootprintConfig(tuple(FootprintPartConfig(f"part_{i}", *part) for i, part in enumerate(runtime.geometry.parts)), 0.03)
        results = {}
        for method in ("MPPI", "KMPPI"):
            runtime._reset()
            controller = SamplingNavigationController(method=method, hardware=runtime.profile.agx_controller.continuous,
                                                       footprint=footprint, physics_dt=runtime.dt,
                                                       config_path=ROOT / "constraint_traverse/configs/benchmark.yaml", logger=runtime.logger)
            initial = controller.read_state(runtime.robot, runtime.drive)
            latencies = []
            previous = None
            for _ in range(120):
                controller.step(robot=runtime.robot, drive=runtime.drive, grid=grid, goal_world_xy=[1.6, 0], now=runtime.world.current_time)
                if controller.last_plan is not None and controller.last_plan is not previous:
                    previous = controller.last_plan
                    latencies.append(previous.planning_sec)
                runtime.world.step(render=False)
            final = controller.read_state(runtime.robot, runtime.drive)
            results[method] = {"status": controller.status, "forward_distance_m": float(final[0] - initial[0]),
                               "planning_cycles": len(latencies), "planning_mean_sec": float(np.mean(latencies)) if latencies else None}
        events = []
        panel = _GoalMapPanel(ui, image_size_px=128, command_sink=events.append, bind_keyboard=False)
        for index, mode in enumerate(LOW_LEVEL_MODES):
            panel._mode_combo.model.get_item_value_model().set_value(index)
        results["mode_callbacks"] = events.copy()
        before = len(events)
        panel.update_controls("MPPI", "TEST")
        results["programmatic_mode_update_emits_no_command"] = len(events) == before
        panel.shutdown()
        results["passed"] = (all(results[method]["forward_distance_m"] > 0.1 for method in ("MPPI", "KMPPI"))
                             and ("SET_MODE", "MPPI") in events and ("SET_MODE", "KMPPI") in events
                             and results["programmatic_mode_update_emits_no_command"])
        path = output / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_report.json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"passed": results["passed"], "report": str(path)}, ensure_ascii=False))
        runtime.subscription = None
        return 0 if results["passed"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
