from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ..runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, setup_logging
from .nav import _build_simulation_app_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Isaac Sim with the CAPTAINav Task Editor extension enabled"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument("--scene-usd", default=os.getenv("TASK_EDITOR_SCENE_USD", "usd/environment/hospital_flat.usd"), help="Scene USD to load for a new task")
    parser.add_argument("--robot-usd", default=os.getenv("TASK_EDITOR_ROBOT_USD", ""), help="Robot USD to load for a new task")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config.logging.level)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ["VLN_TASK_EDITOR_TASK_ID"] = ""
    os.environ["VLN_TASK_EDITOR_SCENE_USD"] = str(args.scene_usd).strip()
    os.environ["VLN_TASK_EDITOR_ROBOT_USD"] = (
        str(args.robot_usd).strip() or str(config.robot.asset.usd_path)
    )
    os.environ["VLN_TASK_EDITOR_AUTO_LOAD"] = os.getenv("TASK_EDITOR_AUTO_LOAD", "1")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(_build_simulation_app_settings(config=config, logger=logger))
    try:
        _enable_task_editor_extension(logger=logger)
        logger.info(
            "[TaskEditorLauncher] Isaac Sim running with vln.nav.task_editor enabled. mode=%s scene=%s auto_load=%s",
            "new-task",
            str(args.scene_usd).strip(),
            os.environ["VLN_TASK_EDITOR_AUTO_LOAD"],
        )
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.001)
    finally:
        simulation_app.close()
    return 0


def _enable_task_editor_extension(*, logger) -> None:
    import omni.kit.app

    ext_root = PROJECT_ROOT / "exts"
    manager = omni.kit.app.get_app().get_extension_manager()
    if hasattr(manager, "add_path"):
        manager.add_path(str(ext_root))
    else:
        logger.warning("[TaskEditorLauncher] extension manager has no add_path(); extension path may need manual registration: %s", ext_root)

    extension_id = "vln.nav.task_editor"
    manager.set_extension_enabled_immediate(extension_id, True)
    logger.info("[TaskEditorLauncher] enabled extension=%s ext_path=%s", extension_id, ext_root)


if __name__ == "__main__":
    raise SystemExit(main())
