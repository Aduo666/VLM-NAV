from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys

import yaml

from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.runtime.isaac_runtime_utils import set_local_xform, wait_for_stage_assets
from vln_nav.runtime.scene_lighting import add_indoor_lighting


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "configs/navigation.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "configs/episodes.json"


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("episode_recorder")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _resolve_project_path(path_text: str) -> Path:
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _json_scene_path(scene_path: Path) -> str:
    """Store project-local scenes as relative paths, external scenes as absolute paths."""
    scene_path = scene_path.resolve()
    try:
        return scene_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return scene_path.as_posix()


def _load_navigation_settings() -> dict:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid navigation config: {SETTINGS_PATH}")

    simulation = data.get("simulation")
    scene = data.get("scene")
    if not isinstance(simulation, dict):
        raise ValueError("navigation.yaml 缺少 simulation 配置")
    if not isinstance(scene, dict):
        raise ValueError("navigation.yaml 缺少 scene 配置")

    required_scene_fields = {
        "prim_path",
        "add_default_ground_plane",
        "stage_load_timeout_sec",
        "add_indoor_lighting",
    }
    missing = required_scene_fields - set(scene)
    if missing:
        raise ValueError(
            f"navigation.yaml scene 缺少字段：{sorted(missing)}"
        )

    return data


def _load_episode_database(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": 1,
            "episodes": [],
        }

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or set(data) != {"schema_version", "episodes"}:
        raise ValueError(
            "episodes.json 顶层必须且只能包含 schema_version 和 episodes"
        )

    if data["schema_version"] != 1:
        raise ValueError(
            f"不支持的 schema_version={data['schema_version']!r}，当前只支持 1"
        )

    if not isinstance(data["episodes"], list):
        raise ValueError("episodes.json 的 episodes 必须是数组")

    return data


def _atomic_write_episode_database(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _world_pose_from_stage(
    stage,
    robot_root_path: str,
) -> tuple[float, float, float]:
    """Read robot-root world X/Y/yaw directly from USD, including manual gizmo edits."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(robot_root_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"机器人根 Prim 不存在：{robot_root_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = xform_cache.GetLocalToWorldTransform(prim)
    translation = world_matrix.ExtractTranslation()

    transform = Gf.Transform(world_matrix)
    quat = transform.GetRotation().GetQuat()

    w = float(quat.GetReal())
    imag = quat.GetImaginary()
    x = float(imag[0])
    y = float(imag[1])
    z = float(imag[2])

    yaw_rad = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    yaw_deg = math.degrees(yaw_rad)
    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

    return (
        float(translation[0]),
        float(translation[1]),
        float(yaw_deg),
    )


class EpisodeRecorderPanel:
    def __init__(
        self,
        ui,
        *,
        stage,
        robot_root_path: str,
        scene_json_path: str,
        output_path: Path,
        initial_episode_id: str,
        initial_name: str,
        initial_instruction: str,
        logger: logging.Logger,
    ) -> None:
        self._ui = ui
        self._stage = stage
        self._robot_root_path = str(robot_root_path)
        self._scene_json_path = str(scene_json_path)
        self._output_path = Path(output_path)
        self._logger = logger

        self._episode_id_model = ui.SimpleStringModel(str(initial_episode_id))
        self._name_model = ui.SimpleStringModel(str(initial_name))
        self._instruction_model = ui.SimpleStringModel(str(initial_instruction))

        self.window = ui.Window(
            "Navigation Episode Recorder",
            width=620,
            height=430,
            visible=True,
        )

        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Scene")
                ui.Label(self._scene_json_path)

                ui.Label("Output JSON")
                ui.Label(str(self._output_path))

                ui.Separator()

                ui.Label("Episode ID")
                ui.StringField(model=self._episode_id_model)

                ui.Label("Name")
                ui.StringField(model=self._name_model)

                ui.Label("Instruction")
                ui.StringField(
                    model=self._instruction_model,
                    height=60,
                )

                ui.Separator()

                self._pose_label = ui.Label("Pose: waiting...")
                self._status_label = ui.Label(
                    "拖拽机器人后点击 Record Current Pose。"
                )

                with ui.HStack(height=36, spacing=8):
                    ui.Button(
                        "Select Robot",
                        clicked_fn=self._select_robot,
                    )
                    ui.Button(
                        "Record Current Pose",
                        clicked_fn=self._record_current_pose,
                    )
                    ui.Button(
                        "Refresh Pose",
                        clicked_fn=self.refresh_pose,
                    )

        self._select_robot()
        self.refresh_pose()

    def refresh_pose(self) -> None:
        try:
            x_m, y_m, yaw_deg = _world_pose_from_stage(
                self._stage,
                self._robot_root_path,
            )
            self._pose_label.text = (
                f"Pose: x={x_m:.3f} m, "
                f"y={y_m:.3f} m, "
                f"yaw={yaw_deg:.2f} deg"
            )
        except Exception as exc:
            self._pose_label.text = f"Pose unavailable: {exc}"

    def shutdown(self) -> None:
        self.window.visible = False

    def _select_robot(self) -> None:
        try:
            import omni.usd

            context = omni.usd.get_context()
            selection = context.get_selection()
            selection.set_selected_prim_paths(
                [self._robot_root_path],
                True,
            )
            self._status_label.text = (
                f"已选择 {self._robot_root_path}，"
                "可使用 Transform gizmo 拖拽/旋转机器人。"
            )
        except Exception as exc:
            self._status_label.text = f"选择机器人失败：{exc}"
            self._logger.exception(
                "[EpisodeRecorder] failed to select robot"
            )

    def _record_current_pose(self) -> None:
        try:
            episode_id = (
                self._episode_id_model.get_value_as_string().strip()
            )
            name = self._name_model.get_value_as_string().strip()
            instruction = (
                self._instruction_model.get_value_as_string().strip()
            )

            if not episode_id:
                raise ValueError("Episode ID 不能为空")
            if not name:
                raise ValueError("Name 不能为空")
            if not instruction:
                raise ValueError("Instruction 不能为空")

            x_m, y_m, yaw_deg = _world_pose_from_stage(
                self._stage,
                self._robot_root_path,
            )

            database = _load_episode_database(self._output_path)

            duplicate = next(
                (
                    episode
                    for episode in database["episodes"]
                    if isinstance(episode, dict)
                    and str(episode.get("id", "")).strip() == episode_id
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(
                    f"Episode ID 已存在：{episode_id}"
                )

            episode = {
                "id": episode_id,
                "name": name,
                "usd_path": self._scene_json_path,
                "start_pose": {
                    "x_m": round(x_m, 4),
                    "y_m": round(y_m, 4),
                    "yaw_deg": round(yaw_deg, 2),
                },
                "instruction": instruction,
            }

            database["episodes"].append(episode)
            _atomic_write_episode_database(
                self._output_path,
                database,
            )

            self._status_label.text = (
                f"已追加 {episode_id}: "
                f"x={x_m:.3f}, "
                f"y={y_m:.3f}, "
                f"yaw={yaw_deg:.2f}°"
            )

            self._logger.info(
                "[EpisodeRecorder] saved "
                "id=%s scene=%s "
                "pose=[%.3f,%.3f,%.2f] output=%s",
                episode_id,
                self._scene_json_path,
                x_m,
                y_m,
                yaw_deg,
                self._output_path,
            )

        except Exception as exc:
            self._status_label.text = f"记录失败：{exc}"
            self._logger.exception(
                "[EpisodeRecorder] record failed"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "加载指定场景和 AGX，"
            "通过 Isaac Sim UI 手动摆放机器人并记录 VLN episode。"
        )
    )

    parser.add_argument(
        "--scene",
        required=True,
        help=(
            "要记录的场景 USD/USDA 路径。"
            "相对路径按工程根目录解析。"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Episode JSON 输出路径；"
            "默认 configs/episodes.json。"
        ),
    )

    parser.add_argument(
        "--episode-id",
        default="",
        help="预填 Episode ID。",
    )
    parser.add_argument(
        "--name",
        default="",
        help="预填 Episode 名称。",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="预填导航指令。",
    )

    parser.add_argument(
        "--initial-x",
        type=float,
        default=0.0,
        help="机器人加载后的初始 X，仅用于方便拖拽，默认 0。",
    )
    parser.add_argument(
        "--initial-y",
        type=float,
        default=0.0,
        help="机器人加载后的初始 Y，仅用于方便拖拽，默认 0。",
    )
    parser.add_argument(
        "--initial-yaw",
        type=float,
        default=0.0,
        help="机器人加载后的初始 yaw，仅用于方便拖拽，默认 0 度。",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    settings = _load_navigation_settings()
    profile = load_agx_profile(project_root=PROJECT_ROOT)
    logger = _make_logger()

    scene_settings = settings["scene"]

    scene_path = _resolve_project_path(args.scene)
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Scene USD 不存在：{scene_path}"
        )

    output_path = _resolve_project_path(args.output)
    scene_json_path = _json_scene_path(scene_path)

    from isaacsim import SimulationApp

    simulation_settings = settings["simulation"]
    app = SimulationApp(
        {
            "headless": False,
            "active_gpu": simulation_settings.get("active_gpu"),
            "physics_gpu": simulation_settings.get("physics_gpu"),
            "multi_gpu": bool(
                simulation_settings.get("multi_gpu", False)
            ),
        }
    )

    panel = None

    try:
        import omni.ui as ui
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
            is_stage_loading,
        )

        # World 只用于保持与主导航一致的 stage/unit 配置。
        # Recorder 不启动导航，也不需要播放物理仿真。
        world = World(
            physics_dt=float(profile.simulation.physics_dt),
            rendering_dt=float(profile.simulation.rendering_dt),
            stage_units_in_meters=1.0,
        )

        stage = get_current_stage()

        add_reference_to_stage(
            str(scene_path),
            str(scene_settings["prim_path"]),
        )

        if bool(scene_settings["add_default_ground_plane"]):
            world.scene.add_default_ground_plane()

        add_reference_to_stage(
            str(profile.asset.usd_path),
            str(profile.asset.prim_path),
        )

        wait_for_stage_assets(
            simulation_app=app,
            is_stage_loading=is_stage_loading,
            timeout_sec=float(
                scene_settings["stage_load_timeout_sec"]
            ),
            logger=logger,
            label="episode recorder scene",
        )

        if bool(scene_settings["add_indoor_lighting"]):
            try:
                add_indoor_lighting(
                    stage=stage,
                    environment_prim_path=str(
                        scene_settings["prim_path"]
                    ),
                    logger=logger,
                )
            except Exception:
                logger.exception(
                    "[EpisodeRecorder] "
                    "failed to add indoor lighting"
                )

        # 最新 navigation.yaml 已不再保存 start pose。
        # Recorder 只给机器人一个方便手动摆放的初始位姿，
        # 真正写入 episodes.json 的位姿来自点击按钮时的当前 USD world pose。
        set_local_xform(
            stage=stage,
            prim_path=profile.asset.prim_path,
            translation=[
                float(args.initial_x),
                float(args.initial_y),
                float(profile.spawn_z_m),
            ],
            rotation_deg=[
                0.0,
                0.0,
                float(args.initial_yaw),
            ],
        )

        panel = EpisodeRecorderPanel(
            ui,
            stage=stage,
            robot_root_path=profile.asset.prim_path,
            scene_json_path=scene_json_path,
            output_path=output_path,
            initial_episode_id=args.episode_id,
            initial_name=args.name,
            initial_instruction=args.instruction,
            logger=logger,
        )

        logger.info(
            "[EpisodeRecorder] scene=%s",
            scene_path,
        )
        logger.info(
            "[EpisodeRecorder] robot=%s",
            profile.asset.prim_path,
        )
        logger.info(
            "[EpisodeRecorder] output=%s",
            output_path,
        )
        logger.info(
            "[EpisodeRecorder] "
            "保持仿真暂停；手动拖拽机器人后点击 "
            "Record Current Pose。"
        )

        # 先提交一次 UI。
        app.update()

        while app.is_running():
            panel.refresh_pose()
            app.update()

    except BaseException:
        import traceback
        traceback.print_exc()
        raise

    finally:
        if panel is not None:
            panel.shutdown()
        app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())