from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import carb
import omni.ext
import omni.ui as ui


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXTENSION_ROOT.parents[1]
DEFAULT_TASKS_PATH = PROJECT_ROOT / "configs" / "instructions.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_ROBOT_USD_PATH = (
    PROJECT_ROOT
    / "src"
    / "ros2_car_model_ws"
    / "src"
    / "1557"
    / "urdf"
    / "1557_all"
    / "1557_all.usd"
)
DEFAULT_ROBOT_PRIM_PATH = "/World/Robot"
DEFAULT_SCENE_PRIM_PATH = "/World/Environment"


class TaskEditorExtension(omni.ext.IExt):
    """Isaac Sim UI extension for authoring VLN task JSON entries."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window = None
        self._status_label = None
        self._tasks_data = _load_tasks_data(DEFAULT_TASKS_PATH)
        self._build_ui()
        self._set_status("Ready. Load scene/robot, move /World/Robot, then save task.")
        carb.log_info("[VLM-NAV Task Editor] panel created")
        if _env_bool("VLN_TASK_EDITOR_AUTO_LOAD", default=False):
            self._on_load_scene_robot()

    def on_shutdown(self) -> None:
        self._window = None
        self._status_label = None

    def _build_ui(self) -> None:
        requested_task_id = os.getenv("VLN_TASK_EDITOR_TASK_ID", "").strip()
        selected_task = _task_by_id(self._tasks_data, requested_task_id) if requested_task_id else None
        default_task_id = str(selected_task.get("id", _next_task_id(self._tasks_data))) if selected_task else _next_task_id(self._tasks_data)
        default_scene = (
            str(selected_task.get("scene_usd_path", ""))
            if selected_task
            else os.getenv("VLN_TASK_EDITOR_SCENE_USD", "usd/environment/hospital_flat.usd")
        )
        default_instruction = str(selected_task.get("instruction", "")) if selected_task else ""
        default_robot = (
            os.getenv("VLN_TASK_EDITOR_ROBOT_USD", "").strip()
            or str(DEFAULT_ROBOT_USD_PATH.relative_to(PROJECT_ROOT))
        )

        self._task_id_model = ui.SimpleStringModel(default_task_id)
        self._scene_model = ui.SimpleStringModel(default_scene)
        self._robot_model = ui.SimpleStringModel(default_robot)
        self._instruction_model = ui.SimpleStringModel(default_instruction)
        self._force_z_zero_model = ui.SimpleBoolModel(True)
        self._status_model = ui.SimpleStringModel("")

        self._window = ui.Window("VLM-NAV Task Editor", width=560, height=390, visible=True)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Author configs/instructions.json from the current /World/Robot pose.", height=24)
                ui.Label(f"Project: {PROJECT_ROOT}", height=20)
                with ui.HStack(height=28):
                    ui.Label("Task ID", width=120)
                    ui.StringField(model=self._task_id_model)
                with ui.HStack(height=28):
                    ui.Label("Scene USD", width=120)
                    ui.StringField(model=self._scene_model)
                with ui.HStack(height=28):
                    ui.Label("Robot USD", width=120)
                    ui.StringField(model=self._robot_model)
                ui.Label("Instruction", height=22)
                ui.StringField(model=self._instruction_model, height=58)
                with ui.HStack(height=28):
                    ui.Label("Force saved Z to 0.0", width=170)
                    ui.CheckBox(model=self._force_z_zero_model)
                with ui.HStack(height=34, spacing=8):
                    ui.Button("Load Scene + Robot", clicked_fn=self._on_load_scene_robot, height=30)
                    ui.Button("Use Current Pose", clicked_fn=self._on_refresh_pose, height=30)
                    ui.Button("Save / Update Task", clicked_fn=self._on_save_task, height=30)
                ui.Label("Saved pose uses /World/Robot root. Z is forced to 0.0 unless unchecked.", height=22)
                self._status_label = ui.Label("", height=72, word_wrap=True)

    def _on_load_scene_robot(self) -> None:
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading

            stage = get_current_stage()
            scene_path = _resolve_project_path(self._scene_model.as_string)
            robot_path = _resolve_project_path(self._robot_model.as_string)
            carb.log_info(f"[VLM-NAV Task Editor] loading scene={scene_path} robot={robot_path}")
            if not scene_path.exists():
                self._set_status(f"ERROR: Scene USD not found: {scene_path}")
                carb.log_error(f"[VLM-NAV Task Editor] scene USD not found: {scene_path}")
                return
            if not robot_path.exists():
                self._set_status(f"ERROR: Robot USD not found: {robot_path}")
                carb.log_error(f"[VLM-NAV Task Editor] robot USD not found: {robot_path}")
                return

            add_reference_to_stage(str(scene_path), DEFAULT_SCENE_PRIM_PATH)
            add_reference_to_stage(str(robot_path), DEFAULT_ROBOT_PRIM_PATH)

            task = _task_by_id(self._tasks_data, self._task_id_model.as_string)
            start_position = _task_list(task, "robot_start_position", [1.5, 4.0, 0.0])
            start_orientation = _task_list(task, "robot_start_orientation_deg", [0.0, 0.0, 0.0])
            _set_local_xform(
                stage=stage,
                prim_path=DEFAULT_ROBOT_PRIM_PATH,
                translation=start_position,
                rotation_deg=start_orientation,
            )
            _focus_viewport()
            self._set_status(
                f"Loaded scene and robot. Move {DEFAULT_ROBOT_PRIM_PATH} manually, then Save / Update Task."
            )
            carb.log_info("[VLM-NAV Task Editor] scene and robot loaded")
            _ = is_stage_loading
        except Exception as exc:
            self._set_status(f"ERROR loading scene/robot: {exc}")
            carb.log_error(f"[VLM-NAV Task Editor] failed to load scene/robot: {exc}")

    def _on_refresh_pose(self) -> None:
        try:
            from isaacsim.core.utils.stage import get_current_stage

            position, rotation = _get_robot_root_pose(stage=get_current_stage(), prim_path=DEFAULT_ROBOT_PRIM_PATH)
            if bool(self._force_z_zero_model.as_bool):
                position[2] = 0.0
            self._set_status(f"Current pose: position={_round_list(position)} rpy_deg={_round_list(rotation)}")
        except Exception as exc:
            self._set_status(f"ERROR reading pose: {exc}")

    def _on_save_task(self) -> None:
        try:
            from isaacsim.core.utils.stage import get_current_stage

            task_id = str(self._task_id_model.as_string).strip()
            scene_path = str(self._scene_model.as_string).strip()
            instruction = str(self._instruction_model.as_string).strip()
            if not task_id:
                self._set_status("ERROR: Task ID is empty.")
                return
            if not scene_path:
                self._set_status("ERROR: Scene USD is empty.")
                return
            if not instruction:
                self._set_status("ERROR: Instruction is empty.")
                return

            position, rotation = _get_robot_root_pose(stage=get_current_stage(), prim_path=DEFAULT_ROBOT_PRIM_PATH)
            if bool(self._force_z_zero_model.as_bool):
                position[2] = 0.0

            task = {
                "id": task_id,
                "scene_usd_path": scene_path,
                "robot_start_position": _round_list(position),
                "robot_start_orientation_deg": _round_list(rotation),
                "instruction": instruction,
            }
            _upsert_task(tasks_path=DEFAULT_TASKS_PATH, tasks_data=self._tasks_data, task=task)
            self._set_status(f"Saved {task_id} to {DEFAULT_TASKS_PATH}: pos={task['robot_start_position']} rpy={task['robot_start_orientation_deg']}")
            carb.log_info(f"[VLM-NAV Task Editor] saved task={task_id}")
        except Exception as exc:
            self._set_status(f"ERROR saving task: {exc}")
            carb.log_error(f"[VLM-NAV Task Editor] failed to save task: {exc}")

    def _set_status(self, text: str) -> None:
        self._status_model.set_value(str(text))
        if self._status_label is not None:
            self._status_label.text = str(text)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_tasks_data(tasks_path: Path) -> dict:
    if not tasks_path.exists():
        return {"schema_version": "1.0", "tasks": []}
    with tasks_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"schema_version": "1.0", "tasks": []}
    tasks = data.setdefault("tasks", [])
    if not isinstance(tasks, list):
        data["tasks"] = []
    data.setdefault("schema_version", "1.0")
    return data


def _first_task(tasks_data: dict) -> dict | None:
    for task in tasks_data.get("tasks", []):
        if isinstance(task, dict) and bool(task.get("enabled", True)):
            return task
    return None


def _task_by_id(tasks_data: dict, task_id: str) -> dict | None:
    wanted = str(task_id).strip()
    if not wanted:
        return None
    for task in tasks_data.get("tasks", []):
        if isinstance(task, dict) and str(task.get("id", "")).strip() == wanted:
            return task
    return None


def _task_list(task: dict | None, key: str, default: list[float]) -> list[float]:
    if task is None:
        return list(default)
    value = task.get(key, default)
    if not isinstance(value, list) or len(value) < len(default):
        return list(default)
    return [float(value[index]) for index in range(len(default))]


def _upsert_task(*, tasks_path: Path, tasks_data: dict, task: dict) -> None:
    tasks = tasks_data.setdefault("tasks", [])
    task_id = str(task["id"])
    for index, existing in enumerate(tasks):
        if isinstance(existing, dict) and str(existing.get("id", "")) == task_id:
            tasks[index] = task
            break
    else:
        tasks.append(task)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{tasks_path.name}.", suffix=".tmp", dir=str(tasks_path.parent))
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(tasks_data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, tasks_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _next_task_id(tasks_data: dict) -> str:
    existing = {
        str(task.get("id", "")).strip()
        for task in tasks_data.get("tasks", [])
        if isinstance(task, dict)
    }
    index = len(existing) + 1
    while True:
        candidate = f"hospital_{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def _set_local_xform(*, stage, prim_path: str, translation: list[float], rotation_deg: list[float]) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()
    translate_op.Set(tuple(float(value) for value in translation[:3]))
    rotate_op.Set(tuple(float(value) for value in rotation_deg[:3]))


def _get_robot_root_pose(*, stage, prim_path: str) -> tuple[list[float], list[float]]:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Robot prim not found: {prim_path}. Click Load Scene + Robot first.")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    yaw_deg = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    return [float(translation[0]), float(translation[1]), float(translation[2])], [0.0, 0.0, float(yaw_deg)]


def _focus_viewport() -> None:
    try:
        from pxr import Sdf
        import omni.kit.viewport.utility as viewport_utils

        viewport = viewport_utils.get_active_viewport()
        if viewport is not None:
            viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
    except Exception:
        return


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
