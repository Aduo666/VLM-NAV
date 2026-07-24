from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np

from vln_nav.app.nav import (
    _bind_viewport_to_camera,
    _get_joint_names,
    _get_world_pose,
    _resolve_robot_relative_prim_path,
    _set_local_xform,
    _wait_for_stage_assets,
)
from vln_nav.control.g1_policy_controller import G1PolicyController
from vln_nav.control.locomotion import create_locomotion_controller
from vln_nav.runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from vln_nav.runtime.robot_profile import RobotProfile, load_robot_profile


_PROFILE_NAME = "unitree_g1_29dof"
_AUTO_RESET_DELAY_SEC = 0.5
_MAX_AUTO_RESETS = 3


@dataclass
class _MotionTelemetry:
    actual_linear_x_mps: float = 0.0
    actual_angular_z_radps: float = 0.0
    pelvis_z_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    policy_max_action: float = 0.0
    joint_tracking_error_rad: float = 0.0


class _G1ControlPanel:
    _KEY_ACTIONS = {
        "UP": "FORWARD",
        "DOWN": "BACKWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }

    def __init__(
        self,
        *,
        default_forward_mps: float,
        default_backward_mps: float,
        vln_turn_radps: float,
        command_limits: dict[str, tuple[float, float]],
    ) -> None:
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self._carb_input = carb.input
        self._ui = ui
        self._held_actions: dict[str, str] = {}
        self._press_order: list[str] = []
        self._reset_requested = False
        self._status = "INITIALIZING"
        self._applied_action = "STOP"
        self._applied_linear_x_mps = 0.0
        self._applied_angular_z_radps = 0.0

        max_forward = max(0.01, float(command_limits["linear_x_mps"][1]))
        max_backward = max(0.01, abs(float(command_limits["linear_x_mps"][0])))
        turn_limit = min(
            abs(float(command_limits["angular_z_radps"][0])),
            abs(float(command_limits["angular_z_radps"][1])),
        )
        self._vln_turn_radps = float(np.clip(vln_turn_radps, 0.0, turn_limit))
        self._forward_model = ui.SimpleFloatModel(
            float(np.clip(default_forward_mps, 0.0, max_forward))
        )
        self._backward_model = ui.SimpleFloatModel(
            float(np.clip(default_backward_mps, 0.0, max_backward))
        )
        self._window = ui.Window(
            "G1 Locomotion Policy Test",
            width=390,
            height=610,
            visible=True,
            dockPreference=ui.DockPreference.LEFT,
        )
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Arrow keys or buttons: latest pressed direction wins")
                ui.Label("Release all directions to command STOP")
                ui.Separator()
                with ui.HStack(height=42):
                    ui.Spacer()
                    self._motion_button("UP", "FORWARD")
                    ui.Spacer()
                with ui.HStack(height=42):
                    self._motion_button("LEFT", "TURN_LEFT")
                    ui.Spacer()
                    self._motion_button("RIGHT", "TURN_RIGHT")
                with ui.HStack(height=42):
                    ui.Spacer()
                    self._motion_button("DOWN", "BACKWARD")
                    ui.Spacer()
                ui.Button("STOP", height=34, clicked_fn=self.clear_motion)
                ui.Button("RESET SIMULATION", height=34, clicked_fn=self.request_reset)
                ui.Separator()
                self._slider_row(
                    "Forward m/s",
                    self._forward_model,
                    minimum=0.0,
                    maximum=max_forward,
                )
                self._slider_row(
                    "Backward m/s",
                    self._backward_model,
                    minimum=0.0,
                    maximum=max_backward,
                )
                ui.Label(
                    "VLN turn speed: "
                    f"{self._vln_turn_radps:.3f} rad/s "
                    "(robot profile turn_command)"
                )
                ui.Separator()
                self._status_label = ui.Label("Status: INITIALIZING")
                self._command_label = ui.Label("Command: STOP")
                self._actual_velocity_label = ui.Label("Actual velocity: x=0.000 m/s yaw=0.000 rad/s")
                self._pose_label = ui.Label("Pelvis: z=0.000 m roll=0.00 deg pitch=0.00 deg")
                self._policy_label = ui.Label("Policy: max_action=0.000 joint_error=0.000 rad")
                self._reset_label = ui.Label(
                    f"Automatic resets: 0/{_MAX_AUTO_RESETS}"
                )

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac Sim app window is unavailable")
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._keyboard_subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

    @property
    def action(self) -> str:
        if not self._press_order:
            return "STOP"
        return self._held_actions.get(self._press_order[-1], "STOP")

    @property
    def command(self) -> tuple[float, float]:
        action = self.action
        if action == "FORWARD":
            return float(self._forward_model.as_float), 0.0
        if action == "BACKWARD":
            return -float(self._backward_model.as_float), 0.0
        if action == "TURN_LEFT":
            return 0.0, self._vln_turn_radps
        if action == "TURN_RIGHT":
            return 0.0, -self._vln_turn_radps
        return 0.0, 0.0

    def consume_reset_request(self) -> bool:
        requested = bool(self._reset_requested)
        self._reset_requested = False
        return requested

    def request_reset(self) -> None:
        self._reset_requested = True

    def clear_motion(self) -> None:
        self._held_actions.clear()
        self._press_order.clear()

    def set_status(self, status: str) -> None:
        self._status = str(status)

    def set_applied_command(
        self,
        *,
        action: str,
        linear_x_mps: float,
        angular_z_radps: float,
    ) -> None:
        self._applied_action = str(action)
        self._applied_linear_x_mps = float(linear_x_mps)
        self._applied_angular_z_radps = float(angular_z_radps)

    def update(
        self,
        *,
        telemetry: _MotionTelemetry,
        auto_reset_count: int,
    ) -> None:
        self._status_label.text = f"Status: {self._status}"
        self._command_label.text = (
            f"Command: {self._applied_action} "
            f"(x={self._applied_linear_x_mps:.3f} m/s, "
            f"yaw={self._applied_angular_z_radps:.3f} rad/s)"
        )
        self._actual_velocity_label.text = (
            "Actual velocity: "
            f"x={telemetry.actual_linear_x_mps:.3f} m/s "
            f"yaw={telemetry.actual_angular_z_radps:.3f} rad/s"
        )
        self._pose_label.text = (
            "Pelvis: "
            f"z={telemetry.pelvis_z_m:.3f} m "
            f"roll={telemetry.roll_deg:.2f} deg "
            f"pitch={telemetry.pitch_deg:.2f} deg"
        )
        self._policy_label.text = (
            "Policy: "
            f"max_action={telemetry.policy_max_action:.3f} "
            f"joint_error={telemetry.joint_tracking_error_rad:.3f} rad"
        )
        self._reset_label.text = (
            f"Automatic resets: {int(auto_reset_count)}/{_MAX_AUTO_RESETS}"
        )

    def shutdown(self) -> None:
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_subscription,
            )
            self._keyboard_subscription = None
        self._window.visible = False
        self._window = None

    def _motion_button(self, label: str, action: str):
        return self._ui.Button(
            label,
            width=95,
            mouse_pressed_fn=lambda *_args: self._press(f"button:{action}", action),
            mouse_released_fn=lambda *_args: self._release(f"button:{action}"),
        )

    def _slider_row(self, label: str, model, *, minimum: float, maximum: float) -> None:
        with self._ui.HStack(height=30):
            self._ui.Label(label, width=115)
            self._ui.FloatSlider(
                model=model,
                min=float(minimum),
                max=float(maximum),
                step=0.01,
            )

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        key_name = str(event.input.name)
        action = self._KEY_ACTIONS.get(key_name)
        if action is None:
            return True
        source = f"key:{key_name}"
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            self._press(source, action)
        elif (
            event.type == self._carb_input.KeyboardEventType.KEY_REPEAT
            and source not in self._held_actions
        ):
            self._press(source, action)
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._release(source)
        return True

    def _press(self, source: str, action: str) -> None:
        self._held_actions[source] = action
        if source in self._press_order:
            self._press_order.remove(source)
        self._press_order.append(source)

    def _release(self, source: str) -> None:
        self._held_actions.pop(source, None)
        if source in self._press_order:
            self._press_order.remove(source)


def main() -> int:
    config = load_config(DEFAULT_CONFIG_PATH)
    profile = load_robot_profile(profile_name=_PROFILE_NAME, project_root=PROJECT_ROOT)
    logger = _create_logger(config.logging.level)
    sys.argv = [sys.argv[0]]

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": False,
            "width": int(config.simulation.width),
            "height": int(config.simulation.height),
            "active_gpu": config.simulation.active_gpu,
            "physics_gpu": config.simulation.physics_gpu,
            "multi_gpu": bool(config.simulation.multi_gpu),
        }
    )
    try:
        return _run(simulation_app=simulation_app, config=config, profile=profile, logger=logger)
    finally:
        simulation_app.close()


def _run(*, simulation_app, config, profile: RobotProfile, logger: logging.Logger) -> int:
    try:
        from isaacsim.core.prims import SingleArticulation as RobotArticulation
    except ImportError:
        from isaacsim.core.prims import Articulation as RobotArticulation
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from pxr import UsdLux

    print(f"[G1PolicyTest] Loading {profile.asset.usd_path}", flush=True)
    world = World(
        physics_dt=float(profile.simulation.physics_dt),
        rendering_dt=float(profile.simulation.rendering_dt),
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    light = UsdLux.DomeLight.Define(stage, "/World/G1PolicyTestLight")
    light.CreateIntensityAttr().Set(1500.0)

    add_reference_to_stage(str(profile.asset.usd_path), profile.asset.prim_path)
    _wait_for_stage_assets(
        simulation_app=simulation_app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(config.scene.stage_load_timeout_sec),
        logger=logger,
        label="G1 policy-test robot",
    )
    _set_local_xform(
        stage=stage,
        prim_path=profile.asset.prim_path,
        translation=[0.0, 0.0, float(profile.spawn_z_m)],
        rotation_deg=[0.0, 0.0, 0.0],
    )

    articulation_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.asset.articulation_path,
    )
    robot = world.scene.add(
        RobotArticulation(
            prim_path=articulation_path,
            name="g1_policy_test_robot",
            reset_xform_properties=False,
        )
    )
    world.reset()
    robot.initialize()
    controller = create_locomotion_controller(
        profile=profile,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=_get_joint_names(robot),
        logger=logger,
    )
    if not isinstance(controller, G1PolicyController):
        raise RuntimeError("G1 policy test created a non-G1 locomotion controller")

    observer_path = _resolve_robot_relative_prim_path(
        robot_root_path=profile.asset.prim_path,
        prim_path=profile.sensors.observer_camera_prim_path,
    )
    observer_prim = stage.GetPrimAtPath(observer_path)
    if not observer_prim or not observer_prim.IsValid() or observer_prim.GetTypeName() != "Camera":
        raise RuntimeError(f"G1 observer camera is missing or invalid: {observer_path}")
    _bind_viewport_to_camera(camera_prim_path=observer_path, logger=logger, enabled=True)

    command_limits = controller.command_limits
    panel = _G1ControlPanel(
        default_forward_mps=float(profile.primitives.forward_command),
        default_backward_mps=min(
            0.5 * float(profile.primitives.forward_command),
            abs(float(command_limits["linear_x_mps"][0])),
        ),
        vln_turn_radps=float(profile.primitives.turn_command),
        command_limits=command_limits,
    )
    print(
        "[G1PolicyTest] Ready: arrow keys control motion; release all keys to STOP.",
        flush=True,
    )

    physics_dt = float(profile.simulation.physics_dt)
    render_stride = max(
        1,
        int(round(float(profile.simulation.rendering_dt) / physics_dt)),
    )
    sim_time_sec = 0.0
    step_index = 0
    standup_until_sec = float(profile.g1_controller.standup_duration_sec)
    fall_started_at_sec: float | None = None
    reset_due_at_sec: float | None = None
    auto_reset_count = 0
    failure_latched = False

    try:
        while simulation_app.is_running():
            if panel.consume_reset_request():
                auto_reset_count = 0
                failure_latched = False
                fall_started_at_sec = None
                reset_due_at_sec = None
                _reset_world(world=world, robot=robot, controller=controller)
                panel.clear_motion()
                standup_until_sec = sim_time_sec + float(
                    profile.g1_controller.standup_duration_sec
                )
                print("[G1PolicyTest] Manual full simulation reset.", flush=True)

            if reset_due_at_sec is not None and sim_time_sec >= reset_due_at_sec:
                auto_reset_count += 1
                _reset_world(world=world, robot=robot, controller=controller)
                panel.clear_motion()
                fall_started_at_sec = None
                reset_due_at_sec = None
                standup_until_sec = sim_time_sec + float(
                    profile.g1_controller.standup_duration_sec
                )
                print(
                    f"[G1PolicyTest] Automatic full reset "
                    f"{auto_reset_count}/{_MAX_AUTO_RESETS}.",
                    flush=True,
                )

            standing_up = sim_time_sec < standup_until_sec
            if failure_latched:
                panel.set_status("FAILED: FALL LIMIT REACHED - PRESS RESET")
                linear_x_mps, angular_z_radps = 0.0, 0.0
            elif reset_due_at_sec is not None:
                panel.set_status("FALL CONFIRMED - RESETTING IN 0.5 SEC")
                linear_x_mps, angular_z_radps = 0.0, 0.0
            elif standing_up:
                panel.set_status("STANDING UP")
                linear_x_mps, angular_z_radps = 0.0, 0.0
            else:
                panel.set_status("READY")
                linear_x_mps, angular_z_radps = panel.command

            applied_action = (
                panel.action
                if abs(linear_x_mps) > 1.0e-6 or abs(angular_z_radps) > 1.0e-6
                else "STOP"
            )
            panel.set_applied_command(
                action=applied_action,
                linear_x_mps=linear_x_mps,
                angular_z_radps=angular_z_radps,
            )
            controller.set_velocity_command(
                linear_x_mps=linear_x_mps,
                angular_z_radps=angular_z_radps,
            )
            controller.advance_physics()
            world.step(render=False)
            sim_time_sec += physics_dt
            step_index += 1

            telemetry = _read_telemetry(robot=robot, controller=controller)
            if (
                bool(profile.safety.fall_detection_enabled)
                and not standing_up
                and reset_due_at_sec is None
                and not failure_latched
            ):
                fallen = (
                    telemetry.pelvis_z_m < float(profile.safety.min_base_height_m)
                    or abs(telemetry.roll_deg) > float(profile.safety.max_roll_pitch_deg)
                    or abs(telemetry.pitch_deg) > float(profile.safety.max_roll_pitch_deg)
                )
                if fallen:
                    if fall_started_at_sec is None:
                        fall_started_at_sec = sim_time_sec
                    elif (
                        sim_time_sec - fall_started_at_sec
                        >= float(profile.safety.confirmation_sec)
                    ):
                        panel.clear_motion()
                        if auto_reset_count < _MAX_AUTO_RESETS:
                            reset_due_at_sec = sim_time_sec + _AUTO_RESET_DELAY_SEC
                            print(
                                "[G1PolicyTest] Fall confirmed; full reset scheduled.",
                                flush=True,
                            )
                        else:
                            failure_latched = True
                            print(
                                "[G1PolicyTest] Fall limit reached; press RESET SIMULATION.",
                                flush=True,
                            )
                else:
                    fall_started_at_sec = None

            if step_index % render_stride == 0:
                panel.update(
                    telemetry=telemetry,
                    auto_reset_count=auto_reset_count,
                )
                world.render()
    finally:
        controller.set_velocity_command(linear_x_mps=0.0, angular_z_radps=0.0)
        panel.shutdown()
    return 0


def _reset_world(*, world, robot, controller: G1PolicyController) -> None:
    controller.set_velocity_command(linear_x_mps=0.0, angular_z_radps=0.0)
    world.reset()
    robot.initialize()
    controller.reset()


def _read_telemetry(*, robot, controller: G1PolicyController) -> _MotionTelemetry:
    position, orientation = _get_world_pose(robot)
    rotation_world_from_body = _rotation_matrix_wxyz(orientation)
    linear_world = np.asarray(robot.get_linear_velocity(), dtype=np.float64).reshape(3)
    angular_world = np.asarray(robot.get_angular_velocity(), dtype=np.float64).reshape(3)
    linear_body = rotation_world_from_body.T @ linear_world
    angular_body = rotation_world_from_body.T @ angular_world
    roll_deg, pitch_deg = _roll_pitch_deg(orientation)
    readback = controller.readback()
    policy_action = readback.get("policy_action", {})
    joint_tracking = readback.get("joint_tracking", {})
    return _MotionTelemetry(
        actual_linear_x_mps=float(linear_body[0]),
        actual_angular_z_radps=float(angular_body[2]),
        pelvis_z_m=float(position[2]),
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        policy_max_action=max(
            (abs(float(value)) for value in policy_action.values()),
            default=0.0,
        ),
        joint_tracking_error_rad=float(joint_tracking.get("max_error_rad", 0.0)),
    )


def _rotation_matrix_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if quaternion.size < 4:
        return np.eye(3, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion[:4]))
    if norm <= 1.0e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion[:4] / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _roll_pitch_deg(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    rotation = _rotation_matrix_wxyz(quaternion_wxyz)
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    return float(np.degrees(roll)), float(np.degrees(pitch))


def _create_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("g1_policy_test")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
