from __future__ import annotations

from dataclasses import dataclass
import math

from vln_nav.control.trajectory import VelocityCommand


@dataclass(frozen=True)
class TeleopConfig:
    max_linear_mps: float
    max_angular_radps: float
    linear_accel_mps2: float
    angular_accel_radps2: float
    linear_decel_mps2: float
    angular_decel_radps2: float

    @classmethod
    def from_mapping(cls, data: dict) -> "TeleopConfig":
        expected = {
            "max_linear_mps",
            "max_angular_radps",
            "linear_accel_mps2",
            "angular_accel_radps2",
            "linear_decel_mps2",
            "angular_decel_radps2",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"teleop requires exactly: {sorted(expected)}")
        config = cls(**{name: float(data[name]) for name in expected})
        if not all(math.isfinite(value) and value > 0 for value in config.__dict__.values()):
            raise ValueError("teleop values must be finite positive numbers")
        return config


class ContinuousTeleop:
    """方向键或 WASD 生成平滑的车体线速度与角速度。"""

    _FORWARD_KEYS = {"W", "UP"}
    _BACKWARD_KEYS = {"S", "DOWN"}
    _LEFT_KEYS = {"A", "LEFT"}
    _RIGHT_KEYS = {"D", "RIGHT"}
    _KEYS = _FORWARD_KEYS | _BACKWARD_KEYS | _LEFT_KEYS | _RIGHT_KEYS

    def __init__(self, config: TeleopConfig) -> None:
        self.config = config
        self._pressed_keys: set[str] = set()
        self._linear = 0.0
        self._angular = 0.0
        self._subscription = None
        self._input = None
        self._keyboard = None

    def bind_keyboard(self) -> None:
        import carb.input
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac Sim app window is unavailable")
        keyboard = app_window.get_keyboard()
        self._keyboard = keyboard
        self._input = carb.input.acquire_input_interface()
        self._carb_input = carb.input
        self._subscription = self._input.subscribe_to_keyboard_events(
            keyboard, self._on_keyboard_event
        )

    def shutdown(self) -> None:
        if (
            self._subscription is not None
            and self._input is not None
            and self._keyboard is not None
        ):
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._subscription,
            )
        self._subscription = None
        self._input = None
        self._keyboard = None

    def reset(self) -> None:
        self._pressed_keys.clear()
        self._linear = 0.0
        self._angular = 0.0

    @property
    def command(self) -> VelocityCommand:
        return VelocityCommand(float(self._linear), float(self._angular))

    def step(self, dt: float) -> VelocityCommand:
        dt = max(float(dt), 0.0)
        forward = int(bool(self._pressed_keys & self._FORWARD_KEYS)) - int(
            bool(self._pressed_keys & self._BACKWARD_KEYS)
        )
        turn = int(bool(self._pressed_keys & self._LEFT_KEYS)) - int(
            bool(self._pressed_keys & self._RIGHT_KEYS)
        )
        target_linear = float(forward) * self.config.max_linear_mps
        target_angular = float(turn) * self.config.max_angular_radps
        self._linear = _approach(
            self._linear,
            target_linear,
            (self.config.linear_accel_mps2 if forward else self.config.linear_decel_mps2) * dt,
        )
        self._angular = _approach(
            self._angular,
            target_angular,
            (self.config.angular_accel_radps2 if turn else self.config.angular_decel_radps2) * dt,
        )
        return self.command

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        key_input = event.input
        key_name = str(getattr(key_input, "name", key_input))
        if key_name not in self._KEYS:
            return True
        if event.type in {
            self._carb_input.KeyboardEventType.KEY_PRESS,
            self._carb_input.KeyboardEventType.KEY_REPEAT,
        }:
            self._pressed_keys.add(key_name)
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key_name)
        return True


class TeleopClock:
    def __init__(self, period_sec: float, timestamp_sec: float) -> None:
        if not math.isfinite(period_sec) or period_sec <= 0:
            raise ValueError("控制周期必须为有限正数")
        self.period_sec = float(period_sec)
        self.reset(timestamp_sec)

    def reset(self, timestamp_sec: float) -> None:
        self._last_update = float(timestamp_sec)

    def step(
        self, teleop: ContinuousTeleop, timestamp_sec: float, *, stopped: bool = False,
    ) -> VelocityCommand:
        now = float(timestamp_sec)
        if stopped:
            teleop.reset()
            self.reset(now)
            return VelocityCommand()
        elapsed = now - self._last_update
        if elapsed < 0:
            self.reset(now)
            return teleop.command
        if elapsed + 1.0e-9 >= self.period_sec:
            self._last_update = now
            return teleop.step(elapsed)
        return teleop.command


def _approach(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return target
