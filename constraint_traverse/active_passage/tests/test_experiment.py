from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from constraint_traverse.active_passage.__main__ import parse_args, run_application
from constraint_traverse.active_passage.contracts import CameraFrame, RealObservation, RobotState
from constraint_traverse.active_passage.experiment import Experiment
from constraint_traverse.active_passage.recording import inspect_record
from constraint_traverse.active_passage.state import RandomState, TrialSnapshot, robot_errors


class Backend:
    def __init__(self):
        self.timestamp = 2.0
        self.goal = np.asarray([1.0, 0.0, 0.0])
        self.stopped = 0
        self.fail_reset = False
        self.generator = np.random.default_rng(17)
        self.position = 0.0

    def robot(self):
        transform = np.eye(4)
        transform[0, 3] = self.position
        return RobotState(self.timestamp, transform, np.zeros(6), ("wheel",), [0.1], [0.2])

    def observation(self):
        robot = self.robot()
        depth = np.asarray([[np.nan, np.inf], [0.0, 1.0]], dtype=np.float32)
        frame = CameraFrame("front", self.timestamp, np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8),
                            depth, np.eye(3), np.eye(4))
        return RealObservation("frame", frame, robot)

    def capture_snapshot(self):
        return TrialSnapshot(self.robot(), {}, {}, {}, {}, {"grid": np.zeros((2, 2))},
            (self.observation(),), self.goal.copy(), RandomState.capture(self.generator), 0,
            self.timestamp, self.timestamp)

    def stop_motion(self):
        self.stopped += 1

    def restore_snapshot(self, snapshot):
        self.position = snapshot.robot.world_from_base[0, 3]
        self.timestamp = snapshot.robot.timestamp_sec
        self.goal = snapshot.goal_world_xyz.copy()
        snapshot.random.restore(self.generator)
        errors = robot_errors(snapshot.robot, self.robot())
        errors.update(rigid_body_error=0, scene_transform_error=0, controller_error=0, map_error=0)
        if self.fail_reset:
            errors["position_m"] = 1.0
        return errors


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="active-passage-unit-")
        self.addCleanup(self.directory.cleanup)
        self.backend = Backend()
        self.experiment = Experiment(self.backend, Path(self.directory.name), {"scene": "test"}, timeout_sec=1)
        self.addCleanup(self.experiment.close)

    def sample(self):
        self.experiment.sample(self.backend.robot(), [0.2, 0.1], [0.18, 0.09],
                               (self.backend.observation(),), physical_time_sec=0.2)

    def test_stop_retains_frames_invalid_depth_and_control(self):
        path = self.experiment.start()
        self.sample()
        self.experiment.stop()
        result = inspect_record(path)
        self.assertEqual(result["status"], "STOPPED")
        self.assertEqual(result["states"], 1)
        self.assertEqual(result["observations"], 2)
        with np.load(path / "frames/000000.npz", allow_pickle=False) as frame:
            np.testing.assert_array_equal(frame["rgb"][0, 0], [255, 0, 0])
            self.assertTrue(np.isnan(frame["depth_m"][0, 0]))
            self.assertTrue(np.isposinf(frame["depth_m"][0, 1]))
            self.assertEqual(frame["depth_m"][1, 0], 0)

    def test_timeout_is_saved_without_completion(self):
        path = self.experiment.start()
        self.backend.timestamp += 1.1
        self.sample()
        self.assertIsNone(self.experiment.recorder)
        self.assertEqual(inspect_record(path)["status"], "TIMEOUT")

    def test_error_and_user_exit_keep_trials(self):
        for reason in ("ERROR", "USER_EXIT", "INTERRUPTED"):
            path = self.experiment.start()
            self.sample()
            self.experiment.close(reason, details={"message": "test"})
            self.assertEqual(inspect_record(path)["status"], reason)

    def test_reset_finishes_record_and_restores_start_goal_and_clock(self):
        path = self.experiment.start()
        self.backend.position = 3
        self.backend.timestamp = 4
        self.backend.goal[:] = 7
        self.experiment.reset()
        self.assertEqual(inspect_record(path)["status"], "RESET_BY_USER")
        self.assertEqual(self.backend.position, 0)
        self.assertEqual(self.backend.timestamp, 2)
        np.testing.assert_array_equal(self.backend.goal, [1, 0, 0])

    def test_failed_reset_is_reported(self):
        self.experiment.save_start()
        self.backend.fail_reset = True
        with self.assertRaises(RuntimeError):
            self.experiment.reset()
        self.assertEqual(self.experiment.status, "RESET_FAILED")

    def test_missing_start_and_missing_goal_reject_operation(self):
        with self.assertRaises(ValueError):
            self.experiment.reset()
        self.backend.goal = None
        with self.assertRaises(ValueError):
            self.experiment.start()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_clock_regression_does_not_write_bad_sample(self):
        path = self.experiment.start()
        self.backend.timestamp -= 0.1
        with self.assertRaises(ValueError):
            self.sample()
        self.backend.timestamp = 2
        self.experiment.stop()
        self.assertEqual(inspect_record(path)["states"], 0)

    def test_corrupted_frame_index_is_detected(self):
        path = self.experiment.start()
        self.experiment.stop()
        (path / "frames/000000.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_record(path)

    def test_unrecorded_map_evidence_is_detected(self):
        path = self.experiment.start()
        self.experiment.recorder.event("MAP_UPDATE", self.backend.timestamp,
            update=dict(observations=[dict(observation_id="missing", source="arm", timestamp_sec=2.0)]))
        self.experiment.stop()
        with self.assertRaises(ValueError):
            inspect_record(path)


class StateTests(unittest.TestCase):
    def test_random_sequences_repeat_without_mutating_snapshot(self):
        generator = np.random.default_rng(12)
        state = RandomState.capture(generator)
        expected = None
        for _ in range(3):
            state.restore(generator)
            actual = [random.random(), np.random.random(), generator.random()]
            if expected is None:
                expected = actual
            np.testing.assert_array_equal(actual, expected)

    def test_rotation_error_handles_half_turn(self):
        first = Backend().robot()
        transform = np.eye(4)
        transform[0, 0] = transform[1, 1] = -1
        second = RobotState(2, transform, np.zeros(6), ("wheel",), [0.1], [0.2])
        self.assertAlmostEqual(robot_errors(first, second)["orientation_rad"], np.pi)

    def test_joint_order_mismatch_is_rejected(self):
        first = Backend().robot()
        second = RobotState(2, np.eye(4), np.zeros(6), ("other",), [0.1], [0.2])
        with self.assertRaises(ValueError):
            robot_errors(first, second)

    def test_mutating_live_goal_does_not_change_snapshot(self):
        backend = Backend()
        state = backend.capture_snapshot()
        backend.goal[:] = 8
        np.testing.assert_array_equal(state.goal_world_xyz, [1, 0, 0])


class CommandTests(unittest.TestCase):
    def test_temporary_data_is_removed_before_framework_exit(self):
        events = []

        class Application:
            def __init__(self, config):
                pass

            def close(self, **kwargs):
                events.append("application_close")
                raise SystemExit(0)

        class Runtime:
            def __init__(self, *args):
                pass

            def initialize(self):
                pass

            def run(self):
                return 0

            def close(self, wait=False):
                events.append("runtime_close")

        isaacsim, omni, kit, kit_app, runtime = [ModuleType(name) for name in
            ("isaacsim", "omni", "omni.kit", "omni.kit.app", "constraint_traverse.active_passage.simulation.runtime")]
        isaacsim.SimulationApp = Application
        omni.kit, kit.app = kit, kit_app
        stream = SimpleNamespace(create_subscription_to_pop_by_type=lambda *args, **kwargs: object())
        kit_app.get_app = lambda: SimpleNamespace(get_shutdown_event_stream=lambda: stream)
        runtime.ExperimentRuntime = Runtime
        with patch.dict(sys.modules, {m.__name__: m for m in (isaacsim, omni, kit, kit_app, runtime)}):
            with self.assertRaises(SystemExit):
                run_application(parse_args(["run"]), cleanup_output=lambda: events.append("temporary_cleanup"))
        self.assertEqual(events, ["runtime_close", "temporary_cleanup", "application_close"])

    def test_default_run_uses_interactive_window(self):
        args = parse_args(["run"])
        self.assertFalse(args.headless)

    def test_check_uses_headless_and_temporary_output(self):
        args = parse_args(["check-sim"])
        self.assertTrue(args.headless)
        self.assertIsNone(args.output)
        self.assertEqual(args.resets, 10)

    def test_invalid_numbers_and_missing_scene_are_rejected(self):
        for values in (["run", "--timeout", "nan"], ["run", "--gpu", "-1"],
                       ["run", "--start", "inf", "0", "0", "0"],
                       ["run", "--usd", "/nonexistent-scene.usd"], ["check-sim", "--resets", "0"]):
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(values)


if __name__ == "__main__":
    unittest.main()
