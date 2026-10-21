from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vln_nav.active_perception.bev_goal_map import BevGoalMap
from vln_nav.active_perception.camera_geometry import world_from_pose_wxyz
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.active_perception.local_geometry import LocalVoxelMap, LocalVoxelMapConfig
from vln_nav.control.controller import SteeringDriveController
from vln_nav.control.trajectory import AgxVelocityAdapter, VelocityCommand
from vln_nav.runtime.agx_profile import load_agx_profile

from .bev import BevConfig, snapshot_to_robot_bev
from .collect import DEFAULT_CONFIG_PATH, PROJECT_ROOT, _load_config
from .episode import EpisodeSession
from .map_state import load_snapshot, verify_episode_bevs
from .recorder import (
    ControlRecord, EpisodeRecorder, SampleRecord, quantize_depth_m,
    recover_episode_directory, validate_episode_directory,
)
from .runtime import (
    ConstraintTraverseCollector, RuntimeOverrides, _apply_velocity_command,
    _complete_requested_episode,
)
from .sampling import PhysicsHistory, PhysicsState, capture_aligned_frame, pair_aligned_frames
from .teleop import ContinuousTeleop, TeleopClock, TeleopConfig
from .transforms import BodyMotionEstimator, quaternion_wxyz_from_yaw
from .ui import CollectorPanel
from .validate import _find_episode_dirs


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = _load_config(DEFAULT_CONFIG_PATH)
        self.map_config = LocalVoxelMapConfig(
            size_x_m=4, size_y_m=4, min_z_m=-0.2, max_z_m=1.2,
            resolution_m=0.1, pixel_stride=1,
        )
        self.local_map = LocalVoxelMap(self.map_config)
        self.local_map.reset(np.array([0., 0., .32]))
        self.bev_config = BevConfig(
            forward_m=1, backward_m=1, left_m=1, right_m=1, resolution_m=.1,
            obstacle_min_height_m=.05, obstacle_max_height_m=1,
        )
        self.intrinsics = np.array([[10., 0., 2.], [0., 10., 2.], [0., 0., 1.]])
        self.front_transform = np.array([
            [0., 0., 1., .1], [-1., 0., 0., 0.], [0., -1., 0., .2], [0., 0., 0., 1.],
        ])
        self.down_transform = np.diag([1., -1., -1., 1.])
        self.down_transform[2, 3] = .2
        self.depth = quantize_depth_m(np.full((5, 5), 1.2344, dtype=np.float32))
        self.local_map.integrate_batch(
            [self.observation("front", 9., np.array([-.2, 0., .32]))],
            center_world_xyz=np.array([-.2, 0., .32]),
        )
        self.initial_map = self.local_map.snapshot()
        self.meta = {
            "physics_dt_sec": 1 / 60,
            "policy_period_sec": 1 / 15,
            "teleop": self.config["teleop"],
            "applied_action_limits": {"max_linear_mps": 1., "max_angular_radps": .5},
            "cameras": {"width": 5, "height": 5, "frequency_hz": 15, "max_frame_skew_sec": .08},
            "bev": asdict(self.bev_config),
            "voxel_map": asdict(self.map_config),
            "collection": self.config["collection"],
        }

    def observation(self, source, timestamp, position, orientation=None):
        if orientation is None:
            orientation = np.array([1., 0., 0., 0.])
        base = world_from_pose_wxyz(position, orientation)
        transform = self.front_transform if source == "front" else self.down_transform
        return TimedDepthObservation(source, timestamp, self.depth, self.intrinsics, base @ transform)

    def recorder(self):
        recorder = EpisodeRecorder(output_root=self.root, scene_path=Path("scene.usda"))
        self.addCleanup(recorder.close)
        recorder.begin(
            start_pose_world=np.zeros(3), start_position_world_xyz=np.array([0., 0., .32]),
            start_time_sec=10., initial_map=self.initial_map, extra_meta=self.meta,
        )
        return recorder

    def record_episode(self, recorder, *, front_lag_steps=0):
        samples = []
        for tick in range(21):
            timestamp = 10 + tick / 60
            position = np.array([min(tick, 16) * .003, 0., .32])
            yaw = min(tick, 16) * .003
            pose = np.array([position[0], 0., yaw])
            orientation = quaternion_wxyz_from_yaw(yaw).astype(np.float64)
            velocity = np.array([
                .18 * np.cos((tick - 1) * .003) if 0 < tick <= 16 else 0.,
                .18 if 0 < tick <= 16 else 0.,
            ])
            action = np.array([.3, .18]) if tick < 16 else np.zeros(2)
            recorder.append_control(ControlRecord(
                timestamp_sec=timestamp, physics_step=1000 + tick,
                pose_world_xyyaw=pose, world_position_xyz=position,
                world_orientation_wxyz=orientation, measured_velocity_vw=velocity,
                human_action_vw=action, applied_action_vw=action,
                steering_position=np.zeros(4), steering_velocity=np.zeros(4), wheel_velocity=np.zeros(4),
            ))
            if tick % 4:
                continue
            front_tick = max(0, tick - front_lag_steps)
            front_time = 10 + front_tick / 60
            front_position = np.array([min(front_tick, 16) * .003, 0., .32])
            front_orientation = quaternion_wxyz_from_yaw(min(front_tick, 16) * .003).astype(np.float64)
            integrated = tick % 8 == 0
            if integrated:
                self.local_map.integrate_batch(
                    [
                        self.observation("front", front_time, front_position, front_orientation),
                        self.observation("down", timestamp, position, orientation),
                    ],
                    center_world_xyz=position,
                )
            snapshot = self.local_map.snapshot()
            sample = SampleRecord(
                sample_time_sec=timestamp, front_timestamp_sec=front_time, down_timestamp_sec=timestamp,
                front_rgb=np.full((5, 5, 3), 120 + front_tick, dtype=np.uint8),
                front_depth_m=self.depth, down_depth_m=self.depth,
                bev=snapshot_to_robot_bev(snapshot, robot_world_xyyaw=pose, floor_z_m=0., config=self.bev_config),
                pose_world_xyyaw=pose, world_position_xyz=position, world_orientation_wxyz=orientation,
                measured_velocity_vw=velocity, human_action_vw=action, applied_action_vw=action,
                steering_position=np.zeros(4), steering_velocity=np.zeros(4), wheel_velocity=np.zeros(4),
                front_intrinsics=self.intrinsics, down_intrinsics=self.intrinsics,
                front_base_from_camera=self.front_transform, down_base_from_camera=self.down_transform,
                front_world_from_camera=world_from_pose_wxyz(front_position, front_orientation) @ self.front_transform,
                down_world_from_camera=world_from_pose_wxyz(position, orientation) @ self.down_transform,
                map_version=snapshot.version, state_timestamp_sec=timestamp, action_timestamp_sec=timestamp,
                physics_step=1000 + tick, action_physics_step=1000 + tick,
                front_physics_step=1000 + front_tick, down_physics_step=1000 + tick,
                map_integrated=integrated, floor_z_m=0.,
            )
            recorder.append(sample)
            samples.append(sample)
        return samples

    def finish(self, recorder, samples):
        return recorder.finish(
            success=True, end_pose_world=samples[-1].pose_world_xyyaw,
            end_time_sec=samples[-1].sample_time_sec, success_requested_time_sec=10 + 19 / 60,
        )

    def test_record_validate_and_reconstruct_with_map_history(self):
        recorder = self.recorder()
        samples = self.record_episode(recorder)
        path = self.finish(recorder, samples)
        self.assertEqual(validate_episode_directory(path), [])
        self.assertEqual(verify_episode_bevs(path), [])
        meta = json.loads((path / "meta.json").read_text())
        self.assertEqual(meta["goal_world_xy"], samples[-1].pose_world_xyyaw[:2].tolist())
        self.assertTrue(meta["valid"])
        self.assertEqual(meta["control_count"], 21)
        self.assertEqual(load_snapshot(path / "initial_map.npz").version, self.initial_map.version)

    def test_asynchronous_camera_episode_validates_and_reconstructs(self):
        recorder = self.recorder()
        samples = self.record_episode(recorder, front_lag_steps=1)
        path = recorder.finish(
            success=True, end_pose_world=samples[-1].pose_world_xyyaw,
            end_time_sec=samples[-1].sample_time_sec, success_requested_time_sec=10 + 18 / 60,
        )
        self.assertNotEqual(samples[-1].front_physics_step, samples[-1].down_physics_step)
        self.assertEqual(validate_episode_directory(path), [])
        self.assertEqual(verify_episode_bevs(path), [])

    def test_close_preserves_committed_data_as_incomplete(self):
        recorder = self.recorder()
        self.record_episode(recorder)
        path = recorder.episode_dir
        recorder.close()
        meta = json.loads((path / "meta.json").read_text())
        self.assertEqual(meta["status"], "incomplete")
        self.assertFalse(meta["success"])
        self.assertIsNone(meta["goal_world_xy"])
        self.assertEqual(meta["sample_count"], 6)
        with np.load(path / "controls.npz") as controls:
            self.assertEqual(len(controls["timestamp_sec"]), 21)
        self.assertIn(path, _find_episode_dirs(self.root))

    def test_recovery_keeps_prefix_when_last_chunk_is_corrupt(self):
        recorder = self.recorder()
        self.record_episode(recorder)
        path = recorder.episode_dir
        recorder.close()
        sorted((path / "chunks").glob("*.npz"))[-1].write_bytes(b"interrupted")
        recover_episode_directory(path)
        meta = json.loads((path / "meta.json").read_text())
        self.assertEqual(meta["sample_count"], 5)
        self.assertEqual(meta["control_count"], 17)
        self.assertFalse(meta["success"])
        self.assertTrue(any("recovery stopped" in error for error in meta["validation_errors"]))

    def test_finish_interruption_preserves_manual_success(self):
        recorder = self.recorder()
        samples = self.record_episode(recorder)
        temporary = recorder.episode_dir
        with patch.object(Path, "rename", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.finish(recorder, samples)
        recorder.close()
        self.assertFalse(temporary.exists())
        path = temporary.with_name("episode_000001")
        meta = json.loads((path / "meta.json").read_text())
        self.assertTrue(meta["success"])
        self.assertEqual(validate_episode_directory(path), [])

    def test_validation_detects_state_time_shift(self):
        recorder = self.recorder()
        path = self.finish(recorder, self.record_episode(recorder))
        with np.load(path / "state.npz") as data:
            values = {name: data[name] for name in data.files}
        values["state_timestamp_sec"][2] += .01
        np.savez_compressed(path / "state.npz", **values)
        errors = validate_episode_directory(path)
        self.assertTrue(any("state_timestamp_sec" in error for error in errors))

    def test_validation_detects_missing_physics_step(self):
        recorder = self.recorder()
        path = self.finish(recorder, self.record_episode(recorder))
        with np.load(path / "controls.npz") as data:
            values = {name: data[name] for name in data.files}
        values["physics_step"][7] += 1
        np.savez_compressed(path / "controls.npz", **values)
        self.assertTrue(any("missing or unordered steps" in error for error in validate_episode_directory(path)))

    def test_validation_detects_velocity_and_pose_disagreement(self):
        recorder = self.recorder()
        path = self.finish(recorder, self.record_episode(recorder))
        with np.load(path / "controls.npz") as data:
            values = {name: data[name] for name in data.files}
        values["measured_velocity_vw"][1, 0] = .9
        np.savez_compressed(path / "controls.npz", **values)
        self.assertTrue(any("velocity integrals" in error for error in validate_episode_directory(path)))

    def test_offline_validation_reports_saved_invalid_status(self):
        recorder = self.recorder()
        path = self.finish(recorder, self.record_episode(recorder))
        meta = json.loads((path / "meta.json").read_text())
        meta["valid"] = False
        meta["validation_errors"] = ["recorded sensor failure"]
        (path / "meta.json").write_text(json.dumps(meta))
        self.assertIn("recorded sensor failure", validate_episode_directory(path))

    def test_success_waits_for_terminal_observation(self):
        recorder = self.recorder()
        samples = self.record_episode(recorder)
        session = EpisodeSession(robot_placed=True, goal_world_xy=np.array([9., 9.]))
        session.begin(start_pose_world=np.zeros(3), start_position_world_xyz=np.array([0., 0., .32]), timestamp_sec=10.)
        session.sample_count = len(samples)
        session.success_requested_at_sec = samples[-1].sample_time_sec
        self.assertIsNone(_complete_requested_episode(session, recorder, samples[-1], self.config["collection"]))
        session.success_requested_at_sec = 10 + 19 / 60
        self.assertIsNone(_complete_requested_episode(
            session, recorder,
            replace(samples[-1], front_timestamp_sec=10 + 19 / 60),
            self.config["collection"],
        ))
        self.assertIsNone(_complete_requested_episode(session, recorder, samples[-2], self.config["collection"]))
        self.assertTrue(recorder.active)
        path = _complete_requested_episode(session, recorder, samples[-1], self.config["collection"])
        meta = json.loads((path / "meta.json").read_text())
        self.assertEqual(meta["goal_world_xy"], [.048, 0.])
        self.assertEqual(validate_episode_directory(path), [])

    def test_start_preserves_current_map_and_accepts_optional_hint(self):
        recorder = EpisodeRecorder(output_root=self.root, scene_path=Path("scene.usda"))
        self.addCleanup(recorder.close)
        session = EpisodeSession(robot_placed=True)
        collector = ConstraintTraverseCollector(
            app=None, project_root=PROJECT_ROOT, config=self.config,
            overrides=RuntimeOverrides(Path("scene.usda"), self.root), logger=logging.getLogger("test"),
        )
        controls = SimpleNamespace(reset=lambda: None, stop=lambda: None)
        mapping = {"camera": self.meta["cameras"], "map": {**asdict(self.map_config), "integration_period_sec": .25}, "max_frame_skew_sec": .08}
        collector._handle_ui_command(
            command="START", session=session, recorder=recorder, episode_map=self.local_map,
            teleop=controls, adapter=controls, robot=None, pose=np.zeros(3), position=np.array([0., 0., .32]),
            now=10., measured_v=0., measured_w=0., motion=BodyMotionEstimator(), current_episode_id="",
            scene_path=Path("scene.usda"), profile=load_agx_profile(project_root=PROJECT_ROOT),
            world=SimpleNamespace(get_physics_dt=lambda: 1 / 60), stage=None, articulation_path="",
            mapping_state={"last_integration_time": 9.}, mapping_cfg=mapping,
            front_path="/front", down_path="/down", last_bev=np.zeros((3, 20, 20), dtype=np.uint8),
        )
        self.assertTrue(session.recording)
        self.assertEqual(self.local_map.version, self.initial_map.version)
        saved = load_snapshot(recorder.episode_dir / "initial_map.npz")
        np.testing.assert_array_equal(saved.log_odds, self.initial_map.log_odds)
        meta = json.loads((recorder.episode_dir / "meta.json").read_text())
        self.assertIsNone(meta["goal_world_xy"])

    def test_camera_uses_cached_rgb_and_timestamped_state(self):
        history = PhysicsHistory()
        state = PhysicsState(
            10., 1000, np.array([0., 0., .32]), np.array([1., 0., 0., 0.]), np.zeros(3),
            np.zeros(2), np.zeros(4), np.zeros(4), np.zeros(4),
            world_from_pose_wxyz(np.array([0., 0., .32]), np.array([1., 0., 0., 0.])) @ self.front_transform,
            np.eye(4),
        )
        history.append(state)
        history.append(replace(state, timestamp_sec=10 + 1 / 60, physics_step=1001, position=np.array([5., 0., .32])))
        frame = {"rendering_time": 10., "rgb": np.full((5, 5, 4), 123, dtype=np.uint8), "distance_to_image_plane": self.depth}
        camera = SimpleNamespace(get_current_frame=lambda **kwargs: frame, get_intrinsics_matrix=lambda: self.intrinsics)
        result, matched = capture_aligned_frame(source="front", camera=camera, history=history, after_sec=9.)
        self.assertEqual(matched.physics_step, 1000)
        np.testing.assert_array_equal(result.rgb, frame["rgb"][..., :3])
        np.testing.assert_allclose(result.base_from_camera, self.front_transform)
        frame["rendering_time"] = 8.
        self.assertIsNone(capture_aligned_frame(source="front", camera=camera, history=history, after_sec=7.))

    def test_camera_pair_keeps_individual_exposure_times(self):
        history = PhysicsHistory()
        state = PhysicsState(
            10., 1000, np.array([0., 0., .32]), np.array([1., 0., 0., 0.]), np.zeros(3),
            np.zeros(2), np.zeros(4), np.zeros(4), np.zeros(4),
            world_from_pose_wxyz(np.array([0., 0., .32]), np.array([1., 0., 0., 0.])) @ self.front_transform,
            world_from_pose_wxyz(np.array([0., 0., .32]), np.array([1., 0., 0., 0.])) @ self.down_transform,
        )
        history.append(state)
        history.append(replace(state, timestamp_sec=10 + 1 / 60, physics_step=1001))
        front_data = {"rendering_time": 10., "rgb": np.full((5, 5, 4), 123, dtype=np.uint8), "distance_to_image_plane": self.depth}
        down_data = {**front_data, "rendering_time": 10 + 1 / 60}
        front_camera = SimpleNamespace(get_current_frame=lambda **kwargs: front_data, get_intrinsics_matrix=lambda: self.intrinsics)
        down_camera = SimpleNamespace(get_current_frame=lambda **kwargs: down_data, get_intrinsics_matrix=lambda: self.intrinsics)
        front = capture_aligned_frame(source="front", camera=front_camera, history=history, after_sec=9.)
        down = capture_aligned_frame(source="down", camera=down_camera, history=history, after_sec=9.)
        pair = pair_aligned_frames(front, down, max_skew_sec=.08)
        self.assertEqual(pair.front.frame_index, 1000)
        self.assertEqual(pair.down.frame_index, 1001)
        self.assertEqual(pair.state.physics_step, 1001)
        self.assertIsNone(pair_aligned_frames(front, down, max_skew_sec=.001))
        self.assertIsNone(pair_aligned_frames(front, down, max_skew_sec=.08, not_before_sec=10 + 1 / 60))

    def test_teleop_clock_controls_all_arrow_keys_without_camera_frames(self):
        event_types = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        for key, axis, sign in (("UP", 0, 1), ("DOWN", 0, -1), ("LEFT", 1, 1), ("RIGHT", 1, -1)):
            with self.subTest(key=key):
                teleop = ContinuousTeleop(TeleopConfig.from_mapping(self.config["teleop"]))
                teleop._carb_input = SimpleNamespace(KeyboardEventType=event_types)
                clock = TeleopClock(1 / 15, 0.)
                teleop._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name=key), type=event_types.KEY_PRESS))
                for tick in range(1, 61):
                    command = clock.step(teleop, tick / 60)
                values = [command.linear_mps, command.angular_radps]
                self.assertGreater(sign * values[axis], .45)
                teleop._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name=key), type=event_types.KEY_RELEASE))
                for tick in range(61, 121):
                    command = clock.step(teleop, tick / 60)
                self.assertAlmostEqual(command.linear_mps, 0.)
                self.assertAlmostEqual(command.angular_radps, 0.)

    def test_combined_keys_keep_motion_with_either_press_order(self):
        controller = load_agx_profile(project_root=PROJECT_ROOT).agx_controller.continuous
        event_types = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        for movement, direction, linear_sign, angular_sign in (
            ("UP", "LEFT", 1, 1), ("UP", "RIGHT", 1, -1),
            ("DOWN", "LEFT", -1, 1), ("DOWN", "RIGHT", -1, -1),
            ("W", "A", 1, 1), ("W", "D", 1, -1),
            ("S", "A", -1, 1), ("S", "D", -1, -1),
        ):
            for first, second in ((movement, direction), (direction, movement)):
                with self.subTest(first=first, second=second):
                    writes = []
                    drive = SimpleNamespace(
                        readback=lambda: {"steering_position": {
                            wheel.steering_joint: 0. for wheel in controller.wheels
                        }},
                        apply_continuous_joint_targets=lambda motors, steering: writes.append((motors, steering)),
                    )
                    adapter = AgxVelocityAdapter(drive, controller)
                    teleop = ContinuousTeleop(TeleopConfig.from_mapping(self.config["teleop"]))
                    teleop._carb_input = SimpleNamespace(KeyboardEventType=event_types)
                    clock = TeleopClock(1 / 15, 0.)
                    timestamp = 0.

                    def key_event(key, event_type):
                        teleop._on_keyboard_event(SimpleNamespace(
                            input=SimpleNamespace(name=key), type=event_type,
                        ))

                    def advance():
                        nonlocal timestamp
                        timestamp += 1 / 15
                        desired = clock.step(teleop, timestamp)
                        write_count = len(writes)
                        applied, _ = _apply_velocity_command(adapter, desired)
                        self.assertEqual(len(writes), write_count + 1)
                        self.assertLessEqual(
                            max(map(abs, writes[-1][1].values())),
                            controller.max_steering_rad + 1e-9,
                        )
                        return applied

                    key_event(first, event_types.KEY_PRESS)
                    for _ in range(15):
                        advance()
                    for remaining, released in ((second, first), (first, second)):
                        key_event(released, event_types.KEY_PRESS)
                        key_event(remaining, event_types.KEY_PRESS)
                        for _ in range(15):
                            applied = advance()
                            self.assertGreater(linear_sign * applied.linear_mps, 0.)
                            self.assertGreater(angular_sign * applied.angular_radps, 0.)
                        key_event(released, event_types.KEY_RELEASE)
                        for _ in range(15):
                            applied = advance()
                        if remaining == movement:
                            self.assertGreater(linear_sign * applied.linear_mps, .45)
                            self.assertAlmostEqual(applied.angular_radps, 0.)
                        else:
                            self.assertAlmostEqual(applied.linear_mps, 0.)
                            self.assertGreater(angular_sign * applied.angular_radps, .45)
                    key_event(first, event_types.KEY_RELEASE)
                    for _ in range(15):
                        applied = advance()
                    self.assertEqual(applied, VelocityCommand())

    def test_low_speed_turn_preserves_translation(self):
        controller = load_agx_profile(project_root=PROJECT_ROOT).agx_controller.continuous
        for linear, angular in ((.04067796468734741, -.18983051180839539), (.06101695075631142, .4067796468734741)):
            for linear_sign in (-1, 1):
                with self.subTest(linear=linear * linear_sign, angular=angular):
                    writes = []
                    drive = SimpleNamespace(
                        readback=lambda: {"steering_position": {
                            wheel.steering_joint: 0. for wheel in controller.wheels
                        }},
                        apply_continuous_joint_targets=lambda motors, steering: writes.append((motors, steering)),
                    )
                    desired = VelocityCommand(linear * linear_sign, angular)
                    applied, limited = _apply_velocity_command(AgxVelocityAdapter(drive, controller), desired)
                    self.assertTrue(limited)
                    self.assertAlmostEqual(applied.linear_mps, desired.linear_mps)
                    self.assertGreater(applied.angular_radps * desired.angular_radps, 0.)
                    self.assertLess(abs(applied.angular_radps), abs(desired.angular_radps))
                    self.assertEqual(len(writes), 1)

    def test_teleop_reads_all_held_arrow_keys_when_events_are_missing(self):
        held = {"UP", "LEFT"}
        teleop = ContinuousTeleop(TeleopConfig.from_mapping(self.config["teleop"]))
        teleop._keyboard = object()
        teleop._input = SimpleNamespace(get_keyboard_value=lambda keyboard, key: float(key in held))
        teleop._carb_input = SimpleNamespace(KeyboardInput=SimpleNamespace(**{
            name: name for name in teleop._KEYS
        }))
        teleop._pressed_keys = {"LEFT"}
        for _ in range(15):
            command = teleop.step(1 / 15)
        self.assertAlmostEqual(command.linear_mps, .5)
        self.assertAlmostEqual(command.angular_radps, .5)
        held.remove("UP")
        for _ in range(15):
            command = teleop.step(1 / 15)
        self.assertAlmostEqual(command.linear_mps, 0.)
        self.assertAlmostEqual(command.angular_radps, .5)
        held.clear()
        for _ in range(15):
            command = teleop.step(1 / 15)
        self.assertEqual(command, VelocityCommand())

    def test_teleop_reserves_driving_keys_and_releases_them_for_placement(self):
        event_types = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        teleop = ContinuousTeleop(TeleopConfig.from_mapping(self.config["teleop"]))
        teleop._carb_input = SimpleNamespace(KeyboardEventType=event_types)
        up_event = SimpleNamespace(input=SimpleNamespace(name="UP"), type=event_types.KEY_PRESS)
        self.assertFalse(teleop._on_keyboard_event(up_event))
        self.assertGreater(teleop.step(1 / 15).linear_mps, 0.)
        teleop.set_enabled(False)
        self.assertEqual(teleop.command, VelocityCommand())
        self.assertTrue(teleop._on_keyboard_event(up_event))
        self.assertEqual(teleop.step(1.), VelocityCommand())
        teleop.set_enabled(True)
        self.assertFalse(teleop._on_keyboard_event(up_event))
        self.assertGreater(teleop.step(1 / 15).linear_mps, 0.)

    def test_releasing_all_keys_stops_and_centers_wheels(self):
        controller = load_agx_profile(project_root=PROJECT_ROOT).agx_controller.continuous
        event_types = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        for keys, press_steps in (
            (("LEFT",), 7), (("RIGHT",), 7),
            (("LEFT",), 15), (("RIGHT",), 15),
            (("UP", "LEFT"), 15), (("UP", "RIGHT"), 15),
            (("DOWN", "LEFT"), 15), (("DOWN", "RIGHT"), 15),
        ):
            with self.subTest(keys=keys, press_steps=press_steps):
                writes = []
                drive = SimpleNamespace(
                    readback=lambda: {"steering_position": {
                        wheel.steering_joint: 0. for wheel in controller.wheels
                    }},
                    apply_continuous_joint_targets=lambda motors, steering: writes.append((dict(motors), dict(steering))),
                )
                adapter = AgxVelocityAdapter(drive, controller)
                teleop = ContinuousTeleop(TeleopConfig.from_mapping(self.config["teleop"]))
                teleop._carb_input = SimpleNamespace(KeyboardEventType=event_types)
                for key in keys:
                    teleop._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name=key), type=event_types.KEY_PRESS))
                for _ in range(press_steps):
                    _apply_velocity_command(adapter, teleop.step(1 / 15))
                steering = writes[-1][1]
                self.assertGreater(max(map(abs, steering.values())), .1)
                for key in keys:
                    teleop._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name=key), type=event_types.KEY_RELEASE))
                for _ in range(20):
                    applied, _ = _apply_velocity_command(adapter, teleop.step(1 / 15))
                self.assertEqual(applied, VelocityCommand())
                self.assertTrue(all(value == 0. for value in writes[-1][0].values()))
                self.assertTrue(all(value == 0. for value in writes[-1][1].values()))
                adapter.stop()
                self.assertTrue(all(value == 0. for value in writes[-1][1].values()))
                teleop._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name="UP"), type=event_types.KEY_PRESS))
                for _ in range(15):
                    applied, _ = _apply_velocity_command(adapter, teleop.step(1 / 15))
                self.assertGreater(applied.linear_mps, .45)
                self.assertTrue(all(value == 0. for value in writes[-1][1].values()))

    def test_velocity_adapter_stops_with_all_wheels_forward(self):
        controller = load_agx_profile(project_root=PROJECT_ROOT).agx_controller.continuous
        initial = {wheel.steering_joint: .2 for wheel in controller.wheels}
        writes = []
        drive = SimpleNamespace(
            readback=lambda: {"steering_position": dict(initial)},
            apply_continuous_joint_targets=lambda motors, steering: writes.append((dict(motors), dict(steering))),
        )
        adapter = AgxVelocityAdapter(drive, controller)
        self.assertEqual(adapter.apply(VelocityCommand()), VelocityCommand())
        self.assertTrue(all(value == 0. for value in writes[-1][0].values()))
        self.assertTrue(all(value == 0. for value in writes[-1][1].values()))
        adapter.apply(VelocityCommand(.3, .2))
        self.assertTrue(any(abs(value) > .1 for value in writes[-1][1].values()))
        adapter.stop()
        self.assertTrue(all(value == 0. for value in writes[-1][0].values()))
        self.assertTrue(all(value == 0. for value in writes[-1][1].values()))

    def test_parking_centering_increases_steering_torque_and_restores_it_for_driving(self):
        config = load_agx_profile(project_root=PROJECT_ROOT).agx_controller.continuous
        for centering_effort in (None, 50.):
            with self.subTest(centering_effort=centering_effort):
                drive = SteeringDriveController.__new__(SteeringDriveController)
                drive._wheel_dofs = {wheel.drive_joint: index for index, wheel in enumerate(config.wheels)}
                drive._steering_dofs = {config.wheels[0].steering_joint: 4, config.wheels[1].steering_joint: 5}
                drive._passive_steering_dofs = {config.wheels[2].steering_joint: 6, config.wheels[3].steering_joint: 7}
                drive.steering_centering_max_effort_nm = centering_effort
                drive._steering_centering_active = False
                drive._steering_centering_original_efforts = {}
                properties = {index: SimpleNamespace(max_effort=5., stiffness=100., damping=10.) for index in range(4, 8)}
                writes = []

                def set_properties(dof, value):
                    properties[dof] = value
                    writes.append((dof, value.max_effort))
                    return True

                drive._dc = SimpleNamespace(
                    get_dof_properties=lambda dof: SimpleNamespace(**vars(properties[dof])),
                    set_dof_properties=set_properties,
                )
                motors = {wheel.drive_joint: 0. for wheel in config.wheels}
                steering = {wheel.steering_joint: 0. for wheel in config.wheels}
                with patch.object(drive, "_apply_targets"):
                    drive.apply_continuous_joint_targets(motors, steering)
                    expected = 5. if centering_effort is None else 50.
                    self.assertTrue(all(value.max_effort == expected for value in properties.values()))
                    write_count = len(writes)
                    drive.apply_continuous_joint_targets(motors, steering)
                    self.assertEqual(len(writes), write_count)
                    drive.apply_continuous_joint_targets({name: 1. for name in motors}, steering)
                    self.assertTrue(all(value.max_effort == 5. for value in properties.values()))
                    self.assertTrue(all(value.stiffness == 100. and value.damping == 10. for value in properties.values()))
                    self.assertEqual(len(writes), 0 if centering_effort is None else 8)

    def test_map_click_uses_displayed_pose(self):
        panel = CollectorPanel.__new__(CollectorPanel)
        panel._goal_map = BevGoalMap(np.zeros((80, 80, 4), dtype=np.uint8), np.ones((80, 80), dtype=bool), 10., 1.5)
        panel._map_pose = np.array([1., 2., np.pi / 2])
        panel._image = SimpleNamespace(computed_width=80., computed_height=80., screen_position_x=0., screen_position_y=0.)
        commands = []
        panel._command_sink = commands.append
        panel._on_map_pressed(40., 40., 0, None)
        forward, left = panel._goal_map.pixel_to_local(40, 40)
        self.assertEqual(commands[0][0], "SET_HINT_WORLD")
        np.testing.assert_allclose(commands[0][1:], [1. - left, 2. + forward])


if __name__ == "__main__":
    unittest.main()
