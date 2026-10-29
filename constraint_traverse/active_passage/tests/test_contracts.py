from __future__ import annotations

import unittest

import numpy as np

from constraint_traverse.active_passage.belief import Belief
from constraint_traverse.active_passage.contracts import (
    ArmObservationPlan, BeliefSnapshot, CameraFrame, GeometryHypothesis,
    GroundTruthGeometry, LossSamples, ObservedGeometry, PredictedObservation,
    RealObservation, RobotState, TaskSpec, TraversalPlan, VoxelGeometry,
)


def frame(timestamp=1):
    return CameraFrame("arm_rgbd", timestamp, np.zeros((2, 3, 3), np.uint8), np.ones((2, 3), np.float32),
                       np.asarray([[10., 0, 1], [0, 10, 1], [0, 0, 1]]), np.eye(4))


def robot(timestamp=1):
    return RobotState(timestamp, np.eye(4), np.zeros(6), ("arm1", "arm2"), np.zeros(2), np.zeros(2))


def observation(identifier="obs1", timestamp=1):
    return RealObservation(identifier, frame(timestamp), robot(timestamp))


def grid():
    return VoxelGeometry(np.zeros((2, 3, 4), np.int8), np.zeros(3), 0.05)


class ObservationContractTests(unittest.TestCase):
    def test_images_are_copied_and_read_only(self):
        image = np.zeros((2, 3, 3), np.uint8)
        value = CameraFrame("arm", 1, image, np.ones((2, 3)), np.eye(3), np.eye(4))
        image[:] = 255
        self.assertEqual(int(value.rgb.max()), 0)
        self.assertFalse(value.rgb.flags.writeable)
        self.assertFalse(value.depth_m.flags.writeable)

    def test_joint_state_is_copied_and_read_only(self):
        positions = np.asarray([0., 1.])
        state = RobotState(1, np.eye(4), np.zeros(6), ("a", "b"), positions, np.zeros(2))
        positions[:] = 5
        np.testing.assert_array_equal(state.joint_positions, [0, 1])
        self.assertFalse(state.joint_positions.flags.writeable)

    def test_timestamp_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            RealObservation("obs", frame(1), robot(2))

    def test_invalid_depth_pixels_remain_invalid(self):
        depth = np.asarray([[0, np.nan, np.inf], [1, 1, 1]])
        value = CameraFrame("arm", 1, np.zeros((2, 3, 3), np.uint8), depth, np.eye(3), np.eye(4))
        self.assertTrue(np.isnan(value.depth_m[0, 1]))
        self.assertTrue(np.isposinf(value.depth_m[0, 2]))

    def test_invalid_transform_is_rejected(self):
        transform = np.eye(4)
        transform[0, 0] = 2
        with self.assertRaises(ValueError):
            RobotState(1, transform, np.zeros(6), (), np.empty(0), np.empty(0))

    def test_duplicate_joint_names_are_rejected(self):
        with self.assertRaises(ValueError):
            RobotState(1, np.eye(4), np.zeros(6), ("a", "a"), np.zeros(2), np.zeros(2))

    def test_voxel_values_are_validated_before_integer_conversion(self):
        with self.assertRaises(ValueError):
            VoxelGeometry(np.full((1, 1, 1), 0.5), np.zeros(3), 0.05)

    def test_unknown_voxels_remain_unknown(self):
        value = VoxelGeometry(np.asarray([[[-1, 0, 1]]]), np.zeros(3), 0.05)
        np.testing.assert_array_equal(value.occupancy, [[[-1, 0, 1]]])


class BeliefIsolationTests(unittest.TestCase):
    def test_real_observation_updates_history(self):
        belief = Belief()
        initial = belief.snapshot()
        updated = belief.update(observation())
        self.assertEqual(initial.version, 0)
        self.assertEqual(initial.observations, ())
        self.assertEqual(updated.version, 1)
        self.assertEqual(updated.observations[0].observation_id, "obs1")
        self.assertIsNone(updated.observed_geometry)

    def test_prediction_and_ground_truth_cannot_update_real_belief(self):
        belief = Belief()
        wrong = (PredictedObservation("prediction", "hypothesis", frame()), GroundTruthGeometry("scene_A", grid()))
        for value in wrong:
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                belief.update(value)
        self.assertEqual(belief.snapshot().version, 0)

    def test_ground_truth_cannot_enter_snapshot_map(self):
        with self.assertRaises(TypeError):
            BeliefSnapshot(0, (), GroundTruthGeometry("scene_A", grid()))

    def test_prediction_cannot_enter_real_history(self):
        with self.assertRaises(TypeError):
            BeliefSnapshot(0, (PredictedObservation("p", "h", frame()),))

    def test_ground_truth_wrapper_cannot_be_a_hypothesis_grid(self):
        with self.assertRaises(TypeError):
            GeometryHypothesis("h", GroundTruthGeometry("scene_A", grid()), 1)

    def test_real_fusion_records_observation_provenance(self):
        class Integrator:
            def integrate(self, current, value):
                ids = () if current is None else current.observation_ids
                return ObservedGeometry(grid(), ids + (value.observation_id,))
        belief = Belief(integrator=Integrator())
        belief.update(observation())
        result = belief.update(observation("obs2", 2))
        self.assertEqual(result.observed_geometry.observation_ids, ("obs1", "obs2"))

    def test_wrong_fusion_result_does_not_update_snapshot(self):
        class Integrator:
            def integrate(self, current, value):
                return GroundTruthGeometry("scene_A", grid())
        belief = Belief(integrator=Integrator())
        with self.assertRaises(TypeError):
            belief.update(observation())
        self.assertEqual(belief.snapshot().observations, ())

    def test_unrecorded_map_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            BeliefSnapshot(1, (observation(),), ObservedGeometry(grid(), ("different_observation",)))

    def test_duplicate_and_older_observations_are_rejected(self):
        belief = Belief()
        belief.update(observation("obs1", 2))
        for value in (observation("obs1", 3), observation("obs2", 1)):
            with self.assertRaises(ValueError):
                belief.update(value)
        self.assertEqual(belief.snapshot().version, 1)

    def test_hypothesis_weights_need_normalization(self):
        with self.assertRaises(ValueError):
            BeliefSnapshot(0, (), hypotheses=(GeometryHypothesis("h", grid(), 0.5),))


class PlanContractTests(unittest.TestCase):
    def test_arm_plan_returns_to_stow(self):
        value = ArmObservationPlan(("a", "b"), np.asarray([[0, 0], [0.2, 0.3]]),
            np.asarray([[0.2, 0.3], [0, 0]]), 0.1, 1, False)
        self.assertFalse(value.safety_passed)
        np.testing.assert_array_equal(value.outward_positions[0], value.return_positions[-1])

    def test_discontinuous_arm_round_trip_is_rejected(self):
        for returning in ([[0.5, 0.3], [0, 0]], [[0.2, 0.3], [1, 1]]):
            with self.assertRaises(ValueError):
                ArmObservationPlan(("a", "b"), np.asarray([[0, 0], [0.2, 0.3]]), np.asarray(returning), 0.1, 1, True)

    def test_arm_check_requires_boolean_result(self):
        with self.assertRaises(TypeError):
            ArmObservationPlan(("a",), np.asarray([[0], [1]]), np.asarray([[1], [0]]), 0.1, 1, "false")

    def test_rotation_segment_cannot_translate(self):
        with self.assertRaises(ValueError):
            TraversalPlan("task", np.asarray([[0, 0, 0], [1, 0, 1]]), np.asarray([0, 0]), 1)

    def test_task_goal_has_no_user_heading(self):
        task = TaskSpec("task", "穿过左侧门", "left_door", np.asarray([1, 2, 0]), 0.3)
        np.testing.assert_array_equal(task.goal_world_xyz, [1, 2, 0])

    def test_loss_sample_arrays_are_immutable(self):
        losses = np.asarray([1., 2.])
        result = LossSamples(losses, ("A", "B"), ("s", "s"), np.asarray([2., 2.]))
        losses[:] = 100
        np.testing.assert_array_equal(result.losses, [1, 2])
        np.testing.assert_array_equal(result.weights, [0.5, 0.5])
        self.assertFalse(result.weights.flags.writeable)


if __name__ == "__main__":
    unittest.main()
