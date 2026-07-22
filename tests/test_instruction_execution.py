from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vln_nav.navigation.instruction_manager import InstructionManager, RouteStep
from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.region_transition import (
    RegionTransitionConfig,
    RegionTransitionManager,
)


class _FakeLLM:
    available = True

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def chat(self, **_kwargs):
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return SimpleNamespace(content=json.dumps(payload), failed=False, reason="ok")


class _FakeRawLLM:
    available = True

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def chat(self, **_kwargs):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return SimpleNamespace(content=response, failed=False, reason="ok")


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_instruction_execution")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


class InstructionExecutionTest(unittest.TestCase):
    def test_invalid_json_uses_exactly_one_repair_call(self) -> None:
        repaired_payload = {
            "schema_version": "2.0",
            "scene_context": {"current_region": None, "context_objects": []},
            "navigation_targets": [
                {"canonical_label": "stairs", "source_span": "stairs"},
            ],
            "route_steps": [
                {
                    "step_id": "step_1",
                    "step_type": "GO_TO",
                    "source_span": "Find the stairs",
                    "target": "stairs",
                    "direction": "none",
                    "trigger": {"trigger_type": "NONE", "label": ""},
                    "constraints": [],
                },
            ],
        }
        client = _FakeRawLLM(["not json", json.dumps(repaired_payload)])
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = InstructionManager(
                llm_client=client,
                logger=_logger(),
                cache_path=Path(temp_dir) / "plans.json",
            ).parse("Find the stairs.")

        self.assertEqual(plan.navigation_target_labels, ["stairs"])
        self.assertEqual(plan.parsing_source, "llm_repair")
        self.assertEqual(client.calls, 2)

    def test_hospital_instruction_compiles_to_typed_steps(self) -> None:
        payload = {
            "schema_version": "2.0",
            "scene_context": {
                "current_region": {"canonical_label": "hospital corridor", "source_span": ""},
                "context_objects": [],
            },
            "navigation_targets": [
                {"canonical_label": "wheelchair", "source_span": "wheelchair"},
                {"canonical_label": "stairs", "source_span": "stairs"},
            ],
            "route_steps": [
                {
                    "step_id": "step_1",
                    "step_type": "GO_TO",
                    "source_span": "Go forward to the wheelchair",
                    "target": "wheelchair",
                    "direction": "forward",
                    "trigger": {"trigger_type": "NONE", "label": ""},
                    "constraints": [],
                },
                {
                    "step_id": "step_2",
                    "step_type": "TURN",
                    "source_span": "then turn left",
                    "direction": "left",
                    "trigger": {"trigger_type": "AFTER_PREVIOUS_STEP", "label": ""},
                    "constraints": [],
                },
                {
                    "step_id": "step_3",
                    "step_type": "GO_TO",
                    "source_span": "continue straight to the stairs",
                    "target": "stairs",
                    "direction": "forward",
                    "trigger": {"trigger_type": "AFTER_PREVIOUS_STEP", "label": ""},
                    "constraints": [],
                },
            ],
        }
        client = _FakeLLM([payload])
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InstructionManager(
                llm_client=client,
                logger=_logger(),
                cache_path=Path(temp_dir) / "plans.json",
            )
            plan = manager.parse(
                "Go forward to the wheelchair, then turn left and continue straight to the stairs."
            )

        self.assertEqual([step.step_type for step in plan.route_steps], ["GO_TO", "TURN", "GO_TO"])
        self.assertEqual(plan.navigation_target_labels, ["wheelchair", "stairs"])
        self.assertEqual(plan.route_steps[1].direction, "left")
        self.assertEqual(client.calls, 1)

    def test_context_objects_are_not_navigation_targets(self) -> None:
        payload = {
            "schema_version": "2.0",
            "scene_context": {
                "current_region": {"canonical_label": "kitchen", "source_span": "in the kitchen"},
                "context_objects": [
                    {"canonical_label": "table", "source_span": "table"},
                    {"canonical_label": "chair", "source_span": "chairs"},
                ],
            },
            "navigation_targets": [
                {"canonical_label": "sofa", "source_span": "sofa"},
            ],
            "route_steps": [
                {
                    "step_id": "step_1",
                    "step_type": "REGION_TRANSITION",
                    "source_span": "go out of the kitchen",
                    "from_region": "kitchen",
                    "to_region": "outside_current_region",
                    "transition": "exit",
                    "direction": "none",
                    "trigger": {"trigger_type": "NONE", "label": ""},
                    "constraints": [],
                },
                {
                    "step_id": "step_2",
                    "step_type": "TURN",
                    "source_span": "turn left",
                    "direction": "left",
                    "trigger": {"trigger_type": "AFTER_PREVIOUS_STEP", "label": ""},
                    "constraints": [],
                },
                {
                    "step_id": "step_3",
                    "step_type": "GO_TO",
                    "source_span": "find the sofa",
                    "target": "sofa",
                    "direction": "none",
                    "trigger": {"trigger_type": "AFTER_PREVIOUS_STEP", "label": ""},
                    "constraints": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = InstructionManager(
                llm_client=_FakeLLM([payload]),
                logger=_logger(),
                cache_path=Path(temp_dir) / "plans.json",
            ).parse(
                "You are in a kitchen with a table and chairs. Go out, turn left, and find the sofa."
            )

        self.assertEqual([item.canonical_label for item in plan.scene_context.context_objects], ["table", "chair"])
        self.assertEqual(plan.navigation_target_labels, ["sofa"])
        self.assertEqual(
            [step.step_type for step in plan.route_steps],
            ["REGION_TRANSITION", "TURN", "GO_TO"],
        )

    def test_context_target_overlap_requires_repair(self) -> None:
        invalid = {
            "schema_version": "2.0",
            "scene_context": {
                "current_region": None,
                "context_objects": [
                    {"canonical_label": "wheelchair", "source_span": "wheelchair"},
                ],
            },
            "navigation_targets": [
                {"canonical_label": "wheelchair", "source_span": "wheelchair"},
            ],
            "route_steps": [
                {
                    "step_id": "step_1",
                    "step_type": "GO_TO",
                    "source_span": "find the wheelchair",
                    "target": "wheelchair",
                    "direction": "none",
                    "trigger": {"trigger_type": "NONE", "label": ""},
                    "constraints": [],
                },
            ],
        }
        repaired = json.loads(json.dumps(invalid))
        repaired["scene_context"]["context_objects"] = []
        client = _FakeLLM([invalid, repaired])
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = InstructionManager(
                llm_client=client,
                logger=_logger(),
                cache_path=Path(temp_dir) / "plans.json",
            ).parse("Find the wheelchair.")

        self.assertEqual(client.calls, 2)
        self.assertEqual(plan.parsing_source, "llm_repair")
        self.assertEqual(plan.scene_context.context_objects, ())

    def test_repeated_context_target_overlap_is_canonicalized_after_repair(self) -> None:
        invalid = {
            "schema_version": "2.0",
            "scene_context": {
                "current_region": None,
                "context_objects": [
                    {"canonical_label": "wheelchair", "source_span": "wheelchair"},
                ],
            },
            "navigation_targets": [
                {"canonical_label": "wheelchair", "source_span": "wheelchair"},
            ],
            "route_steps": [
                {
                    "step_id": "step_1",
                    "step_type": "GO_TO",
                    "source_span": "go to the wheelchair",
                    "target": "wheelchair",
                    "direction": "none",
                    "trigger": {"trigger_type": "NONE", "label": ""},
                    "constraints": [],
                },
            ],
        }
        client = _FakeLLM([invalid, invalid])
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = InstructionManager(
                llm_client=client,
                logger=_logger(),
                cache_path=Path(temp_dir) / "plans.json",
            ).parse("Go to the wheelchair.")

        self.assertEqual(client.calls, 2)
        self.assertEqual(plan.parsing_source, "llm_repair")
        self.assertEqual(
            plan.parsing_reason,
            "repaired_after_validation_and_context_normalization",
        )
        self.assertEqual(plan.scene_context.context_objects, ())
        self.assertEqual(plan.navigation_target_labels, ["wheelchair"])

    def test_region_transition_requires_semantics_odometry_and_topology(self) -> None:
        step = RouteStep(
            step_id="step_1",
            step_type="REGION_TRANSITION",
            source_span="exit the kitchen",
            from_region="kitchen",
            to_region="outside_current_region",
            transition="exit",
            completion="REGION_CHANGED",
        )
        manager = RegionTransitionManager(
            config=RegionTransitionConfig(
                min_displacement_m=1.0,
                stable_confirmations=2,
                min_region_confidence=0.55,
            ),
            logger=_logger(),
        )
        manager.begin(
            step=step,
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            topology_node_id=1,
        )
        first = manager.update(
            step=step,
            robot_pose=Pose2D(1.1, 0.0, 0.0),
            topology_node_id=2,
            current_region_label="hallway",
            region_confidence=0.8,
            region_changed=True,
        )
        second = manager.update(
            step=step,
            robot_pose=Pose2D(1.2, 0.0, 0.0),
            topology_node_id=2,
            current_region_label="hallway",
            region_confidence=0.82,
            region_changed=True,
        )

        self.assertFalse(first.completed)
        self.assertTrue(second.completed)
        self.assertEqual(second.stable_confirmation_count, 2)


if __name__ == "__main__":
    unittest.main()
