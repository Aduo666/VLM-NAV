from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vln_nav.app.nav import _write_evaluation_result
from vln_nav.navigation.instruction_manager import (
    InstructionContext,
    RouteStep,
    SceneContext,
    SemanticEntity,
)
from vln_nav.navigation.vln_policy import PolicyStepResult


class EvaluationResultTest(unittest.TestCase):
    def test_success_result_contains_typed_step_status(self) -> None:
        context = InstructionContext(
            raw_text="Walk down the hallway, passing the TV on the right.",
            normalized_text="Walk down the hallway, passing the TV on the right.",
            instruction_id="instruction-1",
            task_type="vln",
            scene_context=SceneContext(),
            navigation_targets=(
                SemanticEntity(
                    canonical_label="bed",
                    source_span="bed",
                ),
            ),
            route_steps=(
                RouteStep(
                    step_id="step_1",
                    step_type="GO_TO",
                    source_span="stop near the bed",
                    target="bed",
                    direction="forward",
                    completion="SEMANTIC_AND_DISTANCE",
                ),
            ),
            parsing_source="test",
            parsing_reason="ok",
        )
        result = PolicyStepResult(
            action_name="STOP",
            speed_scale=0.0,
            decision_source="ROUTE_COMPLETE",
            selected_candidate_id=None,
            local_goal_xy=[0.0, 0.0],
            world_goal_xy=None,
            execution_steps_remaining=0,
            reasoning_summary="complete",
            progress_signal="ADVANCING",
            replanned=False,
            debug_image_path=None,
            route_complete=True,
            current_step_index=1,
            route_step_count=1,
            navigation_status="ROUTE_COMPLETE",
        )
        config = SimpleNamespace(
            instruction=SimpleNamespace(text=context.raw_text),
            scene=SimpleNamespace(usd_path="scene.usd"),
            robot=SimpleNamespace(name="agx"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "episode.json"
            _write_evaluation_result(
                output_path=output_path,
                status="SUCCESS",
                task_id="episode-1",
                config=config,
                instruction_context=context,
                step_result=result,
                sim_time_sec=12.5,
                failure_stage="",
                reason="all typed instruction steps completed",
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "SUCCESS")
        self.assertEqual(payload["route_steps"][0]["step_type"], "GO_TO")
        self.assertEqual(payload["route_steps"][0]["status"], "COMPLETED")
        self.assertEqual(payload["route_steps"][0]["target"], "bed")


if __name__ == "__main__":
    unittest.main()
