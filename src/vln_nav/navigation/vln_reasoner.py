from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .instruction_manager import InstructionContext, RouteStep
from .policy_helpers import candidate_primitive_action_name
from .progress_manager import ProgressState
from .vlm_client import VLMClient


@dataclass
class ReasonerDecision:
    selected_candidate_id: int | None
    selected_action: str | None
    thought: str
    progress_estimation: str
    reason: str
    target_visible: bool | None = None
    target_bearing_hint: str = "unknown"
    target_status: str = ""
    current_region_label: str = "unknown"
    region_confidence: float = 0.0
    region_changed: bool = False


@dataclass
class CandidateChoiceRequest:
    request_name: str
    system_prompt: str
    user_prompt: str
    max_tokens: int
    temperature: float


class VLNReasoner:
    def __init__(
        self,
        *,
        llm_client: VLMClient,
        robot_capability_summary: str,
        logger: logging.Logger,
    ) -> None:
        self._llm = llm_client
        self._robot_capability_summary = (
            str(robot_capability_summary).strip() or "unknown"
        )
        self._logger = logger.getChild("vln_reasoner")

    def estimate_progress(
        self,
        *,
        instruction_context: InstructionContext,
        progress_state: ProgressState,
    ) -> str:
        history_text = progress_state.history_text if progress_state.history_text != "none" else "no_history"
        return (
            f"progress_signal={progress_state.progress_signal}; "
            f"stalled_steps={progress_state.stalled_steps}; "
            f"net_displacement_m={progress_state.net_displacement_m:.2f}; "
            f"history={history_text}"
        )

    def choose_candidate(
        self,
        *,
        instruction_context: InstructionContext,
        candidate_set: CandidateSet,
        progress_state: ProgressState,
        annotated_image_png_bytes: bytes | None,
        target_memory_text: str = "",
        safety_context_text: str = "",
        route_context_text: str = "",
        reacquire_context_text: str = "",
        obstacle_summary_text: str = "",
        semantic_map_summary: str = "",
        current_step: RouteStep | None = None,
    ) -> ReasonerDecision:
        valid_candidates = [candidate for candidate in candidate_set.candidates if candidate.valid]
        if not valid_candidates:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought="No valid candidates are currently available.",
                progress_estimation=f"progress_signal={progress_state.progress_signal}",
                reason="no_valid_candidates",
            )

        progress_estimation = self.estimate_progress(
            instruction_context=instruction_context,
            progress_state=progress_state,
        )
        if not self._llm.available or annotated_image_png_bytes is None:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought="LLM is unavailable or the clean RGB observation is missing, so no decision was produced.",
                progress_estimation=progress_estimation,
                reason="llm_unavailable",
            )

        request_payload = self.build_candidate_choice_request(
            instruction_context=instruction_context,
            candidate_set=candidate_set,
            progress_state=progress_state,
            progress_estimation=progress_estimation,
            target_memory_text=target_memory_text,
            safety_context_text=safety_context_text,
            route_context_text=route_context_text,
            reacquire_context_text=reacquire_context_text,
            obstacle_summary_text=obstacle_summary_text,
            semantic_map_summary=semantic_map_summary,
            current_step=current_step,
        )
        valid_ids = {candidate.slot_id for candidate in valid_candidates}
        user_prompt = request_payload.user_prompt
        last_error = "invalid_llm_output"
        parsed: dict | None = None
        for attempt in range(2):
            result = self._llm.chat(
                request_name=(
                    request_payload.request_name
                    if attempt == 0
                    else f"{request_payload.request_name}_repair"
                ),
                system_prompt=request_payload.system_prompt,
                user_prompt=user_prompt,
                image_png_bytes=annotated_image_png_bytes,
                max_tokens=request_payload.max_tokens,
                temperature=0.0 if attempt else request_payload.temperature,
                json_response=True,
            )
            if result.failed or not result.content:
                last_error = result.reason
                break
            try:
                candidate_payload = _parse_json(result.content)
                candidate_id = int(candidate_payload.get("selected_candidate_id"))
            except Exception:
                last_error = "invalid_llm_output"
                candidate_payload = None
                candidate_id = -1
            if candidate_payload is not None and candidate_id in valid_ids:
                parsed = candidate_payload
                break
            last_error = "invalid_candidate_id"
            if attempt == 0:
                user_prompt = (
                    request_payload.user_prompt
                    + "\n\nYour previous response was invalid. Return JSON only and choose "
                    + f"exactly one selected_candidate_id from {sorted(valid_ids)}. "
                    + f"Previous response: {result.content}"
                )

        if parsed is None:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought=(
                    "LLM failed deterministic candidate validation after one repair. "
                    f"valid_ids={sorted(valid_ids)}"
                ),
                progress_estimation=progress_estimation,
                reason=last_error,
            )

        selected_candidate_id = int(parsed["selected_candidate_id"])
        thought = str(parsed.get("reason", parsed.get("thought", ""))).strip() or "No reason provided."
        current_region_label = str(parsed.get("current_region_label", "unknown")).strip().lower() or "unknown"
        region_confidence = float(parsed.get("region_confidence", 0.0))
        region_changed = _parse_bool(parsed.get("region_changed", False))
        selected_candidate = next(candidate for candidate in valid_candidates if candidate.slot_id == selected_candidate_id)
        selected_action = candidate_primitive_action_name(selected_candidate)

        return ReasonerDecision(
            selected_candidate_id=selected_candidate_id,
            selected_action=selected_action,
            thought=thought,
            progress_estimation=progress_estimation,
            reason="ok",
            target_visible=None,
            target_bearing_hint="unknown",
            target_status="",
            current_region_label=current_region_label,
            region_confidence=max(0.0, min(1.0, region_confidence)),
            region_changed=region_changed,
        )

    def build_candidate_choice_request(
        self,
        *,
        instruction_context: InstructionContext,
        candidate_set: CandidateSet,
        progress_state: ProgressState,
        progress_estimation: str | None = None,
        target_memory_text: str = "",
        safety_context_text: str = "",
        route_context_text: str = "",
        reacquire_context_text: str = "",
        obstacle_summary_text: str = "",
        semantic_map_summary: str = "",
        current_step: RouteStep | None = None,
    ) -> CandidateChoiceRequest:
        prompt_progress_estimation = progress_estimation
        if prompt_progress_estimation is None:
            prompt_progress_estimation = self.estimate_progress(
                instruction_context=instruction_context,
                progress_state=progress_state,
            )
        candidate_lines = "\n".join(_candidate_line(candidate) for candidate in candidate_set.candidates)
        current_target = _step_subject(current_step) or "none"
        current_step_text = _format_route_step(current_step)
        context_region = (
            "unknown"
            if instruction_context.scene_context.current_region is None
            else instruction_context.scene_context.current_region.canonical_label
        )
        context_objects = [
            item.canonical_label for item in instruction_context.scene_context.context_objects
        ]
        region_output_rules = ""
        output_contract = "{reason, selected_candidate_id}"
        if current_step is not None and current_step.step_type == "REGION_TRANSITION":
            output_contract = (
                "{reason, selected_candidate_id, current_region_label, "
                "region_confidence, region_changed}"
            )
            region_output_rules = (
                "\nREGION TRANSITION EVIDENCE:\n"
                "- Classify the robot's CURRENT region from visible architectural and semantic cues.\n"
                "- region_changed is true only when the image supports that the robot has crossed out of the stated start region.\n"
                "- For a named destination region, region_changed requires current_region_label to match it.\n"
                "- Do not claim a transition merely because a doorway is visible; the robot must be on the other side.\n"
                "- region_confidence is a number in [0,1].\n"
            )
        system_prompt = (
            "You are the high-level local VLN decision module for a navigation robot.\n"
            "Inputs are: (a) one clean first-person camera image without overlays, "
            "(b) structured text map/memory/obstacle summaries, and (c) an explicit candidate action table.\n"
            "The occupancy map is NOT shown as an image here; use the structured text map and candidate table for spatial reasoning.\n"
            "\n"
            "ACTION CONTRACT:\n"
            "- You must choose exactly one valid candidate id from Valid IDs.\n"
            "- The runtime deterministically maps that candidate id to its primitive action.\n"
            "- Do not infer target visibility or output a motor action.\n"
            f"- Robot capabilities: {self._robot_capability_summary}.\n"
            "\n"
            "DECISION RULES:\n"
            "1. If the current target/landmark is visible in the first-person image, choose the candidate that centers or approaches it.\n"
            "2. If the target has a confirmed/map position, choose a safe candidate toward that position.\n"
            "3. If the target is not visible or confirmed, explore open/safe unknown space that is consistent with the instruction.\n"
            "4. Avoid candidates marked invalid, blocked, low-clearance, or inconsistent with safety context.\n"
            "5. Avoid repeating a turn direction after recently losing a visible target unless the current image clearly shows the target there.\n"
            "6. Route turns are one-time relations. If route context says a relation turn was consumed, do not repeat it merely because words such as left/right remain in the instruction.\n"
            "7. Treat repetition_penalty=1.0 as a strong reason not to repeat that turn unless current visual evidence requires it.\n"
            "8. Scene context objects are localization evidence, never goals unless the current typed step names them as a target.\n"
            "9. Respect the current typed step and all global instruction constraints.\n"
            f"{region_output_rules}"
            "10. Output valid JSON only, no markdown."
        )
        valid_ids_text = ", ".join(str(candidate.slot_id) for candidate in candidate_set.candidates if candidate.valid)
        user_prompt = (
            f"Instruction: {instruction_context.normalized_text}\n"
            f"Trusted start region: {context_region}\n"
            f"Context objects (not targets): {context_objects or ['none']}\n"
            f"Current typed step: {current_step_text}\n"
            f"Current target: {current_target}\n"
            f"\n"
            "Progress:\n"
            f"{prompt_progress_estimation}\n"
            f"\n"
            "Structured semantic map:\n"
            f"{semantic_map_summary or 'Objects: none mapped yet'}\n"
            f"\n"
            "Route context:\n"
            f"{route_context_text or 'none'}\n"
            f"\n"
            "Target memory:\n"
            f"{target_memory_text or 'none'}\n"
            f"\n"
            "Obstacle summary:\n"
            f"{obstacle_summary_text or 'none'}\n"
            f"\n"
            "Reacquire guard:\n"
            f"{reacquire_context_text or 'none'}\n"
            f"\n"
            "Safety context:\n"
            f"{safety_context_text or 'none'}\n"
            f"\n"
            "Candidate action table:\n"
            f"{candidate_lines}\n"
            f"Valid IDs: [{valid_ids_text}]\n"
            "\n"
            f"Output JSON: {output_contract}\n"
            "selected_candidate_id: integer from Valid IDs\n"
            "reason: one short sentence explaining why this candidate is best"
        )
        return CandidateChoiceRequest(
            request_name="candidate_choice",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
            temperature=0.2,
        )


def _step_subject(step: RouteStep | None) -> str:
    if step is None:
        return ""
    if step.step_type == "GO_TO":
        return step.target
    if step.step_type == "REGION_TRANSITION":
        return step.to_region
    if step.step_type == "TURN":
        return f"turn {step.direction}"
    return ""


def _format_route_step(step: RouteStep | None) -> str:
    if step is None:
        return "none"
    trigger = "none"
    if step.trigger is not None:
        trigger = f"{step.trigger.trigger_type}:{step.trigger.label or 'none'}"
    return (
        f"id={step.step_id}; type={step.step_type}; target={step.target or 'none'}; "
        f"from_region={step.from_region or 'none'}; to_region={step.to_region or 'none'}; "
        f"transition={step.transition or 'none'}; direction={step.direction}; "
        f"trigger={trigger}; completion={step.completion}; "
        f"cues={[f'{cue.role}:{cue.label}:{cue.expected_side}' for cue in step.cues] or ['none']}; "
        f"relations={[f'{relation.target}:{relation.relation}:{relation.reference}' for relation in step.relations] or ['none']}; "
        f"constraints={list(step.constraints) or ['none']}"
    )


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _candidate_line(candidate: WaypointCandidate) -> str:
    side = _candidate_side(candidate)
    primitive_action = candidate_primitive_action_name(candidate)
    occupancy_reason = str(candidate.depth_support.get("occupancy_reason", "not_checked"))
    occupancy_path_length = float(candidate.depth_support.get("occupancy_path_length_m", 0.0))
    free_ratio = float(candidate.depth_support.get("free_ratio", 0.0))
    near_depth_m = float(candidate.depth_support.get("min_depth", 0.0))
    min_clearance_m = float(candidate.depth_support.get("min_clearance_m", near_depth_m))
    unknown_gain = float(candidate.depth_support.get("unknown_gain", 0.0))
    route_alignment = float(candidate.depth_support.get("route_alignment", 0.0))
    repetition_penalty = float(candidate.depth_support.get("repetition_penalty", 0.0))
    anchor_delta_m = candidate.depth_support.get("anchor_distance_delta_m", "none")
    candidate_type = _candidate_type(candidate=candidate, occupancy_reason=occupancy_reason)
    return (
        f"- id={candidate.slot_id} valid={int(candidate.valid)} "
        f"type={candidate_type} label={candidate.label} side={side} "
        f"primitive_action={primitive_action} "
        f"heading_deg={candidate.heading_delta_deg:+.1f} "
        f"distance_m={candidate.travel_distance_m:.2f} "
        f"source=current_frame "
        f"clearance={candidate.clearance_score:.2f} "
        f"free_ratio={free_ratio:.2f} "
        f"min_clearance_m={min_clearance_m:.2f} "
        f"unknown_gain={unknown_gain:.2f} "
        f"route_alignment={route_alignment:.2f} "
        f"repetition_penalty={repetition_penalty:.2f} "
        f"anchor_distance_delta_m={anchor_delta_m} "
        f"median_depth_m={candidate.depth_support.get('median_depth', 0.0):.2f} "
        f"map_reachability={occupancy_reason} "
        f"planned_path_m={occupancy_path_length:.2f}"
    )


def _candidate_type(*, candidate: WaypointCandidate, occupancy_reason: str) -> str:
    explicit_type = str(candidate.depth_support.get("map_candidate_type", "")).strip()
    if explicit_type:
        return explicit_type
    if not bool(candidate.valid):
        return "invalid"
    heading = abs(float(candidate.heading_delta_deg))
    free_ratio = float(candidate.depth_support.get("free_ratio", 0.0))
    clearance = float(candidate.clearance_score)
    if str(occupancy_reason) == "reachable" and heading <= 12.0 and clearance >= 0.55:
        return "safe_forward"
    if str(occupancy_reason) == "reachable" and heading > 12.0:
        return "safe_turn_search"
    if free_ratio >= 0.50 and clearance >= 0.45:
        return "open_frontier"
    return "low_confidence_search"


def _candidate_side(candidate: WaypointCandidate) -> str:
    heading = float(candidate.heading_delta_deg)
    if heading > 10.0:
        return "left"
    if heading < -10.0:
        return "right"
    return "center"


def _parse_json(content: str) -> dict:
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)
