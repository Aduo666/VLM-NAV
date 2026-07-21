from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .instruction_manager import InstructionContext
from .open_nav_client import OpenNavLLMClient
from .progress_manager import ProgressState
from ..runtime.config import PROJECT_ROOT


LAST_CANDIDATE_CHOICE_DUMP_PATH = PROJECT_ROOT / "logs" / "last_candidate_choice_request.json"


@dataclass
class ReasonerDecision:
    selected_candidate_id: int | None
    selected_action: str | None
    thought: str
    progress_estimation: str
    used_fallback: bool
    reason: str
    target_visible: bool | None = None
    target_bearing_hint: str = "unknown"
    target_status: str = ""


@dataclass
class CandidateChoiceRequest:
    request_name: str
    system_prompt: str
    user_prompt: str
    max_tokens: int
    temperature: float


class OpenNavReasoner:
    def __init__(self, *, llm_client: OpenNavLLMClient, logger: logging.Logger) -> None:
        self._llm = llm_client
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
        annotated_image_path: str | None = None,
        target_memory_text: str = "",
        safety_context_text: str = "",
        route_context_text: str = "",
        reacquire_context_text: str = "",
        obstacle_summary_text: str = "",
        semantic_map_summary: str = "",
    ) -> ReasonerDecision:
        valid_candidates = [candidate for candidate in candidate_set.candidates if candidate.valid]
        if not valid_candidates:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought="No valid candidates are currently available.",
                progress_estimation=f"progress_signal={progress_state.progress_signal}",
                used_fallback=False,
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
                thought="LLM is unavailable or candidate debug image is missing, so no decision was produced.",
                progress_estimation=progress_estimation,
                used_fallback=False,
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
        )
        _dump_candidate_choice_request(
            request_payload=request_payload,
            image_path=annotated_image_path,
        )
        result = self._llm.chat(
            request_name=request_payload.request_name,
            system_prompt=request_payload.system_prompt,
            user_prompt=request_payload.user_prompt,
            image_png_bytes=annotated_image_png_bytes,
            max_tokens=request_payload.max_tokens,
            temperature=request_payload.temperature,
        )
        if result.used_fallback or not result.content:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought=f"LLM did not return a usable candidate decision. reason={result.reason}",
                progress_estimation=progress_estimation,
                used_fallback=False,
                reason=result.reason,
            )

        try:
            payload = _parse_json(result.content)
            selected_candidate_id = int(payload.get("selected_candidate_id"))
            thought = str(payload.get("reason", payload.get("thought", ""))).strip() or "No reason provided."
        except Exception:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought="LLM returned non-JSON or malformed JSON for candidate selection.",
                progress_estimation=progress_estimation,
                used_fallback=False,
                reason="invalid_llm_output",
            )

        valid_ids = {candidate.slot_id for candidate in valid_candidates}
        if selected_candidate_id not in valid_ids:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought=f"LLM selected invalid candidate id={selected_candidate_id}. valid_ids={sorted(valid_ids)}",
                progress_estimation=progress_estimation,
                used_fallback=False,
                reason="invalid_candidate_id",
            )

        selected_candidate = next(candidate for candidate in valid_candidates if candidate.slot_id == selected_candidate_id)
        selected_action = _candidate_primitive_action(selected_candidate)

        return ReasonerDecision(
            selected_candidate_id=selected_candidate_id,
            selected_action=selected_action,
            thought=thought,
            progress_estimation=progress_estimation,
            used_fallback=False,
            reason="ok",
            target_visible=None,
            target_bearing_hint="unknown",
            target_status="",
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
    ) -> CandidateChoiceRequest:
        prompt_progress_estimation = progress_estimation
        if prompt_progress_estimation is None:
            prompt_progress_estimation = self.estimate_progress(
                instruction_context=instruction_context,
                progress_state=progress_state,
            )
        candidate_lines = "\n".join(_candidate_line(candidate) for candidate in candidate_set.candidates)
        current_target = instruction_context.goal_phrase or (
            instruction_context.landmarks[0] if instruction_context.landmarks else instruction_context.normalized_text
        )
        system_prompt = (
            "You are the high-level local VLN decision module for a wheeled robot.\n"
            "Inputs are: (a) one clean first-person camera image with numbered candidate markers, "
            "(b) structured text map/memory/obstacle summaries, and (c) an explicit candidate action table.\n"
            "The occupancy map is NOT shown as an image here; use the structured text map and candidate table for spatial reasoning.\n"
            "\n"
            "ACTION CONTRACT:\n"
            "- You must choose exactly one valid candidate id from Valid IDs.\n"
            "- The runtime deterministically maps that candidate id to its primitive action.\n"
            "- Do not infer target visibility or output a motor action.\n"
            "\n"
            "DECISION RULES:\n"
            "1. If the current target/landmark is visible in the first-person image, choose the candidate that centers or approaches it.\n"
            "2. If the target has a confirmed/map position, choose a safe candidate toward that position.\n"
            "3. If the target is not visible or confirmed, explore open/safe unknown space that is consistent with the instruction.\n"
            "4. Avoid candidates marked invalid, blocked, low-clearance, or inconsistent with safety context.\n"
            "5. Avoid repeating a turn direction after recently losing a visible target unless the current image clearly shows the target there.\n"
            "6. Route turns are one-time relations. If route context says a relation turn was consumed, do not repeat it merely because words such as left/right remain in the instruction.\n"
            "7. Treat repetition_penalty=1.0 as a strong reason not to repeat that turn unless current visual evidence requires it.\n"
            "8. Output valid JSON only, no markdown."
        )
        valid_ids_text = ", ".join(str(candidate.slot_id) for candidate in candidate_set.candidates if candidate.valid)
        user_prompt = (
            f"Instruction: {instruction_context.normalized_text}\n"
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
            "Output JSON: {reason, selected_candidate_id}\n"
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


def _candidate_line(candidate: WaypointCandidate) -> str:
    side = _candidate_side(candidate)
    primitive_action = _candidate_primitive_action(candidate)
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
        f"occupancy={occupancy_reason} "
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


def _candidate_primitive_action(candidate: WaypointCandidate) -> str:
    explicit = str(candidate.depth_support.get("primitive_action", "")).strip().upper()
    if explicit in {"TURN_LEFT", "FORWARD", "TURN_RIGHT"}:
        return explicit
    side = _candidate_side(candidate)
    if side == "left":
        return "TURN_LEFT"
    if side == "right":
        return "TURN_RIGHT"
    return "FORWARD"


def _parse_json(content: str) -> dict:
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def _dump_candidate_choice_request(*, request_payload: CandidateChoiceRequest, image_path: str | None) -> None:
    dump_payload = {
        "request_name": request_payload.request_name,
        "system_prompt": request_payload.system_prompt,
        "user_prompt": request_payload.user_prompt,
        "max_tokens": int(request_payload.max_tokens),
        "temperature": float(request_payload.temperature),
        "image_path": str(image_path or ""),
    }
    target = LAST_CANDIDATE_CHOICE_DUMP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dump_payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
