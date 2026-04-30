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
            selected_action = _sanitize_selected_action(payload.get("selected_action"))
            thought = str(payload.get("reason", payload.get("thought", ""))).strip() or "No reason provided."
            target_visible = _parse_optional_bool(payload.get("target_visible"))
            target_bearing_hint = str(payload.get("target_bearing_hint", "unknown")).strip() or "unknown"
            target_status = str(payload.get("target_memory_state", payload.get("target_status", ""))).strip()
            if target_visible is None:
                target_visible = _infer_target_visible_from_text(thought)
            if target_bearing_hint.lower() == "unknown":
                target_bearing_hint = _infer_bearing_from_text(thought)
            if not target_status:
                target_status = thought[:180]
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
        expected_action = _candidate_primitive_action(selected_candidate)
        if selected_action is None:
            selected_action = expected_action
        if selected_action != expected_action:
            return ReasonerDecision(
                selected_candidate_id=None,
                selected_action=None,
                thought=(
                    "LLM returned inconsistent candidate/action pair: "
                    f"candidate_id={selected_candidate_id} expects {expected_action}, got {selected_action}."
                ),
                progress_estimation=progress_estimation,
                used_fallback=False,
                reason="action_candidate_mismatch",
            )

        return ReasonerDecision(
            selected_candidate_id=selected_candidate_id,
            selected_action=selected_action,
            thought=thought,
            progress_estimation=progress_estimation,
            used_fallback=False,
            reason="ok",
            target_visible=target_visible,
            target_bearing_hint=target_bearing_hint,
            target_status=target_status,
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
            "You are the local decision module of a Vision-Language Navigation agent.\n"
            "Your job is to choose exactly one valid motion candidate id from the image and candidate list.\n"
            "The image has two panels: LEFT is a local top-down depth costmap, RIGHT is the current egocentric/front camera view.\n"
            "Candidate ids are drawn on both panels; the same ids appear in the text list. Use the costmap to avoid obstacles and the front view to judge target visibility.\n"
            "Candidate convention: positive heading_deg and side=left mean steering LEFT; negative heading_deg and side=right mean steering RIGHT; side=center means moving generally FORWARD.\n"
            "The selected candidate is executed as one fixed discrete motion primitive: left -> TURN_LEFT, center -> FORWARD, right -> TURN_RIGHT. There is no continuous waypoint tracking.\n"
            "Use only the current target for this step, not every landmark in the full instruction.\n"
            "Important: target_visible and target_bearing_hint must describe the CURRENT IMAGE ONLY. Target memory is historical and must not be reported as current visibility.\n"
            "Hard rules:\n"
            "1. Never output an invalid candidate id.\n"
            "2. If the current target is visible at the extreme left or extreme right edge, choose a same-side valid candidate to bring it into a usable view.\n"
            "3. If the current target is visible but not at the image edge, do not over-center it; choose the valid candidate that approaches it, often CENTER.\n"
            "4. If the current target is near the center/ahead, CENTER is allowed and usually preferred when clear.\n"
            "5. If the current target is not visible in the current image, set target_visible=false and target_bearing_hint=unknown. Then use target memory, recent history, and progress only to choose a search candidate.\n"
            "6. Do not choose left/right merely because the target is slightly off-center. Turn only for edge recovery or clear directional need.\n"
            "7. Direction consistency applies only when target_visible=true: if target_bearing_hint is right, do not select a left-side candidate; if target_bearing_hint is left, do not select a right-side candidate.\n"
            "8. If Safety context reports a blocked action, do not repeat the same unsafe action unless the visual evidence clearly changed.\n"
            "9. If progress_signal is STALLED or SPINNING, do not repeat the recent direction; choose a valid candidate that changes heading or escapes the corner.\n"
            "10. For multi-landmark instructions, use Route context to keep the current landmark consistent with the previous landmark and the instruction's left/right/forward relation.\n"
            "11. If the current target was seen before but is now lost, reacquire it using target memory; do not blindly drive deeper into a dead end.\n"
            "12. Target memory may include grounded_landmark_map. If it has confirmed=1, treat its robot_forward_m/robot_left_m as stronger than vague historical text memory.\n"
            "13. semantic_visible is not the same as grounded_visible. target_visible is only a semantic CURRENT IMAGE judgment; target grounding with bbox+depth is the strong geometric signal.\n"
            "14. If Reacquire guard says LOST_AFTER_VISIBLE is active and the target is still not visible in the current image, do not choose the forbidden action. Do not keep turning in the same direction based only on memory.\n"
            "15. In LOST_AFTER_VISIBLE, if the target is still not visible, prefer a candidate with the opposite turn or a safe non-repeating candidate. If the target is visible again, current visual evidence overrides memory. The runtime may force STOP if you select the forbidden action while target_visible=false.\n"
            "16. Be self-consistent: target_visible, target_bearing_hint, target_memory_state, reason, and selected_candidate_id must agree.\n"
            "Return strict JSON only. No markdown."
        )
        valid_ids_text = ", ".join(str(candidate.slot_id) for candidate in candidate_set.candidates if candidate.valid)
        user_prompt = (
            "## Navigation Context\n"
            f"Instruction: {instruction_context.normalized_text}\n"
            f"Current target: {current_target}\n"
            f"Ordered actions: {instruction_context.actions}\n"
            f"Landmarks: {instruction_context.landmarks}\n"
            f"Directional clauses: {instruction_context.directional_clauses}\n"
            "\n"
            "## Agent State\n"
            f"Route context: {route_context_text or 'none'}\n"
            f"Target memory: {target_memory_text or 'none'}\n"
            f"Safety context: {safety_context_text or 'none'}\n"
            f"Reacquire guard: {reacquire_context_text or 'none'}\n"
            f"Progress estimation: {prompt_progress_estimation}\n"
            f"Recent history: {progress_state.history_text}\n"
            f"Progress signal: {progress_state.progress_signal}\n"
            "\n"
            "## Decision Procedure\n"
            "1. Judge visibility and coarse bearing of Current target only.\n"
            "2. If Current target is not visible in the current image, write target_visible=false and target_bearing_hint=unknown, even if memory says it was last seen left/right.\n"
            "3. If visible at an image edge, select a same-side candidate unless all same-side candidates are invalid.\n"
            "4. If visible within the usable field of view, prefer approaching over perfect centering; CENTER may be correct only when the target is roughly ahead and the costmap is clear.\n"
            "5. Check direction consistency only when visible_now: right target cannot choose left candidate; left target cannot choose right candidate.\n"
            "6. If Safety context says the previous action was blocked, choose a different valid candidate or the safest valid candidate visible in the current frame.\n"
            "7. If progress is STALLED/SPINNING, select an escape/recovery candidate instead of continuing the same direction.\n"
            "8. If not visible but grounded_landmark_map has confirmed=1, choose a candidate that moves the robot toward robot_forward_m/robot_left_m instead of forgetting the landmark.\n"
            "9. If not visible but only weak target memory has a last seen anchor, try to reacquire around that memory bearing; avoid continuing straight into a dead end.\n"
            "10. If Reacquire guard names a Forbidden action this step and target_visible=false, do not output that action even if memory says the target was last seen that way.\n"
            "11. If this is not the first landmark, use the route context to check whether the candidate is plausible relative to the previous landmark and the instruction direction.\n"
            "12. Select exactly one valid candidate id and output the matching selected_action.\n"
            "\n"
            "## Available Candidates\n"
            "(ID matches the numbered red circle marker in the image.)\n"
            f"Valid candidate ids this step: [{valid_ids_text}]\n"
            "Candidates:\n"
            f"{candidate_lines}\n"
            "\n"
            "## Output Format\n"
            "Return exactly one strict JSON object. Do not copy any placeholder text.\n"
            "Use these fields in this order:\n"
            "1. reason: one short sentence, no step-by-step reasoning.\n"
            "2. target_visible: boolean true or false, not a string.\n"
            "3. target_bearing_hint: one of left, center, right, unknown.\n"
            "4. target_memory_state: one of visible_now, lost_recently, not_seen_yet, approaching, uncertain.\n"
            "5. selected_action: one of TURN_LEFT, FORWARD, TURN_RIGHT, matching the primitive_action of selected_candidate_id.\n"
            "6. selected_candidate_id: one valid integer id from the candidate list."
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
    return (
        f"- id={candidate.slot_id} valid={int(candidate.valid)} "
        f"label={candidate.label} side={side} "
        f"primitive_action={primitive_action} "
        f"heading_deg={candidate.heading_delta_deg:+.1f} "
        f"distance_m={candidate.travel_distance_m:.2f} "
        f"source=current_frame "
        f"clearance={candidate.clearance_score:.2f} "
        f"free_ratio={candidate.depth_support.get('free_ratio', 0.0):.2f} "
        f"near_depth_m={candidate.depth_support.get('min_depth', 0.0):.2f} "
        f"median_depth_m={candidate.depth_support.get('median_depth', 0.0):.2f} "
        f"occupancy={occupancy_reason} "
        f"planned_path_m={occupancy_path_length:.2f}"
    )


def _candidate_side(candidate: WaypointCandidate) -> str:
    heading = float(candidate.heading_delta_deg)
    if heading > 10.0:
        return "left"
    if heading < -10.0:
        return "right"
    return "center"


def _candidate_primitive_action(candidate: WaypointCandidate) -> str:
    side = _candidate_side(candidate)
    if side == "left":
        return "TURN_LEFT"
    if side == "right":
        return "TURN_RIGHT"
    return "FORWARD"


def _sanitize_selected_action(value) -> str | None:
    if value is None:
        return None
    action = str(value).strip().upper()
    aliases = {
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
        "STRAIGHT": "FORWARD",
        "CENTER": "FORWARD",
        "GO_FORWARD": "FORWARD",
    }
    action = aliases.get(action, action)
    if action in {"TURN_LEFT", "FORWARD", "TURN_RIGHT"}:
        return action
    return None


def _parse_json(content: str) -> dict:
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def _parse_optional_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "visible"}:
        return True
    if text in {"false", "no", "0", "not_visible", "not visible"}:
        return False
    return None


def _infer_target_visible_from_text(text: str) -> bool | None:
    lowered = str(text).lower()
    if any(phrase in lowered for phrase in ("not visible", "no longer visible", "lost", "cannot see")):
        return False
    if any(phrase in lowered for phrase in ("visible", "i see", "can see", "in view", "appears", "seen")):
        return True
    return None


def _infer_bearing_from_text(text: str) -> str:
    lowered = str(text).lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    if any(token in lowered for token in ("center", "centre", "ahead", "front", "directly")):
        return "center"
    return "unknown"


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
