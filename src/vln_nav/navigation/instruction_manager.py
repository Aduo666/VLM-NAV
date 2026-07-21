from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field


@dataclass
class RouteStep:
    landmark: str
    directive: str
    behavior_type: str = "go_to"
    reference_landmark: str = ""
    relation_direction: str = "none"


@dataclass
class InstructionContext:
    raw_text: str
    normalized_text: str
    instruction_id: str
    task_type: str
    actions: list[str] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    directional_clauses: list[str] = field(default_factory=list)
    goal_phrase: str = ""
    route_steps: list[RouteStep] = field(default_factory=list)
    parsing_source: str = "heuristic"
    parsing_reason: str = "ok"


class InstructionManager:
    def __init__(
        self,
        *,
        llm_client,
        logger: logging.Logger,
        preparse_enabled: bool = True,
        max_tokens: int = 384,
    ) -> None:
        self._llm = llm_client
        self._logger = logger.getChild("instruction_manager")
        self._preparse_enabled = bool(preparse_enabled)
        self._max_tokens = max(128, int(max_tokens))

    def parse(self, raw_instruction: str) -> InstructionContext:
        normalized = _normalize_instruction(raw_instruction)
        fallback = _heuristic_parse(raw_text=str(raw_instruction), normalized=normalized)
        if not self._preparse_enabled or self._llm is None or not bool(getattr(self._llm, "available", False)):
            return fallback

        llm_context = self._parse_with_llm(raw_instruction=str(raw_instruction), normalized=normalized, fallback=fallback)
        if llm_context is not None:
            return llm_context
        return fallback

    def _parse_with_llm(
        self,
        *,
        raw_instruction: str,
        normalized: str,
        fallback: InstructionContext,
    ) -> InstructionContext | None:
        system_prompt = (
            "You convert a natural-language navigation instruction into a structured route plan.\n"
            "Return strict JSON only. No markdown.\n"
            "Each route step must name the landmark/object/place to approach next and preserve any directional phrase.\n"
            "If the instruction is a single-target command, return one route step.\n"
            "If the instruction is multi-stage, preserve temporal order.\n"
            "Use only fields requested below."
        )
        user_prompt = (
            "Instruction:\n"
            f"{raw_instruction}\n\n"
            "Return exactly one JSON object with these fields:\n"
            "{\n"
            '  "task_type": "objectnav" or "vln",\n'
            '  "goal_phrase": "short current navigation goal phrase",\n'
            '  "landmarks": ["ordered", "landmark", "labels"],\n'
            '  "actions": ["ordered clause 1", "ordered clause 2"],\n'
            '  "directional_clauses": ["clauses that contain left/right/forward/back/through/past/toward"],\n'
            '  "route_steps": [\n'
            '    {"landmark": "wheelchair", "directive": "go forward to the wheelchair", "behavior_type": "go_to", '
            '"reference_landmark": "", "relation_direction": "forward"}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "1. landmark labels must be short lowercase nouns.\n"
            "2. route_steps must be in execution order.\n"
            "3. behavior_type must currently be one of: go_to, pass, turn_at, go_through.\n"
            "4. If unsure, prefer behavior_type=go_to.\n"
            "5. Do not invent extra landmarks that are absent from the instruction.\n"
            "6. Keep the original directional meaning inside directive.\n"
            "7. relation_direction must be one of none, left, right, forward, left_then_forward, right_then_forward, through, past.\n"
            "8. reference_landmark is the previously reached landmark that the relation is relative to, or an empty string.\n"
        )
        result = self._llm.chat(
            request_name="instruction_preparse",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_png_bytes=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        if result.used_fallback or not str(result.content).strip():
            self._logger.info(
                "[InstructionParse] source=heuristic reason=llm_%s instruction=%s",
                result.reason,
                normalized,
            )
            return None
        try:
            payload = _extract_json_object(str(result.content))
            parsed = _context_from_payload(
                raw_text=raw_instruction,
                normalized=normalized,
                payload=payload,
                fallback=fallback,
                parsing_source="llm",
                parsing_reason="ok",
            )
            self._logger.info(
                "[InstructionParse] source=llm task_type=%s landmarks=%s route_steps=%s goal=%s",
                parsed.task_type,
                parsed.landmarks,
                [f"{step.behavior_type}:{step.landmark}" for step in parsed.route_steps],
                parsed.goal_phrase,
            )
            return parsed
        except Exception as exc:
            self._logger.warning("[InstructionParse] source=heuristic reason=llm_parse_failed error=%s", exc)
            return None


def _heuristic_parse(*, raw_text: str, normalized: str) -> InstructionContext:
    actions = _extract_ordered_clauses(normalized)
    landmarks = _extract_landmarks(normalized)
    directional_clauses = [
        clause
        for clause in actions
        if re.search(r"\b(left|right|forward|backward|back|straight|ahead|through|past|toward|to|into)\b", clause, flags=re.IGNORECASE)
    ]
    landmarks = [lm for lm in landmarks if _is_valid_landmark(lm)]
    route_steps = _heuristic_route_steps(actions=actions, landmarks=landmarks, normalized=normalized)
    goal_phrase = route_steps[0].landmark if route_steps else (landmarks[0] if landmarks else normalized)
    instruction_id = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "instruction"
    task_type = "objectnav" if len(route_steps) <= 1 else "vln"
    return InstructionContext(
        raw_text=str(raw_text),
        normalized_text=normalized,
        instruction_id=instruction_id,
        task_type=task_type,
        actions=[str(item).strip() for item in actions if str(item).strip()],
        landmarks=[str(item).strip() for item in landmarks if str(item).strip()],
        directional_clauses=[str(item).strip() for item in directional_clauses if str(item).strip()],
        goal_phrase=str(goal_phrase).strip(),
        route_steps=route_steps,
        parsing_source="heuristic",
        parsing_reason="ok",
    )


def _context_from_payload(
    *,
    raw_text: str,
    normalized: str,
    payload: dict,
    fallback: InstructionContext,
    parsing_source: str,
    parsing_reason: str,
) -> InstructionContext:
    actions = _normalize_text_list(payload.get("actions")) or list(fallback.actions)
    landmarks = _normalize_text_list(payload.get("landmarks")) or list(fallback.landmarks)
    directional_clauses = _normalize_text_list(payload.get("directional_clauses")) or list(fallback.directional_clauses)
    route_steps, extra_directionals = _normalize_route_steps(payload.get("route_steps"))
    if extra_directionals:
        for d in extra_directionals:
            if d not in directional_clauses:
                directional_clauses.append(d)
    if not route_steps:
        route_steps = _heuristic_route_steps(actions=actions, landmarks=landmarks, normalized=normalized)
    if not landmarks:
        landmarks = [step.landmark for step in route_steps if step.landmark]
    landmarks = [lm for lm in landmarks if _is_valid_landmark(lm)]
    landmarks = _dedupe_keep_order(landmarks)
    if not actions:
        actions = [step.directive for step in route_steps if step.directive]
    goal_phrase = str(payload.get("goal_phrase", "")).strip()
    if not goal_phrase:
        goal_phrase = route_steps[0].landmark if route_steps else (landmarks[0] if landmarks else fallback.goal_phrase)
    task_type = str(payload.get("task_type", "")).strip().lower()
    if task_type not in {"objectnav", "vln"}:
        task_type = "objectnav" if len(route_steps) <= 1 else "vln"
    instruction_id = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "instruction"
    return InstructionContext(
        raw_text=str(raw_text),
        normalized_text=normalized,
        instruction_id=instruction_id,
        task_type=task_type,
        actions=actions,
        landmarks=landmarks,
        directional_clauses=directional_clauses,
        goal_phrase=str(goal_phrase).strip(),
        route_steps=route_steps,
        parsing_source=str(parsing_source),
        parsing_reason=str(parsing_reason),
    )


def _normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _extract_ordered_clauses(text: str) -> list[str]:
    clauses = re.split(r"\b(?:and then|after that|then|next)\b|[,;]", str(text), flags=re.IGNORECASE)
    return [clause.strip(" .") for clause in clauses if clause.strip(" .")]


def _extract_landmarks(text: str) -> list[str]:
    lowered = str(text).lower()
    known_landmarks = (
        "wheelchair",
        "stairs",
        "stair",
        "door",
        "chair",
        "bed",
        "desk",
        "table",
        "elevator",
        "hallway",
        "corridor",
    )
    landmarks: list[str] = []
    for landmark in known_landmarks:
        if re.search(rf"\b{re.escape(landmark)}s?\b", lowered):
            normalized = "stairs" if landmark == "stair" else landmark
            if normalized not in landmarks:
                landmarks.append(normalized)
    return landmarks


def _heuristic_route_steps(*, actions: list[str], landmarks: list[str], normalized: str) -> list[RouteStep]:
    if not actions and not landmarks:
        return [RouteStep(landmark=normalized, directive=normalized, behavior_type="go_to")]
    steps: list[RouteStep] = []
    remaining_landmarks = [lm for lm in landmarks if _is_valid_landmark(lm)]
    previous_landmark = ""
    for clause in actions:
        clause_lower = clause.lower()
        clause_landmark = ""
        for landmark in remaining_landmarks:
            if landmark and landmark in clause_lower:
                clause_landmark = landmark
                break
        if not clause_landmark and remaining_landmarks:
            clause_landmark = remaining_landmarks[0]
        if clause_landmark in remaining_landmarks:
            remaining_landmarks.remove(clause_landmark)
        behavior = _infer_behavior_type(str(clause))
        # turn_at with no valid landmark → skip (becomes directional only)
        if behavior == "turn_at" and not _is_valid_landmark(clause_landmark):
            continue
        if _is_valid_landmark(clause_landmark):
            steps.append(
                RouteStep(
                    landmark=clause_landmark,
                    directive=str(clause).strip(),
                    behavior_type=behavior,
                    reference_landmark=previous_landmark,
                    relation_direction=_infer_relation_direction(str(clause)),
                )
            )
            previous_landmark = clause_landmark
    for landmark in remaining_landmarks:
        if _is_valid_landmark(landmark):
            steps.append(
                RouteStep(
                    landmark=landmark,
                    directive=f"go to the {landmark}",
                    behavior_type="go_to",
                    reference_landmark=previous_landmark,
                    relation_direction="none",
                )
            )
            previous_landmark = landmark
    if not steps and normalized:
        steps.append(RouteStep(landmark=normalized, directive=normalized, behavior_type="go_to"))
    return steps


def _infer_behavior_type(clause: str) -> str:
    lowered = str(clause).lower()
    if re.search(r"\b(pass|go past|past|go through|through)\b", lowered):
        return "pass"
    if re.search(r"\b(turn at|at the)\b", lowered) and re.search(r"\bturn\b", lowered):
        return "turn_at"
    return "go_to"


def _infer_relation_direction(clause: str) -> str:
    lowered = str(clause).strip().lower()
    has_left = bool(re.search(r"\b(left|turn left)\b|左", lowered))
    has_right = bool(re.search(r"\b(right|turn right)\b|右", lowered))
    has_forward = bool(re.search(r"\b(forward|straight|ahead|continue)\b|前|直", lowered))
    if has_left and has_forward:
        return "left_then_forward"
    if has_right and has_forward:
        return "right_then_forward"
    if has_left:
        return "left"
    if has_right:
        return "right"
    if re.search(r"\b(through|go through)\b|穿过|通过", lowered):
        return "through"
    if re.search(r"\b(pass|past|go past)\b|经过|绕过", lowered):
        return "past"
    if has_forward:
        return "forward"
    return "none"


def _normalize_text_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return _dedupe_keep_order(result)


def _normalize_route_steps(value) -> tuple[list[RouteStep], list[str]]:
    """Return (valid_steps, extracted_directional_clauses).

    turn_at steps with invalid landmarks are converted to directional
    clauses rather than route steps.
    """
    if not isinstance(value, list):
        return [], []
    steps: list[RouteStep] = []
    extracted_directionals: list[str] = []
    previous_landmark = ""
    for item in value:
        if not isinstance(item, dict):
            continue
        landmark = str(item.get("landmark", "")).strip().lower()
        directive = str(item.get("directive", "")).strip()
        behavior_type = str(item.get("behavior_type", "go_to")).strip().lower() or "go_to"
        reference_landmark = str(item.get("reference_landmark", "")).strip().lower()
        relation_direction = str(item.get("relation_direction", "")).strip().lower()
        if behavior_type not in {"go_to", "pass", "turn_at", "go_through"}:
            behavior_type = "go_to"
        if not landmark and not directive:
            continue
        if not landmark:
            landmark = directive.lower()
        if not directive:
            directive = f"go to the {landmark}"
        # turn_at with invalid landmark → convert to directional clause
        if behavior_type == "turn_at" and not _is_valid_landmark(landmark):
            if directive and directive not in extracted_directionals:
                extracted_directionals.append(directive)
            continue
        # skip steps with invalid landmarks
        if not _is_valid_landmark(landmark):
            continue
        if not reference_landmark:
            reference_landmark = previous_landmark
        allowed_relations = {
            "none", "left", "right", "forward", "left_then_forward",
            "right_then_forward", "through", "past",
        }
        if relation_direction not in allowed_relations:
            relation_direction = _infer_relation_direction(directive)
        steps.append(
            RouteStep(
                landmark=landmark,
                directive=directive,
                behavior_type=behavior_type,
                reference_landmark=reference_landmark,
                relation_direction=relation_direction or "none",
            )
        )
        previous_landmark = landmark
    return steps, extracted_directionals


_INVALID_LANDMARKS: set[str] = {
    "none", "null", "unknown", "n/a", "na", "target", "object", "thing",
    "place", "location", "area", "destination", "goal", "",
}


def _is_valid_landmark(label: str) -> bool:
    text = str(label).strip().lower()
    return bool(text) and text not in _INVALID_LANDMARKS and len(text) >= 2


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(str(item).strip())
    return result


def _extract_json_object(text: str) -> dict:
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("json object not found")
    return json.loads(candidate[start : end + 1])
