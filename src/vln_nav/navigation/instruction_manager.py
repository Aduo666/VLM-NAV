from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


INSTRUCTION_SCHEMA_VERSION = "3.1"
INSTRUCTION_PROMPT_VERSION = "2026-07-23.2"
SUPPORTED_STEP_TYPES = {"GO_TO", "TRAVERSE", "REGION_TRANSITION", "TURN"}
SUPPORTED_DIRECTIONS = {"none", "left", "right", "forward"}
SUPPORTED_CUE_ROLES = {"PASS_CUE", "TURN_CUE", "TRANSITION_CUE", "CONTEXT_CUE"}
SUPPORTED_CUE_SIDES = {"none", "left", "right", "center"}
SUPPORTED_RELATIONS = {
    "LEFT_OF",
    "RIGHT_OF",
    "IN_FRONT_OF",
    "BEHIND_TARGET",
    "NEAR",
}
SUPPORTED_TRAVERSE_COMPLETIONS = {
    "CUE_PASSED",
    "REGION_BOUNDARY",
    "NEXT_EVENT_CONFIRMED",
}


class InstructionPlanError(RuntimeError):
    """Raised when no validated semantic instruction graph is available."""


@dataclass(frozen=True)
class SemanticEntity:
    canonical_label: str
    source_span: str


@dataclass(frozen=True)
class SceneContext:
    current_region: SemanticEntity | None = None
    context_objects: tuple[SemanticEntity, ...] = ()


@dataclass(frozen=True)
class StepTrigger:
    trigger_type: str
    label: str = ""


@dataclass(frozen=True)
class RouteCue:
    label: str
    role: str
    expected_side: str = "none"
    source_span: str = ""


@dataclass(frozen=True)
class SemanticRelation:
    target: str
    relation: str
    reference: str
    source_span: str = ""


@dataclass(frozen=True)
class RouteStep:
    step_id: str
    step_type: str
    source_span: str
    target: str = ""
    from_region: str = ""
    to_region: str = ""
    transition: str = ""
    direction: str = "none"
    trigger: StepTrigger | None = None
    completion: str = ""
    cues: tuple[RouteCue, ...] = ()
    relations: tuple[SemanticRelation, ...] = ()
    constraints: tuple[str, ...] = ()

    @property
    def perception_target(self) -> str:
        if self.step_type == "GO_TO":
            return self.target
        for cue in self.cues:
            if _is_groundable_label(cue.label):
                return cue.label
        if self.step_type == "TURN" and self.trigger is not None:
            if (
                self.trigger.trigger_type == "NEAR_OBJECT"
                and _is_groundable_label(self.trigger.label)
            ):
                return self.trigger.label
        return ""

    @property
    def requires_object_grounding(self) -> bool:
        return bool(self.perception_target)


@dataclass(frozen=True)
class InstructionContext:
    raw_text: str
    normalized_text: str
    instruction_id: str
    task_type: str
    scene_context: SceneContext
    navigation_targets: tuple[SemanticEntity, ...]
    route_steps: tuple[RouteStep, ...]
    parsing_source: str
    parsing_reason: str
    schema_version: str = INSTRUCTION_SCHEMA_VERSION

    @property
    def navigation_target_labels(self) -> list[str]:
        return [entity.canonical_label for entity in self.navigation_targets]


class InstructionManager:
    """Compile one English navigation instruction into a validated semantic graph."""

    def __init__(
        self,
        *,
        llm_client,
        logger: logging.Logger,
        max_tokens: int = 768,
        cache_enabled: bool = True,
        cache_path: str | Path = ".cache/instruction_plans.json",
    ) -> None:
        self._llm = llm_client
        self._logger = logger.getChild("instruction_manager")
        self._max_tokens = max(256, int(max_tokens))
        self._cache_enabled = bool(cache_enabled)
        self._cache_path = Path(cache_path).expanduser()
        self._model_name = str(getattr(llm_client, "model_name", "")).strip()

    def parse(self, raw_instruction: str) -> InstructionContext:
        normalized = _normalize_instruction(raw_instruction)
        if not normalized:
            raise InstructionPlanError("[INSTRUCTION_FATAL] instruction is empty")

        cache_key = _instruction_cache_key(normalized, model_name=self._model_name)
        cached = self._load_cached(cache_key=cache_key, raw_instruction=raw_instruction)
        if cached is not None:
            self._logger.info(
                "[PLAN] source=cache task_type=%s steps=%s context_region=%s context_objects=%s targets=%s",
                cached.task_type,
                _step_summaries(cached.route_steps),
                _region_label(cached.scene_context),
                [item.canonical_label for item in cached.scene_context.context_objects],
                cached.navigation_target_labels,
            )
            return cached

        if self._llm is None or not bool(getattr(self._llm, "available", False)):
            raise InstructionPlanError(
                "[INSTRUCTION_FATAL] no validated cache and instruction LLM is unavailable"
            )

        result = self._llm.chat(
            request_name="instruction_parse",
            system_prompt=_instruction_system_prompt(),
            user_prompt=_instruction_user_prompt(normalized),
            image_png_bytes=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
            json_response=True,
        )
        if result.failed or not str(result.content).strip():
            raise InstructionPlanError(
                f"[INSTRUCTION_FATAL] instruction parse failed reason={result.reason}"
            )

        raw_response = str(result.content)
        try:
            payload = _extract_json_object(raw_response)
            protected_destinations = _declared_destination_labels(payload)
            context, errors = _context_from_payload(
                raw_instruction=raw_instruction,
                normalized=normalized,
                payload=payload,
                parsing_source="llm",
                parsing_reason="ok",
            )
        except InstructionPlanError as exc:
            payload = {"raw_response": raw_response}
            protected_destinations = []
            context = None
            errors = [str(exc)]
        if errors:
            context = self._repair_once(
                raw_instruction=raw_instruction,
                normalized=normalized,
                original_payload=payload,
                validation_errors=errors,
                protected_destinations=protected_destinations,
            )
        if context is None:
            raise InstructionPlanError("[INSTRUCTION_FATAL] instruction plan is unavailable")

        self._store_cached(cache_key=cache_key, context=context)
        self._logger.info(
            "[PLAN] source=%s task_type=%s steps=%s context_region=%s context_objects=%s targets=%s",
            context.parsing_source,
            context.task_type,
            _step_summaries(context.route_steps),
            _region_label(context.scene_context),
            [item.canonical_label for item in context.scene_context.context_objects],
            context.navigation_target_labels,
        )
        return context

    def _repair_once(
        self,
        *,
        raw_instruction: str,
        normalized: str,
        original_payload: dict,
        validation_errors: list[str],
        protected_destinations: list[str],
    ) -> InstructionContext:
        self._logger.warning(
            "[INSTRUCTION_REPAIR] validation_errors=%s",
            validation_errors,
        )
        result = self._llm.chat(
            request_name="instruction_repair",
            system_prompt=_instruction_system_prompt(),
            user_prompt=_instruction_repair_prompt(
                normalized=normalized,
                original_payload=original_payload,
                validation_errors=validation_errors,
                protected_destinations=protected_destinations,
            ),
            image_png_bytes=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
            json_response=True,
        )
        if result.failed or not str(result.content).strip():
            raise InstructionPlanError(
                f"[INSTRUCTION_FATAL] instruction repair failed reason={result.reason}; "
                f"validation_errors={validation_errors}"
            )
        repaired_payload = _extract_json_object(str(result.content))
        repaired_payload, removed_labels = _canonicalize_context_target_overlap(repaired_payload)
        if removed_labels:
            self._logger.warning(
                "[INSTRUCTION_NORMALIZE] removed GO_TO targets from repaired "
                "scene_context.context_objects labels=%s",
                removed_labels,
            )
        context, repaired_errors = _context_from_payload(
            raw_instruction=raw_instruction,
            normalized=normalized,
            payload=repaired_payload,
            parsing_source="llm_repair",
            parsing_reason=(
                "repaired_after_validation_and_context_normalization"
                if removed_labels
                else "repaired_after_validation"
            ),
        )
        if repaired_errors:
            raise InstructionPlanError(
                f"[INSTRUCTION_FATAL] repaired instruction is invalid errors={repaired_errors}"
            )
        repaired_targets = set(context.navigation_target_labels)
        missing_destinations = [
            label for label in protected_destinations if label not in repaired_targets
        ]
        if missing_destinations:
            raise InstructionPlanError(
                "[INSTRUCTION_FATAL] repair removed declared navigation destinations "
                f"missing={missing_destinations}"
            )
        return context

    def _load_cached(self, *, cache_key: str, raw_instruction: str) -> InstructionContext | None:
        if not self._cache_enabled or not self._cache_path.exists():
            return None
        try:
            cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            record = cache.get("plans", {}).get(cache_key)
            if not isinstance(record, dict):
                return None
            if str(record.get("schema_version", "")) != INSTRUCTION_SCHEMA_VERSION:
                return None
            if str(record.get("prompt_version", "")) != INSTRUCTION_PROMPT_VERSION:
                return None
            if str(record.get("model_name", "")) != self._model_name:
                return None
            context, errors = _context_from_payload(
                raw_instruction=raw_instruction,
                normalized=_normalize_instruction(raw_instruction),
                payload=record.get("plan", {}),
                parsing_source="cache",
                parsing_reason="validated_cache",
            )
            if errors:
                self._logger.warning("[INSTRUCTION_CACHE] invalid entry ignored errors=%s", errors)
                return None
            return context
        except Exception as exc:
            self._logger.warning("[INSTRUCTION_CACHE] read_failed path=%s error=%s", self._cache_path, exc)
            return None

    def _store_cached(self, *, cache_key: str, context: InstructionContext) -> None:
        if not self._cache_enabled:
            return
        try:
            cache: dict = {"schema_version": INSTRUCTION_SCHEMA_VERSION, "plans": {}}
            if self._cache_path.exists():
                loaded = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("plans"), dict):
                    cache = loaded
            cache["schema_version"] = INSTRUCTION_SCHEMA_VERSION
            cache["prompt_version"] = INSTRUCTION_PROMPT_VERSION
            cache.setdefault("plans", {})[cache_key] = {
                "instruction": context.normalized_text,
                "schema_version": INSTRUCTION_SCHEMA_VERSION,
                "prompt_version": INSTRUCTION_PROMPT_VERSION,
                "model_name": self._model_name,
                "plan": _context_to_payload(context),
            }
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                prefix=f".{self._cache_path.name}.",
                suffix=".tmp",
                dir=str(self._cache_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(cache, handle, ensure_ascii=True, indent=2)
                    handle.write("\n")
                os.replace(temp_path, self._cache_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            self._logger.warning("[INSTRUCTION_CACHE] write_failed path=%s error=%s", self._cache_path, exc)


def _instruction_system_prompt() -> str:
    return (
        "You are an instruction compiler for embodied vision-language navigation.\n"
        "Convert one English navigation instruction into a small ordered route draft.\n"
        "Return strict JSON only, with no markdown or commentary.\n"
        "Keep scene context separate from commanded destinations.\n"
        "The only supported step types are GO_TO, TRAVERSE, REGION_TRANSITION, and TURN.\n"
        "Any phrase that commands travel to, find, approach, reach, or stop near an object is GO_TO that object.\n"
        "TRAVERSE is only directional travel with no object destination.\n"
        "Cue labels must be physical visible entities, never action phrases such as turn left or continue straight.\n"
        "Preserve every commanded destination and preserve temporal order.\n"
        "Do not invent objects, regions, or actions absent from the instruction."
    )


def _instruction_user_prompt(instruction: str) -> str:
    return (
        f"Instruction:\n{instruction}\n\n"
        "Return this compact JSON draft. Python will derive step ids, triggers, "
        "completion rules, cue roles, and the navigation-target list:\n"
        "{\n"
        f'  "schema_version": "{INSTRUCTION_SCHEMA_VERSION}",\n'
        '  "scene_context": {\n'
        '    "current_region": {"canonical_label": "", "source_span": ""},\n'
        '    "context_objects": [{"canonical_label": "", "source_span": ""}]\n'
        "  },\n"
        '  "route_steps": [\n'
        "    {\n"
        '      "step_type": "GO_TO|TRAVERSE|REGION_TRANSITION|TURN",\n'
        '      "source_span": "exact supporting phrase",\n'
        '      "target": "commanded object destination or empty",\n'
        '      "from_region": "region label or empty",\n'
        '      "to_region": "region label, outside_current_region, adjacent_region, or empty",\n'
        '      "transition": "enter|exit|empty",\n'
        '      "direction": "none|left|right|forward",\n'
        '      "cues": [\n'
        "        {\n"
        '          "label": "physical visible entity only",\n'
        '          "cue_type": "pass|turn|transition|context",\n'
        '          "expected_side": "none|left|right|center",\n'
        '          "source_span": "exact supporting phrase"\n'
        "        }\n"
        "      ],\n"
        '      "relations": [\n'
        "        {\n"
        '          "relation": "LEFT_OF|RIGHT_OF|IN_FRONT_OF|BEHIND_TARGET|NEAR",\n'
        '          "reference": "reference object",\n'
        '          "source_span": "exact supporting phrase"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. scene_context is trusted prior information, not an execution target.\n"
        "2. Objects merely describing the initial scene belong only in context_objects.\n"
        "3. 'find X', 'go to X', 'continue straight to X', 'approach X', and 'stop near X' are GO_TO with target=X.\n"
        "4. TRAVERSE has no target. Use it only for travel such as 'walk down the hallway' when no destination object is commanded.\n"
        "5. 'passing X on the right/left' adds physical cue X with cue_type=pass; it does not make X a GO_TO target.\n"
        "6. Leaving or entering a room is REGION_TRANSITION.\n"
        "7. A directional turn is TURN. Bare 'turn left/right' has cues=[]. "
        "'turn left into the doorway' may use doorway as a physical turn cue.\n"
        "8. Never use action text such as 'turn left', 'continue straight', 'go forward', left, or right as a cue label.\n"
        "9. Relations only disambiguate a GO_TO target; the relation target is inferred from that step.\n"
        "10. Use concise lowercase English labels and exact source spans.\n"
        "11. Use empty lists for absent cues and relations; do not emit empty list objects."
    )


def _instruction_repair_prompt(
    *,
    normalized: str,
    original_payload: dict,
    validation_errors: list[str],
    protected_destinations: list[str],
) -> str:
    return (
        _instruction_user_prompt(normalized)
        + "\nThe previous JSON failed deterministic validation. Repair only invalid fields "
        + "without changing instruction meaning or route order.\n"
        + "Never fix validation by deleting a commanded destination. "
        + f"These previously declared destinations must remain GO_TO steps: "
        + f"{json.dumps(protected_destinations, ensure_ascii=True)}.\n"
        + "A phrase commanding travel to an object must remain GO_TO that object. "
        + "Remove GO_TO targets from scene_context.context_objects instead of deleting route steps.\n"
        + f"Validation errors: {json.dumps(validation_errors, ensure_ascii=True)}\n"
        + f"Previous JSON: {json.dumps(original_payload, ensure_ascii=True)}"
    )


def _declared_destination_labels(payload: dict) -> list[str]:
    """Collect destination evidence before deterministic compilation or repair."""
    if not isinstance(payload, dict):
        return []
    labels: list[str] = []
    for item in payload.get("navigation_targets", []):
        if isinstance(item, dict):
            labels.append(_canonical_label(item.get("canonical_label")))
    for step in payload.get("route_steps", []):
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type", "")).strip().upper()
        if step_type in {"GO_TO", "TRAVERSE"}:
            labels.append(_canonical_label(step.get("target")))
    return _dedupe_keep_order(labels)


def _compile_instruction_payload(payload: dict) -> dict:
    """Compile the compact model draft into the strict internal schema."""
    if not isinstance(payload, dict):
        return payload
    compiled = dict(payload)
    compiled["schema_version"] = INSTRUCTION_SCHEMA_VERSION
    route_payload = payload.get("route_steps")
    if not isinstance(route_payload, list):
        return compiled

    compiled_steps: list = []
    navigation_targets: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for index, raw_step in enumerate(route_payload):
        if not isinstance(raw_step, dict):
            compiled_steps.append(raw_step)
            continue
        step = dict(raw_step)
        step_type = str(step.get("step_type", "")).strip().upper()
        target = _canonical_label(step.get("target"))
        if step_type == "TRAVERSE" and not target:
            target = _infer_traverse_destination(
                source_span=str(step.get("source_span", "")),
                cues=step.get("cues"),
            )
        # A directional segment with an object destination is GO_TO, not TRAVERSE.
        if step_type == "TRAVERSE" and target:
            step_type = "GO_TO"
        step["step_type"] = step_type
        step["step_id"] = _canonical_token(step.get("step_id")) or f"step_{index + 1}"
        step["source_span"] = str(step.get("source_span", "")).strip()
        step["target"] = target
        step["from_region"] = _canonical_label(step.get("from_region"))
        step["to_region"] = _canonical_label(step.get("to_region"))
        step["transition"] = str(step.get("transition", "")).strip().lower()
        step["direction"] = (
            str(step.get("direction", "none")).strip().lower() or "none"
        )
        if (
            step_type == "GO_TO"
            and step["direction"] == "none"
            and re.search(r"\b(?:forward|straight)\b", step["source_span"], re.IGNORECASE)
        ):
            step["direction"] = "forward"

        trigger = step.get("trigger")
        if not isinstance(trigger, dict):
            trigger = {
                "trigger_type": "NONE" if index == 0 else "AFTER_PREVIOUS_STEP",
                "label": "",
            }
        step["trigger"] = trigger

        cues = _compile_cues(
            step.get("cues"),
            step_type=step_type,
            to_region=step["to_region"],
            source_span=step["source_span"],
        )
        relations = _compile_relations(step.get("relations"), target=target)
        if step_type == "GO_TO":
            cues = []
            step["completion"] = "SEMANTIC_AND_DISTANCE"
            if target and target not in seen_targets:
                navigation_targets.append(
                    {
                        "canonical_label": target,
                        "source_span": step["source_span"],
                    }
                )
                seen_targets.add(target)
        elif step_type == "TRAVERSE":
            step["completion"] = (
                "CUE_PASSED"
                if any(item.get("role") == "PASS_CUE" for item in cues)
                else "NEXT_EVENT_CONFIRMED"
            )
            relations = []
        elif step_type == "REGION_TRANSITION":
            step["completion"] = (
                "REGION_CHANGED"
                if step["to_region"]
                in {"", "outside_current_region", "adjacent_region", "unknown"}
                else "NAMED_REGION_CONFIRMED"
            )
            relations = []
        elif step_type == "TURN":
            step["completion"] = "HEADING_ALIGNED"
            relations = []
        step["cues"] = cues
        step["relations"] = relations
        step["constraints"] = _clean_string_list(step.get("constraints"))
        compiled_steps.append(step)

    compiled["route_steps"] = compiled_steps
    compiled["navigation_targets"] = navigation_targets
    return compiled


def _compile_cues(
    value,
    *,
    step_type: str,
    to_region: str,
    source_span: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    compiled: list[dict[str, str]] = []
    role_by_type = {
        "pass": "PASS_CUE",
        "turn": "TURN_CUE",
        "transition": "TRANSITION_CUE",
        "context": "CONTEXT_CUE",
    }
    for item in value:
        if not isinstance(item, dict):
            compiled.append(item)
            continue
        label = _canonical_label(item.get("label"))
        if not _is_groundable_label(label):
            continue
        if step_type == "TURN" and label not in _canonical_label(source_span):
            continue
        role = str(item.get("role", "")).strip().upper()
        if role not in SUPPORTED_CUE_ROLES:
            role = role_by_type.get(
                str(item.get("cue_type", "")).strip().lower(),
                "",
            )
        if step_type == "TURN":
            role = "TURN_CUE"
        elif step_type == "TRAVERSE" and role not in {"PASS_CUE", "CONTEXT_CUE"}:
            role = "PASS_CUE"
        elif step_type == "REGION_TRANSITION" and role not in {
            "PASS_CUE",
            "TRANSITION_CUE",
            "CONTEXT_CUE",
        }:
            role = (
                "TRANSITION_CUE"
                if label == _canonical_label(to_region)
                else "PASS_CUE"
            )
        elif role not in SUPPORTED_CUE_ROLES:
            role = "CONTEXT_CUE"
        compiled.append(
            {
                "label": label,
                "role": role,
                "expected_side": (
                    str(item.get("expected_side", "none")).strip().lower()
                    or "none"
                ),
                "source_span": str(item.get("source_span", "")).strip(),
            }
        )
    return compiled


def _infer_traverse_destination(*, source_span: str, cues) -> str:
    """Recover an object destination that the model misplaced in TRAVERSE.cues."""
    source = _canonical_label(source_span)
    if not source or not isinstance(cues, list):
        return ""
    for item in cues:
        if not isinstance(item, dict):
            continue
        label = _canonical_label(item.get("label"))
        if not _is_groundable_label(label):
            continue
        escaped = re.escape(label)
        if re.search(
            rf"\b(?:to|toward|towards|near|approach|reach|find)\s+(?:the\s+)?{escaped}\b",
            source,
        ):
            return label
    return ""


def _compile_relations(value, *, target: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    compiled: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            compiled.append(item)
            continue
        compiled.append(
            {
                "target": target,
                "relation": str(item.get("relation", "")).strip().upper(),
                "reference": _canonical_label(item.get("reference")),
                "source_span": str(item.get("source_span", "")).strip(),
            }
        )
    return compiled


def _canonicalize_context_target_overlap(payload: dict) -> tuple[dict, list[str]]:
    """Remove a redundant context classification after the single LLM repair attempt."""
    if not isinstance(payload, dict):
        return payload, []
    route_steps = payload.get("route_steps")
    scene_context = payload.get("scene_context")
    if not isinstance(route_steps, list) or not isinstance(scene_context, dict):
        return payload, []
    context_objects = scene_context.get("context_objects")
    if not isinstance(context_objects, list):
        return payload, []

    go_to_labels = {
        _canonical_label(step.get("target"))
        for step in route_steps
        if isinstance(step, dict) and str(step.get("step_type", "")).strip().upper() == "GO_TO"
    }
    go_to_labels.discard("")
    if not go_to_labels:
        return payload, []

    kept_objects: list = []
    removed_labels: list[str] = []
    for item in context_objects:
        label = _canonical_label(item.get("canonical_label")) if isinstance(item, dict) else ""
        if label and label in go_to_labels:
            removed_labels.append(label)
            continue
        kept_objects.append(item)
    if not removed_labels:
        return payload, []

    normalized_payload = dict(payload)
    normalized_scene_context = dict(scene_context)
    normalized_scene_context["context_objects"] = kept_objects
    normalized_payload["scene_context"] = normalized_scene_context
    return normalized_payload, _dedupe_keep_order(removed_labels)


def _context_from_payload(
    *,
    raw_instruction: str,
    normalized: str,
    payload: dict,
    parsing_source: str,
    parsing_reason: str,
) -> tuple[InstructionContext, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        payload = {}
        errors.append("root must be an object")
    payload = _compile_instruction_payload(payload)
    if str(payload.get("schema_version", "")).strip() != INSTRUCTION_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {INSTRUCTION_SCHEMA_VERSION}"
        )

    scene_payload = payload.get("scene_context", {})
    if not isinstance(scene_payload, dict):
        scene_payload = {}
        errors.append("scene_context must be an object")
    current_region = _entity_from_value(scene_payload.get("current_region"))
    context_objects = tuple(_entity_list(scene_payload.get("context_objects"), errors, "context_objects"))
    navigation_targets = tuple(
        _entity_list(payload.get("navigation_targets"), errors, "navigation_targets")
    )

    route_payload = payload.get("route_steps")
    if not isinstance(route_payload, list) or not route_payload:
        route_payload = []
        errors.append("route_steps must be a non-empty list")

    route_steps: list[RouteStep] = []
    seen_step_ids: set[str] = set()
    for index, item in enumerate(route_payload):
        prefix = f"route_steps[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        step_id = _canonical_token(item.get("step_id")) or f"step_{index + 1}"
        if step_id in seen_step_ids:
            errors.append(f"{prefix}.step_id must be unique")
        seen_step_ids.add(step_id)
        step_type = str(item.get("step_type", "")).strip().upper()
        if step_type not in SUPPORTED_STEP_TYPES:
            errors.append(f"{prefix}.step_type unsupported: {step_type or 'empty'}")
            continue
        source_span = str(item.get("source_span", "")).strip()
        if not source_span:
            errors.append(f"{prefix}.source_span is required")
        target = _canonical_label(item.get("target"))
        from_region = _canonical_label(item.get("from_region"))
        to_region = _canonical_label(item.get("to_region"))
        transition = str(item.get("transition", "")).strip().lower()
        direction = str(item.get("direction", "none")).strip().lower() or "none"
        if direction not in SUPPORTED_DIRECTIONS:
            errors.append(f"{prefix}.direction unsupported: {direction}")
            direction = "none"
        trigger = _trigger_from_value(item.get("trigger"), errors=errors, prefix=prefix)
        cues = tuple(_cue_list(item.get("cues"), errors=errors, prefix=prefix))
        relations = tuple(_relation_list(item.get("relations"), errors=errors, prefix=prefix))
        constraints = tuple(_clean_string_list(item.get("constraints")))
        requested_completion = str(item.get("completion", "")).strip().upper()

        if step_type == "GO_TO":
            if not target:
                errors.append(f"{prefix}.target is required for GO_TO")
            completion = "SEMANTIC_AND_DISTANCE"
            from_region = ""
            to_region = ""
            transition = ""
            if cues:
                errors.append(f"{prefix}.cues are not supported for GO_TO; use relations")
            for relation in relations:
                if relation.target != target:
                    errors.append(
                        f"{prefix}.relations target must match GO_TO target={target}"
                    )
        elif step_type == "TRAVERSE":
            if direction not in {"forward", "left", "right"}:
                errors.append(f"{prefix}.direction is required for TRAVERSE")
            if requested_completion not in SUPPORTED_TRAVERSE_COMPLETIONS:
                errors.append(
                    f"{prefix}.completion must be one of {sorted(SUPPORTED_TRAVERSE_COMPLETIONS)}"
                )
            completion = requested_completion
            if completion == "CUE_PASSED" and not any(
                cue.role == "PASS_CUE" for cue in cues
            ):
                errors.append(f"{prefix}.CUE_PASSED requires a PASS_CUE")
            invalid_roles = sorted(
                {cue.role for cue in cues if cue.role not in {"PASS_CUE", "CONTEXT_CUE"}}
            )
            if invalid_roles:
                errors.append(
                    f"{prefix}.TRAVERSE cue roles unsupported: {invalid_roles}"
                )
            if relations:
                errors.append(f"{prefix}.relations are only supported for GO_TO")
            target = ""
            from_region = ""
            to_region = ""
            transition = ""
        elif step_type == "REGION_TRANSITION":
            if transition not in {"enter", "exit"}:
                errors.append(f"{prefix}.transition must be enter or exit")
            if not from_region and current_region is not None:
                from_region = current_region.canonical_label
            if not from_region:
                errors.append(f"{prefix}.from_region is required")
            if not to_region:
                to_region = "outside_current_region" if transition == "exit" else "adjacent_region"
            completion = (
                "REGION_CHANGED"
                if to_region in {"outside_current_region", "adjacent_region", "unknown"}
                else "NAMED_REGION_CONFIRMED"
            )
            target = ""
            if relations:
                errors.append(f"{prefix}.relations are only supported for GO_TO")
            invalid_roles = sorted(
                {
                    cue.role
                    for cue in cues
                    if cue.role not in {"PASS_CUE", "TRANSITION_CUE", "CONTEXT_CUE"}
                }
            )
            if invalid_roles:
                errors.append(
                    f"{prefix}.REGION_TRANSITION cue roles unsupported: {invalid_roles}"
                )
        else:
            if direction not in {"left", "right"}:
                errors.append(f"{prefix}.direction must be left or right for TURN")
            completion = "HEADING_ALIGNED"
            target = ""
            from_region = ""
            to_region = ""
            transition = ""
            if relations:
                errors.append(f"{prefix}.relations are only supported for GO_TO")
            invalid_roles = sorted(
                {cue.role for cue in cues if cue.role != "TURN_CUE"}
            )
            if invalid_roles:
                errors.append(f"{prefix}.TURN cue roles unsupported: {invalid_roles}")

        route_steps.append(
            RouteStep(
                step_id=step_id,
                step_type=step_type,
                source_span=source_span,
                target=target,
                from_region=from_region,
                to_region=to_region,
                transition=transition,
                direction=direction,
                trigger=trigger,
                completion=completion,
                cues=cues,
                relations=relations,
                constraints=constraints,
            )
        )

    target_labels = [entity.canonical_label for entity in navigation_targets]
    go_to_labels = [step.target for step in route_steps if step.step_type == "GO_TO" and step.target]
    if target_labels != _dedupe_keep_order(go_to_labels):
        errors.append(
            "navigation_targets must exactly match GO_TO targets in first-use order "
            f"expected={_dedupe_keep_order(go_to_labels)} actual={target_labels}"
        )
    context_labels = {entity.canonical_label for entity in context_objects}
    overlapping_labels = sorted(context_labels.intersection(go_to_labels))
    if overlapping_labels:
        errors.append(
            "scene_context.context_objects must exclude GO_TO navigation targets "
            f"overlap={overlapping_labels}"
        )

    task_type = "objectnav" if len(route_steps) == 1 and route_steps[0].step_type == "GO_TO" else "vln"
    instruction_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    context = InstructionContext(
        raw_text=str(raw_instruction),
        normalized_text=normalized,
        instruction_id=instruction_id,
        task_type=task_type,
        scene_context=SceneContext(
            current_region=current_region,
            context_objects=context_objects,
        ),
        navigation_targets=navigation_targets,
        route_steps=tuple(route_steps),
        parsing_source=parsing_source,
        parsing_reason=parsing_reason,
    )
    return context, _dedupe_keep_order(errors)


def _entity_from_value(value) -> SemanticEntity | None:
    if not isinstance(value, dict):
        return None
    label = _canonical_label(value.get("canonical_label"))
    if not label:
        return None
    return SemanticEntity(
        canonical_label=label,
        source_span=str(value.get("source_span", "")).strip(),
    )


def _entity_list(value, errors: list[str], field_name: str) -> list[SemanticEntity]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return []
    result: list[SemanticEntity] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        entity = _entity_from_value(item)
        if entity is None:
            errors.append(f"{field_name}[{index}] requires canonical_label")
            continue
        if entity.canonical_label in seen:
            continue
        seen.add(entity.canonical_label)
        result.append(entity)
    return result


def _trigger_from_value(value, *, errors: list[str], prefix: str) -> StepTrigger | None:
    if value is None or value == "":
        return None
    if not isinstance(value, dict):
        errors.append(f"{prefix}.trigger must be an object")
        return None
    trigger_type = str(value.get("trigger_type", "NONE")).strip().upper() or "NONE"
    if trigger_type == "NONE":
        return None
    if trigger_type not in {"NEAR_OBJECT", "AFTER_PREVIOUS_STEP"}:
        errors.append(f"{prefix}.trigger_type unsupported: {trigger_type}")
        return None
    label = _canonical_label(value.get("label"))
    if trigger_type == "NEAR_OBJECT" and not label:
        errors.append(f"{prefix}.trigger.label is required for NEAR_OBJECT")
    return StepTrigger(trigger_type=trigger_type, label=label)


def _cue_list(value, *, errors: list[str], prefix: str) -> list[RouteCue]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{prefix}.cues must be a list")
        return []
    result: list[RouteCue] = []
    for index, item in enumerate(value):
        cue_prefix = f"{prefix}.cues[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{cue_prefix} must be an object")
            continue
        label = _canonical_label(item.get("label"))
        role = str(item.get("role", "")).strip().upper()
        expected_side = str(item.get("expected_side", "none")).strip().lower() or "none"
        source_span = str(item.get("source_span", "")).strip()
        if not label:
            errors.append(f"{cue_prefix}.label is required")
        if role not in SUPPORTED_CUE_ROLES:
            errors.append(f"{cue_prefix}.role unsupported: {role or 'empty'}")
        if expected_side not in SUPPORTED_CUE_SIDES:
            errors.append(f"{cue_prefix}.expected_side unsupported: {expected_side}")
        if not source_span:
            errors.append(f"{cue_prefix}.source_span is required")
        if label and role in SUPPORTED_CUE_ROLES and expected_side in SUPPORTED_CUE_SIDES:
            result.append(
                RouteCue(
                    label=label,
                    role=role,
                    expected_side=expected_side,
                    source_span=source_span,
                )
            )
    return result


def _relation_list(
    value,
    *,
    errors: list[str],
    prefix: str,
) -> list[SemanticRelation]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{prefix}.relations must be a list")
        return []
    result: list[SemanticRelation] = []
    for index, item in enumerate(value):
        relation_prefix = f"{prefix}.relations[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{relation_prefix} must be an object")
            continue
        target = _canonical_label(item.get("target"))
        relation = str(item.get("relation", "")).strip().upper()
        reference = _canonical_label(item.get("reference"))
        source_span = str(item.get("source_span", "")).strip()
        if not target:
            errors.append(f"{relation_prefix}.target is required")
        if relation not in SUPPORTED_RELATIONS:
            errors.append(
                f"{relation_prefix}.relation unsupported: {relation or 'empty'}"
            )
        if not reference:
            errors.append(f"{relation_prefix}.reference is required")
        if not source_span:
            errors.append(f"{relation_prefix}.source_span is required")
        if target and relation in SUPPORTED_RELATIONS and reference:
            result.append(
                SemanticRelation(
                    target=target,
                    relation=relation,
                    reference=reference,
                    source_span=source_span,
                )
            )
    return result


def _context_to_payload(context: InstructionContext) -> dict:
    return {
        "schema_version": context.schema_version,
        "scene_context": {
            "current_region": (
                {"canonical_label": "", "source_span": ""}
                if context.scene_context.current_region is None
                else asdict(context.scene_context.current_region)
            ),
            "context_objects": [asdict(item) for item in context.scene_context.context_objects],
        },
        "navigation_targets": [asdict(item) for item in context.navigation_targets],
        "route_steps": [
            {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "source_span": step.source_span,
                "target": step.target,
                "from_region": step.from_region,
                "to_region": step.to_region,
                "transition": step.transition,
                "direction": step.direction,
                "trigger": (
                    {"trigger_type": "NONE", "label": ""}
                    if step.trigger is None
                    else asdict(step.trigger)
                ),
                "completion": step.completion,
                "cues": [asdict(item) for item in step.cues],
                "relations": [asdict(item) for item in step.relations],
                "constraints": list(step.constraints),
            }
            for step in context.route_steps
        ],
    }


def _instruction_cache_key(normalized: str, *, model_name: str) -> str:
    material = (
        f"{INSTRUCTION_SCHEMA_VERSION}|{INSTRUCTION_PROMPT_VERSION}|"
        f"{model_name}|{normalized}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _canonical_label(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_groundable_label(value) -> bool:
    label = _canonical_label(value)
    if not label or label in {
        "left",
        "right",
        "forward",
        "backward",
        "straight",
        "stop",
    }:
        return False
    return re.match(
        r"^(?:turn|go|move|walk|continue|proceed|head|rotate|stop|enter|exit)\b",
        label,
    ) is None


def _canonical_token(value) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _clean_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _region_label(scene_context: SceneContext) -> str:
    if scene_context.current_region is None:
        return "unknown"
    return scene_context.current_region.canonical_label


def _step_summaries(route_steps: tuple[RouteStep, ...]) -> list[str]:
    summaries: list[str] = []
    for step in route_steps:
        subject = step.target or step.to_region or step.direction
        summaries.append(f"{step.step_id}:{step.step_type}:{subject}")
    return summaries


def _extract_json_object(text: str) -> dict:
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise InstructionPlanError("[INSTRUCTION_FATAL] JSON object not found in instruction response")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InstructionPlanError(f"[INSTRUCTION_FATAL] invalid instruction JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InstructionPlanError("[INSTRUCTION_FATAL] instruction response root must be an object")
    return payload
