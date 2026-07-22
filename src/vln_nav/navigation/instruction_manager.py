from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


INSTRUCTION_SCHEMA_VERSION = "2.0"
INSTRUCTION_PROMPT_VERSION = "2026-07-22.1"
SUPPORTED_STEP_TYPES = {"GO_TO", "REGION_TRANSITION", "TURN"}
SUPPORTED_DIRECTIONS = {"none", "left", "right", "forward"}


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
    constraints: tuple[str, ...] = ()

    @property
    def perception_target(self) -> str:
        if self.step_type == "GO_TO":
            return self.target
        if self.step_type == "TURN" and self.trigger is not None:
            if self.trigger.trigger_type == "NEAR_OBJECT":
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

    def parse(self, raw_instruction: str) -> InstructionContext:
        normalized = _normalize_instruction(raw_instruction)
        if not normalized:
            raise InstructionPlanError("[INSTRUCTION_FATAL] instruction is empty")

        cache_key = _instruction_cache_key(normalized)
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
        )
        if result.failed or not str(result.content).strip():
            raise InstructionPlanError(
                f"[INSTRUCTION_FATAL] instruction parse failed reason={result.reason}"
            )

        raw_response = str(result.content)
        try:
            payload = _extract_json_object(raw_response)
            context, errors = _context_from_payload(
                raw_instruction=raw_instruction,
                normalized=normalized,
                payload=payload,
                parsing_source="llm",
                parsing_reason="ok",
            )
        except InstructionPlanError as exc:
            payload = {"raw_response": raw_response}
            context = None
            errors = [str(exc)]
        if errors:
            context = self._repair_once(
                raw_instruction=raw_instruction,
                normalized=normalized,
                original_payload=payload,
                validation_errors=errors,
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
            ),
            image_png_bytes=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        if result.failed or not str(result.content).strip():
            raise InstructionPlanError(
                f"[INSTRUCTION_FATAL] instruction repair failed reason={result.reason}; "
                f"validation_errors={validation_errors}"
            )
        repaired_payload = _extract_json_object(str(result.content))
        context, repaired_errors = _context_from_payload(
            raw_instruction=raw_instruction,
            normalized=normalized,
            payload=repaired_payload,
            parsing_source="llm_repair",
            parsing_reason="repaired_after_validation",
        )
        if repaired_errors:
            raise InstructionPlanError(
                f"[INSTRUCTION_FATAL] repaired instruction is invalid errors={repaired_errors}"
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
        "Convert one English navigation instruction into a typed semantic execution graph.\n"
        "Return strict JSON only, with no markdown or commentary.\n"
        "Never turn scene-description objects into navigation targets unless the instruction explicitly commands the robot to find, approach, or navigate to them.\n"
        "Distinguish trusted starting context, context objects, action triggers, regions, and true navigation targets.\n"
        "The only supported step types are GO_TO, REGION_TRANSITION, and TURN.\n"
        "Preserve temporal order and explicit directional relations.\n"
        "Do not invent objects, regions, or actions absent from the instruction."
    )


def _instruction_user_prompt(instruction: str) -> str:
    return (
        f"Instruction:\n{instruction}\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        f'  "schema_version": "{INSTRUCTION_SCHEMA_VERSION}",\n'
        '  "scene_context": {\n'
        '    "current_region": {"canonical_label": "", "source_span": ""},\n'
        '    "context_objects": [{"canonical_label": "", "source_span": ""}]\n'
        "  },\n"
        '  "navigation_targets": [{"canonical_label": "", "source_span": ""}],\n'
        '  "route_steps": [\n'
        "    {\n"
        '      "step_id": "step_1",\n'
        '      "step_type": "GO_TO|REGION_TRANSITION|TURN",\n'
        '      "source_span": "exact supporting phrase",\n'
        '      "target": "object label or empty",\n'
        '      "from_region": "region label or empty",\n'
        '      "to_region": "region label, outside_current_region, adjacent_region, or empty",\n'
        '      "transition": "enter|exit|empty",\n'
        '      "direction": "none|left|right|forward",\n'
        '      "trigger": {"trigger_type": "NONE|NEAR_OBJECT|AFTER_PREVIOUS_STEP", "label": ""},\n'
        '      "completion": "SEMANTIC_AND_DISTANCE|REGION_CHANGED|NAMED_REGION_CONFIRMED|HEADING_ALIGNED",\n'
        '      "constraints": ["short global semantic constraint"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. scene_context is trusted prior information, not an execution target.\n"
        "2. Objects merely described as nearby belong only in context_objects. Never duplicate a GO_TO target in context_objects.\n"
        "3. find/go to/navigate to/approach an object becomes GO_TO with SEMANTIC_AND_DISTANCE.\n"
        "4. Leaving or entering a room becomes REGION_TRANSITION.\n"
        "5. A directional turn becomes TURN; if it follows another step, use AFTER_PREVIOUS_STEP.\n"
        "6. 'turn at/near object' uses NEAR_OBJECT and does not make that object a GO_TO target.\n"
        "7. 'go straight to X' is GO_TO(X) with direction=forward, not an infinite forward step.\n"
        "8. navigation_targets contains exactly the GO_TO targets, in first-use order.\n"
        "9. Use concise lowercase English canonical labels.\n"
        "10. If no current region is stated, return empty strings for current_region."
    )


def _instruction_repair_prompt(
    *,
    normalized: str,
    original_payload: dict,
    validation_errors: list[str],
) -> str:
    return (
        _instruction_user_prompt(normalized)
        + "\nThe previous JSON failed deterministic validation. Repair it without changing the instruction meaning.\n"
        + f"Validation errors: {json.dumps(validation_errors, ensure_ascii=True)}\n"
        + f"Previous JSON: {json.dumps(original_payload, ensure_ascii=True)}"
    )


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
        constraints = tuple(_clean_string_list(item.get("constraints")))

        if step_type == "GO_TO":
            if not target:
                errors.append(f"{prefix}.target is required for GO_TO")
            completion = "SEMANTIC_AND_DISTANCE"
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
        else:
            if direction not in {"left", "right"}:
                errors.append(f"{prefix}.direction must be left or right for TURN")
            completion = "HEADING_ALIGNED"
            target = ""
            from_region = ""
            to_region = ""
            transition = ""

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
                "constraints": list(step.constraints),
            }
            for step in context.route_steps
        ],
    }


def _instruction_cache_key(normalized: str) -> str:
    material = f"{INSTRUCTION_SCHEMA_VERSION}|{INSTRUCTION_PROMPT_VERSION}|{normalized}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _canonical_label(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


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
