from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field


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


class InstructionManager:
    def __init__(self, *, llm_client, logger: logging.Logger) -> None:
        self._logger = logger.getChild("instruction_manager")

    def parse(self, raw_instruction: str) -> InstructionContext:
        normalized = _normalize_instruction(raw_instruction)
        # 指令原文仍交给主决策 VLM；这里只做轻量本地摘要，不额外发一次 LLM 请求。
        actions = _extract_ordered_clauses(normalized)
        landmarks = _extract_landmarks(normalized)
        directional_clauses = [
            clause for clause in actions if re.search(r"\b(left|right|forward|backward|back|straight|toward|to)\b", clause)
        ]
        goal_phrase = landmarks[0] if landmarks else normalized
        instruction_id = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "instruction"
        return InstructionContext(
            raw_text=str(raw_instruction),
            normalized_text=normalized,
            instruction_id=instruction_id,
            task_type="vln",
            actions=[str(item).strip() for item in actions if str(item).strip()],
            landmarks=[str(item).strip() for item in landmarks if str(item).strip()],
            directional_clauses=[str(item).strip() for item in directional_clauses if str(item).strip()],
            goal_phrase=goal_phrase,
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
