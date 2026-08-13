from __future__ import annotations

import re

from .schemas import PerceptionQuery


_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_REGION_LABELS = {
    "bathroom",
    "bedroom",
    "corridor",
    "doorway",
    "entrance",
    "hall",
    "hallway",
    "kitchen",
    "lobby",
    "room",
    "staircase",
}
_KNOWN_LABELS = (
    "wall mounted tv",
    "patterned glass door",
    "dark wooden cabinet",
    "yellow warning sign",
    "dining table",
    "coffee table",
    "main hall",
    "open kitchen",
    "glass door",
    "warning sign",
    "wheelchair",
    "staircase",
    "stairs",
    "cabinet",
    "column",
    "doorway",
    "entrance",
    "kitchen",
    "bedroom",
    "bathroom",
    "hallway",
    "corridor",
    "lobby",
    "window",
    "chair",
    "table",
    "sofa",
    "couch",
    "bed",
    "door",
    "television",
    "tv",
)
_CANONICAL_ALIASES = {
    "stairs": "staircase",
    "television": "tv",
    "open kitchen": "kitchen",
    "main hall": "hall",
    "dark wooden cabinet": "cabinet",
    "wall mounted tv": "tv",
    "patterned glass door": "door",
    "glass door": "door",
    "yellow warning sign": "warning sign",
}


class QueryCompiler:
    """Compile open-vocabulary visual queries without an instruction LLM."""

    def compile(self, instruction: str) -> tuple[PerceptionQuery, ...]:
        words = [match.group(0) for match in _TOKEN_PATTERN.finditer(str(instruction))]
        lowered = [word.lower() for word in words]
        normalized_words = [_singular_token(word) for word in lowered]
        candidates: list[tuple[int, int, str, str]] = []
        for phrase in _KNOWN_LABELS:
            phrase_words = phrase.split()
            width = len(phrase_words)
            for start in range(0, len(lowered) - width + 1):
                candidate_words = normalized_words[start : start + width]
                if candidate_words == phrase_words:
                    surface = " ".join(lowered[start : start + width])
                    candidates.append((start, start + width, surface, phrase))

        # Keep the most specific phrase at each token position. This prevents
        # "glass door" and "door" from becoming duplicate detector queries.
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        occupied_tokens: set[int] = set()
        selected: list[tuple[int, int, str, str]] = []
        seen_labels: set[str] = set()
        for start, end, surface, matched_phrase in candidates:
            canonical = _CANONICAL_ALIASES.get(matched_phrase, matched_phrase)
            if canonical in seen_labels or any(index in occupied_tokens for index in range(start, end)):
                continue
            selected.append((start, end, surface, matched_phrase))
            seen_labels.add(canonical)
            occupied_tokens.update(range(start, end))

        result: list[PerceptionQuery] = []
        for index, (start, end, surface, matched_phrase) in enumerate(selected):
            canonical = _CANONICAL_ALIASES.get(matched_phrase, matched_phrase)
            semantic_kind = "REGION" if canonical in _REGION_LABELS else "OBJECT"
            result.append(
                PerceptionQuery(
                    query_id=f"q{index:02d}",
                    text=surface,
                    canonical_label=canonical,
                    token_start=int(start),
                    token_end=int(end),
                    semantic_kind=semantic_kind,
                )
            )
        return tuple(result)


def _singular_token(token: str) -> str:
    aliases = {
        "bathrooms": "bathroom",
        "bedrooms": "bedroom",
        "cabinets": "cabinet",
        "chairs": "chair",
        "columns": "column",
        "corridors": "corridor",
        "doors": "door",
        "doorways": "doorway",
        "entrances": "entrance",
        "hallways": "hallway",
        "kitchens": "kitchen",
        "rooms": "room",
        "signs": "sign",
        "sofas": "sofa",
        "tables": "table",
        "televisions": "television",
        "windows": "window",
    }
    return aliases.get(str(token), str(token))
