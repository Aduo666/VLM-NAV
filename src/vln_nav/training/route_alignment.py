from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .schemas import (
    CanonicalEpisode,
    ReplayPose,
    RouteAlignmentEvidence,
    RouteAlignmentRecord,
)


_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CLAUSE_PATTERN = re.compile(r"[^,.;]+")


@dataclass(frozen=True)
class InstructionSpan:
    span_id: str
    token_start: int
    token_end: int
    text: str
    event_type: str
    direction: str


@dataclass(frozen=True)
class RouteEvent:
    event_type: str
    start_frame_index: int
    peak_frame_index: int
    end_frame_index: int
    label: str = ""
    direction: str = ""
    confidence: float = 1.0
    details: str = ""


@dataclass(frozen=True)
class SceneEntity:
    label: str
    instance_name: str
    world_xyz: tuple[float, float, float]


def align_instruction_to_route(
    *,
    episode: CanonicalEpisode,
    replay: list[ReplayPose],
) -> tuple[RouteAlignmentRecord, ...]:
    """Produce monotonic weak/strong alignment without an instruction LLM."""

    spans = compile_instruction_spans(episode.instruction)
    events = extract_route_events(episode=episode, replay=replay)
    entities = load_scene_entities(episode.scene_usd_path)
    if not spans or not replay:
        return ()
    result: list[RouteAlignmentRecord] = []
    event_cursor = 0
    fallback_cursor = 0
    for span_index, span in enumerate(spans):
        matched_index = _find_compatible_event(
            events,
            span=span,
            start_index=event_cursor,
            minimum_frame_index=fallback_cursor,
        )
        object_event = None
        if matched_index is None and span.event_type in {"PASS", "APPROACH"}:
            object_event = _object_event_for_span(
                span=span,
                entities=entities,
                replay=replay,
                minimum_frame_index=fallback_cursor,
            )
        if matched_index is not None:
            event = events[matched_index]
            event_cursor = matched_index + 1
            fallback_cursor = max(fallback_cursor, event.peak_frame_index)
            confidence = min(float(event.confidence), 0.98)
            evidence = (
                RouteAlignmentEvidence(
                    kind="route_event",
                    frame_index=int(event.peak_frame_index),
                    confidence=confidence,
                    details=event.details or event.event_type,
                ),
            )
            start_frame = event.start_frame_index
            peak_frame = event.peak_frame_index
            end_frame = event.end_frame_index
        elif object_event is not None:
            fallback_cursor = max(fallback_cursor, object_event.peak_frame_index)
            confidence = float(object_event.confidence)
            evidence = (
                RouteAlignmentEvidence(
                    kind="scene_entity_route_proximity",
                    frame_index=object_event.peak_frame_index,
                    confidence=confidence,
                    details=object_event.details,
                ),
            )
            start_frame = object_event.start_frame_index
            peak_frame = object_event.peak_frame_index
            end_frame = object_event.end_frame_index
        else:
            fraction = float(span_index + 1) / float(len(spans) + 1)
            estimated = max(fallback_cursor, int(round(fraction * float(len(replay) - 1))))
            estimated = min(estimated, len(replay) - 1)
            fallback_cursor = estimated
            confidence = 0.30
            evidence = (
                RouteAlignmentEvidence(
                    kind="monotonic_position_prior",
                    frame_index=estimated,
                    confidence=confidence,
                    details="no matching geometric or room-transition event",
                ),
            )
            start_frame = max(0, estimated - 1)
            peak_frame = estimated
            end_frame = min(len(replay) - 1, estimated + 1)
        result.append(
            RouteAlignmentRecord(
                span_id=span.span_id,
                token_start=span.token_start,
                token_end=span.token_end,
                text=span.text,
                event_type=span.event_type,
                start_frame_index=int(start_frame),
                peak_frame_index=int(peak_frame),
                end_frame_index=int(end_frame),
                confidence=float(confidence),
                evidence=evidence,
            )
        )
    return tuple(result)


def compile_instruction_spans(instruction: str) -> tuple[InstructionSpan, ...]:
    word_matches = list(_WORD_PATTERN.finditer(str(instruction)))
    spans: list[InstructionSpan] = []
    for clause in _CLAUSE_PATTERN.finditer(str(instruction)):
        text = clause.group(0).strip()
        if not text:
            continue
        covered = [
            index
            for index, word in enumerate(word_matches)
            if clause.start() <= word.start() < clause.end()
        ]
        if not covered:
            continue
        lowered = text.lower()
        event_type = _event_type(lowered)
        if event_type == "STOP_AT":
            text = str(instruction)[clause.start() :].strip()
            covered = list(range(int(covered[0]), len(word_matches)))
        spans.append(
            InstructionSpan(
                span_id=f"span_{len(spans):02d}",
                token_start=int(covered[0]),
                token_end=int(covered[-1] + 1),
                text=text,
                event_type=event_type,
                direction=_direction(lowered),
            )
        )
        if spans[-1].event_type == "STOP_AT":
            break
    return tuple(spans)


def load_scene_entities(scene_usd_path: str | Path) -> tuple[SceneEntity, ...]:
    """Read top-level object instance translations from InteriorNav USDA text."""

    path = Path(scene_usd_path)
    if not path.is_file() or path.suffix.lower() not in {".usd", ".usda"}:
        return ()
    xform_pattern = re.compile(r'^\s*def Xform "([^"]+)"')
    translation_pattern = re.compile(
        r"xformOp:translate\s*=\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\)"
    )
    instance_pattern = re.compile(r"^(.+)_([0-9]{4})$")
    pending_name = ""
    result: list[SceneEntity] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        xform_match = xform_pattern.search(line)
        if xform_match is not None:
            pending_name = xform_match.group(1)
            continue
        if not pending_name:
            continue
        translation_match = translation_pattern.search(line)
        if translation_match is None:
            continue
        instance_match = instance_pattern.match(pending_name)
        if instance_match is not None:
            label = _canonical_entity_label(instance_match.group(1))
            result.append(
                SceneEntity(
                    label=label,
                    instance_name=pending_name,
                    world_xyz=tuple(float(translation_match.group(index)) for index in (1, 2, 3)),
                )
            )
        pending_name = ""
    return tuple(result)


def extract_route_events(
    *,
    episode: CanonicalEpisode,
    replay: list[ReplayPose],
) -> tuple[RouteEvent, ...]:
    events: list[RouteEvent] = []
    index = 0
    while index < len(replay):
        motion = replay[index].motion_type
        if motion not in {"TURN_LEFT", "TURN_RIGHT"}:
            index += 1
            continue
        start = index
        direction = "left" if motion == "TURN_LEFT" else "right"
        while index + 1 < len(replay) and replay[index + 1].motion_type == motion:
            index += 1
        events.append(
            RouteEvent(
                event_type="TURN",
                start_frame_index=start,
                peak_frame_index=index,
                end_frame_index=index,
                direction=direction,
                confidence=0.95,
                details=f"canonical_path_turn direction={direction}",
            )
        )
        index += 1

    previous_room = ""
    transition_start = 0
    for pose in replay:
        room = _room_at_pose(episode=episode, pose=pose)
        if room and room != previous_room:
            events.append(
                RouteEvent(
                    event_type="ENTER_REGION",
                    start_frame_index=max(0, pose.frame_index - 1),
                    peak_frame_index=pose.frame_index,
                    end_frame_index=min(len(replay) - 1, pose.frame_index + 1),
                    label=room,
                    confidence=0.92,
                    details=f"room_polygon_entry label={room}",
                )
            )
            transition_start = pose.frame_index
        if previous_room and room != previous_room:
            events.append(
                RouteEvent(
                    event_type="EXIT_REGION",
                    start_frame_index=max(0, pose.frame_index - 1),
                    peak_frame_index=pose.frame_index,
                    end_frame_index=min(len(replay) - 1, pose.frame_index + 1),
                    label=previous_room,
                    confidence=0.92,
                    details=f"room_polygon_exit label={previous_room}",
                )
            )
        previous_room = room
    events.append(
        RouteEvent(
            event_type="STOP_AT",
            start_frame_index=max(transition_start, len(replay) - 2),
            peak_frame_index=len(replay) - 1,
            end_frame_index=len(replay) - 1,
            confidence=0.98,
            details="reference_route_endpoint",
        )
    )
    events.sort(key=lambda event: (event.peak_frame_index, event.start_frame_index))
    return tuple(events)


def progress_alignment_at_frame(
    alignments: tuple[RouteAlignmentRecord, ...],
    *,
    frame_index: int,
    token_count: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    RouteAlignmentRecord | None,
    float,
]:
    """Build weak semantic-progress supervision for one causal replay frame.

    ``completed_prefix_weights`` is monotonic by construction. It is a training
    target only and is never copied into either student observation contract.
    """

    completed = np.zeros((max(0, int(token_count)),), dtype=np.float32)
    active_weights = np.zeros_like(completed)
    if not alignments:
        empty = tuple(float(value) for value in completed)
        return empty, empty, None, 0.0

    for alignment in alignments:
        start_frame = int(alignment.start_frame_index)
        end_frame = max(start_frame + 1, int(alignment.end_frame_index))
        if int(frame_index) < start_frame:
            span_completion = 0.0
        elif int(frame_index) >= int(alignment.end_frame_index):
            span_completion = 1.0
        else:
            span_completion = float(
                np.clip(
                    (float(frame_index) - float(start_frame))
                    / float(end_frame - start_frame),
                    0.0,
                    1.0,
                )
            )
        token_start = max(0, min(completed.size, int(alignment.token_start)))
        token_end = max(token_start, min(completed.size, int(alignment.token_end)))
        if token_end > token_start:
            completed[token_start:token_end] = np.maximum(
                completed[token_start:token_end],
                span_completion,
            )
    if completed.size > 1:
        completed = np.minimum.accumulate(completed)

    distances = [abs(int(item.peak_frame_index) - int(frame_index)) for item in alignments]
    active = alignments[int(np.argmin(np.asarray(distances, dtype=np.int32)))]
    start = max(0, min(active_weights.size, int(active.token_start)))
    end = max(start, min(active_weights.size, int(active.token_end)))
    if end > start:
        active_weights[start:end] = 1.0 / float(end - start)
    if int(frame_index) >= int(active.end_frame_index):
        completion = 1.0
    elif int(frame_index) <= int(active.start_frame_index):
        completion = 0.0
    else:
        duration = max(1, int(active.end_frame_index) - int(active.start_frame_index))
        completion = float(
            np.clip(
                (float(frame_index) - float(active.start_frame_index)) / float(duration),
                0.0,
                1.0,
            )
        )
    return (
        tuple(float(value) for value in completed),
        tuple(float(value) for value in active_weights),
        active,
        completion,
    )


def _find_compatible_event(
    events: tuple[RouteEvent, ...],
    *,
    span: InstructionSpan,
    start_index: int,
    minimum_frame_index: int,
) -> int | None:
    for index in range(max(0, int(start_index)), len(events)):
        event = events[index]
        if event.peak_frame_index < int(minimum_frame_index):
            continue
        if event.event_type != span.event_type:
            continue
        if span.direction and event.direction and span.direction != event.direction:
            continue
        if span.event_type in {"ENTER_REGION", "EXIT_REGION"} and not _region_label_matches(
            span_text=span.text,
            event_label=event.label,
        ):
            continue
        return index
    return None


def _event_type(text: str) -> str:
    if re.search(r"\b(stop|stops|finish|finishes|terminate|terminates)\b", text):
        return "STOP_AT"
    if re.search(r"\b(exit|exits|leave|leaves)\b", text):
        return "EXIT_REGION"
    if re.search(r"\b(enter|enters|entering|go through|walk through|proceed through|proceeds through)\b", text):
        return "ENTER_REGION"
    if re.search(r"\b(turn|turns|turning|veer|veers|veering)\b", text):
        return "TURN"
    if re.search(r"\b(pass|passes|passing)\b", text):
        return "PASS"
    if re.search(r"\b(find|approach|toward|towards|face|near)\b", text):
        return "APPROACH"
    return "CONTINUE"


def _direction(text: str) -> str:
    if re.search(r"\bleft\b", text):
        return "left"
    if re.search(r"\bright\b", text):
        return "right"
    if re.search(r"\b(back|behind)\b", text):
        return "back"
    if re.search(r"\b(straight|forward|ahead)\b", text):
        return "forward"
    return ""


def _room_at_pose(*, episode: CanonicalEpisode, pose: ReplayPose) -> str:
    x, y = pose.position_xyz[:2]
    for room in episode.rooms:
        if _point_in_polygon(float(x), float(y), room.polygon_xy):
            return room.label
    return ""


def _object_event_for_span(
    *,
    span: InstructionSpan,
    entities: tuple[SceneEntity, ...],
    replay: list[ReplayPose],
    minimum_frame_index: int,
) -> RouteEvent | None:
    normalized = span.text.lower().replace("-", " ")
    labels = {
        entity.label
        for entity in entities
        if re.search(rf"\b{re.escape(entity.label)}s?\b", normalized)
    }
    if not labels:
        return None
    expected_sides = {label: _expected_side_for_label(normalized, label) for label in labels}
    best: tuple[float, int, list[tuple[SceneEntity, float]]] | None = None
    start_frame = max(0, int(minimum_frame_index) - 2)
    for frame_index in range(start_frame, len(replay)):
        pose = replay[frame_index]
        matched_entities: list[tuple[SceneEntity, float]] = []
        costs: list[float] = []
        cosine = float(np.cos(pose.yaw_rad))
        sine = float(np.sin(pose.yaw_rad))
        for label in sorted(labels):
            label_best: tuple[float, SceneEntity, float] | None = None
            expected_side = expected_sides[label]
            for entity in entities:
                if entity.label != label:
                    continue
                delta_x = float(entity.world_xyz[0] - pose.position_xyz[0])
                delta_y = float(entity.world_xyz[1] - pose.position_xyz[1])
                forward = cosine * delta_x + sine * delta_y
                left = -sine * delta_x + cosine * delta_y
                distance = float(np.hypot(delta_x, delta_y))
                if distance > 3.0 or forward < -1.0:
                    continue
                side_penalty = 0.0
                if expected_side == "left" and left < 0.10:
                    side_penalty = 1.5
                elif expected_side == "right" and left > -0.10:
                    side_penalty = 1.5
                cost = distance + side_penalty + (0.5 if forward < -0.25 else 0.0)
                if label_best is None or cost < label_best[0]:
                    label_best = cost, entity, distance
            if label_best is not None:
                costs.append(label_best[0])
                matched_entities.append((label_best[1], label_best[2]))
        if not costs:
            continue
        coverage_penalty = 2.0 * float(len(labels) - len(costs))
        score = float(np.mean(costs)) + coverage_penalty
        if best is None or score < best[0]:
            best = score, frame_index, matched_entities
    if best is None:
        return None
    _, frame_index, matched_entities = best
    mean_distance = float(np.mean([item[1] for item in matched_entities]))
    coverage = float(len(matched_entities)) / float(len(labels))
    confidence = (0.88 if mean_distance <= 1.5 else 0.65) * coverage
    entity_summary = ",".join(
        f"{entity.instance_name}:{distance:.2f}m"
        for entity, distance in matched_entities
    )
    return RouteEvent(
        event_type=span.event_type,
        start_frame_index=max(0, frame_index - 2),
        peak_frame_index=frame_index,
        end_frame_index=min(len(replay) - 1, frame_index + 2),
        label="+".join(entity.label for entity, _ in matched_entities),
        confidence=confidence,
        details=(
            f"scene_entities={entity_summary} coverage={len(matched_entities)}/{len(labels)}"
        ),
    )


def _canonical_entity_label(label: str) -> str:
    normalized = str(label).strip().lower().replace("-", "_")
    aliases = {
        "television": "tv",
        "closestool": "toilet",
        "dining_table": "table",
        "coffee_table": "table",
    }
    return aliases.get(normalized, normalized.replace("_", " "))


def _region_label_matches(*, span_text: str, event_label: str) -> bool:
    text = str(span_text).lower()
    label = str(event_label).lower()
    aliases = {
        "hall": {"hall", "living room"},
        "main hall": {"hall", "living room"},
        "lobby": {"lobby", "living room"},
        "corridor": {"corridor", "hallway"},
        "hallway": {"corridor", "hallway"},
    }
    mentioned: set[str] = set()
    for region in (
        "main hall",
        "living room",
        "hallway",
        "corridor",
        "bedroom",
        "bathroom",
        "kitchen",
        "balcony",
        "lobby",
        "hall",
    ):
        if re.search(rf"\b{re.escape(region)}\b", text):
            mentioned.update(aliases.get(region, {region}))
    return not mentioned or label in mentioned


def _expected_side_for_label(text: str, label: str) -> str:
    match = re.search(rf"\b{re.escape(label)}s?\b", text)
    if match is None:
        return ""
    tail = text[match.end() :]
    clause_end = re.search(r"\b(?:and|then|while)\b|[,.;]", tail)
    segment = tail if clause_end is None else tail[: clause_end.start()]
    if re.search(r"\bleft\b", segment):
        return "left"
    if re.search(r"\bright\b", segment):
        return "right"
    return ""


def _point_in_polygon(
    x: float,
    y: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        intersects = (y0 > y) != (y1 > y)
        if intersects:
            denominator = y1 - y0
            if abs(denominator) <= 1.0e-12:
                previous = current
                continue
            boundary_x = (x1 - x0) * (y - y0) / denominator + x0
            if x < boundary_x:
                inside = not inside
        previous = current
    return inside
