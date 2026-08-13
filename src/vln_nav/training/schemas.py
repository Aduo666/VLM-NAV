from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_EPISODE_SCHEMA_VERSION = "captainav-canonical-episode-v1"
EXPERT_REPLAY_SCHEMA_VERSION = "captainav-expert-replay-v1"
TRAINING_SAMPLE_SCHEMA_VERSION = "captainav-training-sample-v2"


class JsonRecord:
    """Small JSON-friendly base used by all persisted training records."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoomRegion(JsonRecord):
    label: str
    polygon_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CanonicalEpisode(JsonRecord):
    schema_version: str
    source_dataset: str
    split: str
    episode_id: str
    scene_id: str
    scene_usd_path: str
    instruction: str
    instruction_words: tuple[str, ...]
    source_instruction_token_ids: tuple[int, ...]
    start_xyz: tuple[float, float, float]
    start_rotation_wxyz: tuple[float, float, float, float]
    reference_path_xyz: tuple[tuple[float, float, float], ...]
    goal_xyz: tuple[float, float, float]
    goal_radius_m: float
    geodesic_distance_m: float | None = None
    rooms: tuple[RoomRegion, ...] = ()


@dataclass(frozen=True)
class ReplayPose(JsonRecord):
    frame_index: int
    timestamp_sec: float
    position_xyz: tuple[float, float, float]
    yaw_rad: float
    path_progress: float
    source_segment_index: int
    motion_type: str
    delta_distance_m: float
    delta_yaw_rad: float


@dataclass(frozen=True)
class CameraCalibration(JsonRecord):
    width: int
    height: int
    intrinsics: tuple[tuple[float, float, float], ...]
    camera_view_matrix_ros: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class ExpertReplayFrame(JsonRecord):
    schema_version: str
    episode_id: str
    pose: ReplayPose
    rgb_path: str
    depth_path: str
    calibration: CameraCalibration


@dataclass(frozen=True)
class PerceptionQuery(JsonRecord):
    query_id: str
    text: str
    canonical_label: str
    token_start: int
    token_end: int
    semantic_kind: str


@dataclass(frozen=True)
class ObservedLandmark(JsonRecord):
    track_id: str
    label: str
    query_id: str
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    detector_confidence: float
    depth_m: float | None
    estimated_world_xy: tuple[float, float] | None
    observation_count: int
    last_seen_frame_index: int
    verification_status: str
    source: str


@dataclass(frozen=True)
class FrontierToken(JsonRecord):
    frontier_id: int
    boundary_world_xy: tuple[float, float]
    navigation_goal_world_xy: tuple[float, float]
    local_forward_m: float
    local_left_m: float
    distance_m: float
    bearing_rad: float
    cell_count: int
    information_gain: int
    path_cost_m: float
    obstacle_clearance_m: float
    reachable: bool


@dataclass(frozen=True)
class CausalObservationState(JsonRecord):
    """Runtime-available state. Privileged future information is forbidden."""

    episode_id: str
    frame_index: int
    timestamp_sec: float
    robot_pose_xyyaw: tuple[float, float, float]
    rgb_history_paths: tuple[str, ...]
    local_map_path: str
    global_map_path: str
    trajectory_world_xy: tuple[tuple[float, float], ...]
    observed_landmarks: tuple[ObservedLandmark, ...]
    frontiers: tuple[FrontierToken, ...]


@dataclass(frozen=True)
class RouteAlignmentEvidence(JsonRecord):
    kind: str
    frame_index: int
    confidence: float
    details: str


@dataclass(frozen=True)
class RouteAlignmentRecord(JsonRecord):
    span_id: str
    token_start: int
    token_end: int
    text: str
    event_type: str
    start_frame_index: int
    peak_frame_index: int
    end_frame_index: int
    confidence: float
    evidence: tuple[RouteAlignmentEvidence, ...] = ()


@dataclass(frozen=True)
class StudentSample(JsonRecord):
    """Information available to the learned policy at inference time."""

    instruction: str
    instruction_words: tuple[str, ...]
    observation: CausalObservationState
    query_phrases: tuple[PerceptionQuery, ...]


@dataclass(frozen=True)
class OracleTargets(JsonRecord):
    """Supervision derived from the reference route, never a model input."""

    active_span_weights: tuple[float, ...]
    event_type: str
    completion_probability: float
    semantic_progress: float
    navigation_mode: str
    frontier_ids: tuple[int, ...]
    frontier_value_targets: tuple[float, ...]
    selected_frontier_id: int | None


@dataclass(frozen=True)
class DebugOracle(JsonRecord):
    """Privileged data retained only for auditing labels and replay quality."""

    future_reference_path_xy: tuple[tuple[float, float], ...]
    nearest_reference_path_index: int
    route_alignment_path: str
    active_span_id: str | None
    oracle_reason: str
    oracle_crossing_world_xy: tuple[float, float] | None
    oracle_confidence: float


@dataclass(frozen=True)
class TrainingSampleV2(JsonRecord):
    schema_version: str
    sample_id: str
    student: StudentSample
    targets: OracleTargets
    debug_oracle: DebugOracle
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
