"""Stable output contracts for the downstream rally referee layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REFEREE_CONTRACT_VERSION = 1
REFEREE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScoreDecision:
    """One idempotent score-settlement decision."""

    rally_id: int
    status: str
    rally_winner: str | None
    point_awarded_to: str | None
    score_before: dict[str, int]
    score_after: dict[str, int]
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "status": self.status,
            "rally_winner": self.rally_winner,
            "point_awarded_to": self.point_awarded_to,
            "score_before": dict(self.score_before),
            "score_after": dict(self.score_after),
            "duplicate": self.duplicate,
        }


@dataclass(frozen=True)
class HitRecord:
    """One confirmed hit/route record exposed to downstream consumers."""

    hit_count: int
    hit_half: str
    hit_side: str
    route_destination_half: str
    route_destination_side: str
    hit_start_frame_index: int
    hit_start_timestamp_s: float
    hit_end_frame_index: int | None = None
    hit_end_timestamp_s: float | None = None
    event_type_0811: str = "NONE"
    first_landing_frame_index: int | None = None
    first_landing_timestamp_s: float | None = None
    first_landing_position: tuple[float, float] | None = None
    first_landing_inout: str | None = None
    decision_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_count": self.hit_count,
            "hit_half": self.hit_half,
            "hit_side": self.hit_side,
            "route_destination_half": self.route_destination_half,
            "route_destination_side": self.route_destination_side,
            "hit_start_frame_index": self.hit_start_frame_index,
            "hit_start_timestamp_s": self.hit_start_timestamp_s,
            "hit_starttime": self.hit_start_timestamp_s,
            "hit_end_frame_index": self.hit_end_frame_index,
            "hit_end_timestamp_s": self.hit_end_timestamp_s,
            "hit_endtime": self.hit_end_timestamp_s,
            "event_type": self.event_type_0811,
            "event_type_0811": self.event_type_0811,
            "first_landing_frame_index": self.first_landing_frame_index,
            "first_landing_timestamp_s": self.first_landing_timestamp_s,
            "firstlanding_time": self.first_landing_timestamp_s,
            "first_landing_position": (
                list(self.first_landing_position)
                if self.first_landing_position is not None
                else None
            ),
            "firstlanding_position": (
                list(self.first_landing_position)
                if self.first_landing_position is not None
                else None
            ),
            "first_landing_inout": self.first_landing_inout,
            "firstlanding_inout": self.first_landing_inout,
            "decision_latency_ms": self.decision_latency_ms,
        }


@dataclass(frozen=True)
class RallyResult:
    """The single terminal result emitted for one rally."""

    rally_id: int
    start_frame_index: int
    start_timestamp_s: float
    end_frame_index: int
    end_timestamp_s: float
    server_half: str
    server_side: str
    leg_index: int
    terminal_event: str
    fault_half: str | None
    fault_side: str | None
    rally_winner_half: str | None
    rally_winner: str | None
    point_awarded_to: str | None
    score_before: dict[str, int]
    score_after: dict[str, int]
    status: str
    confidence: float
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    server_player_centers_court_xy: tuple[tuple[float, float], ...] = ()
    server_player_center_outside_court: bool | None = None
    next_server_half: str | None = None
    next_server_side: str | None = None
    expected_server_side: str | None = None
    server_side_match: bool | None = None
    event_type_0811: str = "NONE"
    rally_over: str = "Y"
    manual_confirmation_required: bool = False
    decision_latency_ms: float | None = None
    hit_records: tuple[HitRecord, ...] = ()
    contract_version: int = REFEREE_CONTRACT_VERSION
    schema_version: int = REFEREE_SCHEMA_VERSION

    @property
    def result_id(self) -> str:
        return f"rally-{self.rally_id:06d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "rally_id": self.rally_id,
            "start_frame_index": self.start_frame_index,
            "start_timestamp_s": self.start_timestamp_s,
            "end_frame_index": self.end_frame_index,
            "end_timestamp_s": self.end_timestamp_s,
            "server_half": self.server_half,
            "server_side": self.server_side,
            "leg_index": self.leg_index,
            "terminal_event": self.terminal_event,
            "fault_half": self.fault_half,
            "fault_side": self.fault_side,
            "rally_winner_half": self.rally_winner_half,
            "rally_winner": self.rally_winner,
            "point_awarded_to": self.point_awarded_to,
            "score_before": dict(self.score_before),
            "score_after": dict(self.score_after),
            "status": self.status,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "server_player_centers_court_xy": [
                list(point) for point in self.server_player_centers_court_xy
            ],
            "server_player_center_outside_court": (
                self.server_player_center_outside_court
            ),
            "next_server_half": self.next_server_half,
            "next_server_side": self.next_server_side,
            "expected_server_side": self.expected_server_side,
            "server_side_match": self.server_side_match,
            "event_type": self.event_type_0811,
            "event_type_0811": self.event_type_0811,
            "rally_over": self.rally_over,
            "manual_confirmation_required": self.manual_confirmation_required,
            "decision_latency_ms": self.decision_latency_ms,
            "hit_count": len(self.hit_records),
            "hit_records": [record.to_dict() for record in self.hit_records],
        }


@dataclass
class RefereeFrameResult:
    """Per-frame state snapshot; ``rally_result`` exists only at termination."""

    phase: str
    frame_index: int
    timestamp_s: float
    rally_id: int | None
    physical_half: str | None
    physical_side: str | None
    server_half: str | None
    server_side: str | None
    route_origin_half: str | None
    route_origin_side: str | None
    route_destination_half: str | None
    route_destination_side: str | None
    leg_index: int
    net_crossed: bool
    first_bounce_half: str | None
    score: dict[str, int]
    eligible_player_centers_court_xy: list[tuple[float, float]] = field(
        default_factory=list
    )
    eligible_player_center_outside_court: bool = False
    service_owner_half: str | None = None
    service_owner_side: str | None = None
    expected_server_side: str | None = None
    server_side_match: bool | None = None
    target_ball_state_0811: str = "target_ball_lost"
    event_type_0811: str = "NONE"
    rally_over: str = "NONE"
    manual_confirmation_required: bool = False
    decision_latency_ms: dict[str, float] = field(default_factory=dict)
    hit_records: list[HitRecord] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rally_result: RallyResult | None = None
    contract_version: int = REFEREE_CONTRACT_VERSION
    schema_version: int = REFEREE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "phase": self.phase,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "rally_id": self.rally_id,
            "physical_half": self.physical_half,
            "physical_side": self.physical_side,
            "server_half": self.server_half,
            "server_side": self.server_side,
            "route_origin_half": self.route_origin_half,
            "route_origin_side": self.route_origin_side,
            "route_destination_half": self.route_destination_half,
            "route_destination_side": self.route_destination_side,
            "leg_index": self.leg_index,
            "net_crossed": self.net_crossed,
            "first_bounce_half": self.first_bounce_half,
            "score": dict(self.score),
            "eligible_player_centers_court_xy": [
                list(point) for point in self.eligible_player_centers_court_xy
            ],
            "eligible_player_center_outside_court": (
                self.eligible_player_center_outside_court
            ),
            "service_owner_half": self.service_owner_half,
            "service_owner_side": self.service_owner_side,
            "expected_server_side": self.expected_server_side,
            "server_side_match": self.server_side_match,
            "target_ball_state": self.target_ball_state_0811,
            "target_ball_state_0811": self.target_ball_state_0811,
            "event_type": self.event_type_0811,
            "event_type_0811": self.event_type_0811,
            "rally_over": self.rally_over,
            "manual_confirmation_required": self.manual_confirmation_required,
            "decision_latency_ms": dict(self.decision_latency_ms),
            "hit_count": len(self.hit_records),
            "hit_records": [record.to_dict() for record in self.hit_records],
            "events": list(self.events),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "rally_result": (self.rally_result.to_dict() if self.rally_result is not None else None),
        }
