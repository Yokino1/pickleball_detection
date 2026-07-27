"""Global single-ball arbitration across synchronized left/right cameras."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..types import BallTrack


SIDES = ("left", "right")
DEFAULT_PRIMARY_OBSERVATION_SOURCES = frozenset(
    {"yolo", "onnxruntime", "detector"}
)


@dataclass
class GlobalBallSelection:
    """One global ball decision for a synchronized frame pair."""

    global_track_id: int
    active_side: Optional[str]
    state: str
    track: Optional[BallTrack]
    local_track_id: Optional[int]
    switched_side: bool = False
    previous_side: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "global_track_id": self.global_track_id,
            "active_side": self.active_side,
            "state": self.state,
            "track": self.track.to_dict() if self.track is not None else None,
            "local_track_id": self.local_track_id,
            "switched_side": self.switched_side,
            "previous_side": self.previous_side,
        }


class CrossCameraHandoffAdvisor:
    """Turn an observed net-bound trajectory into a short receiving-side ROI."""

    def __init__(
        self,
        net_margin_ratio: float = 0.22,
        receiving_band_ratio: float = 0.30,
        min_toward_net_speed_px_per_second: float = 220.0,
        alert_duration_ms: float = 180.0,
        reference_frame_width: float = 1280.0,
        frame_scale_overrides: Optional[dict[str, float]] = None,
        left_net_edge: str = "right",
        right_net_edge: str = "left",
    ):
        self.net_margin_ratio = min(0.5, max(0.01, float(net_margin_ratio)))
        self.receiving_band_ratio = min(
            1.0, max(0.05, float(receiving_band_ratio))
        )
        self.min_toward_net_speed_px_per_second = max(
            0.0, float(min_toward_net_speed_px_per_second)
        )
        self.alert_duration_s = max(0.0, float(alert_duration_ms) / 1000.0)
        self.reference_frame_width = max(1.0, float(reference_frame_width))
        self.frame_scale_overrides = {
            side: max(0.25, float(value))
            for side, value in (frame_scale_overrides or {}).items()
            if side in SIDES
        }
        self.net_edges = {
            "left": self._validate_edge(left_net_edge),
            "right": self._validate_edge(right_net_edge),
        }
        self.target_side: Optional[str] = None
        self.source_side: Optional[str] = None
        self.expires_at_s = -1.0
        self.trigger_count = 0

    def update(
        self,
        selection: GlobalBallSelection,
        *,
        timestamp_s: float,
        frame_widths: dict[str, int],
    ) -> None:
        if (
            selection.track is None
            or selection.active_side not in SIDES
            or selection.track.center is None
            or selection.track.velocity is None
        ):
            return
        side = selection.active_side
        width = max(1, int(frame_widths[side]))
        center_x = float(selection.track.center[0])
        velocity_x = float(selection.track.velocity[0])
        edge = self.net_edges[side]
        near_net = (
            center_x >= width * (1.0 - self.net_margin_ratio)
            if edge == "right"
            else center_x <= width * self.net_margin_ratio
        )
        frame_scale = self.frame_scale_overrides.get(
            side,
            max(0.25, width / self.reference_frame_width),
        )
        normalized_velocity_x = velocity_x / frame_scale
        toward_net = (
            normalized_velocity_x >= self.min_toward_net_speed_px_per_second
            if edge == "right"
            else normalized_velocity_x <= -self.min_toward_net_speed_px_per_second
        )
        if not (near_net and toward_net):
            return
        target_side = "right" if side == "left" else "left"
        if self.target_side != target_side or timestamp_s > self.expires_at_s:
            self.trigger_count += 1
        self.source_side = side
        self.target_side = target_side
        self.expires_at_s = float(timestamp_s) + self.alert_duration_s

    def search_roi(
        self,
        side: str,
        *,
        timestamp_s: float,
        frame_width: int,
        frame_height: int,
    ) -> Optional[list[float]]:
        if side != self.target_side or timestamp_s > self.expires_at_s:
            return None
        width = max(1, int(frame_width))
        height = max(1, int(frame_height))
        edge = self.net_edges[side]
        band_width = max(1.0, width * self.receiving_band_ratio)
        if edge == "left":
            return [0.0, 0.0, band_width, float(height)]
        return [float(width) - band_width, 0.0, float(width), float(height)]

    def diagnostics(self, timestamp_s: float) -> dict:
        active = (
            self.target_side is not None
            and float(timestamp_s) <= self.expires_at_s
        )
        return {
            "active": active,
            "source_side": self.source_side if active else None,
            "target_side": self.target_side if active else None,
            "expires_at_s": round(self.expires_at_s, 6) if active else None,
            "trigger_count": self.trigger_count,
        }

    @staticmethod
    def _validate_edge(value: str) -> str:
        edge = str(value).strip().lower()
        if edge not in {"left", "right"}:
            raise ValueError("Camera net edge must be 'left' or 'right'")
        return edge


class CrossHalfBallCoordinator:
    """Select one global ball from two independent local tracker outputs.

    Phase one deliberately does not compare image coordinates between cameras.
    It uses synchronized observations and active-side hysteresis only. A later
    phase can add calibrated handoff prediction without changing this contract.
    """

    def __init__(
        self,
        global_track_id: int = 1,
        switch_min_missing_ms: float = 20.0,
        max_continuity_speed_px_per_second: float = 0.0,
        reference_frame_width: float = 1280.0,
        frame_scale_overrides: Optional[dict[str, float]] = None,
        strict_handoff: bool = False,
        receiver_confirmation_hits: int = 2,
        switch_lock_ms: float = 100.0,
        left_net_edge: str = "right",
        right_net_edge: str = "left",
        observation_first: bool = False,
        primary_observation_sources: Optional[list[str]] = None,
    ):
        self.global_track_id = int(global_track_id)
        self.switch_min_missing_ms = max(0.0, float(switch_min_missing_ms))
        self.max_continuity_speed_px_per_second = max(
            0.0,
            float(max_continuity_speed_px_per_second),
        )
        self.reference_frame_width = max(1.0, float(reference_frame_width))
        self.frame_scale_overrides = {
            side: max(0.25, float(value))
            for side, value in (frame_scale_overrides or {}).items()
            if side in SIDES
        }
        self.strict_handoff = bool(strict_handoff)
        self.receiver_confirmation_hits = max(
            1,
            int(receiver_confirmation_hits),
        )
        self.switch_lock_s = max(0.0, float(switch_lock_ms) / 1000.0)
        self.net_edges = {
            "left": CrossCameraHandoffAdvisor._validate_edge(left_net_edge),
            "right": CrossCameraHandoffAdvisor._validate_edge(right_net_edge),
        }
        self.observation_first = bool(observation_first)
        self.primary_observation_sources = frozenset(
            str(value).strip().lower()
            for value in (
                primary_observation_sources
                or DEFAULT_PRIMARY_OBSERVATION_SOURCES
            )
            if str(value).strip()
        )
        self.active_side: Optional[str] = None
        self.active_local_track_id: Optional[int] = None
        self.state = "no_ball"
        self.handoff_count = 0
        self.continuity_gate_rejections = 0
        self.handoff_rejections = 0
        self.handoff_timeouts = 0
        self.prediction_boundary_rejections = 0
        self.observation_preemptions = 0
        self.same_side_observation_preemptions = 0
        self._last_output_by_side: dict[str, tuple[float, float, float]] = {}
        self._fallback_timestamp_s = 0.0
        self._handoff_source_side: Optional[str] = None
        self._handoff_target_side: Optional[str] = None
        self._handoff_expires_at_s = -1.0
        self._handoff_target_roi: Optional[list[float]] = None
        self._confirmation_track_id: Optional[int] = None
        self._confirmation_hits = 0
        self._confirmation_last: Optional[tuple[float, float, float]] = None
        self._switch_lock_until_s = -1.0

    def update(
        self,
        left_tracks: list[BallTrack],
        right_tracks: list[BallTrack],
        *,
        timestamp_s: Optional[float] = None,
        frame_widths: Optional[dict[str, int]] = None,
        handoff: Optional[dict] = None,
    ) -> GlobalBallSelection:
        timestamp = self._resolve_timestamp(timestamp_s)
        widths = frame_widths or {
            "left": int(self.reference_frame_width),
            "right": int(self.reference_frame_width),
        }
        by_side = {"left": left_tracks, "right": right_tracks}
        previous_side = self.active_side
        switched = False
        rejected_active_key: Optional[tuple[str, Optional[int]]] = None
        if self.strict_handoff:
            self._update_handoff_window(handoff, timestamp)

        if self.active_side is None:
            initial = self._best_observed_across_sides(by_side)
            if initial is not None:
                self._activate(*initial)
                self.state = f"{self.active_side}_active"
        else:
            active_track = self._find_track(
                by_side[self.active_side],
                self.active_local_track_id,
            )
            rejected_track_id = None
            if (
                active_track is not None
                and self.strict_handoff
                and self._prediction_crossed_net_exit(
                    self.active_side,
                    active_track,
                    widths,
                )
            ):
                rejected_track_id = active_track.track_id
                rejected_active_key = (
                    self.active_side,
                    self.active_local_track_id,
                )
                active_track = None
                self.prediction_boundary_rejections += 1
            if (
                active_track is not None
                and not (
                    self.observation_first
                    and self._is_primary_observation(active_track)
                )
                and not self._is_continuous(
                    self.active_side,
                    active_track,
                    timestamp,
                    widths,
                )
            ):
                rejected_track_id = active_track.track_id
                rejected_active_key = (
                    self.active_side,
                    self.active_local_track_id,
                )
                active_track = None
                self.continuity_gate_rejections += 1
            other_side = "right" if self.active_side == "left" else "left"
            other_observed = self._best_observed(by_side[other_side])
            other_primary_observed = self._best_primary_observed(
                by_side[other_side]
            )
            same_side_primary_observed = self._best_primary_observed(
                by_side[self.active_side],
                excluded_track_id=self.active_local_track_id,
            )

            if active_track is None:
                same_side_observed = self._best_continuous_observed(
                    self.active_side,
                    by_side[self.active_side],
                    timestamp,
                    widths,
                    excluded_track_id=rejected_track_id,
                )
                if same_side_observed is not None:
                    self._activate(self.active_side, same_side_observed)
                    self._cancel_confirmation()
                    self._refresh_active_state(timestamp)
                elif self._observation_can_preempt(
                    other_primary_observed,
                    timestamp,
                ):
                    switched = self._preempt_with_observation(
                        other_side,
                        other_primary_observed,
                        timestamp,
                    )
                elif other_observed is not None:
                    switched = self._maybe_switch_to_receiver(
                        other_side,
                        other_observed,
                        timestamp,
                        widths,
                    )
                elif self.strict_handoff:
                    self._cancel_confirmation()
                    self._refresh_active_state(timestamp)
            elif active_track.status == "observed":
                if self.strict_handoff:
                    self._cancel_confirmation()
                    self._refresh_active_state(timestamp)
            elif (
                active_track.status == "predicted"
                and same_side_primary_observed is not None
            ):
                self._activate(self.active_side, same_side_primary_observed)
                self._cancel_confirmation()
                self._refresh_active_state(timestamp)
                self.same_side_observation_preemptions += 1
            elif self._observation_can_preempt(
                other_primary_observed,
                timestamp,
            ):
                switched = self._preempt_with_observation(
                    other_side,
                    other_primary_observed,
                    timestamp,
                )
            elif (
                active_track.status == "predicted"
                and other_observed is not None
                and active_track.missing_time_ms >= self.switch_min_missing_ms
            ):
                switched = self._maybe_switch_to_receiver(
                    other_side,
                    other_observed,
                    timestamp,
                    widths,
                )
            elif self.strict_handoff:
                self._cancel_confirmation()
                self._refresh_active_state(timestamp)

        selected = None
        if self.active_side is not None:
            selected = self._find_track(
                by_side[self.active_side],
                self.active_local_track_id,
            )
            if rejected_active_key == (
                self.active_side,
                self.active_local_track_id,
            ):
                selected = None

        if switched:
            self.handoff_count += 1
        if selected is not None and self.active_side is not None:
            self._remember_output(
                self.active_side,
                selected,
                timestamp,
            )
        global_track = (
            replace(selected, track_id=self.global_track_id)
            if selected is not None
            else None
        )
        if self.strict_handoff:
            state = self.state
            if global_track is None and state in {
                "left_active",
                "right_active",
            }:
                state = "temporarily_lost"
        else:
            state = (
                f"{self.active_side}_active"
                if global_track is not None
                else "temporarily_lost"
                if self.active_side is not None
                else "no_ball"
            )
        return GlobalBallSelection(
            global_track_id=self.global_track_id,
            active_side=self.active_side,
            state=state,
            track=global_track,
            local_track_id=(
                selected.track_id if selected is not None else self.active_local_track_id
            ),
            switched_side=switched,
            previous_side=previous_side if switched else None,
        )

    def diagnostics(self) -> dict:
        return {
            "active_side": self.active_side,
            "active_local_track_id": self.active_local_track_id,
            "global_track_id": self.global_track_id,
            "handoff_count": self.handoff_count,
            "strict_handoff": self.strict_handoff,
            "observation_first": self.observation_first,
            "primary_observation_sources": sorted(
                self.primary_observation_sources
            ),
            "state": self.state,
            "switch_min_missing_ms": self.switch_min_missing_ms,
            "receiver_confirmation_hits": self.receiver_confirmation_hits,
            "confirmation_hits": self._confirmation_hits,
            "confirmation_track_id": self._confirmation_track_id,
            "handoff_source_side": self._handoff_source_side,
            "handoff_target_side": self._handoff_target_side,
            "handoff_expires_at_s": (
                round(self._handoff_expires_at_s, 6)
                if self._handoff_target_side is not None
                else None
            ),
            "handoff_target_roi": (
                list(self._handoff_target_roi)
                if self._handoff_target_roi is not None
                else None
            ),
            "switch_lock_until_s": (
                round(self._switch_lock_until_s, 6)
                if self._switch_lock_until_s >= 0.0
                else None
            ),
            "handoff_rejections": self.handoff_rejections,
            "handoff_timeouts": self.handoff_timeouts,
            "prediction_boundary_rejections": (
                self.prediction_boundary_rejections
            ),
            "observation_preemptions": self.observation_preemptions,
            "same_side_observation_preemptions": (
                self.same_side_observation_preemptions
            ),
            "max_continuity_speed_px_per_second": (
                self.max_continuity_speed_px_per_second
            ),
            "continuity_gate_rejections": self.continuity_gate_rejections,
        }

    def _maybe_switch_to_receiver(
        self,
        other_side: str,
        other_observed: BallTrack,
        timestamp_s: float,
        frame_widths: dict[str, int],
    ) -> bool:
        if not self.strict_handoff:
            self._activate(other_side, other_observed)
            return True
        if not self._handoff_allows(other_side, timestamp_s):
            self.handoff_rejections += 1
            self._cancel_confirmation()
            self._refresh_active_state(timestamp_s)
            return False
        if not self._track_is_in_handoff_roi(other_observed):
            self.handoff_rejections += 1
            self._cancel_confirmation()
            self._refresh_active_state(timestamp_s)
            return False
        if not self._receiver_candidate_is_continuous(
            other_side,
            other_observed,
            timestamp_s,
            frame_widths,
        ):
            self.continuity_gate_rejections += 1
            self._start_confirmation(other_side, other_observed, timestamp_s)
            return False
        if self._confirmation_track_id == other_observed.track_id:
            self._confirmation_hits += 1
        else:
            self._start_confirmation(other_side, other_observed, timestamp_s)
        self._remember_confirmation(other_observed, timestamp_s)
        self.state = f"{other_side}_confirming"
        if self._confirmation_hits < self.receiver_confirmation_hits:
            return False
        self._activate(other_side, other_observed)
        self.state = f"{other_side}_active"
        self._switch_lock_until_s = timestamp_s + self.switch_lock_s
        self._clear_handoff_window()
        return True

    def _observation_can_preempt(
        self,
        track: Optional[BallTrack],
        timestamp_s: float,
    ) -> bool:
        return (
            self.observation_first
            and track is not None
            and self._is_primary_observation(track)
            and timestamp_s >= self._switch_lock_until_s
        )

    def _preempt_with_observation(
        self,
        side: str,
        track: BallTrack,
        timestamp_s: float,
    ) -> bool:
        self._activate(side, track)
        self.state = f"{side}_active"
        self._switch_lock_until_s = timestamp_s + self.switch_lock_s
        self._clear_handoff_window()
        self.observation_preemptions += 1
        return True

    def _update_handoff_window(
        self,
        handoff: Optional[dict],
        timestamp_s: float,
    ) -> None:
        if (
            self._handoff_target_side is not None
            and timestamp_s > self._handoff_expires_at_s
        ):
            self.handoff_timeouts += 1
            self._clear_handoff_window()
            self._refresh_active_state(timestamp_s)
        if timestamp_s < self._switch_lock_until_s or not handoff:
            return
        if not bool(handoff.get("active", False)):
            return
        source_side = handoff.get("source_side")
        target_side = handoff.get("target_side")
        expires_at_s = handoff.get("expires_at_s")
        target_roi = handoff.get("target_roi")
        if (
            source_side not in SIDES
            or target_side not in SIDES
            or source_side == target_side
            or source_side != self.active_side
            or expires_at_s is None
            or float(expires_at_s) < timestamp_s
            or not isinstance(target_roi, (list, tuple))
            or len(target_roi) != 4
        ):
            return
        self._handoff_source_side = source_side
        self._handoff_target_side = target_side
        self._handoff_expires_at_s = float(expires_at_s)
        self._handoff_target_roi = [float(value) for value in target_roi]
        if self._confirmation_hits == 0:
            self.state = f"{source_side}_to_{target_side}_armed"

    def _handoff_allows(self, target_side: str, timestamp_s: float) -> bool:
        return (
            timestamp_s >= self._switch_lock_until_s
            and self._handoff_source_side == self.active_side
            and self._handoff_target_side == target_side
            and timestamp_s <= self._handoff_expires_at_s
        )

    def _receiver_candidate_is_continuous(
        self,
        side: str,
        track: BallTrack,
        timestamp_s: float,
        frame_widths: dict[str, int],
    ) -> bool:
        if (
            self._confirmation_last is None
            or self._confirmation_track_id != track.track_id
            or track.center is None
            or self.max_continuity_speed_px_per_second <= 0.0
        ):
            return True
        elapsed_s = timestamp_s - self._confirmation_last[2]
        if elapsed_s <= 0.0:
            return False
        distance = float(
            np.hypot(
                float(track.center[0]) - self._confirmation_last[0],
                float(track.center[1]) - self._confirmation_last[1],
            )
        )
        width = max(
            1,
            int(frame_widths.get(side, self.reference_frame_width)),
        )
        scale = self.frame_scale_overrides.get(
            side,
            max(0.25, width / self.reference_frame_width),
        )
        return (
            distance / elapsed_s
            <= self.max_continuity_speed_px_per_second * scale
        )

    def _track_is_in_handoff_roi(self, track: BallTrack) -> bool:
        if track.center is None or self._handoff_target_roi is None:
            return False
        return (
            self._handoff_target_roi[0]
            <= float(track.center[0])
            <= self._handoff_target_roi[2]
            and self._handoff_target_roi[1]
            <= float(track.center[1])
            <= self._handoff_target_roi[3]
        )

    def _start_confirmation(
        self,
        side: str,
        track: BallTrack,
        timestamp_s: float,
    ) -> None:
        self._confirmation_track_id = track.track_id
        self._confirmation_hits = 1
        self._remember_confirmation(track, timestamp_s)
        self.state = f"{side}_confirming"

    def _remember_confirmation(
        self,
        track: BallTrack,
        timestamp_s: float,
    ) -> None:
        if track.center is None:
            self._confirmation_last = None
            return
        self._confirmation_last = (
            float(track.center[0]),
            float(track.center[1]),
            float(timestamp_s),
        )

    def _cancel_confirmation(self) -> None:
        self._confirmation_track_id = None
        self._confirmation_hits = 0
        self._confirmation_last = None

    def _clear_handoff_window(self) -> None:
        self._handoff_source_side = None
        self._handoff_target_side = None
        self._handoff_expires_at_s = -1.0
        self._handoff_target_roi = None
        self._cancel_confirmation()

    def _refresh_active_state(self, timestamp_s: float) -> None:
        if self.active_side is None:
            self.state = "no_ball"
            return
        if self._handoff_allows(
            self._handoff_target_side or "",
            timestamp_s,
        ):
            self.state = (
                f"{self.active_side}_to_"
                f"{self._handoff_target_side}_armed"
            )
        else:
            self.state = f"{self.active_side}_active"

    def _best_continuous_observed(
        self,
        side: str,
        tracks: list[BallTrack],
        timestamp_s: float,
        frame_widths: dict[str, int],
        *,
        excluded_track_id: Optional[int] = None,
    ) -> Optional[BallTrack]:
        candidates = []
        for track in tracks:
            if track.status != "observed" or track.track_id == excluded_track_id:
                continue
            if (
                self.observation_first
                and self._is_primary_observation(track)
            ):
                candidates.append(track)
            elif self._is_continuous(
                side,
                track,
                timestamp_s,
                frame_widths,
            ):
                candidates.append(track)
            else:
                self.continuity_gate_rejections += 1
        return max(candidates, key=self._score, default=None)

    def _is_continuous(
        self,
        side: str,
        track: BallTrack,
        timestamp_s: float,
        frame_widths: dict[str, int],
    ) -> bool:
        previous = self._last_output_by_side.get(side)
        if (
            previous is None
            or track.center is None
            or self.max_continuity_speed_px_per_second <= 0.0
        ):
            return True
        elapsed_s = float(timestamp_s) - previous[2]
        if elapsed_s <= 0.0:
            return True
        distance = float(
            np.hypot(
                float(track.center[0]) - previous[0],
                float(track.center[1]) - previous[1],
            )
        )
        width = max(
            1,
            int(frame_widths.get(side, self.reference_frame_width)),
        )
        scale = self.frame_scale_overrides.get(
            side,
            max(0.25, width / self.reference_frame_width),
        )
        return (
            distance / elapsed_s
            <= self.max_continuity_speed_px_per_second * scale
        )

    def _prediction_crossed_net_exit(
        self,
        side: str,
        track: BallTrack,
        frame_widths: dict[str, int],
    ) -> bool:
        if track.status != "predicted" or track.center is None:
            return False
        width = max(
            1,
            int(frame_widths.get(side, self.reference_frame_width)),
        )
        center_x = float(track.center[0])
        edge = self.net_edges[side]
        return center_x >= float(width) if edge == "right" else center_x < 0.0

    def _remember_output(
        self,
        side: str,
        track: BallTrack,
        timestamp_s: float,
    ) -> None:
        if track.center is None:
            return
        self._last_output_by_side[side] = (
            float(track.center[0]),
            float(track.center[1]),
            float(timestamp_s),
        )

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None:
            timestamp = self._fallback_timestamp_s
            self._fallback_timestamp_s += 1.0 / 30.0
            return timestamp
        timestamp = float(timestamp_s)
        self._fallback_timestamp_s = timestamp + 1.0 / 30.0
        return timestamp

    def _activate(self, side: str, track: BallTrack) -> None:
        if side not in SIDES:
            raise ValueError(f"Unsupported side: {side}")
        self.active_side = side
        self.active_local_track_id = track.track_id

    @staticmethod
    def _find_track(
        tracks: list[BallTrack],
        track_id: Optional[int],
    ) -> Optional[BallTrack]:
        if track_id is None:
            return None
        return next((track for track in tracks if track.track_id == track_id), None)

    @classmethod
    def _best_observed_across_sides(
        cls,
        by_side: dict[str, list[BallTrack]],
    ) -> Optional[tuple[str, BallTrack]]:
        candidates = [
            (side, track)
            for side in SIDES
            if (track := cls._best_observed(by_side[side])) is not None
        ]
        return max(candidates, key=lambda item: cls._score(item[1]), default=None)

    @classmethod
    def _best_observed(cls, tracks: list[BallTrack]) -> Optional[BallTrack]:
        observed = [track for track in tracks if track.status == "observed"]
        return max(observed, key=cls._score, default=None)

    def _best_primary_observed(
        self,
        tracks: list[BallTrack],
        *,
        excluded_track_id: Optional[int] = None,
    ) -> Optional[BallTrack]:
        observed = [
            track
            for track in tracks
            if self._is_primary_observation(track)
            and track.track_id != excluded_track_id
        ]
        return max(observed, key=self._score, default=None)

    def _is_primary_observation(self, track: BallTrack) -> bool:
        return (
            track.status == "observed"
            and track.confirmed
            and self._observation_source(track)
            in self.primary_observation_sources
        )

    @staticmethod
    def _observation_source(track: BallTrack) -> str:
        return str(
            track.observation_source or track.source
        ).strip().lower()

    @staticmethod
    def _score(track: BallTrack) -> float:
        return (
            2.0 * float(track.confidence)
            + 0.02 * min(int(track.hits), 20)
            + 0.25 * float(track.motion_confirmed)
            + 0.10 * float(track.confirmed)
        )
