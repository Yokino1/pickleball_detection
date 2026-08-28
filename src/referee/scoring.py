"""Idempotent score bookkeeping, separate from rally interpretation."""

from __future__ import annotations

from dataclasses import replace

from .types import ScoreDecision


class ScoreRecorder:
    """Award at most one point for each rally ID.

    ``rally_point`` awards every resolvable Demo rally to its winner. The optional
    ``server_only`` mode is kept behind the same result contract for later rule
    selection; it does not change rally-winner inference.
    """

    VALID_MODES = {"rally_point", "server_only"}

    def __init__(
        self,
        *,
        teams: tuple[str, str] = ("left", "right"),
        scoring_mode: str = "rally_point",
        initial_score: dict[str, int] | None = None,
    ) -> None:
        if len(teams) != 2 or teams[0] == teams[1]:
            raise ValueError("teams must contain two distinct names")
        if scoring_mode not in self.VALID_MODES:
            raise ValueError(
                f"unsupported scoring_mode={scoring_mode!r}; expected one of {sorted(self.VALID_MODES)}"
            )
        self.teams = tuple(str(team) for team in teams)
        self.scoring_mode = scoring_mode
        supplied = dict(initial_score or {})
        self._score = {team: max(0, int(supplied.get(team, 0))) for team in self.teams}
        self._decisions: dict[int, ScoreDecision] = {}
        self._duplicate_attempts = 0

    @property
    def score(self) -> dict[str, int]:
        return dict(self._score)

    def award(
        self,
        rally_id: int,
        *,
        rally_winner: str | None,
        server_side: str | None,
        status: str = "confirmed",
    ) -> ScoreDecision:
        """Settle one rally, returning the original decision on duplicates."""
        rally_id = int(rally_id)
        if rally_id in self._decisions:
            self._duplicate_attempts += 1
            return replace(self._decisions[rally_id], duplicate=True)

        before = self.score
        point_awarded_to: str | None = None
        if status in {"confirmed", "demo_inferred"} and rally_winner in self.teams:
            if self.scoring_mode == "rally_point":
                point_awarded_to = rally_winner
            elif rally_winner == server_side:
                point_awarded_to = rally_winner
        if point_awarded_to is not None:
            self._score[point_awarded_to] += 1

        decision = ScoreDecision(
            rally_id=rally_id,
            status=str(status),
            rally_winner=rally_winner,
            point_awarded_to=point_awarded_to,
            score_before=before,
            score_after=self.score,
        )
        self._decisions[rally_id] = decision
        return decision

    def diagnostics(self) -> dict:
        return {
            "scoring_mode": self.scoring_mode,
            "score": self.score,
            "settled_rallies": len(self._decisions),
            "settled_rally_ids": sorted(self._decisions),
            "duplicate_award_attempts": self._duplicate_attempts,
        }
