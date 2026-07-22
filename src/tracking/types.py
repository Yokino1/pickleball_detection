"""
Shared data structures for the side-view pickleball video analysis system.

All types are designed to be JSON-serializable so that FrameResult
can be written directly to JSONL without custom encoders.

Coordinate conventions
----------------------
- Image coordinates are in pixels, origin at top-left.
- Bounding boxes use [x1, y1, x2, y2] (inclusive).
- Court coordinates are in feet, origin at far-baseline-left.
- Missing / unknown values are represented as None.
- "unknown" state is represented explicitly, never guessed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JSON helper – keeps numpy scalars and other awkward types out of json.dumps
# ---------------------------------------------------------------------------

def _sanitize(value: Any) -> Any:
    """Return *value* converted to a plain Python type safe for json.dumps.

    Handles:
    - numpy floats / ints  →  float / int
    - enum values          →  str
    - lists/tuples         →  list (recursively sanitised)
    """
    if isinstance(value, enum.Enum):
        return value.value

    # numpy scalars (np.float32, np.int64, …) – checked by name so we
    # don't need a numpy import just for this helper.
    t = type(value).__module__
    if t.startswith("numpy"):
        if hasattr(value, "item"):
            return value.item()
        return float(value)

    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]

    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}

    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BallStatus(str, enum.Enum):
    """Three simple ball states — no complex lifecycle."""
    OBSERVED = "observed"
    PREDICTED = "predicted"
    ABSENT = "absent"


class ProjectionStatus(str, enum.Enum):
    """Court projection state for the current frame."""
    NONE = "none"
    HOMOGRAPHY = "homography"
    CACHED = "cached"
    UNAVAILABLE = "unavailable"


class VisibleEvent(str, enum.Enum):
    """Candidate events the system can detect.

    All events are **candidates** — hypotheses that a human reviewer
    (or later stages) can confirm or reject.  No strong conclusions.
    """

    BALL_SEEN = "ball_seen"
    BALL_MISSING = "ball_missing"
    BOUNCE_CANDIDATE = "bounce_candidate"
    LEAVE_FRAME = "leave_frame"
    PROJECTION_UNAVAILABLE = "projection_unavailable"



# ---------------------------------------------------------------------------
# Detector outputs
# ---------------------------------------------------------------------------

@dataclass
class BallDetection:
    """A single ball detection from the detector.

    All coordinates are in full-frame image pixels.  When a detector runs
    on an ROI crop, the coordinates MUST be mapped back to full-frame
    before constructing this object (add roi.x1, roi.y1 offsets).
    """

    bbox: list[float]          # [x1, y1, x2, y2] in image pixels
    center: list[float]        # [cx, cy] in image pixels
    confidence: float          # 0.0 – 1.0
    source: str = "unknown"    # "yolo", "tracknet", "hsv", "mock", …

    def to_dict(self) -> dict:
        return {
            "bbox": _sanitize(self.bbox),
            "center": _sanitize(self.center),
            "confidence": _sanitize(self.confidence),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Ball track (simplified — 3 states)
# ---------------------------------------------------------------------------

@dataclass
class BallTrack:
    """Ball tracking result for a single frame.

    Three status values::

        observed   ← detection accepted this frame
        predicted  ← no detection, short-term Kalman predict
        absent     ← no reliable ball position
    """

    track_id: Optional[int] = None                # stable ID within one video/stream
    status: str = "absent"                       # "observed" | "predicted" | "absent"
    center: Optional[list[float]] = None          # [cx, cy] in image pixels
    bbox: Optional[list[float]] = None            # [x1, y1, x2, y2] in image pixels
    confidence: float = 0.0
    velocity: Optional[list[float]] = None        # [vx, vy] in pixels / frame
    missing_frames: int = 0                       # consecutive frames without detection
    source: str = "none"                          # "detector" | "prediction" | "none"
    roi: Optional[list[float]] = None             # [x1, y1, x2, y2] suggested search region
    age: int = 0                                  # frames since this track was created
    hits: int = 0                                 # accepted detector observations
    confirmed: bool = False                       # enough observations to trust this track

    def to_dict(self) -> dict:
        return {
            "track_id": _sanitize(self.track_id) if self.track_id is not None else None,
            "status": self.status,
            "center": _sanitize(self.center) if self.center is not None else None,
            "bbox": _sanitize(self.bbox) if self.bbox is not None else None,
            "confidence": _sanitize(self.confidence),
            "velocity": _sanitize(self.velocity) if self.velocity is not None else None,
            "missing_frames": _sanitize(self.missing_frames),
            "source": self.source,
            "roi": _sanitize(self.roi) if self.roi is not None else None,
            "age": _sanitize(self.age),
            "hits": _sanitize(self.hits),
            "confirmed": self.confirmed,
        }


# ---------------------------------------------------------------------------
# Court structures
# ---------------------------------------------------------------------------

@dataclass
class CourtInfo:
    """Per-frame court state.

    The layout is always "pickleball_14pt" — the mini-court can always
    be drawn.  projection_status tells you whether image→court mapping
    is available for this frame.
    """

    layout: str = "pickleball_14pt"
    projection_status: str = "none"
    visible_keypoints: list[int] = field(default_factory=list)
    homography_available: bool = False
    ball_court_xy: Optional[list[float]] = None   # [x, y] in court feet
    ball_zone: Optional[str] = None               # Deprecated projection output; currently not computed.
    projection_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layout": self.layout,
            "projection_status": self.projection_status,
            "visible_keypoints": _sanitize(self.visible_keypoints),
            "homography_available": self.homography_available,
            "ball_court_xy": _sanitize(self.ball_court_xy) if self.ball_court_xy is not None else None,
            "ball_zone": self.ball_zone,
            "projection_warnings": _sanitize(self.projection_warnings),
        }


# ---------------------------------------------------------------------------
# Player detection
# ---------------------------------------------------------------------------

@dataclass
class PlayerDetection:
    """A single player detection."""

    bbox: list[float]                      # [x1, y1, x2, y2] in image pixels
    track_id: Optional[int] = None         # tracking ID if available
    confidence: float = 0.0
    foot_point: Optional[list[float]] = None  # [x, y] estimated foot position

    def to_dict(self) -> dict:
        return {
            "bbox": _sanitize(self.bbox),
            "track_id": _sanitize(self.track_id) if self.track_id is not None else None,
            "confidence": _sanitize(self.confidence),
            "foot_point": _sanitize(self.foot_point) if self.foot_point is not None else None,
        }


# ---------------------------------------------------------------------------
# Per-frame aggregate
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Complete per-frame record emitted by the pipeline."""

    frame_index: int
    timestamp: Optional[float] = None

    # Ball
    ball_detections: list[BallDetection] = field(default_factory=list)
    ball_tracks: list[BallTrack] = field(default_factory=list)
    # Deprecated single-track view retained for legacy side-view consumers.
    ball_track: Optional[BallTrack] = None

    # Court
    court: Optional[CourtInfo] = None

    # Players (empty list when disabled)
    players: list[PlayerDetection] = field(default_factory=list)

    # Events active in this frame
    events: list[str] = field(default_factory=list)

    # Diagnostics
    diagnostics: dict = field(default_factory=lambda: {
        "detector_error": False,
        "dropped_detection_reason": None,
        "overlay_warnings": [],
    })

    # Image dimensions
    width: int = 1280
    height: int = 720

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSONL writing."""
        result: dict = {
            "frame_index": _sanitize(self.frame_index),
            "timestamp": _sanitize(self.timestamp),
            "width": _sanitize(self.width),
            "height": _sanitize(self.height),

            # New fields
            "ball_detections": [d.to_dict() for d in self.ball_detections],
            "ball_tracks": [track.to_dict() for track in self.ball_tracks],
            "ball_track": self.ball_track.to_dict() if self.ball_track else None,
            "court": self.court.to_dict() if self.court else None,
            "players": [p.to_dict() for p in self.players],
            "events": _sanitize(self.events),
            "diagnostics": _sanitize(self.diagnostics),
        }
        return result
