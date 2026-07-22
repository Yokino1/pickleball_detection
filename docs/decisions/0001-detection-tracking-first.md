# ADR 0001: Detection and Tracking First

Status: Accepted

## Context

The inherited side-view pipeline combined single-ball tracking, court calibration, coordinate projection,
player detection and event logic. Several court-tracking dependencies were absent, and the product priority
changed to reliable ball detection and tracking on an edge board.

## Decision

The maintained main path is now `BallDetector -> MultiBallTracker -> FrameResult.ball_tracks`. Court
projection remains optional legacy code and cannot be a dependency of the main video application.

## Consequences

- Detection and tracking can be tested and deployed independently.
- The output contract supports multiple simultaneous balls and predicted states.
- Coordinate projection is deferred and can later consume track centers without changing detector ownership.
- The legacy single-track field remains temporarily for compatibility.
