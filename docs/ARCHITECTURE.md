# Architecture

## Main pipeline

```text
video frame
    |
    v
CameraMotionEstimator
    |-- sparse background optical flow
    |-- RANSAC global translation
    v
BallDetector (PT or ONNX)
    |
    | 0-N BallDetection
    v
MultiBallTracker
    |-- high-confidence association and track creation
    |-- low-confidence recovery for existing tracks
    |-- camera-compensated motion confirmation
    |-- impact/reversal recovery with bounded gap growth
    |-- per-track constant-velocity Kalman prediction
    |-- expiry and later re-identification with a new ID
    v
FrameResult.ball_tracks[]
    |-- JSONL output
    `-- observed/predicted overlay and per-ID trail
```

`apps/track_video.py` only depends on `src/tracking`. It does not import court calibration or projection.

## Module ownership

- `ball_detector.py`: detector protocol and Ultralytics implementation.
- `onnx_detector.py`: Torch-free ONNX Runtime implementation and post-processing.
- `multi_ball_tracker.py`: track lifecycle, association and motion prediction.
- `camera_motion.py`: robust inter-frame camera translation estimation.
- `ball_pipeline.py`: per-frame orchestration and timing diagnostics.
- `types.py`: stable JSON-serializable contracts.
- `overlay.py`: presentation only; it must not change tracking state.

## Track lifecycle

1. A detection above `high_conf` starts a tentative track.
2. After `min_hits`, the track is confirmed.
3. High-confidence detections are associated first.
4. Detections between `low_conf` and `high_conf` may recover an existing track but cannot create one.
5. A missed track remains visible as `predicted` for `max_predict_frames`.
6. It remains internally recoverable until `max_missing_frames`, then expires.
7. A later unmatched detection creates a new ID. No visual re-identification is currently claimed.

The reference configuration also clusters duplicate ball boxes, requires motion in both raw and
camera-compensated coordinates before exposing a new track, sleeps stationary tracks, recovers bounded
impact/reversal jumps and rejects physically implausible displacement. All valid moving tracks are emitted;
the number of balls is not configured manually.

## Supported boundary

`FrameResult.ball_tracks` is the current multi-object contract. `FrameResult.ball_track` remains only for
legacy single-ball consumers and is populated by the new pipeline only when exactly one track is emitted.

The inherited court projection, player detection, event logic and old single-ball pipeline are stored under
`legacy/handoff_projection/`. They are historical reference only: active code must not import from `legacy`
and archived code is excluded from compile, lint, tests and release packages.

`tools/check_project_refs.py` enforces the active/legacy boundary and validates local Markdown links.
