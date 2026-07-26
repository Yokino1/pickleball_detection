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
TemporalMotionFilter (optional)
    |-- camera-compensated frame difference
    |-- local motion evidence
    v
MultiBallTracker
    |-- high-confidence association and track creation
    |-- low-confidence recovery for existing tracks
    |-- camera-compensated motion confirmation
    |-- impact/reversal recovery with bounded gap growth
    |-- configurable CV or CA Kalman prediction
    |-- expiry and later re-identification with a new ID
    v
FrameResult.ball_tracks[]
    |-- JSONL output
    `-- observed/predicted overlay and per-ID trail
```

## Person-contact experiment

`tracking_person_contact.yaml` adds an optional side branch without changing the
three maintained ball-only configurations:

```text
frame
  +--> ball detector every frame ---------------------------+
  |                                                         |
  +--> person detector every 5 frames                       |
         -> PersonBoxTracker every frame                    |
         -> PlayerSelector                                  |
         -> eligible player boxes                           |
                                                            v
                              contact-gated MultiBallTracker
```

The person detector is low frequency on both desktop and board runs. Person boxes
are propagated every frame, so contact evidence remains available on the four
frames between detector calls. Only boxes marked `eligible_player` are passed to
the impact-recovery gate. Person detection never creates or predicts a ball.

`apps/track_video.py` only depends on `src/tracking`. It does not import court calibration or projection.

## Module ownership

- `ball_detector.py`: detector protocol and Ultralytics implementation.
- `onnx_detector.py`: Torch-free ONNX Runtime implementation and post-processing.
- `multi_ball_tracker.py`: track lifecycle, association, optional CV/CA Kalman models and physical gating.
- `motion_models.py`: constant-velocity and constant-acceleration Kalman implementations.
- `camera_motion.py`: robust inter-frame camera translation estimation.
- `temporal_motion.py`: camera-compensated consecutive-frame motion evidence for detector candidates.
- `person_detector.py`: optional Ultralytics person inference.
- `person_tracking.py`: low-frequency box continuation and player/spectator selection.
- `ball_pipeline.py`: per-frame orchestration and timing diagnostics.
- `types.py`: stable JSON-serializable contracts.
- `overlay.py`: presentation only; it must not change tracking state.
- `factory.py`: shared detector, tracker and pipeline assembly.
- `fast_motion.py`: ROI-gated consecutive high-speed motion proposals.
- `run_manifest.py`: reproducible code/config/input/output metadata.
- `dual_camera/coordinator.py`: global single-ball arbitration and handoff advice.
- `dual_camera/runner.py`: synchronized two-stream processing.
- `dual_camera/rendering.py`: side-by-side presentation and header layout.
- `dual_camera/artifacts.py`: output names, partial promotion and integrity validation.

The dependency direction is `apps/tools -> factory/core`. Core modules never import an
application entry point. `apps/track_video.py` and `apps/track_dual_halves.py` are thin
CLIs and do not own tracking policy.

## Dual-camera experiment and robot target

The dual-camera path keeps independent pixel coordinate systems:

```text
left camera  -> local pipeline --+
                                 +-> global single-ball coordinator -> global JSONL
right camera -> local pipeline --+
```

A net-bound trajectory activates a short receiving-side entry band. The receiver
still runs full-frame YOLO every frame; after a miss it retries YOLO on that band.
Fast-motion proposals are allowed only in an activated handoff ROI or an existing
fast-track prediction ROI. No uncalibrated coordinate mapping between cameras is
performed.

The current `dual_camera/runner.py` is an offline paired-file regression runner.
It requires matching file metadata, reads the same frame index from both inputs,
then invokes the shared detector for the left and right pipelines sequentially.
This preserves offline time alignment but is not the final live-camera runtime.

The robot target is two physical 60 FPS cameras:

```text
left capture worker  --+
                       +-> timestamp pairer -> inference scheduler
right capture worker --+                         |
                                                  +-> left local pipeline
                                                  +-> right local pipeline
                                                           |
                                                           v
                                                global ball coordinator
```

At 60 FPS, one frame pair arrives every 16.67 ms. Per-frame ball detection creates
120 ball images/s of inference demand. Person detection every five frames creates
another 24 images/s. One scheduler owns the accelerator and may implement batch 2
or controlled batch-1 sequencing according to measurements on the final board.
Independent processes must not compete for one NPU by default.

Live capture, timestamp synchronization, bounded queues and accelerator scheduling
belong under `src/runtime/`. They must not be added to the offline runner. The
accepted boundary and provisional latency gates are recorded in
`docs/decisions/0002-dual-camera-60fps-edge-runtime.md`.

## Track lifecycle

1. A detection above `high_conf` starts a tentative track.
2. After `min_hits`, the track is confirmed.
3. High-confidence detections are associated first.
4. Detections between `low_conf` and `high_conf` may recover an existing track but cannot create one.
5. A missed track remains visible as `predicted` for `max_prediction_ms`; fast tracks use the shorter `fast_max_prediction_ms`.
6. It remains internally recoverable until `max_missing_ms`, then expires.
7. A later unmatched detection creates a new ID. No visual re-identification is currently claimed.

The mainline configuration clusters duplicate ball boxes, requires motion in both raw and
camera-compensated coordinates before exposing a new track, sleeps stationary tracks, recovers bounded
impact/reversal jumps and rejects physically implausible displacement. The maintained desktop variants are
documented in `docs/VERSIONS.md`.

## Supported boundary

`FrameResult.ball_tracks` is the current multi-object contract. `FrameResult.ball_track` remains only for
legacy single-ball consumers and is populated by the new pipeline only when exactly one track is emitted.

The inherited court projection, player detection, event logic and old single-ball pipeline are stored under
`legacy/ball_tracking_handoff/`. They are historical reference only: active code must not import from `legacy`
and archived code is excluded from compile, lint, tests and release packages.

The active person-contact experiment is a new implementation under `src/tracking`;
it does not import the historical player detector from `legacy`.

`tools/check_project_refs.py` enforces the active/legacy boundary and validates local Markdown links.
