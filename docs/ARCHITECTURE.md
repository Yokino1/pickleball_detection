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
TemporalMotionFilter (enabled in revision 9)
    |-- camera-compensated frame difference
    |-- local motion evidence
    v
MultiBallTracker
    |-- high-confidence association and track creation
    |-- low-confidence recovery for existing tracks
    |-- camera-compensated motion confirmation
    |-- impact/reversal recovery with bounded gap growth
    |-- close-range primary-observation bounce recovery
    |-- consecutive primary-observation filter-lag correction
    |-- configurable CV or CA Kalman prediction
    |-- expiry and later re-identification with a new ID
    v
FrameResult.ball_tracks[]
    |-- JSONL output
    `-- observed/predicted overlay and per-ID trail
```

## Current official pipeline

`configs/tracking.yaml` revision 9 integrates the ball pipeline with the
low-frequency person-contact side branch:

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

The person detector is low frequency on both desktop and future board runs. Person boxes
are propagated every frame, so contact evidence remains available on the four
frames between detector calls. Only boxes marked `eligible_player` are passed to
the impact-recovery gate. Person detection never creates or predicts a ball.

The former mainline, temporal and physics configurations are archived under
`legacy/ball_tracking_handoff/configs/maintained_history/`. Active code does not
import or depend on them.

`apps/track_video.py` only depends on `src/tracking`. It does not import court calibration or projection.

## Module ownership

完整目录、模块责任、测试归属和文档权威关系统一维护在
[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)。本节只描述运行时架构和关键依赖。

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
- `dual_camera/runner.py`: offline same-index paired-video processing.
- `dual_camera/projection_replay.py`: inference-free court-panel replay from one completed R9 run.
- `dual_camera/rendering.py`: side-by-side presentation and header layout.
- `dual_camera/artifacts.py`: output names, partial promotion and integrity validation.
- `court/layout.py`: canonical 20 x 44 ft court geometry and shared coordinate definition.
- `court/calibration.py`: fixed-camera manual keypoints, homography construction and validation.
- `court/projector.py`: read-only image-to-court projection of the selected global ball.
- `court/events.py`: read-only candidate bounce/out/second-bounce/hit state interpretation.
- `court/factory.py`: shared projector/renderer/event-interpreter assembly.
- `court/renderer.py`: blank court framework, projected point and trail presentation.
- `runtime/frame_packet.py`: capture-timestamped left/right frame contracts.
- `runtime/synchronization/queues.py`: bounded latest-frame queues and drop accounting.
- `runtime/synchronization/pairer.py`: timestamp-nearest pairing, skew checks and stale-frame drops.

`ground_detection/pickleball_court_detector_handoff.py` 是独立的首帧标定上游工具，
不是活动运行时模块。它的输出经人工复核后进入正式配置；`src/court` 和
`src/tracking` 都不得 import 它，也不会在每帧重新检测球场。

The dependency direction is `apps/tools -> factory/core`. Core modules never import an
application entry point. `apps/track_video.py` and `apps/track_dual_halves.py` are thin
CLIs and do not own tracking policy.

The current repository boundary is:

```text
apps/track_dual_halves.py
    -> src/tracking/dual_camera/runner.py          offline paired-file regression
       -> src/tracking/factory.py                 shared model/pipeline assembly
       -> left/right BallTrackingPipeline         independent local state
       -> CrossHalfBallCoordinator                one global ball
       -> src/court/FixedCourtProjector           read-only selected-ball projection
       -> src/court/CourtEventInterpreter         read-only candidate event state
       -> src/court/CourtPanelRenderer            blank framework output

apps/replay_court_projection.py
    -> saved dual_tracking.mp4 + left/right/global JSONL
    -> src/tracking/dual_camera/projection_replay.py
    -> shared src/court assembly                  no model inference
    -> replacement court panel + derived JSONL

future live entry
    -> src/runtime/capture/                       not implemented
    -> src/runtime/synchronization/               contracts implemented
    -> src/runtime/inference/                     not implemented
    -> src/tracking/                              reuse algorithm core
    -> global single-ball selection
    -> shared src/court projection + events       same live frame stream
    -> src/runtime/outputs/                       not implemented
```

`projection_replay.py` is strictly a desktop debugging and offline-regression
optimization. It is not an alternative deployment architecture. The RK3588S
runtime must consume live paired frames and execute detection, tracking, global
selection, per-camera fixed-calibration projection and event interpretation in
one online flow. Production must not depend on pre-generated MP4 or JSONL.

## Dual-camera offline path and robot target

The dual-camera path keeps independent pixel coordinate systems:

```text
left camera  -> local pipeline --+
                                 +-> global single-ball coordinator -> global JSONL
right camera -> local pipeline --+
                                                                    |
                                                                    v
                                                       fixed per-side court projector
                                                                    |
                                                                    v
                                               court JSON fields + blank court panel
```

A net-bound trajectory activates a short receiving-side entry band. The receiver
still runs full-frame YOLO every frame; after a miss it retries YOLO on that band.
Fast-motion proposals are allowed only in an activated handoff ROI or an existing
fast-track prediction ROI. Revision 9 gives a confirmed YOLO/ONNX observation priority
over a prediction or missing output on the other side. Handoff arming, entry-ROI
membership and consecutive confirmation remain mandatory for auxiliary fast-motion
candidates, so motion evidence cannot impersonate a primary detector observation.
Local pixel coordinates are never directly compared between cameras. When fixed-camera
calibration is enabled, each side independently maps only the already selected global
ball into the shared ground-plane court coordinate system; the projection has no feedback
path into either local pipeline or the coordinator. In strict mode, a
source-side prediction that passes its configured net-facing image edge is retained
only inside the local tracker; it is suppressed from global output and rendering.
A confirmed side switch clears both rendering trails so points from different camera
pixel spaces are never joined.

Observation-first begins after the local pipeline has accepted a model detection.
Raw YOLO/ONNX boxes are still subject to confidence filtering, duplicate removal,
temporal-motion filtering, track confirmation and local association. The offline
coordinator also allows a confirmed primary observation on the other side to preempt
an old-side prediction without an armed handoff. Production must reconcile this
recall-first behavior with a stricter cross-camera transition state machine.

The current `dual_camera/runner.py` is an offline paired-file regression runner.
It requires matching file metadata, reads the same frame index from both inputs,
then invokes the shared detector for the left and right pipelines sequentially.
This preserves offline time alignment but is not the final live-camera runtime.
Thus the current application-level path is paired and synchronized but serial:
left read, right read, left pipeline, right pipeline, coordinator, court projection,
render and write. OpenCV or CUDA libraries may use internal worker threads, but the
Python runner does not run the two camera pipelines in a thread pool or separate
processes.
For the maintained pre-cropped offline inputs it derives a shared physical scale
from the sum of the left and right image widths. Explicit per-camera overrides
remain available for independently calibrated real cameras.

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

`src/runtime` now provides hardware-independent `FramePacket`, `FramePair`,
`BoundedLatestQueue` and `TimestampPairer` components. Camera drivers, RKNN
contexts and robot I/O remain intentionally unimplemented until their concrete
interfaces are available.

## Track lifecycle

1. A detection above `high_conf` starts a tentative track.
2. After `min_hits`, the track is confirmed.
3. High-confidence detections are associated first.
4. Detections between `low_conf` and `high_conf` may recover an existing track but cannot create one.
5. A missed track remains visible as `predicted` for `max_prediction_ms`; fast tracks use the shorter `fast_max_prediction_ms`.
6. It remains internally recoverable until `max_missing_ms`, then expires.
7. A later unmatched detection creates a new ID. No visual re-identification is currently claimed.

The current official configuration clusters duplicate ball boxes, requires motion in both raw and
camera-compensated coordinates before exposing a new track, sleeps stationary tracks, recovers bounded
impact/reversal jumps and rejects physically implausible displacement. The current revision and archived
desktop predecessors are documented in `docs/VERSIONS.md`.

## Supported boundary

`FrameResult.ball_tracks` is the current multi-object contract. `FrameResult.ball_track` remains only for
legacy single-ball consumers and is populated by the new pipeline only when exactly one track is emitted.

The inherited court projection, player detection, event logic and old single-ball pipeline are stored under
`legacy/ball_tracking_handoff/`. They are historical reference only: active code must not import from `legacy`
and archived code is excluded from compile, lint, tests and release packages.

The active person-contact implementation lives under `src/tracking`;
it does not import the historical player detector from `legacy`.

`tools/check_project_refs.py` enforces the active/legacy boundary and validates local Markdown links.
The current implementation state, known gaps and handoff checklist are maintained in
`docs/HANDOFF.md`.
