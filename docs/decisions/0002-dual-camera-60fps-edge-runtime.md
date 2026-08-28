# ADR 0002: Dual-Camera 60 FPS Edge Runtime

Status: Accepted; RK3588S SoC confirmed; complete board environment pending

## Context

The product target is two physical cameras, one per half court, both capturing at
60 FPS. The current `apps/track_dual_halves.py` path is an offline regression
runner: it reads an already synchronized file pair, processes the left pipeline
and then the right pipeline, and combines their results.

The SoC is confirmed as Rockchip RK3588S. The exact complete board/carrier,
memory, camera interface, operating system, NPU runtime, cooling and power mode
are not yet frozen.

At 60 FPS the frame-pair period is 16.67 ms. Running the ball detector on every
frame means 120 ball images per second. Running the person detector every five
frames on both cameras adds 24 person images per second.

## Decision

The live robot runtime will use:

```text
two capture workers
    -> hardware/system capture timestamps
    -> bounded frame-pair synchronizer
    -> one accelerator-owned inference scheduler
    -> two independent local tracking pipelines
    -> one cross-camera single-ball coordinator
    -> robot telemetry/control output
```

- Capture is concurrent; tracking consumes timestamp-matched frame pairs.
- The inference scheduler owns the accelerator and exposes a paired-frame
  contract. It may use batch 2, controlled batch-1 sequencing, or multiple
  contexts only after board benchmarks.
- Two independent processes must not load competing copies of the same model on
  one NPU by default.
- The live queue is bounded and latest-frame-first. Old frames are dropped
  instead of allowing control latency to grow without bound.
- Left and right trackers retain independent pixel coordinate systems and
  camera-specific scale, play-area and spectator-exclusion configuration.
- Production cross-camera switching must use a time-bounded state machine. The
  offline revision-9 coordinator already applies strict arming, entry ROI,
  consecutive confirmation and locking to auxiliary `fast_motion` candidates,
  but a confirmed primary YOLO/ONNX observation may still preempt an old-side
  prediction without an armed handoff. That recall-first exception must be
  reconciled with production transition evidence before robot release.
- Rendering, MP4 encoding and verbose JSONL are diagnostics and must not block
  the control path.

## Provisional acceptance gates

These gates are measured on the final board, camera drivers and power mode:

- sustained input: two 60 FPS streams without unbounded queue growth;
- sustained inference demand: 120 ball images/s plus 24 person images/s;
- frame-pair processing p95 at or below 16.67 ms, or an explicitly validated
  pipelined equivalent with no accumulating latency;
- capture timestamp skew and dropped-pair counts recorded for every run;
- 30-minute soak without memory growth, thermal collapse or timestamp drift;
- every precision supported by the frozen RKNN toolchain compared with the ONNX
  reference on the same fixed dual-camera regression set.

## Code ownership

- `src/tracking/`: detector-independent tracking algorithms and data contracts.
- `src/tracking/dual_camera/`: cross-camera arbitration, handoff policy, offline
  paired-file runner, rendering and run artifacts.
- `src/runtime/`: future live capture, timestamp synchronization, inference
  scheduling and non-blocking robot output.
- `apps/`: thin offline and live entry points.

The offline runner remains the reproducible regression baseline. Live-camera
concerns must not be added directly to its frame loop.

## Consequences

- Batch 2 is a benchmark candidate, not a hard-coded assumption.
- The current ONNX Runtime edge profile remains a portable reference, not the
  RK3588S release runtime.
- A release RKNN profile is created only after the complete board environment and converted
  model are available.
- Full-frame 960-pixel inference on both 60 FPS streams is not assumed feasible;
  model size, input size and runtime are selected from measured recall and
  latency.
