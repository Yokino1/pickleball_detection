# Live Runtime Boundary

This directory is reserved for the robot runtime and intentionally contains no
camera- or board-specific implementation yet.

Future modules belong here:

```text
capture/              concurrent camera acquisition and capture timestamps
synchronization/      bounded frame pairing, skew accounting and stale-frame drop
inference/            RKNN/ONNX paired-frame scheduling and accelerator ownership
outputs/              non-blocking robot telemetry and optional diagnostics
```

Detection, tracking, player selection and cross-camera handoff policy remain
under `src/tracking/`. The existing `src/tracking/dual_camera/runner.py` is the
offline paired-video regression runner and must not become the live camera event
loop.

See `docs/decisions/0002-dual-camera-60fps-edge-runtime.md` before adding runtime
code.
