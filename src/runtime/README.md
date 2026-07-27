# Live Runtime Boundary

This directory owns reusable robot-runtime contracts. It still contains no
camera-driver, RKNN-context or robot-control implementation.

Implemented modules:

```text
frame_packet.py       timestamped left/right frame and paired-frame contracts
synchronization/      bounded queues, timestamp pairing and stale-frame accounting
```

The synchronization components are hardware-independent and accept frames from
future concurrent capture workers. They reject duplicate or out-of-order
packets, pair frames within a configured capture-time tolerance, discard the
older side when skew is too large and expose capacity/stale-drop diagnostics.

Future board-specific modules belong here:

```text
capture/              concurrent camera acquisition and hardware timestamps
inference/            RKNN paired-frame scheduling and accelerator ownership
outputs/              non-blocking robot telemetry and optional diagnostics
```

Detection, tracking, player selection and cross-camera handoff policy remain
under `src/tracking/`. The existing `src/tracking/dual_camera/runner.py` is the
offline paired-video regression runner and must not become the live camera event
loop.

See `docs/decisions/0002-dual-camera-60fps-edge-runtime.md` before adding runtime
code.
