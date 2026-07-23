# Changelog

## Unreleased

### Added

- Multi-ball tracking with stable IDs, two-stage confidence association and Kalman gap prediction.
- Detection-only video pipeline and overlay independent of court projection.
- Torch-free ONNX Runtime detector for edge deployment.
- Clean training and edge configurations.
- Reproducible train, export, validation and runtime benchmark tools.
- Core tracking tests and project maintenance documentation.
- Evaluation-first side-view acceptance plan and automated project-reference checks.
- Folder-based batch video processing with model reuse, resumable outputs and recursive discovery.
- Ball-specific duplicate suppression, motion confirmation, stationary-track sleep, physical displacement
  gating and configurable active-track output limits.
- Sparse-optical-flow camera motion compensation and per-frame track-state diagnostics.
- Bounded impact/reversal reassociation, adaptive high-speed prediction and slow rolling-ball confirmation.

### Changed

- The primary product objective is detection and tracking; court coordinate projection is deferred.
- `FrameResult` now exposes `ball_tracks` while retaining the legacy `ball_track` field.
- Runtime, training and development dependencies are separated.
- Inherited court projection and old single-ball code moved to `legacy/handoff_projection/`.
- Local videos are organized into `data/sideview_raw` and `data/reference`; generated debug outputs are not
  retained as project artifacts.
- Tracking now emits every valid moving ball by default and suppresses static scene or screen-fixed candidates.
