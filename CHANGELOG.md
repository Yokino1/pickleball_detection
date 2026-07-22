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

### Changed

- The primary product objective is detection and tracking; court coordinate projection is deferred.
- `FrameResult` now exposes `ball_tracks` while retaining the legacy `ball_track` field.
- Runtime, training and development dependencies are separated.
- Inherited court projection and old single-ball code moved to `legacy/handoff_projection/`.
