# Changelog

Last updated: 2026-07-26

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
- Camera-compensated consecutive-frame motion filtering with per-candidate diagnostics.
- Flight-direction gating and a minimum missing-time guard for impact recovery.
- Optional six-state constant-acceleration Kalman tracking, NIS/acceleration gates and
  speed-continuous prediction horizons for physics-constrained A/B testing.
- Dedicated `tracking_temporal.yaml` and `tracking_physics.yaml` experiment configurations.
- Matching `run_mainline_tracking.cmd`, `run_temporal_tracking.cmd` and
  `run_physics_tracking.cmd` launchers.
- `docs/VERSIONS.md` as the single comparison entry for the maintained tracking variants.
- Fixed-colour overlay regression coverage and temporal/physics tracking regression tests.
- A design branch for person-contact-gated recovery, perspective-aware local scaling and
  low-frequency dual-model board deployment.
- Player-versus-spectator selection using play-area foot-point priors, persistent person tracks,
  spectator exclusion regions and match-size limits before contact gating.
- A runnable person-contact experiment with a dedicated config and CMD launcher, YOLO person
  inference every five frames, per-frame person-box continuation and contact-gated impact recovery.
- Synchronized dual-camera single-ball coordination with receiving-side ROI handoff and conservative
  high-speed motion proposals for short YOLO gaps.
- Per-run manifests containing Git state, configuration identity, input metadata, output files and
  runtime summaries.
- A maintained output taxonomy, experiment registry and project maintenance guide.
- An accepted dual-camera 60 FPS robot-runtime architecture decision with a
  dedicated live-runtime ownership boundary.

### Changed

- The primary product objective is detection and tracking; court coordinate projection is deferred.
- `FrameResult` now exposes `ball_tracks` while retaining the legacy `ball_track` field.
- Runtime, training and development dependencies are separated.
- Inherited court projection and old single-ball code moved to `legacy/ball_tracking_handoff/`.
- Local videos are organized into `data/sideview_raw` and `data/reference`; generated debug outputs are not
  retained as project artifacts.
- Tracking now defaults to one primary output track while retaining internal recovery candidates; setting
  `max_output_tracks` to `0` restores all valid moving tracks.
- Overlay history is limited to the latest 10 points in the reference and edge configurations.
- Tracking velocity, motion gates and Kalman prediction now use real frame timestamps and pixels per
  second; visible prediction and internal recovery limits are configured in milliseconds.
- Maintained desktop runs are separated into mainline, temporal and physics configurations with matching
  CMD launchers and named output directories.
- `tracking.yaml` is the CV mainline and no longer enables temporal filtering implicitly;
  `tracking_temporal.yaml` enables the lightweight frame-difference filter explicitly.
- Single-target overlays use one fluorescent-green colour for every ID, including predictions and labels.
- The former `motion_tracking_v3` result directory is named `motion_tracking_mainline`.
- Project documentation now points to the three maintained variants and a consolidated next-step plan.
- Shared component construction moved from application code to `src/tracking/factory.py`.
- Dual-camera synchronization, coordination, rendering and artifact handling were split into focused
  modules under `src/tracking/dual_camera/`.
- Dual-camera runs now refuse accidental overwrite unless `--overwrite` is explicitly provided.
- Dual-camera stream scale overrides preserve pixels-per-metre after half-court cropping, so tracker,
  fast-motion and handoff speed gates no longer shrink with crop width.
- The person-contact profile continues person inference and contact gating while hiding person boxes
  in rendered videos by default.
- Global dual-camera output now rejects physically impossible frame-to-frame jumps even when the
  selected local track ID changes.
- Person-contact recovery now expires after 120 ms and requires an observed or predicted endpoint
  near the player box; an arbitrarily long crossing segment is no longer contact evidence.
- The deployment target is recorded as two 60 FPS physical cameras and an
  RK3588S-class candidate board, with the exact SKU still requiring confirmation.
- The existing paired-file runner is explicitly classified as an offline
  regression tool; live capture, timestamp synchronization and accelerator
  scheduling belong under `src/runtime/`.

### Removed

- Intermediate `motion_tracking_v4_solid` and `motion_tracking_v6_time_based` results.
- Old `sideview_results`, single-target/time-based previews and `_smoke_*` outputs.
- The obsolete `configs/legacy_simple_tracking.yaml` tuning configuration.
- The duplicate `docs/ROADMAP.md`; active research work is consolidated in `docs/NEXT_STEPS.md`.
- Local Python, pytest, Ruff and misplaced Ultralytics cache directories.

### Validation

- 58 unit tests pass, including dual-camera coordination, ROI retry, fast-motion proposals,
  run manifests and smoke-retention tooling.
- Active Python imports, syntax and local Markdown links pass `tools/check_project_refs.py`.
- Configuration assembly verifies:
  - mainline: constant-velocity KF without temporal filtering;
  - temporal: constant-velocity KF with temporal filtering;
  - physics: constant-acceleration KF with temporal filtering and physical gates.
