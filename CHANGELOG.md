# Changelog

Last updated: 2026-07-27

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
- A strict experimental cross-camera handoff state machine requiring an armed
  source-to-target window, receiver entry-ROI membership, consecutive receiver
  confirmation and post-switch locking.
- Hardware-independent live-runtime frame contracts, bounded latest-frame queues
  and timestamp pairing with skew, stale-drop, capacity-drop and out-of-order
  diagnostics.
- A handoff snapshot documenting current implementation, unresolved observation
  and cross-camera boundaries, quantization prerequisites and Git asset controls.

### Changed

- Promoted `pickleball_tracking` revision 9 after full `test`/`test_2`
  diagnostic replay. A confirmed same-camera YOLO/ONNX observation now
  immediately replaces an old predicted local ID; this does not count as a
  cross-camera side switch and resets the rendered trail through the existing
  local-ID-change rule.
- Added a 55-reference-pixel consecutive-primary-observation recovery. It only
  corrects a lagging Kalman/NIS state when the previous and current observations
  are both primary-model results, consecutive, under the speed ceiling and
  mutually direction-consistent. Raw 90-degree turns, fast-motion proposals,
  missing-frame recovery and distant points cannot use this path.
- Aligned `direction_gate_min_hits` with the four-observation CA maturity
  boundary. The strict 35-pixel bounce signature may now operate inside an
  eligible-player contact zone, because valid ground bounces can occur near a
  player's feet; all existing speed, time, vertical-reversal and horizontal
  continuity bounds remain active.
- Added `primary_continuity_recoveries` and
  `same_side_observation_preemptions` diagnostics plus regression coverage for
  same-side observation priority, contact-zone bounce recovery and consecutive
  primary-observation filter correction.
- Promoted `pickleball_tracking` revision 8 as a deliberately narrow refinement
  of revision 7. A high-confidence YOLO/ONNX observation may now reset a
  descending track into an upward rebound only when it is within 35 reference
  pixels of the last observed landing point, occurs within 80 ms, preserves
  material horizontal direction, stays outside eligible-player contact zones
  and remains under the existing global speed ceiling. The cross-camera
  coordinator and revision-7 observation-first policy are unchanged. A fixed,
  non-extending 80 ms stabilization window lets subsequent nearby upward model
  observations absorb the first rebound frames without reopening general
  impact recovery.
- Added `bounce_recoveries` tracker diagnostics and close-versus-distant
  rebound regression tests.
- Promoted `pickleball_tracking` revision 7 in `configs/tracking.yaml` to the
  only maintained algorithm profile. It integrates temporal evidence, CA/NIS
  physics constraints, person-contact gating, observation-first arbitration and
  strict auxiliary handoff. Revision 7 also derives scale from paired-crop total
  width and breaks rendering trails at local-ID or physical discontinuities.
- Archived the retired mainline, temporal and physics YAML/CMD entry points under
  `legacy/ball_tracking_handoff/`; active apps and default scripts no longer
  depend on them.
- Confirmed the target SoC as RK3588S. Complete board/carrier, memory, camera,
  cooling, calibration and measured RKNN runtime parameters remain deployment
  records rather than assumed configuration values.
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
- Earlier desktop A/B work separated mainline, temporal and physics configurations;
  those profiles are now retained only as archived history.
- `tracking.yaml` now represents revision 7 and enables the integrated maintained
  pipeline rather than the retired CV-only mainline.
- Single-target overlays use one fluorescent-green colour for every ID, including predictions and labels.
- The former `motion_tracking_v3` result directory is named `motion_tracking_mainline`.
- Project documentation now points to one maintained revision and clearly labels
  the three former desktop variants as archived history.
- Shared component construction moved from application code to `src/tracking/factory.py`.
- Dual-camera synchronization, coordination, rendering and artifact handling were split into focused
  modules under `src/tracking/dual_camera/`.
- Dual-camera runs now refuse accidental overwrite unless `--overwrite` is explicitly provided.
- Offline paired-crop streams derive one physical scale from
  `(left_width + right_width) / reference_frame_width`; this produces 1.0 for
  the 1280-wide `test`/`test_2` sources and 3.0 for a 3840-wide source.
- The integrated revision-7 profile continues person inference and contact gating while hiding person boxes
  in rendered videos by default.
- Global continuity gating remains mandatory for predictions and auxiliary
  candidates. Confirmed primary observations remain observation-first, while
  revision 7 breaks the rendered trail instead of drawing an impossible segment.
- Person-contact recovery now expires after 120 ms and requires an observed or predicted endpoint
  near the player box; an arbitrarily long crossing segment is no longer contact evidence.
- The deployment target is recorded as two 60 FPS physical cameras and a confirmed
  RK3588S SoC; exact complete-board and installation details still require recording.
- The existing paired-file runner is explicitly classified as an offline
  regression tool; live capture, timestamp synchronization and accelerator
  scheduling belong under `src/runtime/`.
- Revision 5 introduced strict handoff so an arbitrary receiver observation could
  no longer switch the global ID after a source miss.
- Strict handoff suppresses source-side predictions after they pass the configured
  net-facing image boundary, and a confirmed side switch clears both rendering trails.
- Revision 6 introduced observation preemption: confirmed YOLO/ONNX
  observations preempt a prediction or missing output on the other camera, while
  fast-motion proposals still require the strict handoff path.
- Local revision-7 single-ball arbitration ranks observed tracks before predicted
  tracks; quality scoring is applied only inside the same observation class.
- Revision 7 fixes the r6 regression that applied the 3840-source scale `3.0`
  to 1280-source paired videos, which had inflated association and speed gates
  and allowed large observed jumps to inherit an old local ID.
- Rendering now resets the active trail when the camera side changes, the selected
  local track ID changes, or adjacent displayed positions exceed the calibrated
  speed limit. The current model observation remains visible; no smoothing or old
  prediction is allowed to hide it.
- Dual-camera MP4 artifacts now use the neutral `_dual_tracking.mp4` suffix instead
  of the retired experiment-specific `_dual_person_contact.mp4` suffix.
- Audited tracking thresholds, output names, RK3588S status, training-data
  preconditions, quantization commands and benchmark limitations against current
  code and `configs/tracking.yaml`.

### Removed

- Intermediate `motion_tracking_v4_solid` and `motion_tracking_v6_time_based` results.
- Old `sideview_results`, single-target/time-based previews and `_smoke_*` outputs.
- The obsolete `configs/legacy_simple_tracking.yaml` tuning configuration.
- The duplicate `docs/ROADMAP.md`; active research work is consolidated in `docs/NEXT_STEPS.md`.
- Local Python, pytest, Ruff and misplaced Ultralytics cache directories.

### Validation

- 89 unit tests pass, including dual-camera coordination, paired-crop scale,
  same-side primary-observation preemption, bounce recovery and filter-lag
  correction,
  derivation, trail discontinuity handling, runtime synchronization,
  ROI retry, fast-motion proposals,
  run manifests and smoke-retention tooling.
- Active Python imports, syntax, local Markdown links, official profile identity
  and critical documented thresholds pass `tools/check_project_refs.py`.
- Configuration checks enforce `configs/tracking.yaml` as the only maintained
  profile, with identity `pickleball_tracking` revision 9, and reject the three
  retired YAML names if they return to the active config directory.
