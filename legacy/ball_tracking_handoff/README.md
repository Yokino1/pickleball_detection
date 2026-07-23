# Ball Tracking Handoff

This folder contains the pickleball ball tracking code and nearby runtime files for handoff.

## Contents

- `src/tracking/`
  - `ball_detector.py`: ball detector abstraction, including YOLO detector and null detector.
  - `ball_track.py`: simple Kalman-style ball tracker, gating, prediction, ROI handling.
  - `pipeline.py`: per-frame orchestration for ball detection, tracking, court projection, and event detection.
  - `events.py`: ball seen/missing/lost and simple trajectory event logic.
  - `types.py`: shared dataclasses for detections, tracks, frame output, events.
  - `player_detector.py`: optional player detector wrapper; does not drive ball tracking.

- `apps/run_sideview_video.py`
  Main side-view video processing entry point.

- `tools/annotate_video.py`
  Older/utility annotation script with ball detection and overlay logic.

- `src/court/`
  Minimal court projection/rendering helpers needed by the tracking pipeline and overlay:
  `layout.py`, `observation.py`, `projector.py`, `renderer.py`, `zones.py`, `calibration_compat.py`.

- `configs/`
  Existing side-view configs and court calibration examples.

- `artifacts/models/ball_best.pt`
  Current YOLO ball detection model.

- `requirements.txt`
  Python dependency list copied from the project root.

## Typical Run

From this project root, the main command shape is:

```powershell
python apps/run_sideview_video.py --config configs/sideview_test2_preview.yaml
```

If running from inside this handoff folder, keep the same relative layout or add this folder to `PYTHONPATH` so `src.*` imports resolve.

## Important Config Fields

In the side-view YAML configs:

- `models.ball_model`: path to `artifacts/models/ball_best.pt`.
- `models.ball_class_id`: ball class id, currently `0`.
- `tracking.min_conf`: detector confidence threshold.
- `tracking.detector_interval`: frame interval for detector calls.
- `tracking.*`: tracker gate, smoothing, missing-frame, and ROI parameters.
- `court.*`: court calibration / projection settings used to map image ball position to court coordinates.

## Handoff Notes

- The ball tracking core is in `src/tracking/`.
- The ball detector can run as YOLO or `NullBallDetector` when the model is unavailable.
- `SideViewPipeline` is the integration point: detector -> tracker -> optional court projection -> events -> frame result.
- Court tracking/calibration is adjacent infrastructure. Ball tracking can run without a court projector, but court coordinates and zones will be unavailable.
- The model file is included for convenience; verify licensing/storage expectations before sharing outside the project.
