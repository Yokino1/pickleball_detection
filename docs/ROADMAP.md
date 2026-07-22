# Roadmap

## Phase 1: Engineering baseline

- [x] Clean and de-duplicate object detection data.
- [x] Separate ball tracking from court projection.
- [x] Add multi-ball IDs and short-gap Kalman prediction.
- [x] Add PT and Torch-free ONNX inference paths.
- [x] Add tests, versioned configs and maintenance documentation.

## Phase 2: Real-camera baseline

- [ ] Freeze and checksum the inherited `ball_best.pt` checkpoint.
- [ ] Capture representative side-view, oblique and no-ball clips from the intended camera setup.
- [ ] Build an untouched side-view detection set and continuous tracking set.
- [ ] Run the inherited model and publish an error-slice baseline report.

## Phase 3: Targeted quality work

- [ ] Decide from the baseline whether detector fine-tuning is necessary.
- [ ] Build tiny-ball, blur, ground-ball, multi-ball and hard-negative metric slices.
- [ ] Review false negatives and false positives with an error gallery.
- [ ] If needed, fine-tune with target-camera samples plus replay data and compare against the inherited model.
- [ ] Select the smallest model meeting the recall target.

## Phase 4: Tracking quality

- [ ] Capture and annotate 20-30 continuous unseen-camera clips with `track_id`.
- [ ] Measure IDF1/HOTA, ID switches, false tracks/minute and gap recovery.
- [ ] Tune association gates by resolution and FPS.
- [ ] Add camera-cut detection and stream reset behavior.

## Phase 5: Board release

- [ ] Record target board, accelerator, RAM, OS and FPS SLA.
- [ ] Export FP32/FP16/INT8 candidates with representative calibration.
- [ ] Run detector and tracking accuracy regression for every candidate.
- [ ] Benchmark p50/p95 latency, memory, temperature and power on target hardware.
- [ ] Complete soak test, model card and reproducible release bundle.

## Deferred

- Court homography and image-to-court coordinate projection.
- Bounce, in/out and scoring logic.
- Cross-camera identity continuity.
