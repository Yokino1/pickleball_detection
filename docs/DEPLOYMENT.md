# Edge Deployment

## Current target

- Cameras: two physical half-court cameras at 60 FPS each.
- Ball inference: every frame on both streams, or 120 images/s total.
- Person inference: every five frames on both streams, or 24 images/s total.
- Candidate board: Rockchip RK3588S-class hardware; exact SKU, RAM, camera
  interface and power mode are still pending.
- Portable reference: ONNX Runtime.
- Expected release runtime for RK3588S: RKNN after conversion and accuracy
  validation.

The supplied board name must be checked before release. Official Rockchip
material documents RK3588/RK3588S; the project does not treat `RK3688S` as a
confirmed SKU.

The RK3588S-class NPU is not assumed to meet the target solely from its advertised
TOPS. The final decision uses end-to-end measurements including two-camera
preprocessing, ball and person inference, postprocessing, tracking, synchronization
and thermal behavior.

## Decision order

The board model, accelerator, operating system, available RAM and target input FPS must be recorded before
choosing the final export format. Use ONNX Runtime as the portable CPU baseline; switch to the board vendor's
native runtime only after measuring the same test set.

Recommended format by hardware family:

- CPU-only: ONNX Runtime or OpenVINO on supported Intel hardware.
- NVIDIA GPU/Jetson: TensorRT FP16 first, then calibrated INT8.
- Rockchip NPU: RKNN with representative INT8 calibration.
- Other NPUs: use the vendor compiler, but retain ONNX as the reference artifact.

The current `tracking_edge.yaml` is a single-stream ONNX Runtime portability
baseline. It is not yet the dual-camera RKNN release profile. Do not add a release
RKNN YAML until the board SKU and converted model artifact are fixed.

## Dual-camera 60 FPS budget

One synchronized frame pair arrives every:

```text
1000 / 60 = 16.67 ms
```

The production runtime must own the accelerator in one inference scheduler.
Benchmark these implementations behind the same paired-frame interface:

1. RKNN batch 2, if the converted model and runtime support it efficiently.
2. Controlled batch-1 left/right sequencing with one loaded model.
3. Multiple RKNN contexts only if board measurements show a benefit without
   unstable memory or latency.

Do not start with two independent Python processes that each load ball and person
models onto the same NPU.

The live input queue must be bounded. When processing falls behind, stale frames
are dropped and counted; they are not accumulated. Tracking and handoff use
capture timestamps rather than `frame_index / fps`.

Provisional acceptance targets:

- sustained 60 synchronized frame pairs/s;
- p95 pair processing at or below 16.67 ms, or a measured pipelined equivalent
  with no accumulating queue delay;
- capture skew, dropped frames and dropped pairs reported;
- control output isolated from MP4 encoding and verbose diagnostic writes;
- 30-minute thermal and memory soak on the final power mode.

## Export

FP32 reference:

```powershell
python tools/export_model.py --model artifacts/models/best.pt --format onnx --imgsz 640
```

INT8 candidate using the real pickleball dataset for calibration:

```powershell
python tools/export_model.py `
  --model artifacts/models/best.pt --format onnx --imgsz 640 `
  --precision int8 --data datasets/cleaned_ball_detection/data.yaml
```

The export script records model size, versions, precision and calibration settings beside the model. Current
Ultralytics export supports quantized ONNX and requires representative calibration data for INT8. Do not use
a generic calibration dataset for the release artifact.

Reference: [Ultralytics model export documentation](https://docs.ultralytics.com/modes/export/).

## Quantization gate

Validate FP32 first, then compare the quantized artifact against it:

```powershell
python tools/validate_model.py `
  --model artifacts/models/best_int8.onnx --split test --imgsz 640 `
  --baseline artifacts/benchmarks/fp32_test.json `
  --output artifacts/benchmarks/int8_test.json
```

Provisional release limits until the product SLA is confirmed:

- `mAP50` drop no more than 1.5 percentage points;
- recall drop no more than 2 percentage points;
- tracking IDF1 drop no more than 2 percentage points on the temporal set;
- model artifact preferably no more than 20 MB;
- p95 pipeline latency below the frame budget on the actual board;
- no sustained memory growth during a 30-minute video.

If INT8 fails, try FP16 or mixed precision before increasing the model. Never accept a smaller file based only
on aggregate mAP when distant-ball recall has regressed.

## Board benchmark

Run this command on the board itself:

```powershell
python tools/benchmark_runtime.py `
  --config configs/tracking_edge.yaml `
  --frames 1000 `
  --output artifacts/benchmarks/board_runtime.json
```

Record the exact board SKU, power mode, runtime/provider version, cooling state, camera resolution and model
checksum in the release model card.

The existing benchmark is a single-stream portability check. Before release it
must be extended with a live or recorded dual-stream 60 FPS benchmark that reports
pair latency, throughput, synchronization skew, queue depth and drops.

Reference: [Ultralytics benchmark documentation](https://docs.ultralytics.com/modes/benchmark/).
