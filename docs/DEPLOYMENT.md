# Edge Deployment

## Current target

- Cameras: two physical half-court cameras at 60 FPS each.
- Ball inference: every frame on both streams, or 120 images/s total.
- Person inference: every five frames on both streams, or 24 images/s total.
- Confirmed SoC: Rockchip RK3588S; exact carrier/complete device, RAM, camera
  interface and power mode are still pending.
- Portable reference: ONNX Runtime.
- Expected release runtime for RK3588S: RKNN after conversion and accuracy
  validation.

The SoC name is confirmed as RK3588S. Release records must still identify the
complete board/carrier, memory, operating system, camera interfaces, cooling and
power mode.

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
RKNN YAML until the complete board environment and converted model artifact are fixed.
Its tracker parameters are not functionally equivalent to `configs/tracking.yaml`
revision 9, so it must not be used to claim revision-9 end-to-end accuracy.

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

### Preconditions

Before any export or quantization run, record:

- source PT SHA-256 and model card;
- exact Git revision and `configs/tracking.yaml` profile/revision;
- exact Ultralytics, ONNX/ONNX Runtime and later RKNN Toolkit/runtime versions;
- input size, NMS setting and calibration dataset SHA-256/inventory;
- fixed detector validation set and fixed continuous tracking regression set.

`datasets/cleaned_ball_detection/data.yaml` is a local asset path convention and
is not guaranteed to exist in a fresh checkout. The export/validation commands
must not be run until the local dataset manifest and referenced images have been
verified. Dataset files, model binaries and generated engines are not committed
to ordinary Git.

### Portable ONNX reference

Export the inherited source model using its actual repository name. Compare
precision at the same input size before changing resolution. Revision 9 currently
uses 960, so the first portable reference is:

```powershell
D:\anacondaa\envs\torch-cu128\python.exe tools\export_model.py `
  --model artifacts/models/ball_best.pt `
  --format onnx --precision fp32 --imgsz 960
```

Create an ONNX INT8 candidate with representative pickleball calibration data:

```powershell
D:\anacondaa\envs\torch-cu128\python.exe tools\export_model.py `
  --model artifacts/models/ball_best.pt --format onnx --imgsz 960 `
  --precision int8 --data datasets/cleaned_ball_detection/data.yaml
```

The helper writes metadata beside the exported artifact, including model size,
Ultralytics version, precision, input size and calibration settings. Keep this
metadata with the candidate. Do not use a generic calibration dataset.

An ONNX INT8 file is a portable comparison artifact; it is not an RKNN release
artifact. RKNN conversion, operator compatibility, preprocessing, output decoding
and NPU accuracy must be validated separately with the exact RKNN toolchain and
RK3588S runtime.

Reference: [Ultralytics model export documentation](https://docs.ultralytics.com/modes/export/).

## Quantization gate

Validate FP32 first, then compare the quantized artifact against it:

```powershell
D:\anacondaa\envs\torch-cu128\python.exe tools\validate_model.py `
  --model <fp32-reference.onnx> `
  --data datasets/cleaned_ball_detection/data.yaml `
  --split test --imgsz 960 `
  --output artifacts/benchmarks/fp32_test.json

D:\anacondaa\envs\torch-cu128\python.exe tools\validate_model.py `
  --model <candidate-int8.onnx> `
  --data datasets/cleaned_ball_detection/data.yaml `
  --split test --imgsz 960 `
  --baseline artifacts/benchmarks/fp32_test.json `
  --output artifacts/benchmarks/int8_test.json
```

Generate `fp32_test.json` first with the same dataset, split and input size. The
current validation helper checks detector `mAP50` and recall deltas only. It does
not calculate IDF1/HOTA, dual-camera handoff accuracy, person-contact behavior or
RKNN end-to-end latency; those require separate fixed-video regression reports.

Provisional release limits until the product SLA is confirmed:

- `mAP50` drop no more than 1.5 percentage points;
- recall drop no more than 2 percentage points;
- tracking IDF1 drop no more than 2 percentage points on the temporal set
  (manual/external evaluator until a repository tool exists);
- model artifact preferably no more than 20 MB;
- p95 pipeline latency below the frame budget on the actual board;
- no sustained memory growth during a 30-minute video.

If INT8 fails, evaluate a higher-precision mode only if the exact RKNN
toolchain/target supports it, or retain a measured ONNX/CPU fallback. Do not
invent an FP16 or mixed-precision release mode from desktop behavior. Never
accept a smaller file based only on aggregate mAP when distant-ball recall has
regressed.

## Board benchmark

The existing command is only a single-stream detector plus basic-tracker
microbenchmark:

```powershell
D:\anacondaa\envs\torch-cu128\python.exe tools\benchmark_runtime.py `
  --config configs/tracking_edge.yaml `
  --frames 1000 `
  --output artifacts/benchmarks/board_runtime.json
```

Record the exact board SKU, power mode, runtime/provider version, cooling state, camera resolution and model
checksum in the release model card.

It does not run the revision-9 temporal filter, camera-motion estimator, person
detector, dual-camera coordinator, video encoding or live queues. Therefore its
throughput cannot be used as the 60 FPS robot acceptance result.

Before release, add a live or recorded dual-stream RKNN benchmark that reports:

- capture and pair timestamps, p50/p95/p99 skew and dropped frames/pairs;
- preprocessing, ball inference, person inference, postprocessing, tracking and
  coordinator latency separately;
- end-to-end control-output age, queue depth and stale-frame drops;
- model/context count, batch strategy, NPU core policy and memory;
- temperature, throttling and throughput over a 30-minute soak.

Reference: [Ultralytics benchmark documentation](https://docs.ultralytics.com/modes/benchmark/).
