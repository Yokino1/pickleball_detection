# Edge Deployment

## Decision order

The board model, accelerator, operating system, available RAM and target input FPS must be recorded before
choosing the final export format. Use ONNX Runtime as the portable CPU baseline; switch to the board vendor's
native runtime only after measuring the same test set.

Recommended format by hardware family:

- CPU-only: ONNX Runtime or OpenVINO on supported Intel hardware.
- NVIDIA GPU/Jetson: TensorRT FP16 first, then calibrated INT8.
- Rockchip NPU: RKNN with representative INT8 calibration.
- Other NPUs: use the vendor compiler, but retain ONNX as the reference artifact.

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

Reference: [Ultralytics benchmark documentation](https://docs.ultralytics.com/modes/benchmark/).
