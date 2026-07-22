# Training and Evaluation

## Dataset

Use only:

```text
datasets/cleaned_ball_detection/data.yaml
```

It contains 23,007 images, 25,141 boxes and 1,294 no-ball negatives. Splits are grouped by source clip;
adjacent frames never cross train, validation and test boundaries.

## Evaluation-first strategy

Do not retrain merely because cleaned data is available. Freeze the inherited `artifacts/models/ball_best.pt`
as the first baseline and evaluate it on videos captured from the intended side-view camera position. The
checkpoint is a one-class YOLO detector trained at `imgsz=1280`; its stored class name is `item`, but runtime
maps class `0` to pickleball and accepts 0-N detections per frame.

The cleaned dataset contains 2,045 multi-ball images (8.89% of all images), but is biased toward rear and
broadcast-style views. It cannot replace a side-view acceptance set.

Fine-tuning is justified only when the error review shows a detector problem:

- sustained side-view false negatives: add real side-view positives;
- background false positives: add no-ball and confusing-object negatives from the target camera;
- missed nearby balls: add side-view multi-ball scenes with every visible ball labelled;
- only short detection gaps: tune and evaluate tracking before changing detector weights.

When fine-tuning is required, mix new target-camera samples with a smaller replay sample from the cleaned
dataset to retain rear-view and multi-ball coverage. Split by complete video clip and keep final acceptance
clips untouched.

## Optional new-model baseline

Only after the inherited-model report exists, compare a nano detector at 640, 768 and 960 before increasing
model size. Small-ball recall often benefits more from input resolution and clean labels than a larger
backbone.

```powershell
python tools/train_detector.py `
  --model yolo26n.pt `
  --data datasets/cleaned_ball_detection/data.yaml `
  --imgsz 960 `
  --epochs 150 `
  --name pickleball_nano_960
```

Training output belongs under `artifacts/training/`. Keep the best checkpoint, `args.yaml`, metric plots and
the exact dataset cleaning report together.

## Detector acceptance

Evaluate on the untouched test split:

```powershell
python tools/validate_model.py `
  --model artifacts/training/pickleball_nano_960/weights/best.pt `
  --split test --imgsz 960 `
  --output artifacts/benchmarks/fp32_test.json
```

Track at least `mAP50-95`, `mAP50`, precision and recall. For this project, recall on tiny and motion-blurred
balls is more important than a small precision gain. In addition to aggregate metrics, maintain manual slices:

- distant/tiny ball;
- motion blur;
- ball on ground;
- partial occlusion;
- indoor, outdoor and night;
- multiple balls;
- hard negatives such as shoes, lights, logos and tennis/padel balls.

## Tracking evaluation gap

The current Roboflow exports do not contain reliable temporal `track_id` labels. Before calling tracking
production-ready, annotate 20-30 continuous clips from unseen cameras and venues. Record at least:

- IDF1 or HOTA;
- ID switches;
- track recall;
- longest recovered detection gap;
- false tracks per minute;
- observed versus predicted frame ratio.

Do not tune tracker gates on the final tracking test clips.

See [NEXT_STEPS.md](NEXT_STEPS.md) for the acceptance sequence and decision rules.
