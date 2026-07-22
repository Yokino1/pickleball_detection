# Model Card: inherited-ball-best

## Identity

- Status: frozen inherited baseline; not a production release
- Received checkpoint: `ball_best.pt`
- SHA-256: `7160047ffe947d31cb77601129bc73a1d39e2bb7af54967f5c8825d7d60bf9c8`
- File size: 20,645,947 bytes
- Task: YOLO object detection
- Stored classes: `{0: item}`
- Stored training input size: 1280
- Stored training mode: single class
- Metadata recorded: 2026-07-22

## Provenance

The checkpoint was inherited with the project. Its embedded data path points to
`/home/disk/kongchaoran/code/DTDP/metal_defect/pickleball/pickleball.yaml`, which is not part of this
repository. Exact training code, dataset revision, metrics and code revision are unknown.

## Intended baseline use

Runtime maps class `0` to pickleball. The detector may output multiple class-0 boxes per frame; multi-object
identity and short-gap prediction are supplied by `MultiBallTracker`.

Use this checkpoint as the first comparison on real side-view acceptance clips. Do not overwrite it. Store
fine-tuned candidates under a new training-run name and require a separate model card.

## Missing acceptance evidence

- Real target-camera side-view detection report
- Continuous multi-ball tracking report
- Tiny, blur, ground-ball, occlusion, multi-ball and no-ball slice metrics
- Quantized accuracy comparison
- Target-board latency, memory, thermal and soak reports
