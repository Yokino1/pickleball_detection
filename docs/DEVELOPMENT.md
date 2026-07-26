# Development and Maintenance

## Supported workflow

1. Put behavior changes in `src/tracking`, not in the video loop.
2. Add or update tests before tuning configuration values.
3. Keep run-specific values in versioned YAML with `schema_version`.
4. Write generated models, metrics and videos under `artifacts/` or `outputs/`.
5. Update `CHANGELOG.md` and the model card for every release candidate.
6. Treat `legacy/ball_tracking_handoff` as read-only history; do not import it from active code.
7. Keep CLIs thin; shared construction belongs in `src/tracking/factory.py`.
8. Store meaningful experiment metadata in `experiments/` and generated files under the categorized
   `outputs/` layout.
9. Keep live camera capture, timestamp pairing, accelerator scheduling and robot I/O under
   `src/runtime/`; the paired-file runner remains an offline regression tool.

Run local checks:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1 -Python python
```

With development dependencies installed:

```powershell
ruff check src apps tools tests
pytest -q
```

## Configuration policy

- `configs/tracking.yaml`: mainline CV tracking without temporal filtering.
- `configs/tracking_temporal.yaml`: mainline plus lightweight consecutive-frame filtering.
- `configs/tracking_physics.yaml`: temporal filtering plus CA/NIS physics constraints.
- `configs/tracking_edge.yaml`: board-oriented ONNX Runtime run.
- Do not add a new YAML for every tuning attempt; use an output-side experiment note until a version is promoted.
- Every maintained configuration declares `profile.name`, `profile.revision` and `profile.status`.
- Never silently change configuration semantics; increment `schema_version` when compatibility breaks.
- Tracker thresholds consume detector outputs, so record detector confidence and input size with tracker metrics.

## Version control policy

- The repository root is this project directory, not the outer `D:\ball` workspace.
- Track source, configuration, tests, documentation and small metadata reports.
- Ignore raw videos, datasets, checkpoints, exported engines and generated outputs.
- Keep commits scoped to one behavior or maintenance concern.
- Do not commit datasets, videos, generated model formats or benchmark outputs.
- Review changes for detection recall, ID stability, runtime and output-schema compatibility.
- A pull request changing association or prediction must add a regression sequence or synthetic unit test.
- Run `scripts/check.ps1` before every merge; CI applies the same import-boundary and link checks.
- Never overwrite a completed run implicitly; use a new run ID or explicit `--overwrite`.

Detailed run naming, manifest and smoke retention rules are in
[MAINTENANCE.md](MAINTENANCE.md).

## Release checklist

- All unit tests pass.
- Clean test-set detector report exists.
- Temporal tracking report exists.
- Quantized model passes the precision gate.
- Runtime report comes from the target board and target power mode.
- Model card includes dataset version, code revision, configuration and known limitations.
- A 30-minute soak test completes without failure or unbounded memory growth.
