# Development and Maintenance

## Supported workflow

1. Put behavior changes in `src/tracking`, not in the video loop.
2. Add or update tests before tuning configuration values.
3. Keep run-specific values in versioned YAML with `schema_version`.
4. Write generated models, metrics and videos under `artifacts/` or `outputs/`.
5. Update `CHANGELOG.md` and the model card for every release candidate.
6. Treat `legacy/handoff_projection` as read-only history; do not import it from active code.

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

- `configs/tracking.yaml`: quality-oriented desktop/reference run.
- `configs/tracking_edge.yaml`: board-oriented ONNX Runtime run.
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

## Release checklist

- All unit tests pass.
- Clean test-set detector report exists.
- Temporal tracking report exists.
- Quantized model passes the precision gate.
- Runtime report comes from the target board and target power mode.
- Model card includes dataset version, code revision, configuration and known limitations.
- A 30-minute soak test completes without failure or unbounded memory growth.
