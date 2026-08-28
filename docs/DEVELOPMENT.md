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
10. Update the authoritative document named in `MAINTENANCE.md` whenever a
    default threshold, CLI, output contract, module boundary or deployment claim changes.

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

- `configs/tracking.yaml`: the only maintained algorithm profile, currently revision 9.
- `configs/tracking_edge.yaml`: a single-stream ONNX Runtime portability and deployment-research profile,
  not a second algorithm version or an RKNN release profile.
- The retired mainline, temporal and physics YAML files live under
  `legacy/ball_tracking_handoff/configs/maintained_history/` for explicit historical regression only.
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
- Never use `git add .`; stage reviewed source, config, test and documentation paths explicitly.
- Keep runtime commands as pasteable Conda CMD blocks in documentation; do not
  generate active `run_*.cmd` launcher files.
- Never include local videos, datasets, model binaries, generated outputs or business DOCX files
  in an algorithm commit.

Detailed run naming, manifest and smoke retention rules are in
[MAINTENANCE.md](MAINTENANCE.md).
The current implementation snapshot and unresolved handoff risks are in
[HANDOFF.md](HANDOFF.md).

## Release checklist

- All unit tests pass.
- Clean test-set detector report exists.
- Temporal tracking report exists.
- Quantized model passes the precision gate.
- Runtime report comes from the target board and target power mode.
- Model card includes dataset version, code revision, configuration and known limitations.
- A 30-minute soak test completes without failure or unbounded memory growth.
