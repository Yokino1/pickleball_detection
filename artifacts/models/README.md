# Model Registry

Large model binaries should be stored in controlled artifact storage rather than ordinary Git history.
For every release model, keep a model card based on `docs/MODEL_CARD_TEMPLATE.md` and record:

- SHA-256 checksum;
- source training run and code revision;
- dataset cleaning report;
- export format, precision and input size;
- detector, tracking and target-board benchmark reports.

`ball_best.pt` is the inherited checkpoint. Its frozen metadata is recorded in
`ball_best.model-card.md`. It has not yet passed side-view, temporal tracking or edge quantization acceptance
and must not be treated as the production release model.
