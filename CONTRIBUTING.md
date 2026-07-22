# Contributing

Use `docs/DEVELOPMENT.md` as the working agreement. Every behavior change should include a focused test and
must preserve the JSON output contract unless the configuration/schema version is deliberately incremented.

Before review, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1 -Python python
```

Do not commit source datasets, generated videos, training runs or exported board models.
