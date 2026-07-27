# Contributing

Use `docs/DEVELOPMENT.md` as the working agreement. Every behavior change should include a focused test and
must preserve the JSON output contract unless the configuration/schema version is deliberately incremented.

Before review, run:

```cmd
D:\anacondaa\envs\torch-cu128\python.exe tools\check_project_refs.py
D:\anacondaa\envs\torch-cu128\python.exe -m unittest discover -s tests -v
D:\anacondaa\envs\torch-cu128\python.exe -m compileall -q apps src tools tests
git diff --check
git status --short
```

Do not commit source datasets, generated videos, training runs or exported board models.
Do not use `git add .`; review and stage exact files.
