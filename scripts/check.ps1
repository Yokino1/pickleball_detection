param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python -m compileall -q src apps tools tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python tools/check_project_refs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m unittest discover -s tests -v
exit $LASTEXITCODE
