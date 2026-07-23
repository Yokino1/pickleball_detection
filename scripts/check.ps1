param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

& $Python tools/check_project_refs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m unittest discover -s tests -v
exit $LASTEXITCODE
