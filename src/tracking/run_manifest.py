"""Reproducible metadata records for generated tracking runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return {
            "available": True,
            "revision": revision,
            "branch": branch,
            "dirty": dirty,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "available": False,
            "revision": None,
            "branch": None,
            "dirty": None,
        }


def create_manifest(
    *,
    project_root: Path,
    run_id: str,
    run_type: str,
    config_path: Path,
    config: dict,
    inputs: list[dict],
    parameters: dict,
) -> dict:
    # Keep the source-file identity for backwards compatibility, but also
    # capture the actual in-memory configuration. CLI entry points may apply
    # run-local overrides after loading YAML, so the file hash alone is not a
    # reproducible description of the run.
    effective_config = json.loads(json.dumps(config, ensure_ascii=False))
    profile = dict(effective_config.get("profile", {}))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_type": run_type,
        "status": "running",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "code": collect_git_state(project_root),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "profile": profile,
            "effective_sha256": sha256_json(effective_config),
            "effective": effective_config,
        },
        "inputs": inputs,
        "parameters": parameters,
        "outputs": {},
        "summary": {},
    }


def complete_manifest(
    manifest: dict,
    *,
    outputs: dict,
    summary: dict,
) -> dict:
    completed = dict(manifest)
    completed["status"] = "completed"
    completed["completed_at_utc"] = utc_now()
    completed["outputs"] = outputs
    completed["summary"] = summary
    return completed


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
