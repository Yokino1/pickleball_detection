"""Validate active/legacy boundaries and local Markdown links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIRS = ("apps", "src", "tools", "tests")
FORBIDDEN_IMPORTS = (
    "src.court",
    "src.tracking.ball_track",
    "src.tracking.events",
    "src.tracking.pipeline",
    "src.tracking.player_detector",
    "legacy.handoff_projection",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
CONFIG_STATUSES = {"maintained", "experimental", "deployment"}


def active_python_files() -> list[Path]:
    files: list[Path] = []
    for directory in ACTIVE_DIRS:
        files.extend((ROOT / directory).rglob("*.py"))
    return sorted(path for path in files if path.resolve() != Path(__file__).resolve())


def check_import_boundaries() -> list[str]:
    errors: list[str] = []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_IMPORTS:
            if token in text:
                errors.append(f"{path.relative_to(ROOT)} imports archived module token {token!r}")
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"src", "tools"} and (
            "from apps" in text or "import apps" in text
        ):
            errors.append(f"{relative} imports an application entry point")
        if relative.parts[0] == "apps" and (
            "from apps." in text or "import apps." in text
        ):
            errors.append(f"{relative} imports another application entry point")
    return errors


def check_python_syntax() -> list[str]:
    errors: list[str] = []
    files = active_python_files() + [Path(__file__).resolve()]
    for path in sorted(set(files)):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)} has invalid syntax: {exc}")
    return errors


def check_markdown_links() -> list[str]:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", "outputs", "artifacts"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            target = target.strip("<>")
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)} has missing link target {raw_target!r}")
    return errors


def check_config_profiles() -> list[str]:
    errors: list[str] = []
    names: dict[str, Path] = {}
    for path in sorted((ROOT / "configs").glob("tracking*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        profile = config.get("profile")
        if not isinstance(profile, dict):
            errors.append(f"{path.relative_to(ROOT)} has no profile metadata")
            continue
        name = profile.get("name")
        revision = profile.get("revision")
        status = profile.get("status")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{path.relative_to(ROOT)} has invalid profile.name")
        elif name in names:
            errors.append(
                f"{path.relative_to(ROOT)} duplicates profile.name from "
                f"{names[name].relative_to(ROOT)}"
            )
        else:
            names[name] = path
        if not isinstance(revision, int) or revision < 1:
            errors.append(f"{path.relative_to(ROOT)} has invalid profile.revision")
        if status not in CONFIG_STATUSES:
            errors.append(f"{path.relative_to(ROOT)} has invalid profile.status")
    return errors


def check_maintenance_files() -> list[str]:
    required = (
        ROOT / "configs" / "README.md",
        ROOT / "docs" / "MAINTENANCE.md",
        ROOT / "experiments" / "README.md",
        ROOT / "outputs" / "README.md",
    )
    return [
        f"missing required maintenance file {path.relative_to(ROOT)}"
        for path in required
        if not path.exists()
    ]


def main() -> int:
    errors = (
        check_python_syntax()
        + check_import_boundaries()
        + check_markdown_links()
        + check_config_profiles()
        + check_maintenance_files()
    )
    if errors:
        for error in errors:
            print(f"[reference-error] {error}", file=sys.stderr)
        return 1
    print("[checks] Python syntax, active imports and local Markdown links are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
