"""Validate active/legacy boundaries and local Markdown links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIRS = ("apps", "src", "tools", "tests")
FORBIDDEN_IMPORTS = (
    "src.tracking.ball_track",
    "src.tracking.events",
    "src.tracking.pipeline",
    "src.tracking.player_detector",
    "legacy.handoff_projection",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
CONFIG_STATUSES = {"maintained", "experimental", "deployment"}
OFFICIAL_CONFIG = ROOT / "configs" / "tracking.yaml"
OFFICIAL_PROFILE = {
    "name": "pickleball_tracking",
    "revision": 9,
    "status": "maintained",
}
RETIRED_ACTIVE_CONFIGS = (
    ROOT / "configs" / "tracking_temporal.yaml",
    ROOT / "configs" / "tracking_physics.yaml",
    ROOT / "configs" / "tracking_person_contact.yaml",
)
FORBIDDEN_RUNNER_SCRIPTS = (
    ROOT / "scripts" / "run_tracking.cmd",
    ROOT / "scripts" / "run_single_tracking.cmd",
    ROOT / "scripts" / "run_dual_tracking.cmd",
    ROOT / "scripts" / "run_mainline_tracking.cmd",
    ROOT / "scripts" / "run_temporal_tracking.cmd",
    ROOT / "scripts" / "run_physics_tracking.cmd",
    ROOT / "scripts" / "run_person_contact_tracking.cmd",
)
DOCUMENTED_CONFIG_VALUES = {
    ("detector", "low_conf"): "detector.low_conf",
    ("tracker", "low_conf"): "tracker.low_conf",
    ("tracker", "high_conf"): "tracker.high_conf",
    ("tracker", "max_speed_px_per_second"): "max_speed_px_per_second",
    ("tracker", "max_prediction_ms"): "max_prediction_ms",
    ("tracker", "fast_prediction_speed_px_per_second"): (
        "fast_prediction_speed_px_per_second"
    ),
    ("tracker", "fast_max_prediction_ms"): "fast_max_prediction_ms",
    ("tracker", "impact_recovery_max_missing_ms"): (
        "impact_recovery_max_missing_ms"
    ),
    ("tracker", "bounce_recovery_max_displacement_px"): (
        "bounce_recovery_max_displacement_px"
    ),
    ("tracker", "primary_continuity_gate_px"): (
        "primary_continuity_gate_px"
    ),
    ("tracker", "max_flight_direction_change_deg"): (
        "max_flight_direction_change_deg"
    ),
    ("tracker", "nis_gate_threshold"): "nis_gate_threshold",
}


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
    maintained: list[Path] = []
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
        if status == "maintained":
            maintained.append(path)

    if maintained != [OFFICIAL_CONFIG]:
        rendered = ", ".join(str(path.relative_to(ROOT)) for path in maintained) or "none"
        errors.append(
            "the only maintained tracking profile must be configs/tracking.yaml; "
            f"found {rendered}"
        )
    if OFFICIAL_CONFIG.exists():
        official = yaml.safe_load(OFFICIAL_CONFIG.read_text(encoding="utf-8")) or {}
        profile = official.get("profile", {})
        for key, expected in OFFICIAL_PROFILE.items():
            if profile.get(key) != expected:
                errors.append(
                    f"{OFFICIAL_CONFIG.relative_to(ROOT)} profile.{key} must be "
                    f"{expected!r}, found {profile.get(key)!r}"
                )
    for path in RETIRED_ACTIVE_CONFIGS:
        if path.exists():
            errors.append(
                f"{path.relative_to(ROOT)} is retired and must remain in legacy history"
            )
    return errors


def check_documented_config_values() -> list[str]:
    if not OFFICIAL_CONFIG.exists():
        return []
    config = yaml.safe_load(OFFICIAL_CONFIG.read_text(encoding="utf-8")) or {}
    rules_path = ROOT / "docs" / "TRACKING_RULES.md"
    if not rules_path.exists():
        return ["missing docs/TRACKING_RULES.md"]
    rules = rules_path.read_text(encoding="utf-8")
    errors: list[str] = []
    for keys, label in DOCUMENTED_CONFIG_VALUES.items():
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                errors.append(
                    f"{OFFICIAL_CONFIG.relative_to(ROOT)} is missing {'.'.join(keys)}"
                )
                break
            value = value[key]
        else:
            rendered = (
                str(int(value))
                if isinstance(value, float) and value.is_integer()
                else str(value)
            )
            token = f"`{label}: {rendered}`"
            matches = re.findall(
                rf"`{re.escape(label)}:\s*([^`]+)`",
                rules,
            )
            documented = False
            for match in matches:
                try:
                    documented = float(match.strip()) == float(value)
                except (TypeError, ValueError):
                    documented = match.strip() == rendered
                if documented:
                    break
            if not documented:
                errors.append(
                    f"{rules_path.relative_to(ROOT)} must document {token} "
                    "from configs/tracking.yaml"
                )
    return errors


def check_maintenance_files() -> list[str]:
    required = (
        ROOT / "configs" / "README.md",
        ROOT / "docs" / "MAINTENANCE.md",
        ROOT / "experiments" / "README.md",
        ROOT / "outputs" / "README.md",
    )
    errors = [
        f"missing required maintenance file {path.relative_to(ROOT)}"
        for path in required
        if not path.exists()
    ]
    errors.extend(
        f"{path.relative_to(ROOT)} must not be generated; publish pasteable "
        "Conda CMD commands in documentation"
        for path in FORBIDDEN_RUNNER_SCRIPTS
        if path.exists()
    )
    return errors


def main() -> int:
    errors = (
        check_python_syntax()
        + check_import_boundaries()
        + check_markdown_links()
        + check_config_profiles()
        + check_documented_config_values()
        + check_maintenance_files()
    )
    if errors:
        for error in errors:
            print(f"[reference-error] {error}", file=sys.stderr)
        return 1
    print(
        "[checks] syntax, active imports, Markdown links, profile identity "
        "and documented thresholds are valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
