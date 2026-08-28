"""List or remove expired smoke-run directories safely."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_ROOT = PROJECT_ROOT / "outputs" / "smoke"


def find_expired_runs(
    smoke_root: Path,
    *,
    older_than_days: float,
    now_s: float | None = None,
) -> list[Path]:
    if not smoke_root.exists():
        return []
    cutoff_s = (time.time() if now_s is None else now_s) - (
        max(0.0, older_than_days) * 86400.0
    )
    return sorted(
        path
        for path in smoke_root.iterdir()
        if path.is_dir() and path.stat().st_mtime < cutoff_s
    )


def _is_direct_child(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path.parent == resolved_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run by default; only manages outputs/smoke children"
    )
    parser.add_argument("--older-than-days", type=float, default=7.0)
    parser.add_argument(
        "--smoke-root",
        type=Path,
        default=DEFAULT_SMOKE_ROOT,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove the listed directories",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    smoke_root = args.smoke_root.resolve()
    expected_root = DEFAULT_SMOKE_ROOT.resolve()
    if smoke_root != expected_root:
        raise ValueError(
            f"Refusing non-standard smoke root: {smoke_root}; "
            f"expected {expected_root}"
        )

    expired = find_expired_runs(
        smoke_root,
        older_than_days=args.older_than_days,
    )
    if not expired:
        print("[smoke] no expired run directories")
        return 0

    action = "remove" if args.apply else "would remove"
    for path in expired:
        if not _is_direct_child(path, smoke_root):
            raise RuntimeError(f"Unsafe smoke path: {path}")
        print(f"[smoke] {action}: {path}")
        if args.apply:
            shutil.rmtree(path)
    if not args.apply:
        print("[smoke] dry run only; add --apply to remove listed directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
