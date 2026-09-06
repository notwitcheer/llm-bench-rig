#!/usr/bin/env python3
"""Backfill a provenance block into existing results/<slug>/meta.json files.

Old rows were produced before provenance was recorded, so the build, server flags
and template hash are gone. What can still be recovered offline is the gguf sha256
(when the file named in meta.json still exists), the current python and package
versions, and a marker that says this block was reconstructed after the fact:

    "provenance": {"source": "backfill", "harness_git": {"sha": "unknown", ...}, ...}

Usage:
    python3 scripts/backfill_provenance.py results/            # every slug
    python3 scripts/backfill_provenance.py results/some-slug   # one slug
    python3 scripts/backfill_provenance.py results/ --force    # overwrite existing blocks

Rows that already carry a `provenance` block with source "run" are left alone
unless --force is given. Never touches anything but meta.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.provenance import attach_to_meta, gguf_sha256, package_versions  # noqa: E402


def backfill_block(meta: dict) -> dict:
    """Build the offline provenance block for one meta.json dict (pure apart from hashing)."""
    block = {
        "source": "backfill",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": package_versions(),
        "harness_git": {"sha": "unknown", "dirty": None,
                        "note": "row predates provenance recording"},
        "server_command": None,
        "server_props": {"error": "not recoverable offline"},
        "quality_config": None,
    }
    path = meta.get("path")
    if path and Path(path).expanduser().is_file():
        block["gguf"] = gguf_sha256(path)
    else:
        block["gguf"] = {"sha256": None, "skipped": True,
                         "reason": "model file not found" if path else "no path in meta"}
    return block


def backfill_dir(slug_dir: Path, force: bool = False) -> str:
    meta_path = slug_dir / "meta.json"
    if not meta_path.exists():
        return "no meta.json"
    meta = json.loads(meta_path.read_text() or "{}")
    existing = meta.get("provenance")
    if isinstance(existing, dict) and existing.get("source") == "run" and not force:
        return "kept run provenance"
    attach_to_meta(meta_path, backfill_block(meta))
    return "backfilled"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", help="results root or one results/<slug> directory")
    ap.add_argument("--force", action="store_true", help="overwrite run-time provenance too")
    args = ap.parse_args(argv)
    target = Path(args.target)
    if (target / "meta.json").exists():
        dirs = [target]
    else:
        dirs = sorted(d for d in target.iterdir() if d.is_dir())
    for d in dirs:
        print(f"{d.name}: {backfill_dir(d, force=args.force)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
