#!/usr/bin/env python3
"""Deterministic seed generator for the agenttest harness (rebuilt 2026-10-22).

Writes a fresh work tree for the 10 frozen tasks. Content comes ONLY from the frozen files in
seed_data/ (no randomness): those files were recovered from the surviving Muse Glimmer work folder
(2026-10-20) or reconstructed from its transcripts, see seed_data/PROVENANCE.json. The only
transformation is replacing the {WORK} placeholder (t09/run_report.sh) with the work root.

usage: gen_seeds.py [--work /opt/data/tmp/sparkagent/work]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "seed_data")
TASK_IDS = [f"t{i:02d}" for i in range(1, 11)]


def default_work():
    return json.load(open(os.path.join(HERE, "tasks.json")))["work_root"]


def seed_files():
    """(relative path, bytes) for every seed file, sorted, placeholder NOT yet substituted."""
    out = []
    for dp, _dn, fns in os.walk(SEED):
        for fn in fns:
            if fn == "PROVENANCE.json":
                continue
            p = os.path.join(dp, fn)
            out.append((os.path.relpath(p, SEED), open(p, "rb").read()))
    return sorted(out)


def render(raw: bytes, work: str) -> bytes:
    return raw.replace(b"{WORK}", work.encode())


def expected_bytes(rel: str, work: str) -> bytes:
    return render(open(os.path.join(SEED, rel), "rb").read(), work)


def build(work: str) -> dict:
    work = os.path.abspath(work)
    # safety: only ever wipe a dedicated tmp work tree
    if not (work.startswith("/opt/data/tmp/") and os.path.basename(work).startswith(("work", "proof"))):
        sys.exit(f"refusing to wipe {work}")
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "smoke"))
    for t in TASK_IDS:
        os.makedirs(os.path.join(work, t))
    manifest = {}
    for rel, raw in seed_files():
        dst = os.path.join(work, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        b = render(raw, work)
        with open(dst, "wb") as f:
            f.write(b)
        os.chmod(dst, 0o644)
        manifest[rel] = hashlib.sha256(b).hexdigest()
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    m = build(a.work or default_work())
    print(json.dumps({"work": a.work or default_work(), "files": m}, indent=1))
