"""Freeze the toolcall-v1 id list (BFCL v4 subset) BEFORE any model output exists.

Run once on capsule in ~/toolcall-v1/code with ../.venv/bin/python freeze_toolcall_v1.py <outdir>.
Writes <outdir>/test_case_ids_to_generate.json (BFCL --run-ids format, phase 1),
<outdir>/test_case_ids_phase2.json (S5 phase-2 reserve), <outdir>/manifest.json (per-item sets + tags,
seed, method, data pin), <outdir>/FROZEN.sha256. Never edit after the first leg; additions = toolcall-v2.

Sets (design doc 2026-10-15-toolcall-slice-design.md, "the slice I recommend"):
  S1 parallel pile-up: all of parallel, parallel_multiple, live_parallel, live_parallel_multiple (440)
  S2 nested schemas: every item of the 8 call-graded single-turn categories whose offered schema has an
     object inside an object or an array of objects (67; 7 already in S1)
  S3 call-formation controls: seeded 200 of non-nested simple_python + 100 of non-nested multiple
  S4 abstain: seeded 100 of irrelevance
  S5 multi_turn_base: seeded 100 phase 1, stratified on ground-truth calls per entry (<=4 / 5-8 / >=9),
     allocation round(n_stratum/2) (Python round, half to even) = 26 / 64 / 10; other 100 = phase 2
Pre-freeze exclusion (decided 2026-10-15 before any model output, from the parser round-trip): an
item whose canonical ground truth passes a string where the offered schema types the parameter as
array/object/integer/number/boolean cannot be answered with a schema-typed Qwen3.8 tool call; it is
dropped from S1/S2/S3 and listed in manifest["excluded"] (S3 samples from the pool after exclusion).
Seeding: random.Random(SEED + k).sample(sorted pool, n), one k per draw, pools sorted by BFCL id order.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import urllib.request
from collections import Counter
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bfcl_eval  # noqa: E402
from bfcl_eval.utils import load_ground_truth_entry, sort_key  # noqa: E402
from bfcl_items import _entries, _gt, tools_for, unrepresentable  # noqa: E402

LIST_NAME = "toolcall-v1"
SEED = 20261015
BFCL_COMMIT = "6ea57973"
CATS8 = ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple",
         "live_parallel", "live_parallel_multiple"]
S1_CATS = ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]
OBJ, ARR = {"dict", "object"}, {"array", "list", "tuple"}


def nested_schema(fn: dict) -> bool:
    """object inside an object (a dict-typed parameter WITH properties) or an array of objects."""
    def walk(v, depth):
        t = v.get("type")
        if t in OBJ and depth >= 1 and v.get("properties"):
            return True
        if t in ARR and isinstance(v.get("items"), dict) and v["items"].get("type") in OBJ:
            return True
        for sub in (v.get("properties") or {}).values():
            if isinstance(sub, dict) and walk(sub, depth + 1):
                return True
        if isinstance(v.get("items"), dict) and walk(v["items"], depth + 1):
            return True
        return False
    return any(walk(v, 1) for v in fn["parameters"].get("properties", {}).values())


def _vals(v):
    return v if isinstance(v, list) else [v]


def json_arg(gt) -> bool:
    """ground truth passes an array or object argument (any acceptable value of any parameter)."""
    return any(isinstance(x, (list, dict)) for call in gt for _, a in call.items() for vals in a.values() for x in _vals(vals))


def gt_nested(gt) -> bool:
    """ground truth passes a nested argument (an object, or an array holding arrays/objects)."""
    def nv(x):
        return isinstance(x, dict) or (isinstance(x, list) and any(isinstance(i, (dict, list)) for i in x))
    return any(nv(x) for call in gt for _, a in call.items() for vals in a.values() for x in _vals(vals))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict] = {}

    def add(i, s, **tags):
        d = items.setdefault(i, {"id": i, "category": i.rsplit("_", 1)[0], "sets": []})
        if s not in d["sets"]:
            d["sets"].append(s)
        d.update(tags)

    excluded = {}
    for c in CATS8:
        for i, e in _entries(c).items():
            bad = unrepresentable(_gt(c)[i]["ground_truth"], tools_for(e), e["function"])
            if bad:
                excluded[i] = bad

    def tags_single(i):
        c = i.rsplit("_", 1)[0]
        e = _entries(c)[i]
        g = _gt(c)[i]["ground_truth"] if c != "irrelevance" else []
        return dict(n_calls=len(g), nested=any(nested_schema(f) for f in e["function"]),
                    json_arg=json_arg(g), gt_nested=gt_nested(g), n_functions_offered=len(e["function"]))

    # S1
    for c in S1_CATS:
        for i in _entries(c):
            if i not in excluded:
                add(i, "S1", **tags_single(i))
    # S2
    for c in CATS8:
        for i, e in _entries(c).items():
            if i not in excluded and any(nested_schema(f) for f in e["function"]):
                add(i, "S2", **tags_single(i))
    # S3
    k = 0
    for c, n in (("simple_python", 200), ("multiple", 100)):
        pool = sorted([i for i, e in _entries(c).items() if i not in excluded and not any(nested_schema(f) for f in e["function"])], key=sort_key_id)
        k += 1
        for i in random.Random(SEED + k).sample(pool, n):
            add(i, "S3", **tags_single(i))
    # S4
    pool = sorted(_entries("irrelevance"), key=sort_key_id)
    k += 1
    for i in random.Random(SEED + k).sample(pool, 100):
        add(i, "S4", **tags_single(i))
    # S5
    mt_gt = {e["id"]: e["ground_truth"] for e in load_ground_truth_entry("multi_turn_base")}
    strata: dict[str, list[str]] = {"<=4": [], "5-8": [], ">=9": []}
    for i in sorted(mt_gt, key=sort_key_id):
        n = sum(len(t) for t in mt_gt[i])
        strata["<=4" if n <= 4 else ("5-8" if n <= 8 else ">=9")].append(i)
    phase1, phase2 = [], []
    for name, pool in strata.items():
        k += 1
        take = round(len(pool) / 2)
        pick = set(random.Random(SEED + k).sample(pool, take))
        phase1 += [i for i in pool if i in pick]
        phase2 += [i for i in pool if i not in pick]
        for i in pool:
            n = sum(len(t) for t in mt_gt[i])
            add(i, "S5" if i in pick else "S5_phase2", gt_calls=n, gt_turns=len(mt_gt[i]), stratum=name)
    assert len(phase1) == 100 and len(phase2) == 100, (len(phase1), len(phase2))

    # BFCL run-ids files
    run_ids: dict[str, list[str]] = {}
    reserve: dict[str, list[str]] = {}
    for d in sorted(items.values(), key=lambda d: sort_key_id(d["id"])):
        if d["sets"] == ["S5_phase2"]:
            reserve.setdefault(d["category"], []).append(d["id"])
        else:
            run_ids.setdefault(d["category"], []).append(d["id"])

    # data pin: package data == gorilla @ BFCL_COMMIT
    data_dir = Path(bfcl_eval.__file__).parent / "data"
    pin = {}
    files = [f"BFCL_v4_{c}.json" for c in CATS8 + ["irrelevance", "multi_turn_base"]]
    files += [f"possible_answer/BFCL_v4_{c}.json" for c in CATS8 + ["multi_turn_base"]]
    files += sorted("multi_turn_func_doc/" + p.name for p in (data_dir / "multi_turn_func_doc").glob("*.json"))
    for f in files:
        local = sha256_file(data_dir / f)
        url = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_COMMIT}/berkeley-function-call-leaderboard/bfcl_eval/data/{f}"
        try:
            remote = hashlib.sha256(urllib.request.urlopen(url, timeout=60).read()).hexdigest()
        except Exception as ex:  # noqa: BLE001
            remote = f"FETCH_FAILED {ex}"
        pin[f] = {"sha256": local, "matches_git_commit": local == remote}

    counts = Counter(s for d in items.values() for s in d["sets"])
    n_single_requests = sum(1 for d in items.values() if any(s in ("S1", "S2", "S3", "S4") for s in d["sets"]))
    manifest = {
        "list": LIST_NAME, "seed": SEED, "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "bfcl_eval_version": metadata.version("bfcl-eval"), "bfcl_git_commit": BFCL_COMMIT,
        "method": __doc__, "set_counts": dict(counts), "single_turn_requests": n_single_requests,
        "tag_definitions": {
            "nested": "offered schema has a dict-typed parameter with properties below the top level, or an array of objects (nested_schema())",
            "json_arg": "ground truth passes an array or object argument (any acceptable value)",
            "gt_nested": "ground truth passes an object or an array holding arrays/objects",
            "n_calls": "number of calls in the single-turn ground truth",
            "gt_calls/gt_turns": "multi-turn ground-truth calls summed over turns / number of turns"},
        "excluded": excluded, "data_pin": pin, "items": sorted(items.values(), key=lambda d: sort_key_id(d["id"])),
    }
    p_ids = outdir / "test_case_ids_to_generate.json"
    p_res = outdir / "test_case_ids_phase2.json"
    p_man = outdir / "manifest.json"
    p_ids.write_text(json.dumps(run_ids, indent=2) + "\n")
    p_res.write_text(json.dumps(reserve, indent=2) + "\n")
    p_man.write_text(json.dumps(manifest, indent=1) + "\n")
    (outdir / "FROZEN.sha256").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in (p_ids, p_res, p_man)))
    print(json.dumps({"excluded": excluded, "set_counts": dict(counts), "single_turn_requests": n_single_requests,
                      "run_ids_per_category": {c: len(v) for c, v in run_ids.items()},
                      "data_pin_all_match": all(v["matches_git_commit"] for v in pin.values()),
                      "S2_gt_nested": sum(1 for d in items.values() if "S2" in d["sets"] and d["gt_nested"]),
                      "S1_by_calls": dict(Counter(min(d["n_calls"], 4) for d in items.values() if "S1" in d["sets"]))}, indent=1))


def sort_key_id(i: str):
    return sort_key({"id": i})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
