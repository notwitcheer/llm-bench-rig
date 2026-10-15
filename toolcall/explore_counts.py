"""exploratory: reproduce the design doc's nesting / call counts before freezing (no output is frozen here)."""
import json, sys
sys.path.insert(0, ".")
from bfcl_items import _entries, _gt, all_ids

CATS8 = ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
OBJ = {"dict", "object"}
ARR = {"array", "list", "tuple"}


def nested_A(p):  # object inside object (a dict-typed property under a dict-typed param) or array of objects, any depth >= 1 below params
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
    return any(walk(v, 1) for v in p["parameters"].get("properties", {}).values())


def nested_B(p):  # top-level dict param with properties, or array of dicts
    for v in p["parameters"].get("properties", {}).values():
        t = v.get("type")
        if t in OBJ:
            return True
        if t in ARR and isinstance(v.get("items"), dict) and v["items"].get("type") in OBJ:
            return True
    return False


def nested_C(p):  # dict param WITH properties, or array of dicts
    for v in p["parameters"].get("properties", {}).values():
        t = v.get("type")
        if t in OBJ and v.get("properties"):
            return True
        if t in ARR and isinstance(v.get("items"), dict) and v["items"].get("type") in OBJ:
            return True
    return False


for name, fn in [("A", nested_A), ("B", nested_B), ("C", nested_C)]:
    per = [sum(1 for e in _entries(c).values() if any(fn(f) for f in e["function"])) for c in CATS8]
    print(name, sum(per), per)


def first_val(vals):
    return vals[0] if isinstance(vals, list) and vals else vals


def gt_json_arg(gt):
    return any(isinstance(first_val(vals), (list, dict)) for call in gt for _, args in call.items() for vals in args.values())


def is_nested_val(x):
    if isinstance(x, dict):
        return True
    if isinstance(x, list):
        return any(isinstance(i, (dict, list)) for i in x)
    return False


def gt_nested_arg(gt):
    for call in gt:
        for _, args in call.items():
            for vals in args.values():
                if isinstance(vals, list) and any(is_nested_val(v) for v in vals):
                    return True
                if isinstance(vals, dict):
                    return True
    return False

ja = sum(1 for c in CATS8 for i, g in _gt(c).items() if gt_json_arg(g["ground_truth"]))
print("json_arg (first acceptable value list/dict):", ja)
ja2 = sum(1 for c in CATS8 for i, g in _gt(c).items() if any(isinstance(v, (list, dict)) for call in g["ground_truth"] for _, a in call.items() for vals in a.values() for v in (vals if isinstance(vals, list) else [vals])))
print("json_arg (any acceptable value list/dict):", ja2)
for name, fn in [("A", nested_A), ("C", nested_C)]:
    ids = [i for c in CATS8 for i, e in _entries(c).items() if any(fn(f) for f in e["function"])]
    gn = sum(1 for i in ids if gt_nested_arg(_gt(i.rsplit('_', 1)[0])[i]["ground_truth"]))
    print(name, "gt nested among nested-schema:", gn)
from collections import Counter
for c in ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]:
    print(c, sorted(Counter(len(g["ground_truth"]) for g in _gt(c).values()).items()))
mt = {e["id"]: e for e in __import__("bfcl_eval.utils", fromlist=["x"]).load_ground_truth_entry("multi_turn_base")}
cnt = Counter()
for i, g in mt.items():
    n = sum(len(t) for t in g["ground_truth"])
    cnt["<=4" if n <= 4 else ("5-8" if n <= 8 else ">=9")] += 1
print("multi_turn_base buckets", dict(cnt), len(mt))
print("simple_python non-nested A:", sum(1 for e in _entries("simple_python").values() if not any(nested_A(f) for f in e["function"])),
      "multiple non-nested A:", sum(1 for e in _entries("multiple").values() if not any(nested_A(f) for f in e["function"])))
