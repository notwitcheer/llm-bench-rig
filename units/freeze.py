#!/usr/bin/env python3
"""freeze the units task set: builds tasks.json from tasks_def.py + schemas.json (verbatim Hermes schemas) and writes FROZEN.sha256.
proves, for every one of the 160 forms, that num x unit factor == the task's expected value EXACTLY (Fractions), and prints the magnitude balance
(how far the stated number sits from the expected argument, in powers of ten).
run ONCE, before any model output. The queue item re-checks FROZEN.sha256 and `freeze.py --check` at every DRYRUN.
usage: freeze.py [--check]"""
import hashlib
import json
import math
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks_def as D  # noqa: E402
from ucommon import ROLES, sha  # noqa: E402

FROZEN_FILES = ["tasks_def.py", "schemas.json", "tasks.json", "models.json", "PREREG.md"]
CODE_FILES = ["ucommon.py", "run_units.py", "score.py", "bridge.py", "mock_units.py", "aggregate_selftest.py", "render_orbit.py", "freeze.py", "extract_schemas.py"]   # code hashes: written by `freeze.py --code`, may change after a smoke bug-fix (PARSER_VERSION bump), never the task files


def build():
    src = json.load(open(os.path.join(HERE, "schemas.json")))
    tools = {}
    tname = {}                                                           # registry key in tasks_def.py -> the name the model must call (the schema's own "name")
    for name, meta in D.TOOLS.items():
        if meta["real"]:
            r = src["tools"][name]
            schema = r["schema"]
            tn = schema["name"]
            tools[tn] = {"real": True, "schema": schema, "hermes_pin": src["pin"], "file": r["file"], "schema_line": r["schema_line"], "arg_line": r["arg_line"],
                         "arg": meta["arg"], "unit": r["unit"], "arg_description": schema["parameters"]["properties"][meta["arg"]].get("description"), "note": r["note"]}
            assert meta["arg"] in schema["parameters"]["properties"], name
        else:
            schema = D.SYNTHETIC_SCHEMAS[name]
            tn = schema["name"]
            tools[tn] = {"real": False, "schema": schema, "synthetic": True, "arg": meta["arg"], "unit": meta["fam"],
                         "arg_description": schema["parameters"]["properties"][meta["arg"]]["description"]}
        tname[name] = tn
        tools[tn]["schema_sha256"] = sha(json.dumps(schema, sort_keys=True))
        tools[tn]["arg_type"] = schema["parameters"]["properties"][meta["arg"]]["type"]
    tasks = []
    stats = []
    for t in D.TASKS:
        meta = D.TOOLS[t["tool"]]
        exp = Fraction(t["expected"])
        if meta["argtype"] == "integer":
            assert exp.denominator == 1, (t["id"], "integer argument with a non-integer expected value")
        assert [f["role"] for f in t["forms"]] == ROLES, t["id"]
        forms = []
        for f in t["forms"]:
            fac = D.FAC[meta["fam"]][f["unit"]]
            val = Fraction(f["num"]) * fac
            assert val == exp, (t["id"], f["role"], str(val), str(exp))          # exact conversion or the freeze refuses
            assert "{Q}" in t["template"]
            prompt = t["template"].replace("{Q}", f["text"])
            forms.append({"role": f["role"], "num": f["num"], "unit": f["unit"], "text": f["text"], "prompt_user": prompt, "prompt_sha256": sha(prompt)})
            if f["role"] != "canonical":
                stats.append((t["id"], f["role"], math.log10(abs(Fraction(f["num"]) / exp)) if Fraction(f["num"]) != 0 else 0.0))
        tasks.append({"id": t["id"], "tool": tname[t["tool"]], "real": meta["real"], "arg": meta["arg"], "unit": meta["fam"], "arg_type": meta["argtype"], "must": meta["must"],
                      "expected": t["expected"], "ask": t["ask"], "verb": meta["verb"], "template": t["template"], "forms": forms})
    spec = {"version": "units-v1", "n_tasks": len(tasks), "n_forms": sum(len(t["forms"]) for t in tasks), "system": D.SYSTEM_PROMPT, "roles": ROLES,
            "tools": tools, "tasks": tasks}
    # magnitude balance (stated number vs expected argument): |log10(stated/expected)| for the 3 non-canonical forms of every task
    bal = {"ge1": sum(1 for _, _, x in stats if abs(x) >= 0.999), "lt1": sum(1 for _, _, x in stats if abs(x) < 0.999), "ge1_le3": sum(1 for _, _, x in stats if 0.999 <= abs(x) <= 3.001), "n": len(stats)}
    by_role = {}
    for _, r, x in stats:
        by_role.setdefault(r, []).append(round(abs(x), 2))
    spec["balance"] = {"note": "|log10(stated number / expected argument)| over the 3 non-canonical forms of each task", "counts": bal,
                       "median_by_role": {r: sorted(v)[len(v) // 2] for r, v in by_role.items()}}
    return spec


if __name__ == "__main__":
    spec = build()
    path = os.path.join(HERE, "tasks.json")
    if "--check" in sys.argv:
        cur = json.load(open(path))
        same = json.dumps(cur, sort_keys=True) == json.dumps(spec, sort_keys=True)
        print("tasks.json matches the registry + schemas" if same else "MISMATCH: tasks.json differs from tasks_def.py / schemas.json")
        sys.exit(0 if same else 1)
    json.dump(spec, open(path, "w"), indent=1)
    with open(os.path.join(HERE, "FROZEN.sha256"), "w") as f:
        for rel in FROZEN_FILES:
            f.write(f"{hashlib.sha256(open(os.path.join(HERE, rel), 'rb').read()).hexdigest()}  {rel}\n")
    with open(os.path.join(HERE, "CODE.sha256"), "w") as f:
        for rel in CODE_FILES:
            f.write(f"{hashlib.sha256(open(os.path.join(HERE, rel), 'rb').read()).hexdigest()}  {rel}\n")
    b = spec["balance"]
    print(f"froze {spec['n_tasks']} tasks, {spec['n_forms']} forms, {sum(1 for t in spec['tasks'] if t['real'])} on real Hermes tools, {sum(1 for t in spec['tasks'] if not t['real'])} synthetic; balance {b['counts']} median {b['median_by_role']}")
