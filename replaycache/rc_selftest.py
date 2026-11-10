#!/usr/bin/env python3
"""rc_selftest.py: CPU-only checks of the parser, the statistics, the sandbox and the entry validator. Run inside the item's DRYRUN. Exit 0 = all ok."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as C

bad = 0


def chk(name, cond):
    global bad
    print(("ok   " if cond else "FAIL ") + name)
    bad += (not cond)


chk("parse marker", C.parse_final("so 3 + 4.\n#### 1,234.5")[0] == 1234.5)
chk("parse marker wins over later numbers", C.parse_final("#### 12\nthe 7 apples")[0] == 12.0)
chk("parse last number fallback", C.parse_final("it is 3 then 18")[0] == 18.0 and C.parse_final("it is 3 then 18")[1] == "last_number")
chk("parse nothing", C.parse_final("no digits")[0] is None)
chk("same tolerance", C.same(0.1 + 0.2, 0.3) and not C.same(41, 42) and not C.same(None, 1))
chk("mcnemar 0/5", abs(C.mcnemar_exact(0, 5) - 0.0625) < 1e-12)
chk("mcnemar 3/3", C.mcnemar_exact(3, 3) == 1.0)
lo, hi = C.wilson(180, 200)
chk("wilson 180/200", abs(lo - 0.8506) < 1e-3 and abs(hi - 0.9344) < 1e-3)
chk("wilson 0/0", C.wilson(0, 0) == (0.0, 0.0))
d = tempfile.mkdtemp()


def w(name, src):
    p = os.path.join(d, name)
    open(p, "w").write(src)
    return p


ok, v, why = C.run_solve(w("a.py", "def solve(p):\n    return p['x'] * 2 + 1\n"), {"x": 4})
chk("sandbox runs solve", ok and v == 9.0)
ok, v, why = C.run_solve(w("b.py", "def solve(p):\n    return p['missing']\n"), {"x": 4})
chk("sandbox: missing param is a crash, not a number", (not ok) and "crash" in why)
ok, v, why = C.run_solve(w("c.py", "def solve(p):\n    while True: pass\n"), {"x": 4}, timeout=8)
chk("sandbox: endless loop is cut", not ok)
ok, v, why = C.run_solve(w("d.py", "def solve(p):\n    return 'ten'\n"), {"x": 4})
chk("sandbox: non-number is a miss", not ok)
ok, v, why = C.run_solve(w("e.py", "def solve(p):\n    return float('nan')\n"), {"x": 4})
chk("sandbox: nan is a miss", not ok)
good = "```json\n" + json.dumps({"description": "double plus one", "schema": [{"name": "x", "description": "the number", "example": 4}], "params": {"x": 4}}) + "\n```\n```python\ndef solve(params):\n    return params['x'] * 2 + 1\n```"
pa, why = C.parse_builder(good)
chk("builder reply parses", pa is not None)
chk("builder: no blocks rejected", C.parse_builder("sorry")[0] is None)
chk("builder: forbidden construct rejected", C.parse_builder(good.replace("return params['x'] * 2 + 1", "import os\n    return open('/etc/passwd').read()"))[0] is None)
chk("builder: non-number param rejected", C.parse_builder(good.replace('"params": {"x": 4}', '"params": {"x": "four"}'))[0] is None)
ok, v, why = C.validate_entry(pa, d, 9.0, "t1")
chk("validate: reproduces the source answer", ok and v == 9.0)
ok, v, why = C.validate_entry(pa, d, 10.0, "t2")
chk("validate: wrong answer rejected", not ok)
hard = dict(pa); hard["code"] = "def solve(params):\n    return 9.0\n"
ok, v, why = C.validate_entry(hard, d, 9.0, "t3")
chk("validate: hard-coded answer rejected", (not ok) and "hard-coded" in why)
sel = C.make_selection()
chk("selection: 20 templates, 10 + 10 each, none is instance 0 of main in held-out", len(sel["templates"]) == 20 and all(len(t["held"]) == 10 and len(t["near"]) == 10 and all(h[2] != 0 for h in t["held"]) for t in sel["templates"]))
chk("selection: deterministic", C.make_selection()["templates"][0]["held"] == sel["templates"][0]["held"])
print("selftest:", "FAIL %d" % bad if bad else "ok")
sys.exit(1 if bad else 0)
