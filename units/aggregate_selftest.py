#!/usr/bin/env python3
"""end-to-end CPU selftest of the units harness on SCRIPTED mock answers (mock_units.py). Never model output.
proves: freeze check, driver (resume, deadline exit 3, shapes), exact-rational classifier categories, per-model verdict branches (SLIPS FOUND >= 3, CLEAN 0-2, BROKEN > 20%),
overall branches (SLIPS FOUND / CLEAN / INVALID), the LINE-1 builder (slip line from the worst real slip, clean line), McNemar, and the render input path."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
import ucommon as C  # noqa: E402

PORT = int(os.environ.get("UNITS_MOCK_PORT", "8651"))
PROFILES = ["mock-perfect", "mock-slippy", "mock-clean", "mock-three", "mock-broken", "mock-copycat", "mock-shapes"]
ok_all = True


def check(name, cond, extra=""):
    global ok_all
    print(("OK   " if cond else "BAD  ") + name + (("  " + extra) if extra and not cond else ""))
    ok_all = ok_all and bool(cond)


tmp = tempfile.mkdtemp(prefix="units-selftest-")
mods = json.load(open(os.path.join(HERE, "models.json")))
mods["legs"] = [{"key": p, "label": p, "leg": i + 1, "alias": p, "chat_template_kwargs": {"enable_thinking": False}, "thinking": "mock", "max_tokens": 64, "file": "mock", "sha256": "mock", "bytes": 0, "source": "mock"}
                for i, p in enumerate(PROFILES)]
mods["pair"] = ["mock-perfect", "mock-slippy"]
mp = os.path.join(tmp, "models.json")
json.dump(mods, open(mp, "w"))

r = subprocess.run([sys.executable, os.path.join(HERE, "freeze.py"), "--check"], capture_output=True, text=True)
check("freeze --check: tasks.json matches the registry + schemas", r.returncode == 0, r.stdout)
r = subprocess.run(["sha256sum", "-c", "FROZEN.sha256"], capture_output=True, text=True, cwd=HERE)
check("sha256sum -c FROZEN.sha256", r.returncode == 0 and "FAILED" not in r.stdout, r.stdout[-300:])

srv = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_units.py"), str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    res = os.path.join(tmp, "res")
    run = lambda key, *extra: subprocess.run([sys.executable, os.path.join(HERE, "run_units.py"), "--model", key, "--port", str(PORT), "--out", res, "--models", mp, *extra], capture_output=True, text=True)
    # deadline in the past -> exit 3, no rows
    r = run("mock-perfect", "--deadline-epoch", "1")
    check("driver: deadline in the past exits 3 with no rows", r.returncode == 3 and not os.path.exists(os.path.join(res, "mock-perfect.jsonl")) or (r.returncode == 3 and os.path.getsize(os.path.join(res, "mock-perfect.jsonl")) == 0), r.stdout)
    # partial then resume: only t01 first, then everything; no duplicates
    r = run("mock-perfect", "--tasks", "t01")
    n1 = sum(1 for _ in open(os.path.join(res, "mock-perfect.jsonl")))
    check("driver: --tasks t01 writes 4 rows", r.returncode == 0 and n1 == 4, r.stdout)
    r = run("mock-perfect")
    rows = [json.loads(x) for x in open(os.path.join(res, "mock-perfect.jsonl"))]
    check("driver: resume completes to 160 rows without duplicates", r.returncode == 0 and len(rows) == 160 and len({(x["task"], x["role"]) for x in rows}) == 160, "%d rows rc %s" % (len(rows), r.returncode))
    r = run("mock-perfect")
    check("driver: a finished model is a no-op (0 new rows)", r.returncode == 0 and "0 new rows" in r.stdout, r.stdout)
    for p in PROFILES[1:]:
        r = run(p)
        check("driver: %s runs to 160 rows" % p, r.returncode == 0 and sum(1 for _ in open(os.path.join(res, p + ".jsonl"))) == 160, r.stdout[-200:])
    # dead server -> exit 4
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_units.py"), "--model", "mock-perfect", "--port", "1", "--out", os.path.join(tmp, "res2"), "--models", mp], capture_output=True, text=True)
    check("driver: dead server exits 4", r.returncode == 4)
    # row content: the payload the driver sent is the pre-registered one (greedy, seed, tools, system prompt)
    spec = C.load_tasks()
    t0 = spec["tasks"][0]
    pl = C.build_payload(spec, t0, t0["forms"][0], mods["legs"][0])
    check("payload: temp 0, top_k 1, seed 20260725, one tool, stream false", pl["temperature"] == 0 and pl["top_k"] == 1 and pl["seed"] == 20260725 and len(pl["tools"]) == 1 and pl["stream"] is False and pl["chat_template_kwargs"] == {"enable_thinking": False})
    check("payload: the experiment is not named (no unit/orbit/invariance words in the prompts)", not any(w in json.dumps(pl).lower() for w in ["orbit", "invarian", "experiment", "benchmark", "magnitude", "convert"]))

    # ---- scoring ----
    agg = S.aggregate(res, mp, spec)
    M = agg["models"]
    check("perfect: 160 ok, CLEAN", M["mock-perfect"]["counts"]["ok"] == 160 and M["mock-perfect"]["verdict"] == "CLEAN")
    sl = M["mock-slippy"]
    check("slippy: 8 slips, one each of other_value / malformed / no_call / wrong_tool", sl["counts"]["magnitude_slip"] == 8 and sl["counts"]["other_value"] == 1 and sl["counts"]["malformed"] == 1 and sl["counts"]["no_call"] == 1 and sl["counts"]["wrong_tool"] == 1, str(sl["counts"]))
    check("slippy: SLIPS FOUND, k (value x 10^k == expected) histogram: three values 1000x too big (k -3), one each of -2 -1 1 2 3", sl["verdict"] == "SLIPS FOUND" and sl["slip_k"] == {"-3": 3, "-2": 1, "-1": 1, "1": 1, "2": 1, "3": 1}, str(sl["slip_k"]))
    check("clean: 2 slips -> CLEAN (boundary)", M["mock-clean"]["counts"]["magnitude_slip"] == 2 and M["mock-clean"]["verdict"] == "CLEAN")
    check("three: 3 slips -> SLIPS FOUND (boundary)", M["mock-three"]["counts"]["magnitude_slip"] == 3 and M["mock-three"]["verdict"] == "SLIPS FOUND")
    check("broken: > 20% no_call -> BROKEN", M["mock-broken"]["verdict"] == "BROKEN" and M["mock-broken"]["bad_fraction"] > 0.2, str(M["mock-broken"]["bad_fraction"]))
    cc = M["mock-copycat"]
    check("copycat: 120 slips/other, canonical forms ok (40), copied_stated counted", cc["by_role"]["canonical"]["ok"] == 40 and cc["by_role"]["canonical"]["slips"] == 0 and cc["verdict"] == "SLIPS FOUND" and cc["copied_stated"] > 90, str(cc["by_role"]) + " " + str(cc["copied_stated"]))
    check("copycat: every non-canonical copy is a slip (power of ten) or other_value (e.g. 3/2 min -> 1.5), forms that already equal the expected number stay ok", cc["counts"]["ok"] + cc["counts"]["magnitude_slip"] + cc["counts"]["other_value"] == 160 and cc["counts"]["magnitude_slip"] > 40 and cc["counts"]["other_value"] > 20, str(cc["counts"]))
    sh = M["mock-shapes"]
    check("shapes: dict args, numeric strings and 90.0 floats all score ok", sh["counts"]["ok"] == 160, str(sh["counts"]))
    check("overall with every mock leg: SLIPS FOUND, provisional False", agg["overall"] == "SLIPS FOUND" and agg["provisional"] is False)
    def sub(keys, tag):
        d = os.path.join(tmp, tag)
        os.makedirs(d, exist_ok=True)
        for k in keys:
            shutil.copy(os.path.join(res, k + ".jsonl"), d)
        return S.aggregate(d, mp, spec)
    a = sub(["mock-slippy", "mock-three"], "worst")
    w = a["worst_slip"]
    check("worst slip: largest |k|, upward first, earlier leg, then task id -> slippy e03 words (k -3: asked 10 min, set 10000 min... )", w and w["model_key"] == "mock-slippy" and w["task"] == "e03" and w["k"] == -3, str(w))
    check("overall with every mock leg: worst slip exists", agg["worst_slip"] is not None)
    w = agg["worst_slip"]
    print("      line1:", agg["line1"])
    check("line 1 (slips) is built from data: names the ask, the model and the value set", agg["line1"] and w["ask"] in agg["line1"] and w["model"] in agg["line1"] and w["set"] in agg["line1"])
    check("line 1 contains no em dash", "\u2014" not in (agg["line1"] or ""))

    a = sub(["mock-perfect", "mock-clean"], "clean")
    check("overall CLEAN when every valid model is CLEAN (perfect + clean)", a["overall"] == "CLEAN" and a["provisional"] is True and a["line1"].startswith("local models convert units into tool arguments cleanly when the schema names the unit"), str(a["line1"]))
    a = sub(["mock-broken"], "broken")
    check("overall INVALID when the only model is BROKEN, no line 1", a["overall"] == "INVALID" and a["line1"] is None)
    a = sub(["mock-perfect", "mock-broken"], "pb")
    check("a BROKEN model is excluded: overall CLEAN from the valid one", a["overall"] == "CLEAN")
    a = sub(["mock-clean", "mock-three"], "ct")
    check("one SLIPS FOUND model makes the overall SLIPS FOUND (a CLEAN model beside it)", a["overall"] == "SLIPS FOUND" and a["worst_slip"] is not None)
    a = sub([], "none")
    check("nothing run -> NOT RUN", a["overall"] == "NOT RUN")
    pr = agg["pair_q4_q8"]
    check("McNemar (perfect vs slippy): only_a 12 (8 slips + 4 others), only_b 0, p = 2^-11", pr and pr["only_a_ok"] == 12 and pr["only_b_ok"] == 0 and abs(pr["mcnemar_exact_p"] - 2 * (1 / 2 ** 12)) < 1e-12, str(pr))
    check("mcnemar_exact(5,5) == 1.0 and (0,10) == 2/1024", S.mcnemar_exact(5, 5) == 1.0 and abs(S.mcnemar_exact(0, 10) - 2 / 1024) < 1e-12)
    # classifier unit cases (exact rationals, boundaries)
    t = {"id": "x", "tool": "terminal", "arg": "timeout", "must": {}, "expected": "90"}
    f = {"role": "sub", "num": "90000", "unit": "ms"}
    P = lambda raw, name="terminal", n=1: {"n_calls": n, "name": name, "args_raw": raw, "finish_reason": "tool_calls"}
    cl = lambda raw, **k: C.classify(t, f, P(raw, **k))
    check("classify: 90 ok", cl('{"command":"x","timeout":90}')["cat"] == "ok")
    check("classify: 90.0 ok", cl('{"command":"x","timeout":90.0}')["cat"] == "ok")
    c = cl('{"command":"x","timeout":90000}')
    check("classify: 90000 = slip k=-3 (value x 10^-3 == expected), copied the stated number", c["cat"] == "magnitude_slip" and c["k"] == -3 and c["copied_stated"] is True)
    check("classify: 9 = slip k=1", cl('{"command":"x","timeout":9}')["k"] == 1)
    check("classify: 0.09 = slip k=3", cl('{"command":"x","timeout":0.09}')["k"] == 3)
    check("classify: 91 = other_value", cl('{"command":"x","timeout":91}')["cat"] == "other_value")
    check("classify: 90000000 (10^6 off) = slip k=-6, 10^7 off = other_value", cl('{"command":"x","timeout":90000000}')["k"] == -6 and cl('{"command":"x","timeout":900000000}')["cat"] == "other_value")
    check("classify: negative -90 = other_value", cl('{"command":"x","timeout":-90}')["cat"] == "other_value")
    check("classify: missing arg / bad json / string words = malformed", cl('{"command":"x"}')["cat"] == "malformed" and cl('{"command":')["cat"] == "malformed" and cl('{"command":"x","timeout":"ninety"}')["cat"] == "malformed")
    check("classify: numeric string '90' ok (flagged), 0 calls = no_call, other tool = wrong_tool", cl('{"command":"x","timeout":"90"}')["cat"] == "ok" and cl("", n=0, name=None)["cat"] == "no_call" and cl('{"timeout":90}', name="process_manage")["cat"] == "wrong_tool")
    check("classify: boolean = malformed", cl('{"command":"x","timeout":true}')["cat"] == "malformed")
    t2 = dict(t, tool="process_manage", must={"action": "wait"})
    check("classify: process with action != wait = wrong_tool", C.classify(t2, f, P('{"action":"poll","timeout":90}', name="process_manage"))["cat"] == "wrong_tool" and C.classify(t2, f, P('{"action":"wait","timeout":90}', name="process_manage"))["cat"] == "ok")
    # exactness: a float-looking 0.1+0.2 style value must not be rounded into a match
    t3 = {"id": "y", "tool": "trim_cable", "arg": "length_m", "must": {}, "expected": "1/4"}
    f3 = {"role": "sub", "num": "25", "unit": "cm"}
    check("classify: 0.25 == 1/4 exactly ok, 0.2500001 other_value", C.classify(t3, f3, P('{"spool":"a","length_m":0.25}', name="trim_cable"))["cat"] == "ok" and C.classify(t3, f3, P('{"spool":"a","length_m":0.2500001}', name="trim_cable"))["cat"] == "other_value")
    H = C.humanise
    check("humanise: 90000 s = 25 hours, 1/4 m = 25 centimetres, 1250 cents = $12.50, 6/5 cents = 1.2 cents, 90 s = 1.5 minutes, 7/2000 km... = 3.5 metres", H(90000, "seconds") == "25 hours" and H("1/4", "metres") == "25 centimetres" and H(1250, "cents") == "$12.50" and H("6/5", "cents") == "1.2 cents" and H(90, "seconds") == "1.5 minutes" and H("7/2", "metres") == "3.5 metres" and H(1, "grams") == "1 gram", "%s | %s | %s | %s | %s" % (H(90000, "seconds"), H("1/4", "metres"), H(1250, "cents"), H("6/5", "cents"), H(90, "seconds")))
    # readout renders
    out = os.path.join(tmp, "out")
    subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--res", res, "--out", out, "--models", mp], check=True, capture_output=True)
    ro = open(os.path.join(out, "readout.md")).read()
    check("readout.md written with the verdict, line 1 and the McNemar line", "SLIPS FOUND" in ro and "line 1:" in ro and "McNemar" in ro)
    json.dump({"res": res, "agg": os.path.join(out, "aggregate.json"), "models": mp}, open(os.path.join(tmp, "paths.json"), "w"))
    if os.environ.get("UNITS_KEEP"):
        print("kept", tmp)
        open(os.environ["UNITS_KEEP"], "w").write(tmp)
finally:
    srv.terminate()
    if not os.environ.get("UNITS_KEEP"):
        shutil.rmtree(tmp, ignore_errors=True)
print("UNITS_SELFTEST_OK" if ok_all else "UNITS_SELFTEST_BROKEN")
sys.exit(0 if ok_all else 1)
