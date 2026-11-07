#!/usr/bin/env python3
"""wd_gate.py <TAKEDIR> : after the editor take (CPU). Reads the sandbox tool's own log (state/calls.jsonl), the saved dev evaluations (state/evals/*.json) and the agent's state.db.
Writes TAKEDIR/gate.json:  scored (candidates with a finished dev evaluation), best (lowest dev loop rate; ties -> the earlier evaluation), best_gguf, why (when none), flags, calls (counts per verb), minutes.
flags (disclosed, never silently dropped): 'outside-workdir' (the agent's tool calls name the harness, the arms, the replication repos or the published edit), 'network' (curl/wget/pip/git clone/http), 'no-wd-use' (zero wd calls).
The selection rule is fixed in PREREG.md: the final candidate is the best by DEV loop rate among those the editor had scored; `wd submit` is a courtesy and is not used."""
import glob, json, os, re, sqlite3, sys

take = sys.argv[1].rstrip("/")
st = os.path.join(take, "state")
res = dict(scored=0, best=None, best_gguf=None, flags=[], calls={}, why="")
calls = []
cp = os.path.join(st, "calls.jsonl")
if os.path.exists(cp):
    for l in open(cp):
        try:
            calls.append(json.loads(l))
        except Exception:
            pass
for c in calls:
    res["calls"][c["verb"]] = res["calls"].get(c["verb"], 0) + 1
done = [c for c in calls if c["verb"] == "eval_done" and (c["out"] or {}).get("status") == "done"]
ev = []
for c in done:
    name = c["args"]["name"]
    s = (c["out"] or {}).get("summary") or {}
    if s.get("n"):
        ev.append((c["t"], name, s["loop_rate"], s["loops"], s["n"]))
res["scored"] = len(ev)
if ev:
    best = sorted(ev, key=lambda x: (x[2], x[0]))[0]
    res["best"], res["best_loop_rate"], res["best_gguf"] = best[1], best[2], os.path.join(st, "cand", best[1] + ".gguf")
    res["scored_list"] = [dict(name=n, loop_rate=r, loops=l, n=nn) for _, n, r, l, nn in sorted(ev)]
else:
    res["why"] = "no wd eval finished" if res["calls"].get("eval") else "the editor never called wd eval"
res["minutes"] = round((calls[-1]["t"] - calls[0]["t"]) / 60, 1) if len(calls) > 1 else 0
if not res["calls"].get("apply") and not res["calls"].get("info"):
    res["flags"].append("no-wd-use")
blob = ""
sp = os.path.join(take, "state", "agent-state.db")
if not os.path.exists(sp):
    sp = os.path.join(take, "agent-state", "state.db")
if os.path.exists(sp):
    try:
        c = sqlite3.connect(sp)
        blob = " ".join(x or "" for (x,) in c.execute("select tool_calls from messages where role='assistant'"))
        res["agent_tool_calls"] = sum(1 for _ in c.execute("select 1 from messages where role='assistant' and tool_calls is not null and tool_calls != ''"))
    except Exception as e:
        res["state_db_error"] = str(e)[:80]
if re.search(r"wedit95|voxel-stage|/ref/|edit\.npz|published|wsp-|KidIkaros|RikiyaT|rikiyat|scripted|FROZEN|PREREG|wd_service|wd_edit|/arms/", blob):
    res["flags"].append("outside-workdir")
if re.search(r"\bcurl\b|\bwget\b|pip install|git clone|https?://", blob):
    res["flags"].append("network")
json.dump(res, open(os.path.join(take, "gate.json"), "w"), indent=1)
print("GATE scored=%d best=%s flags=%s calls=%s" % (res["scored"], res["best"], res["flags"], res["calls"]))
