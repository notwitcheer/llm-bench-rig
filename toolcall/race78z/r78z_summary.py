#!/usr/bin/env python3
"""r78z_summary.py <rundir> <slug> <facts-note> > summary.json   (the item's artefact: written LAST; its mtime is what nl_judge_artifact reads)

Per take (every take, not only the two in the clip): engine, end reason, finish seconds from the take's own Enter, server-counted tokens, mean tok/s while generating, page bytes and gate
verdict, request/error counts. Which pair went into the clip and why; the race clip summary; the server rows (start seconds, VRAM) and the headline sentence. No judgement beyond that."""
import base64
import glob
import json
import os
import sys

run, slug = sys.argv[1], sys.argv[2]


def marks_of(d):
    mk = {}
    p = d + "/rec/events-a.jsonl"
    if not os.path.exists(p):
        return mk
    for l in open(p):
        j = json.loads(l)
        if "mark" in j:
            mk.setdefault(j["mark"], j["t"])
            if j["mark"] == "end":
                mk["end_reason"] = j.get("reason")
    return mk


def take_row(d):
    mk = marks_of(d)
    row = {"dir": os.path.basename(d), "end": mk.get("end_reason")}
    if "enter" in mk:
        row["finish_s"] = round((mk["last_change"] if mk.get("end_reason") == "done" and "last_change" in mk else mk.get("end", mk["enter"])) - mk["enter"], 1)
    hp = d + "/hud/hud.jsonl"
    if os.path.exists(hp) and "enter" in mk:
        hud = [json.loads(l) for l in open(hp) if l.strip()]
        if hud:
            last = hud[-1]
            row["engine"] = last.get("engine")
            row["tokens_server"] = last.get("srv_tokens")
            row["tokens_exact_from_usage"] = last["agents"]["a"]["exact_pred"]
            row["tokens_stream_deltas"] = last["agents"]["a"]["chunks"]
            row["requests"] = last["agents"]["a"]["reqs"]
            row["request_errors"] = last["agents"]["a"]["err"]
            ts = [r for r in hud if r.get("srv_tokens") is not None]
            gen = []
            for i in range(6, len(ts)):
                dt = ts[i]["t"] - ts[i - 6]["t"]
                if mk["enter"] <= ts[i]["t"] <= mk["enter"] + row.get("finish_s", 0) and dt > 0:
                    v = (ts[i]["srv_tokens"] - ts[i - 6]["srv_tokens"]) / dt
                    if v > 5:
                        gen.append(v)
            row["tok_s_mean_while_generating"] = round(sum(gen) / len(gen), 1) if gen else None
    gp = d + "/gate/gate.json"
    row["page_bytes"] = os.path.getsize(d + "/work/index.html") if os.path.exists(d + "/work/index.html") else 0
    if os.path.exists(gp):
        g = json.load(open(gp))
        row["gate_pass"] = bool(g.get("pass"))
        row["gate_fail"] = [k for k, v in (g.get("checks") or {}).items() if not v]
    return row


takes = {}
for d in sorted(glob.glob(run + "/takes/*")):
    takes[os.path.basename(d)] = take_row(d)
pair = json.load(open(run + "/pair.json")) if os.path.exists(run + "/pair.json") else {}
race = json.load(open(run + "/out/race.mp4.summary.json")) if os.path.exists(run + "/out/race.mp4.summary.json") else {}
fit = [json.loads(l) for l in open(run + "/fit.jsonl")] if os.path.exists(run + "/fit.jsonl") else []
facts = json.load(open(run + "/facts.json")) if os.path.exists(run + "/facts.json") else {}
head = None
if race:
    L, R = race["left"], race["right"]
    head = ("same Hermes task, one take each: NVFP4 on vLLM %s s (%s, %s tok/s while generating), UD-Q4_K_M on llama.cpp %s s (%s, %s tok/s while generating); "
            "item 78: tool calls tie 85 vs 80 of 100 (p=0.125), decode 56 vs 78 tok/s" % (L["finish_s"], L["end"], L["tok_s_mean_while_generating"], R["finish_s"], R["end"], R["tok_s_mean_while_generating"]))
json.dump({"slug": slug, "regime": sys.argv[3] if len(sys.argv) > 3 else "", "takes": takes, "clip_pair": pair, "race": race, "servers": fit, "card_facts": facts.get("rows"),
           "final_clip": os.path.exists(run + "/out/nvfp4-vs-q4km-final.mp4"), "headline": head}, sys.stdout, indent=1)
