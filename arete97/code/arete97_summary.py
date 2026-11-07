#!/usr/bin/env python3
"""item 97-arete-msm: summary + frozen verdict (stdlib only; applies PREREG.md exactly, nothing is worded by hand).

usage: arete97_summary.py --runs <voxelbench/runs> --prefix <slug prefix, e.g. msm2-h3> --out <dir> [--now-ok]
reads, per take (4: Q1 A1 Q2 A2): gate/gate.json, metrics.json, driver.json, run-config.json, server-metrics.txt, vram-samples.csv,
showcase/orbit-choice.json, path-miss.json, work/index.html.  writes <out>/summary.json, <out>/pair.json, <out>/summary.md.
code KB (HARDENING L26) = bytes of the page's lines of at most 50,000 characters (an inlined library sits on one long line), / 1000.
"""
import argparse, csv, datetime, json, os, re, sys
from pathlib import Path

ARMS = {"q": ("qwen", "qwen38-27b-q6k", "Qwen3.8-27B Q6_K"), "a": ("arete", "arete27b-q6k", "Arete-27B Q6_K")}
TAKES = [(1, "q"), (1, "a"), (2, "q"), (2, "a")]          # interleaved order
LIB_LINE = 50000
RATIO_LEVEL = 1.5


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def slug_candidates(prefix, n, arm):
    base = f"{prefix}-t{n}%s-{ARMS[arm][1]}"
    return [base % s for s in ("", "b", "c")]


def pick_slug(runs, prefix, n, arm):
    """the attempt that has a gate.json (latest wins); None when no attempt finished."""
    found = None
    for s in slug_candidates(prefix, n, arm):
        if (Path(runs) / s / "gate" / "gate.json").is_file():
            found = s
    return found


def code_kb(html):
    try:
        n = 0
        for line in open(html, "rb"):
            if len(line) <= LIB_LINE:
                n += len(line)
        return round(n / 1000, 1)
    except Exception:
        return None


def srv_metrics(p):
    out = {}
    try:
        for line in open(p):
            m = re.match(r"llamacpp:(\w+)\s+([0-9.eE+-]+)", line.strip())
            if m:
                out[m.group(1)] = float(m.group(2))
    except Exception:
        pass
    return out


def vram_peak(p):
    try:
        v = [int(r[1]) for r in csv.reader(open(p)) if len(r) > 1 and r[1].strip().isdigit()]
        return max(v) if v else None
    except Exception:
        return None


def take_record(runs, prefix, n, arm):
    slug = pick_slug(runs, prefix, n, arm)
    rec = {"id": f"t{n}-{arm}", "model": ARMS[arm][0], "take": n, "status": "not_run", "slug": None}
    if not slug:
        return rec
    d = Path(runs) / slug
    g = jload(d / "gate" / "gate.json", {}) or {}
    m = jload(d / "metrics.json", {}) or {}
    dr = jload(d / "driver.json", {}) or {}
    rc = jload(d / "run-config.json", {}) or {}
    sm = srv_metrics(d / "server-metrics.txt")
    oc = jload(d / "showcase" / "orbit-choice.json", {}) or {}
    pm = jload(d / "path-miss.json")
    html = g.get("html") or str(d / "work" / "index.html")
    passed = bool(g.get("pass"))
    tr = m.get("tool_results_by_name") or {}
    pred = sm.get("tokens_predicted_total"); psec = sm.get("tokens_predicted_seconds_total")
    rec.update({
        "status": "pass" if passed else "fail", "slug": slug, "gate_pass": passed,
        "gate_checks": g.get("checks"), "gate_error": g.get("error") or g.get("harness_error") or ("no_html" if g.get("no_html") else None),
        "fps": g.get("fps"), "page_bytes": g.get("bytes"), "code_kb": code_kb(html) if passed or g.get("bytes") else None,
        "api_calls": m.get("api_calls"), "tool_calls": sum(tr.values()) if tr else m.get("tool_calls"),
        "tool_calls_session_row": m.get("tool_calls"), "tool_errors_flagged": m.get("tool_errors_flagged"),
        "compacted_messages": m.get("compacted_messages"), "goal": m.get("goal"),
        "driver_end": dr.get("end"), "wall_s": dr.get("wall_s"), "stall_nudges": dr.get("stall_nudges"),
        "server_tokens_predicted": pred, "server_prompt_tokens": sm.get("prompt_tokens_total"),
        "server_decode_tok_s_mean": round(pred / psec, 1) if pred and psec else None,
        "path_miss": bool(pm), "ctx": rc.get("ctx"), "vram_peak_mib": vram_peak(d / "vram-samples.csv"),
        "orbit": {"script": oc.get("script"), "orbit_px": oc.get("orbit_px"), "method": oc.get("method"),
                  "mp4": str(d / "showcase" / "orbit.mp4") if (d / "showcase" / "orbit.mp4").is_file() else None},
    })
    return rec


def verdict(models):
    q, a = models["qwen"], models["arete"]
    pq, pa = q["passes"], a["passes"]
    kq, ka = q["code_kb_mean_passing"], a["code_kb_mean_passing"]
    ratio = None
    if kq and ka:
        ratio = round(max(kq, ka) / min(kq, ka), 2)
    v = {"passes": {"qwen": pq, "arete": pa}, "code_kb_mean_passing": {"qwen": kq, "arete": ka}, "code_kb_ratio": ratio, "threshold_ratio": RATIO_LEVEL}
    if pq == pa:
        if pq == 0:
            v.update(label="LEVEL", text="LEVEL: both models 0 of 2 gate passes; there is no page to compare")
        elif ratio is not None and ratio <= RATIO_LEVEL:
            v.update(label="LEVEL", text=f"LEVEL: both {pq} of 2 gate passes, mean code size within {RATIO_LEVEL}x (ratio {ratio})")
        else:
            big = "qwen" if kq > ka else "arete"
            v.update(label="AHEAD_ON_SIZE_ONLY", ahead=big,
                     text=f"{big.upper()} ahead on code size only: both {pq} of 2 gate passes, mean code KB {kq} vs {ka} (ratio {ratio} > {RATIO_LEVEL}); size is not quality, the 8-element read decides what is said in public")
    else:
        lead = "qwen" if pq > pa else "arete"; diff = abs(pq - pa)
        if diff == 1:
            v.update(label="NOISE_LEVEL", ahead=lead,
                     text=f"NOISE-LEVEL: {lead.upper()} passed 1 more take ({max(pq, pa)} of 2 vs {min(pq, pa)} of 2); a 1-take difference at n=2 is never reported as better")
        else:
            v.update(label="AHEAD", ahead=lead, text=f"{lead.upper()} AHEAD by {diff} takes ({max(pq, pa)} of 2 vs {min(pq, pa)} of 2 gate passes)")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--prefix", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    takes = [take_record(a.runs, a.prefix, n, arm) for n, arm in TAKES]
    models = {}
    for arm, (key, _slug, label) in ARMS.items():
        mine = [t for t in takes if t["model"] == key]
        passing = [t for t in mine if t["status"] == "pass" and t.get("code_kb")]
        models[key] = {"label": label, "takes_run": sum(t["status"] != "not_run" for t in mine),
                       "passes": sum(t["status"] == "pass" for t in mine),
                       "code_kb_passing": [t["code_kb"] for t in passing],
                       "code_kb_mean_passing": round(sum(t["code_kb"] for t in passing) / len(passing), 1) if passing else None}
    complete = all(t["status"] != "not_run" for t in takes)
    pair = {}
    for key in ("qwen", "arete"):
        first = next((t for t in takes if t["model"] == key and t["status"] == "pass"), None)
        if first:
            pair[key] = {"slug": first["slug"], "take": first["take"], "labelled": first["take"] != 1,
                         "label_note": "" if first["take"] == 1 else "take 1 failed the gate",
                         "code_kb": first["code_kb"], "orbit_mp4": first["orbit"]["mp4"]}
        else:
            pair[key] = None
    pair_ok = bool(pair["qwen"] and pair["arete"] and pair["qwen"]["orbit_mp4"] and pair["arete"]["orbit_mp4"])
    res = {"item": "97-arete-msm", "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "complete": complete, "prereg": "PREREG.md (sha256-frozen in FROZEN.sha256 before any take)",
           "takes": takes, "models": models, "verdict": verdict(models) if complete else {"label": "INCOMPLETE", "text": "INCOMPLETE: not all four takes have a gate.json yet"},
           "pair": pair, "pair_ok": pair_ok,
           "eight_element_read": "manual (mercury, from gate shot_6s + an orbit contact sheet); not computed here",
           "publication": "decided at readout from this file, the 8-element read and the vault note; the item never publishes"}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    (out / "pair.json").write_text(json.dumps({"ok": pair_ok, "pair": pair}, indent=2))
    lines = [f"# 97-arete-msm summary ({res['generated']})", "", f"verdict: {res['verdict']['text']}", "",
             "| take | slug | gate | code KB | api calls | tool calls | wall s | server tokens | end |", "|---|---|---|---|---|---|---|---|---|"]
    for t in takes:
        lines.append(f"| {t['id']} | {t.get('slug')} | {t['status']} | {t.get('code_kb')} | {t.get('api_calls')} | {t.get('tool_calls')} | {t.get('wall_s')} | {t.get('server_tokens_predicted')} | {t.get('driver_end')} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": complete, "passes": {k: v["passes"] for k, v in models.items()}, "verdict": res["verdict"]["text"], "pair_ok": pair_ok}))


if __name__ == "__main__":
    main()
