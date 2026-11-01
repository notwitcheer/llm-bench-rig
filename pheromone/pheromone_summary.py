#!/usr/bin/env python3
"""Summarise the pheromone (pressure-field replay + extension) cells.

usage: pheromone_summary.py <results_dir> [--match <pf_cell> <cv_cell>]

Reads every <cell>.json in results_dir (grid / ablation output of
Govcraft/pressure-field-experiment schedule-experiment), walks it for trial
records (dicts with "solved" and "config"), and prints one line per cell
(and per ablation config): solved/N, Wilson 95% CI, mean ticks, mean tokens.
Writes summary.json + summary.md next to the cells.

--match prints the conversation tick budget that matches the pressure-field
cell's mean tokens per trial: round(pf_mean_tokens / cv_tokens_per_tick),
clamped to [50, 400]. Exit 2 when either cell is missing or empty.
"""
import json, math, os, sys, glob


def trials(obj):
    out = []
    def walk(o):
        if isinstance(o, dict):
            if "solved" in o and "config" in o and "total_ticks" in o:
                out.append(o); return
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(obj)
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tok(t):
    return (t.get("total_prompt_tokens") or 0) + (t.get("total_completion_tokens") or 0)


def load(path):
    try: return trials(json.load(open(path)))
    except Exception: return []


def label(t):
    c = t["config"]
    flags = [k for k in ("decay_enabled", "inhibition_enabled", "examples_enabled", "propagation_enabled") if c.get(k) is False]
    return "no_" + "_no_".join(f.replace("_enabled", "") for f in flags) if flags else "full"


def main():
    d = sys.argv[1]
    if len(sys.argv) >= 5 and sys.argv[2] == "--match":
        pf, cv = load(os.path.join(d, sys.argv[3] + ".json")), load(os.path.join(d, sys.argv[4] + ".json"))
        if not pf or not cv: print("match: source cell missing", file=sys.stderr); sys.exit(2)
        pf_mean = sum(map(tok, pf)) / len(pf)
        cv_per_tick = sum(map(tok, cv)) / max(1, sum(t["total_ticks"] for t in cv))
        print(max(50, min(400, round(pf_mean / max(1.0, cv_per_tick)))))
        return
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(f)[:-5]
        if name.startswith("summary") or name.startswith("run-config"): continue
        ts = load(f)
        if not ts: continue
        groups = {}
        for t in ts:
            key = name if not name.endswith("decay") else f"{name}:{label(t)}"
            groups.setdefault(key, []).append(t)
        for key, g in groups.items():
            k = sum(1 for t in g if t["solved"]); n = len(g); lo, hi = wilson(k, n)
            c = g[0]["config"]
            rows.append({"cell": key, "strategy": c.get("strategy"), "agents": c.get("agent_count"), "meetings": c.get("n"),
                         "solved": k, "n": n, "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
                         "mean_ticks": round(sum(t["total_ticks"] for t in g) / n, 1),
                         "mean_tokens": round(sum(map(tok, g)) / n), "final_models": sorted({t.get("final_model") for t in g})})
    json.dump(rows, open(os.path.join(d, "summary.json"), "w"), indent=1)
    lines = ["cell | strategy | agents | solved/N | rate | 95% CI | mean ticks | mean tokens"]
    for r in rows:
        lines.append(f"{r['cell']} | {r['strategy']} | {r['agents']} | {r['solved']}/{r['n']} | {r['rate']*100:.1f}% | "
                     f"{r['ci95'][0]*100:.1f}-{r['ci95'][1]*100:.1f}% | {r['mean_ticks']} | {r['mean_tokens']}")
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
