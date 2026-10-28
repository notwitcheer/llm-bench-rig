#!/usr/bin/env python3
"""aggregate the L1 C1 rows into the tier table and a readout (CPU only, idempotent, safe on a partial run).

reads <res>/<slug>/summary.json for every row of rows.json that finished (+ optional <res>/fit-q8/<slug>.json), writes into <out>:
  l1-tier-table.json / .csv   one line per finished row: score, groups, tool errors, approval blocks, wall, first-turn wait, measured peak VRAM, tier
  l1-readout.md               per-tier pick under the PRE-REGISTERED rules (see PREREG.md), spread test, paired reads, honest-limits reminder
rules (pre-registered): tier = smallest card in 8/12/16/24/32 GB with peak_mib + 768 <= card * 1024 (card_fit.py convention); the pick for card C = the highest score among the rows that
fit C; a lead of fewer than 3 tasks (of 30) is called a tie and no pick is made; paired exact McNemar (two-sided) between the two best rows of a card is reported, never decisive on one run.
usage: aggregate.py --res DIR --out DIR [--rows rows.json]"""
import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = (8, 12, 16, 24, 32)
HEADROOM = 768


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def tier_of(peak):
    if peak is None:
        return None
    for c in CARDS:
        if peak + HEADROOM <= c * 1024:
            return c
    return 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=os.path.join(HERE, "rows.json"))
    a = ap.parse_args()
    spec = json.load(open(a.rows))
    os.makedirs(a.out, exist_ok=True)
    table, pending = [], []
    for r in spec["rows"]:
        sp = os.path.join(a.res, r["slug"], "summary.json")
        nofit = os.path.join(a.res, r["slug"], "load-failed.json")
        if os.path.isfile(nofit):
            table.append({"slug": r["slug"], "label": r["label"], "nofit": True, "reason": json.load(open(nofit)).get("reason")})
            continue
        if not os.path.isfile(sp):
            pending.append(r["slug"])
            continue
        s = json.load(open(sp))
        fq = os.path.join(a.res, "fit-q8", r["slug"] + ".json")
        q8 = json.load(open(fq)) if os.path.isfile(fq) else None
        peak = s.get("vram_peak_mib")
        meta = dict(s.get("meta", {}))
        stp = os.path.join(a.res, r["slug"], "server-timings.json")
        if os.path.isfile(stp):  # decode p50 over requests of 32+ tokens and the cold first prefill, from the llama-server log
            try:
                st = json.load(open(stp))
                meta["decode_tok_s_p50"] = (st.get("decode_tps_requests_ge32tok") or {}).get("p50")
                big = [q for q in st.get("requests_detail", []) if q.get("prompt_n", 0) >= 2000]
                meta["first_big_prefill_s"] = round(big[0]["prompt_ms"] / 1000, 1) if big else None
            except Exception:  # noqa: BLE001
                pass
        table.append({"slug": r["slug"], "label": r["label"], "family": r["family"], "file_gb": round(r["bytes"] / 1e9, 2), "tier_hyp": r["tier_hyp"], "passed": s["passed"], "n": s["n"],
                      "mean_frac": s["mean_frac"], "by_group": s["by_group"], "tool_errors_other": s["tool_errors_other"], "approval_blocks": s["approval_blocks"],
                      "malformed_calls": s["malformed_calls"], "timeouts": s["timeouts"], "tool_calls": s["tool_calls"], "wall_s_total": s["wall_s_total"], "wall_s_median": s["wall_s_median"],
                      "first_turn_cold_s": s.get("first_assistant_s_cold"), "first_turn_median_s": s.get("first_assistant_s_median"), "vram_peak_mib": peak,
                      "vram_loaded_mib": meta.get("vram_loaded_mib"), "tier": tier_of(peak), "n_cpu_moe": meta.get("n_cpu_moe_used"), "effort": meta.get("effort"),
                      "decode_tok_s_p50": meta.get("decode_tok_s_p50"), "first_big_prefill_s": meta.get("first_big_prefill_s"), "fit_q8_peak_mib": (q8 or {}).get("vram_peak_mib"), "fit_q8_tier": tier_of((q8 or {}).get("vram_peak_mib")),
                      "per_task": s["per_task"], "sampling_applied": meta.get("sampling_applied")})
    done = [t for t in table if not t.get("nofit")]
    json.dump({"rows": table, "pending": pending}, open(os.path.join(a.out, "l1-tier-table.json"), "w"), indent=1)
    cols = ["slug", "label", "passed", "n", "tool_errors_other", "approval_blocks", "malformed_calls", "timeouts", "wall_s_total", "first_turn_cold_s", "first_big_prefill_s", "decode_tok_s_p50", "vram_peak_mib", "tier", "fit_q8_peak_mib", "fit_q8_tier", "file_gb"]
    with open(os.path.join(a.out, "l1-tier-table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in done:
            w.writerow([t.get(c) for c in cols])
    L = ["# L1 readout (generated by aggregate.py; numbers come from <res>/<row>/summary.json, nothing typed by hand)", "",
         f"rows finished {len(done)} of {len(spec['rows'])}; pending: {', '.join(pending) or 'none'}", "",
         "| row | pass /30 | easy A | mid B | hard C | blocks | other tool errors | malformed | timeouts | wall min | first turn cold s | peak MiB @64k | tier | q8_0 KV peak | q8 tier |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in sorted(done, key=lambda x: -x["passed"]):
        g = t["by_group"]
        L.append(f"| {t['label']} | {t['passed']} | {g.get('diff:A')} | {g.get('diff:B')} | {g.get('diff:C')} | {t['approval_blocks']} | {t['tool_errors_other']} | {t['malformed_calls']} | {t['timeouts']} | "
                 f"{t['wall_s_total'] / 60:.0f} | {t['first_turn_cold_s']} | {t['vram_peak_mib']} | {t['tier']} | {t['fit_q8_peak_mib']} | {t['fit_q8_tier']} |")
    L += ["", "## spread test (pre-registered)", ""]
    if done:
        sc = [t["passed"] for t in done]
        small = [t for t in done if t["slug"].startswith("spark")]
        ceiling = bool(small) and small[0]["passed"] >= 27
        L.append(f"scores across rows: min {min(sc)}, max {max(sc)} of 30 (range {max(sc) - min(sc)}). ceiling flag (the 4B Spark row at 27+ of 30): {'YES, the set does not spread, say so' if ceiling else 'no'}. "
                 f"floor flag (best row at 5 or fewer): {'YES' if max(sc) <= 5 else 'no'}.")
    L += ["", "## pick per card (rule: highest score among rows whose measured peak + 768 MiB fits; lead under 3 tasks = tie, no pick)", ""]
    for c in CARDS:
        fit = [t for t in done if t["tier"] is not None and t["tier"] <= c]
        if not fit:
            L.append(f"- {c} GB: no finished row fits at 64k f16 KV yet")
            continue
        fit.sort(key=lambda x: (-x["passed"], x["vram_peak_mib"], x["wall_s_total"]))
        top = fit[0]
        if len(fit) > 1 and top["passed"] - fit[1]["passed"] < 3:
            tied = [t for t in fit if top["passed"] - t["passed"] < 3]
            keys = list(fit[0]["per_task"])
            n01, n10, p = mcnemar([fit[0]["per_task"][k][0] for k in keys], [fit[1]["per_task"].get(k, [0])[0] for k in keys])
            L.append(f"- {c} GB: TIE within 2 tasks: " + ", ".join(f"{t['label']} {t['passed']}/30" for t in tied) + f" (top two paired {n01}/{n10} discordant, exact p {p:.3f}); no pick from one run")
        else:
            L.append(f"- {c} GB: **{top['label']}** {top['passed']}/30 ({top['vram_peak_mib']} MiB peak" + (f", next {fit[1]['label']} {fit[1]['passed']}/30" if len(fit) > 1 else "") + ")")
    L += ["", "## remember when quoting", "",
          "one run per row at the server's sampling (explicit flags, see rows.json), 30 own tasks, not a public benchmark; the 5090 is a stand-in (peak bytes transfer, decode speed does not); headless approval blocks "
          "and other tool errors are reported separately; first-turn wait is the cold first call of the row's first task; f16 KV at -c 65536 unless a column says q8_0."]
    open(os.path.join(a.out, "l1-readout.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
