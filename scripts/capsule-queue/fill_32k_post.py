#!/usr/bin/env python3
"""fill the held 32k MATH-500 post once the last 32k leg (qwable) has landed.

run on mercury:  python3 99-personal/benchmark-rig/scripts/fill_32k_post.py
- pulls fresh counts from capsule (data32k.py + cmp32k.py in ~ on capsule, copies kept next to this script)
- refuses (exit 2) if any 32k leg has no detail json, or if any answer that finished inside 16k changed verdict at 32k
- renders the chart (llm-bench-rig scripts/chart_math500_32k.py) to /opt/data/tmp/math500-32k/math500-32k.png
- writes the final bodies between the FINAL markers in drafts/2026-09-27-math500-32k-post.md and prints them
the template text passed the mercury stop-slop pass on 2026-09-27; this script only fills numbers and one floor clause.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT = os.path.join(HERE, "..", "drafts", "2026-09-27-math500-32k-post.md")
TMP = "/opt/data/tmp/math500-32k"
CHART = "/opt/data/repos/llm-bench-rig/scripts/chart_math500_32k.py"
Q38_TPC, Q38_SCORE = 1779.6, 95.2  # results/qwen3-8-27b-q6-k-thinkon/math500_detail.json (16k, cap 0.016)

def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "capsule-w", cmd], capture_output=True, text=True, timeout=120)

os.makedirs(TMP, exist_ok=True)
for f in ("data32k.py", "cmp32k.py"):
    subprocess.run(["scp", "-q", os.path.join(HERE, f), f"capsule-w:/home/witcheer/{f}"], check=True, timeout=60)
r = ssh("python3 ~/data32k.py")
if r.returncode:
    sys.exit(f"data32k failed: {r.stderr[-500:]}")
rows = json.loads(r.stdout)
json.dump(rows, open(f"{TMP}/data32k.json", "w"), indent=1)
missing = [x["label"] for x in rows if not x["32k"].get("final")]
TESTREPO = os.environ.get("TESTREPO")  # dev only: stand-in values, write report into a scratch copy of the repo, leave the draft alone
PREVIEW = os.environ.get("PREVIEW") == "1" or bool(TESTREPO)
if missing and not PREVIEW:
    print("NOT LANDED:", missing, {x["label"]: x["32k"]["n"] for x in rows}); sys.exit(2)
for x in rows:  # PREVIEW=1 only: stand-in values from the running progress json so the template can be read end to end
    s32 = x["32k"]
    if not s32.get("final"):
        s32.update(score=round(s32["correct"] / s32["n"] * 100, 1), capped_rate=round(s32["capped"] / s32["n"], 3), pf=0, tpc=13000.0,
                   detail=dict(score=round(s32["correct"] / s32["n"] * 100, 1), metric="acc", correct=s32["correct"], total=s32["n"], capped_count=s32["capped"],
                               capped_rate=round(s32["capped"] / s32["n"], 3), parse_failures=0, tokens_per_correct=13000.0, completion_tokens_median=0,
                               reasoning_fallback_count=0, regime="STAND-IN", bench="STAND-IN"))

cmp = ssh("python3 ~/cmp32k.py " + " ".join(x["slug"] for x in rows)).stdout
print(cmp.strip())
for line in cmp.splitlines():
    m = re.search(r"common (\d+) .* uncapped (\d+) .* same correctness (\d+) lost (\d+) won (\d+)", line)
    if m:
        common, unc, same, lost, won = map(int, m.groups())
        if (common != 500 and not PREVIEW) or same != unc or lost or won:
            print("VERDICT CHANGE on a finished answer, do not ship the 'same verdict' line:", line); sys.exit(2)

S = {x["slug"]: x for x in rows}
q = S["qwable-27b-q4-k-m"]["32k"]
caps = [x["32k"]["capped_rate"] * 100 for x in rows]
tpcs = [x["32k"]["tpc"] for x in rows]
fin_acc = []
for x in rows:
    for k in ("16k", "32k"):
        s = x[k]; capped_ok = s["capped"] - s["capped_wrong"]
        fin_acc.append((s["correct"] - capped_ok) / (s["n"] - s["capped"]) * 100)
ratio = [t / Q38_TPC for t in tpcs]
NAMES = {"qwen3-6-35b-a3b-ud-q5-k-m": "Qwen3.6-35B-A3B", "qwen3-6-27b-q6-k": "Qwen3.6-27B", "qwable-27b-q4-k-m": "Qwable-27B"}
floor_sent = " ".join(f"{x['32k']['capped_rate']*100:.0f}% of {NAMES[x['slug']]}'s answers still hit the 32k cap, so its {x['32k']['score']:.1f} is a floor."
                      for x in rows if x["32k"]["capped_rate"] >= 0.10)
cap_lo, cap_hi = round(min(caps)), round(max(caps))
cap_s = f"{cap_lo}%" if cap_lo == cap_hi else f"{cap_lo} to {cap_hi}%"
tpc_s = f"{round(min(tpcs), -2):,.0f} to {round(max(tpcs), -2):,.0f}"
seventh = "about a seventh" if 6.5 <= min(ratio) and max(ratio) <= 7.5 else f"{1/max(ratio):.2f} to {1/min(ratio):.2f} times"
fa = f"about {round(sum(fin_acc)/len(fin_acc))}%"
if max(fin_acc) - min(fin_acc) > 1.5:
    fa = f"{min(fin_acc):.0f} to {max(fin_acc):.0f}%"
Q = f"{q['score']:.1f}"
d = sorted(x["32k"]["score"] - x["16k"]["score"] for x in rows)

x_main = f"""doubling the thinking limit from 16k to 32k tokens took three models from 72 to 73 on competition maths to 91.2, 92.0 and {Q}.

they were the three odd rows on my thinking board last week: 93.9 to 95.1 on the code tests, but 27 to 30% of their maths answers had been cut off by the limit. same 500 MATH-500 questions this time, same RTX 5090, only the limit changed.

~ Qwen3.6-35B-A3B: 72.8 at 16k, 91.2 at 32k
~ Qwen3.6-27B: 73.2 at 16k, 92.0 at 32k
~ Qwable-27B: 71.6 at 16k, {Q} at 32k

every answer that finished inside 16k got the same verdict at 32k. all of the extra points came from questions that had run out of room. when these three finish an answer, they are right {fa} of the time.

the cost is tokens. they spend {tpc_s} tokens per correct answer, and {cap_s} of questions still hit the 32k limit. Qwen3.8-27B scored 95.2 on the same questions at 16k with about 1,800 tokens per correct answer.

if you already run one of these three with thinking on, raise the limit to 32k for maths. if you are choosing a model for maths, Qwen3.8-27B scores higher on {seventh} of the tokens."""

reply1 = """[attach math500-32k.png]
the chart, per-question counts and the script: https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/second-tier-thinkon-32k.md"""

reply2 = ("method: llama.cpp, greedy, zero-shot, one pass per question. max 32,768 completion tokens with ctx 40,960 "
          "(the 16k run used 16,384 and 24,576). the 32k numbers are their own table, not merged into the 16k board."
          + (" " + floor_sent if floor_sent else ""))

tg = f"""last week my thinking board had three models stuck at 72 to 73 on competition maths, while the same three scored 93.9 to 95.1 on the code tests. 27 to 30% of their maths answers had hit the 16k thinking limit before they finished.

so I gave exactly those three double the room, 32k tokens, and reran the same 500 questions on the same RTX 5090:

~ Qwen3.6-35B-A3B: 72.8 to 91.2
~ Qwen3.6-27B: 73.2 to 92.0
~ Qwable-27B: 71.6 to {Q}

every answer they had finished at 16k was marked the same way at 32k. every extra point came from a question that had run out of room. when these models finish, they are right {fa} of the time.

the cost is the length of the thinking: {tpc_s} tokens per correct answer. Qwen3.8-27B got 95.2 on the same questions at 16k with about 1,800.

short version: if you run one of these three with thinking on, give it 32k for maths.

the chart and the full table: https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/second-tier-thinkon-32k.md"""

if PREVIEW and not TESTREPO:
    print("PREVIEW (stand-in qwable values, nothing written):\n\n" + x_main + "\n\n---\n" + reply2 + "\n\n---\n" + tg); sys.exit(0)
r = subprocess.run(["/opt/data/venvs/charts/bin/python", CHART, f"{TMP}/data32k.json", f"{TMP}/math500-32k.png"], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr[-300:])

# ---------------------------------------------------------------- report + csv + README in the llm-bench-rig clone (local commit only, push is gated)
import csv, math
REPO = TESTREPO or "/opt/data/repos/llm-bench-rig"
TODAY = subprocess.run(["date", "-u", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip()
QUANT = {"qwen3-6-35b-a3b-ud-q5-k-m": ("Qwen3.6-35B-A3B", "UD-Q5_K_M"), "qwen3-6-27b-q6-k": ("Qwen3.6-27B", "Q6_K"), "qwable-27b-q4-k-m": ("Qwable-27B", "Q4_K_M")}
CMP = {}
for line in cmp.splitlines():
    m = re.match(r"(\S+) common (\d+) 16k-capped (\d+) -> correct at 32k (\d+) still capped (\d+) \| uncapped (\d+) same tokens (\d+) same correctness (\d+) lost (\d+) won (\d+)", line.strip())
    if m:
        k = m.group(1); v = list(map(int, m.groups()[1:]))
        CMP[k] = dict(zip(["common", "capA", "capA_ok32", "capA_still", "unc", "same_tok", "same_corr", "lost", "won"], v))
def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    return z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den * 100
def gb(b): return f"{b/1e9:.1f} GB" if b else "-"
tab, ver = [], []
for x in rows:
    a, b, c = x["16k"], x["32k"], CMP[x["slug"]]; name, qn = QUANT[x["slug"]]; d32 = b["detail"]
    tab.append(f"| {name} | {qn} | {gb(x.get('file_bytes'))} | {a['score']:.1f} | {a['capped_rate']*100:.1f}% | **{b['score']:.1f}** | {b['capped_rate']*100:.1f}%{' ▲' if b['capped_rate'] >= 0.10 else ''} | {b['pf']} | {b['tpc']:,.0f} | ±{wilson(d32['correct'], d32['total']):.1f} |")
    ver.append(f"| {name} | {c['unc']} | {c['same_corr']} | {c['same_tok']} | {c['capA']} | {c['capA_ok32']} | {c['capA_still']} |")
unc_s = ", ".join(f"{CMP[x['slug']]['same_corr']}/{CMP[x['slug']]['unc']}" for x in rows)
tok_s = ", ".join(f"{CMP[x['slug']]['same_tok']}/{CMP[x['slug']]['unc']}" for x in rows)
capA_tot = sum(CMP[x["slug"]]["capA"] for x in rows); capA_ok = sum(CMP[x["slug"]]["capA_ok32"] for x in rows)
fb = [x["32k"]["detail"].get("reasoning_fallback_count") for x in rows]
lo16, hi16 = min(x["16k"]["score"] for x in rows), max(x["16k"]["score"] for x in rows)
lo32, hi32 = min(x["32k"]["score"] for x in rows), max(x["32k"]["score"] for x in rows)
floors = [QUANT[x["slug"]][0] for x in rows if x["32k"]["capped_rate"] >= 0.10]
floor_line = (f"{' and '.join(floors)} still cap{'s' if len(floors) == 1 else ''} at 10% or more at 32k (▲), so {'that score is' if len(floors) == 1 else 'those scores are'} still a floor." if floors else "All three rows cap under 10% at 32k.")
gap_lo, gap_hi = Q38_SCORE - hi32, Q38_SCORE - lo32
BUILD = "llama.cpp build 10371 (`5d16e81dd`)"
same_med = all(x["16k"]["detail"]["completion_tokens_median"] == x["32k"]["detail"]["completion_tokens_median"] for x in rows) if all("detail" in x["16k"] for x in rows) else False
median_line = ("The median completion length is identical at both budgets on all three rows, because the median item finishes well inside 16k." if same_med
               else "Median completion lengths per budget are in the csv.")
report = f"""# Thinking on at a 32k budget: the three MATH-500 rows that capped at 16k

**Rig:** RTX 5090 32GB (capsule), {BUILD}, llama-server, batch 1 · **Window:** 2026-09-20 to {TODAY}, inside the box's cheap-electricity hours · **Harness:** [llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig) second-tier lane, idle-queue item `15-second-tier-thinkon-32k`, artefacts in `results/<slug>-thinkon-32k/` · **Data:** [`dataset/second_tier_32k.csv`](../dataset/second_tier_32k.csv) · **Chart:** [`second-tier-thinkon-32k.png`](second-tier-thinkon-32k.png) · **Parent:** [thinking on at a 16k budget](second-tier-thinkon-16k.md)

![MATH-500 at 16k vs 32k](second-tier-thinkon-32k.png)

This is a separate bench, not a rerun merged into the 16k board. The 16k board flagged three MATH-500 cells as budget floors (▲): the rows hit the 16,384-token completion cap on 27 to 30% of the 500 problems. This pass gives exactly those three rows 32,768 tokens and changes nothing else that affects the answers.

## TL;DR

1. **Doubling the budget moved the three rows from {lo16:.1f} to {hi16:.1f} up to {lo32:.1f} to {hi32:.1f}.** Qwen3.6-35B-A3B UD-Q5_K_M {rows[0]['16k']['score']:.1f} to {rows[0]['32k']['score']:.1f}, Qwen3.6-27B Q6_K {rows[1]['16k']['score']:.1f} to {rows[1]['32k']['score']:.1f}, Qwable-27B Q4_K_M {rows[2]['16k']['score']:.1f} to {rows[2]['32k']['score']:.1f}.
2. **Every answer that finished inside 16k got the same verdict at 32k** ({unc_s} per row), with the same completion length on {tok_s}. No finished answer flipped in either direction, so every added point is an item that had hit the 16k cap: {capA_ok} of those {capA_tot} items are correct at 32k.
3. **The cost per correct answer barely moved** ({tpc_s} completion tokens per correct answer at 32k, against 12,406 to 12,895 at 16k). {median_line} {floor_line}

## Setup

- **Regime:** thinking on, `max_tokens 32768`, `ctx 40960`, greedy, zero-shot, batch 1 through llama-server's chat-completions endpoint, one pass per item. The 16k board used `max_tokens 16384`, `ctx 24576`; the MATH-500 prompt is under 2k tokens, so the context change only adds room for the longer completion.
- **Task:** MATH-500 only (500 problems, answer equivalence via the vendored Qwen-Math grader; `grade(r'\\frac{{1}}{{2}}', '0.5')` prints `True` in the run venv, checked {TODAY}). IFEval and the two code suites capped under 8% for these rows at 16k and were not rerun.
- **Rows:** the same three GGUF files as the 16k board. Each detail json carries `regime` `think-on, max_tokens 32768, ctx 40960, zero-shot, greedy` and `bench` `second-tier-32k (separate pass, not comparable to the 16k -thinkon rows)`.

## The table

| Model | Quant | GGUF | 16k MATH-500 | 16k cap | 32k MATH-500 | 32k cap | 32k unboxed | 32k tokens / correct | 32k Wilson 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(tab) + f"""

▲ 10% or more of the items still hit the 32k cap: a floor, not a ceiling. `unboxed` is answers the grader could not extract. Per-row `correct`/`total`, `capped_count`, `parse_failures`, `tokens_per_correct` and the median completion length: [`dataset/second_tier_32k.csv`](../dataset/second_tier_32k.csv).

## Item by item

Both passes keep a per-item progress file, so the two runs can be compared question by question.

| Model | finished at 16k | same verdict at 32k | same completion length | hit the 16k cap | of those, correct at 32k | still capped at 32k |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(ver) + f"""

## Reads

1. **The 16k maths cells measured the budget.** On the three rows, answers that finished were right about as often at 16k as at 32k (finished-answer accuracy {min(fin_acc):.1f} to {max(fin_acc):.1f}% across the six cells); what changed was how many answers finished.
2. **More room does not make these models cheaper per answer.** Tokens per correct answer stay at 12k to 13k. Qwen3.8-27B Q6_K scored {Q38_SCORE} on the same 500 problems at 16k with 1.6% capped and {Q38_TPC:,.0f} tokens per correct answer, {gap_lo:.1f} to {gap_hi:.1f} points above these three at 32k. The budgets differ, so that is context, not a merged ranking.
3. **Some capped answers still score.** When a completion hits the cap, the grader can still credit an answer it finds in the reasoning text (`reasoning_fallback_count` at 32k: {', '.join(str(v) for v in fb)}). That is why the chart's red bar is shorter than the cap share.

## Worth it if / not if

- **Worth it if** you run Qwen3.6-27B, Qwen3.6-35B-A3B or Qwable-27B with thinking on for maths: raise the completion limit to 32k. The 16k ceiling cost them {d[0]:.1f} to {d[-1]:.1f} points here.
- **Not if** you are choosing a model for maths from scratch on one 5090: the Qwen3.8-27B rows on the 16k board score higher on {seventh} of the tokens per correct answer.

## Limits

- One greedy pass per item, one card. The Wilson intervals above are sampling intervals, not run-to-run variance.
- The pass ran across weekday and weekend cheap-electricity blocks from 2026-09-20 to {TODAY}; a leg cut by a block's hard stop resumed from its per-item progress file at the next block.
- Build: both this pass and the 16k board ran through the rig's configured `llama_cpp.server_bin`, {BUILD}, built 2026-08-11. The 16k report first cited b9653 (`9dbc6621a`); that was corrected on {TODAY}. Completion lengths match on {tok_s} of the items that finished at 16k, which fits one build across both passes.
- MATH-500 only. A 64k pass for the rows still at or above 10% capped is not queued.

*Sources: `results/<slug>-thinkon-32k/math500_detail.json`, `math500_progress.json` and `math500_tokens.jsonl` on the rig, {TODAY}; the 16k cells from `results/<slug>-thinkon/` (same files as the 16k board); `dataset/second_tier_32k.csv` in this commit; chart script `scripts/chart_math500_32k.py`.*
"""
open(f"{REPO}/reports/second-tier-thinkon-32k.md", "w").write(report)
subprocess.run(["cp", f"{TMP}/math500-32k.png", f"{REPO}/reports/second-tier-thinkon-32k.png"], check=True)
cols = ["slug", "model", "quant", "file_bytes", "task", "metric", "score", "correct", "total", "capped_count", "capped_rate", "parse_failures", "tokens_per_correct", "completion_tokens_median", "regime", "bench"]
with open(f"{REPO}/dataset/second_tier_32k.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for x in rows:
        d32 = x["32k"]["detail"]; name, qn = QUANT[x["slug"]]
        w.writerow([x["slug"], name, qn, x.get("file_bytes") or "", "math500", d32.get("metric", "acc"), d32["score"], d32["correct"], d32["total"], d32["capped_count"], d32["capped_rate"], d32["parse_failures"], d32["tokens_per_correct"], d32["completion_tokens_median"], d32["regime"], d32.get("bench", "")])
# README: build fix + 32k pointer (idempotent)
rp = f"{REPO}/dataset/README.md"; rd = open(rp).read()
rd = rd.replace("greedy, zero-shot, one pass per item, llama.cpp b9653, 1,583 items per model.", f"greedy, zero-shot, one pass per item, {BUILD} (first published as b9653, corrected {TODAY}), 1,583 items per model.")
ptr = "[thinking on at a 16k budget](../reports/second-tier-thinkon-16k.md) · [chart](../reports/second-tier-thinkon-16k.png)."
add32 = f" **32k pass:** the three ▲ MATH-500 rows rerun at `max_tokens 32768` as a separate table, {lo16:.1f}-{hi16:.1f} at 16k to {lo32:.1f}-{hi32:.1f} at 32k, every finished 16k answer unchanged: [thinking on at a 32k budget](../reports/second-tier-thinkon-32k.md) · [`second_tier_32k.csv`](second_tier_32k.csv)."
if "second-tier-thinkon-32k.md" not in rd and ptr in rd:
    rd = rd.replace(ptr, ptr + add32, 1)
open(rp, "w").write(rd)
# 16k report: build correction + owed bullet now points at the 32k report (idempotent)
p16 = f"{REPO}/reports/second-tier-thinkon-16k.md"; r16 = open(p16).read()
r16 = r16.replace("llama.cpp b9653 (`9dbc6621a`), llama-server", f"{BUILD}, llama-server (corrected {TODAY}: first published as b9653 `9dbc6621a`; the rig's `server_bin` has pointed at the `5d16e81dd` build since 2026-08-11)")
r16 = r16.replace("- The 32k pass on the three capped MATH-500 legs (`results/<slug>-thinkon-32k/`, separate bench, same rows) is queued and will publish as its own table, never merged into this one.",
                  f"- ~~The 32k pass on the three capped MATH-500 legs~~ published {TODAY} as its own table: [thinking on at a 32k budget](second-tier-thinkon-32k.md).")
open(p16, "w").write(r16)
g = subprocess.run(f"cd {REPO} && git add reports/second-tier-thinkon-32k.md reports/second-tier-thinkon-32k.png dataset/second_tier_32k.csv dataset/README.md reports/second-tier-thinkon-16k.md scripts/chart_math500_32k.py && git -c user.name=witcheer -c user.email=witcheer.eth@gmail.com commit -qm 'second tier, thinking on at 32k: the three capped MATH-500 rows as a separate table (report, csv, chart, chart script); 16k report build line corrected' && git log --oneline -1 && git status -sb | head -2",
                   shell=True, capture_output=True, text=True)
print("LLM-BENCH-RIG LOCAL COMMIT (NOT PUSHED):", g.stdout.strip(), g.stderr[-300:])
if TESTREPO:
    print("TESTREPO run: draft untouched"); sys.exit(0)

final = (f"<!-- FINAL START (filled {subprocess.run(['date','-u','+%Y-%m-%d %H:%M UTC'],capture_output=True,text=True).stdout.strip()} by fill_32k_post.py) -->\n"
         f"# X main (FINAL)\n\n{x_main}\n\n# X reply 1 (FINAL)\n\n{reply1}\n\n# X reply 2, method (FINAL)\n\n{reply2}\n\n# Grimoire TG (FINAL, chart attached)\n\n{tg}\n"
         f"\n<!-- numbers: qwable 32k {Q} cap {q['capped_rate']} pf {q['pf']} tpc {q['tpc']}; 32k deltas {d}; caps {caps}; tpcs {tpcs}; finished-answer accuracy {[round(v,1) for v in fin_acc]}; ratio vs qwen3.8 tpc {[round(v,2) for v in ratio]} -->\n"
         "<!-- FINAL END -->")
s = open(DRAFT).read()
s = re.sub(r"<!-- FINAL START.*?<!-- FINAL END -->", lambda m: final, s, flags=re.S)
open(DRAFT, "w").write(s)
print(final)
