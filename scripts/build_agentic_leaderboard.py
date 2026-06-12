"""Regenerate the Agentic Score leaderboard artifacts from results/<slug>/agentic_native.json.
Writes leaderboard/{leaderboard.json, leaderboard.csv, README.md}. Re-run after every new model.

Adding a model = run it through lib/agentic/native (writes results/<slug>/agentic_native.json),
add a META entry below, re-run this script, re-upload the leaderboard/ dir to the HF dataset."""
import json, csv
from collections import Counter
from pathlib import Path

DATE = "2026-06-12"  # stamp; bench rig has no live clock in workflow contexts

# slug -> display metadata (the result JSON carries the metrics; this carries identity)
META = {
    "qwen3-5-35b-base":    {"name": "Qwen3.5-35B-A3B (base)", "params": "35B-A3B", "quant": "Q4_K_M",
                            "repo": "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF", "note": "generalist base, no agentic post-train"},
    "granite-4-1-30b":     {"name": "Granite-4.1-30b", "params": "30B", "quant": "UD-Q4_K_XL",
                            "repo": "unsloth/granite-4.1-30b-GGUF", "note": "leanest competent agent on the board"},
    "nex-n2-mini":         {"name": "Nex-N2-mini", "params": "35B-A3B", "quant": "Q4_K_M",
                            "repo": "eramax/Nex-N2-mini-gguf", "note": "agentic post-train of Qwen3.5; Adaptive Thinking"},
    "kimi-linear-48b-a3b": {"name": "Kimi-Linear-48B-A3B", "params": "48B-A3B", "quant": "Q4_K_M",
                            "repo": "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF",
                            "note": "linear-attention; runs natively on one 5090; long-context untested here"},
    "qwen3-6-27b":         {"name": "Qwen3.6-27B", "params": "27B", "quant": "Q6_K",
                            "repo": "unsloth/Qwen3.6-27B-GGUF", "note": "dense 27B; the model Donald itself runs"},
    "nemotron-cascade-2-30b": {"name": "Nemotron-Cascade-2-30B", "params": "30B-A3B", "quant": "Q4_K_M",
                            "repo": "bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF",
                            "note": "Nvidia; the rig's prior speed-king"},
    "qwopus-glm-18b":      {"name": "Qwopus-GLM-18B", "params": "18B", "quant": "Q6_K",
                            "repo": "KyleHessling1/Qwopus-GLM-18B-Merged-GGUF", "note": "GLM-based community merge"},
    "qwopus3-6-27b-coder": {"name": "Qwopus3.6-27B-Coder", "params": "27B", "quant": "Q5_K_M",
                            "repo": "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF",
                            "note": "coder SFT of Qwopus3.6-v2; trained on Hermes agent traces — partially in-distribution for this bench (see reality anchor)"},
}

rows = []
for slug, meta in META.items():
    p = Path(f"results/{slug}/agentic_native.json")
    if not p.exists():
        print(f"  skip {slug} (no result)"); continue
    d = json.loads(p.read_text())
    ax, ok = Counter(), Counter()
    for r in d["details"]:
        ax[r["axis"]] += 1; ok[r["axis"]] += 1 if r["success"] else 0
    rows.append({
        "model": meta["name"], "params": meta["params"], "quant": meta["quant"],
        "agentic_score": d["score"], "task_success_pct": d["task_success_pct"],
        "tool_eff": d["tool_eff"], "stable_pct": d["stable_pct"], "tokens_per_task": d["tokens_per_task"],
        "chain": f"{ok['chain']}/{ax['chain']}", "multistep": f"{ok['multistep']}/{ax['multistep']}",
        "coding": f"{ok['coding']}/{ax['coding']}", "gate": "pass", "hf_repo": meta["repo"],
        "date": DATE, "note": meta["note"],
    })
    # long-context (separate sub-score; missing file -> "-", VRAM wall -> "OOM")
    def _lc(tier):
        f = Path(f"results/{slug}/agentic_longctx_{tier}.json")
        if not f.exists():
            return "-"
        d2 = json.loads(f.read_text())
        return "OOM" if d2.get("reach") == "wall" else f"{d2['success_pct']:.0f}%"
    rows[-1]["lc32k"] = _lc("32k")
    rows[-1]["lc128k"] = _lc("128k")
    rows[-1]["reach"] = ("128k" if rows[-1]["lc128k"] not in ("-", "OOM")
                         else "32k" if rows[-1]["lc32k"] not in ("-", "OOM") else "-")
rows.sort(key=lambda r: r["agentic_score"], reverse=True)

out = Path("leaderboard"); out.mkdir(exist_ok=True)
(out / "leaderboard.json").write_text(json.dumps(rows, indent=2))
with (out / "leaderboard.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def md_table(rows):
    head = ("| # | model | params | Agentic Score | success | tool-eff | tokens/task | "
            "chain | multistep | coding | lc@32k | lc@128k |")
    sep = "|" + "---|" * 12
    body = []
    for i, r in enumerate(rows, 1):
        star = " 🏆" if i == 1 else ""
        body.append(f"| {i} | **{r['model']}**{star} | {r['params']} | **{r['agentic_score']}** | "
                    f"{r['task_success_pct']:.0f}% | {r['tool_eff']:.2f} | {r['tokens_per_task']:.0f} | "
                    f"{r['chain']} | {r['multistep']} | {r['coding']} | "
                    f"{r.get('lc32k', '-')} | {r.get('lc128k', '-')} |")
    return "\n".join([head, sep] + body)

readme = f"""---
license: mit
tags:
- agentic
- tool-calling
- llm-leaderboard
- benchmark
- rtx-5090
pretty_name: Agentic Score Leaderboard (one RTX 5090)
configs:
- config_name: default
  data_files: leaderboard.csv
---

# 🛠️ Agentic Score Leaderboard — one RTX 5090

How well do local models actually **drive a tool-using agent loop**? Not single-call function-calling
benchmarks — a real loop: native OpenAI tool-calling through `llama-server`, multi-step deterministic
tasks, programmatic verification. Everything runs on a **single RTX 5090 32GB**.

Updated **{DATE}** · llama.cpp b9562 · `--jinja` native tool-calling · temp 0.

## Leaderboard

{md_table(rows)}

*tool-eff = tool calls vs optimal (1.0 = no wasted calls). tokens/task = avg completion tokens (lower =
leaner). A sub-5% score gap is a tie.*

![efficiency frontier](agentic-leaderboard-frontier.png)

![agentic score](agentic-leaderboard-score.png)

![long-context reach](agentic-leaderboard-longctx.png)

## The Agentic Score (0–100)

Aggregate over 36 deterministic short-context tasks across five axes (tool-use chains, multi-step
dependencies, sandboxed coding, **error-recovery**, **distractor-robustness**), weighted as below.
A separate **long-context** axis (needle-in-a-document at 32K / 128K) is reported in the `lc@` columns,
not blended into the score (so a 128K VRAM wall doesn't corrupt it):

| axis | weight | measures |
|---|---|---|
| Task success | 0.50 | % tasks completed correctly (programmatic check) |
| Tool efficiency | 0.20 | tool calls vs optimal; malformed/wasted calls penalized |
| Token efficiency | 0.15 | avg tokens/task (efficiency at equal success) |
| Loop stability | 0.15 | completes without stalling / exceeding the step cap |

Calibration-grade (synthetic, deterministic, re-runnable) — and **reality-anchored**: across all 8 models on
30 real SWE-bench Verified bugs, the synthetic score predicts real-bug *rank* (Spearman ρ=0.76) and
moderately predicts resolve rate (Pearson r=0.59). Two known failure modes, both caught by the anchor:
it over-ranks models that drive tools fluently but don't commit fixes (Nemotron-Cascade-2: synthetic #5,
real last), and it over-ranks models whose training data overlaps the bench's flavor (Qwopus3.6-27B-Coder:
trained on Hermes agent traces, posts a perfect 100 synthetic — then resolves fewer real bugs than its own
base model, 57% vs 63%). Read the top of the board with the anchor open (see `reality-anchor/`).
Harness + unit tests: **[notwitcheer/llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig)**
(`lib/agentic/native/`).

## Notes per model

""" + "\n".join(f"- **{r['model']}** ({r['hf_repo']}): {r['note']}" for r in rows) + """

## How it grows

Each model: served Donald-safe on the 5090, hard-gated for parseable native `tool_calls`, then run
through the harness. New models are appended as they're benched. Companion write-ups live in the
[rtx-5090-benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks) dataset.
"""
(out / "README.md").write_text(readme)
print("wrote leaderboard/{leaderboard.json, leaderboard.csv, README.md} —", len(rows), "models")
for r in rows:
    print(f"  {r['agentic_score']:5}  {r['model']}")
