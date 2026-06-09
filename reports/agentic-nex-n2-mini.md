# Agentic post-training, measured: Nex-N2-mini vs its base — on one RTX 5090

**Rig:** single RTX 5090 32GB · llama.cpp b9562 · native OpenAI tool-calling (`--jinja`) · temp 0
**Subjects:** `Nex-N2-mini` (35B-A3B, agentic post-train) vs its base `Qwen3.5-35B-A3B`, both Q4_K_M
**Harness:** the rig's new **Agentic Score** — a model-agnostic native tool-calling loop over 15
deterministic, programmatically-checked tasks (tool-use chains, multi-step dependencies, sandboxed
coding). Calibration-grade, not SWE-bench; a sub-5% gap is a tie.

---

## The finding

Nex-N2's headline claim is that "Adaptive Thinking" cuts ~20% of tokens at zero performance loss. On
this rig, **the token claim is not just true — it's an understatement.** Nex-N2-mini does the same
agentic work in **93 tokens/task vs the base's 264 — a 65% cut, ~2.8x leaner.** But "zero performance
loss" does **not** hold: the post-train trades **13 points of task success** for that efficiency.

| axis | base Qwen3.5-35B-A3B | Nex-N2-mini | read |
|---|---|---|---|
| **Agentic Score** | **98.0** | 92.0 | base wins overall |
| Task success | **100%** (15/15) | 86.7% (13/15) | base completes more |
| Tool efficiency | 0.90 | **0.933** | Nex slightly tighter |
| Loop stability | 100% | 100% | tie — neither stalls |
| **Tokens / task** | 264.3 | **93.3** | **Nex −65%** |

Per axis, Nex's misses are concentrated: chain 4/5, multistep **5/5**, coding 4/5. The base is clean 5/5/5.

## The mechanism (why the post-train loses success)

Both failures trace to the *same* cause — terse Adaptive Thinking skips the sanity-check a longer pass
would have caught:

1. **`chain_vram_sq`** — Nex searched (got 32GB), then called `calc("32^2")`. The calculator is Python
   `eval`, where `^` is **XOR**, so it returned **34**. Nex trusted the surprising number and answered
   34. The base used a power/multiply expression and got 1024. *A model that paused on "34 ≠ 32²" would
   have caught it; Nex didn't pause.*
2. **`coding_sum_evens`** — Nex's code `print()`ed the answer instead of assigning `result` (the tool's
   documented contract), got an error, then **guessed 90** (correct: 110). The base followed the
   contract and verified in the sandbox.

Same tools, same prompts, temp 0, both reproduced on a re-run. This is the agentic cost of aggressive
brevity: fewer tokens, fewer self-checks.

## Methodology note (the harness, suspected first)

The first Nex run scored 73% — and was **wrong**. Traces showed the mock `web_search` only matched an
exact key, so the model's reasonable paraphrases ("RTX 5090 specs VRAM GB") returned *"no match"* and it
looped to the step cap; a second task tripped a safety refusal. Those were **harness artifacts, not
model weakness.** Fixed (search now tolerates phrasing like a real engine; tasks ground via named tools;
the safety-trap task reframed), re-ran both models on the identical corrected set. The numbers above are
from the corrected harness. Suspect the harness before the model — every time.

## Worth it?

- **Reach for Nex-N2-mini** if tokens, latency, or serving cost dominate and your tasks are
  well-specified — it does the same tool-driving for a third of the tokens, with tighter tool use and
  zero stalls.
- **Reach for the base** if maximum task completion matters more than token budget — it self-checks
  surprising tool outputs and honors tool contracts that the terse post-train skips.

Adaptive Thinking is a real, large efficiency win. Just not a free one.

---

*Harness: `lib/agentic/native/` in notwitcheer/llm-bench-rig (17 unit tests). Charts:
`reports/agentic-nex-n2-mini.png`, `reports/agentic-tokens.png`.*
