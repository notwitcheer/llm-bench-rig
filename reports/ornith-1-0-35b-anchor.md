# Ornith-1.0-35B's self-written scaffold doesn't survive a different harness: 5/12 on real bugs, below its own base's 7/12

**Rig:** one RTX 5090 32GB · llama.cpp b9653 · Qwen3.5-35B-A3B-arch (`qwen35moe`) · Q4_K_M · native OpenAI tool-calling loop (40-step, `docker exec` repo tools) · think-off · temp 0 · official SWE-bench harness
**Subject:** Ornith-1.0-35B (DeepReinforce, MIT) — the RL coder whose headline mechanism is that it *co-optimizes the solution rollouts and a task-specific agent scaffold jointly*, baking the scaffold into the weights. The 35B claims **75.6 SWE-bench Verified** (the family's 82.4 belongs to the unrunnable 397B flagship). DeepReinforce evaluated it in the **OpenHands** harness at temp 1.0, 256K context.

## The question

Ornith's whole pitch is that the orchestration is *in the model* now. If that's real, the model should carry its advantage into a *different* harness — not just the lenient OpenHands setup it was measured in. So: hold the bugs, the harness, the quant, and the thinking mode fixed; change only the model; compare Ornith-35B against the exact base it was post-trained from (Qwen3.5-35B-A3B). This is the rig's standard real-bug anchor — the one that has caught every coding tune so far.

## Result: a regression, and a strict subset

| model (think-off, Q4_K_M, same harness) | resolved | empty patches |
|---|---|---|
| base Qwen3.5-35B-A3B | **7/12** | 3 |
| Ornith-1.0-35B | **5/12** | 4 |

Ornith resolves a **strict subset** of the base: `django-16082`, `flask-5014`, `pytest-6202`, `scikit-learn-14141`, `sympy-22914`. It recovers **nothing** the base missed, and it **loses two bugs the base solved** — `astropy-12907` and `xarray-3677`. This is the same shape as every prior coding tune on the anchor (Qwopus 17 vs 19, Qwable 11 vs 19, AgentWorld 14 vs 16; only pi-tune, trained on real agent traces, improved).

## Why it loses the two: tool-call JSON fragility, not reasoning

The two regressions have a single, concrete cause. On `astropy-12907` and `xarray-3677`, Ornith repeatedly emitted a `bash` tool call whose `arguments` packed a multi-line `python -c "..."` script with **literal unescaped newlines and nested quotes** — invalid JSON. llama-server's strict `--jinja` tool-call parser rejected it with a 500. The base, on the same two bugs, drove the same harness cleanly and fixed both.

This first showed up as something worse: the un-hardened loop *crashed the whole instance* on a 500 (recorded `steps=-1`, empty), which would have unfairly zeroed Ornith on bugs it never really attempted — the rig's classic "low score = harness bug." So the loop was hardened mid-run to catch the 500, feed the parse error back to the model, and let it retry (what a production harness like OpenHands does). The fix is **inert for the base** — the base never triggered a 500, so its banked 7/12 is untouched — and it only gave Ornith the fair recovery. **Even with the fair retry, Ornith still failed both:** it spent its full 40-step budget re-emitting unparseable commands, ballooning context to ~26–29k tokens, and produced no patch. So the loss is genuine, not a harness artifact: under a strict tool-call protocol, Ornith's output format is fragile where its base's is not.

## The harness-as-variable reading

This is the sharpest version of a finding the rig keeps surfacing: **an agentic-coding number is a property of the model *and* the harness around it, not the model alone.** Ornith's 75.6 was measured in OpenHands — a lenient harness (temp 1.0, large context, tolerant tool-call handling) and, per Ornith's own thesis, the kind of scaffold the model was trained to drive. Strip that away to the rig's strict neutral loop and the self-scaffold model is **more fragile than the base it was trained from**. The RL didn't produce a harness-independent capability uplift; it produced a model coupled to a forgiving scaffold's leniency. That is the precise inverse of what "bake the scaffold into the weights" promises — and it's exactly the question the previous harness-as-variable result (omp vs the native loop: scaffold worth a +1/12 give-up lever on a fixed model) set up.

## Honest caveats

- **n = 12, single seed.** The +/−1-2 bug deltas sit inside the noise; the signal is the *direction* (a regression, a strict subset) plus the mechanism (format fragility), not a precise resolve rate.
- **This is not a refutation of the 75.6.** That number is a different harness (OpenHands), temperature (1.0), and context — and, by design, Ornith's own scaffold. This measures whether the self-scaffold training transferred to a *neutral strict* harness. It did not.
- **think-off, to match the base's banked number.** Ornith is an RL reasoning coder; a think-on pass might lift it (and might also help it format tool calls). But the base was also think-off, so the comparison is a fair same-constraint one. A think-on A/B against a think-on base is the separate scoped follow-on (the AgentWorld t081 pattern).
- **The fragility is partly harness-strictness-dependent.** A lenient parser would likely tolerate the multi-line-bash JSON. So read this as: *Ornith-35B needs a forgiving harness to hit its numbers* — which is the deployability finding, not a disqualification.
- **The 397B flagship (the real 82.4) is untested** — 794GB, datacenter-only. This is the runnable 35B against its size-appropriate base.
- **Minor build drift:** base ran on an earlier llama.cpp (~b9562, June 10), Ornith on b9653; same arch family, same quant.

## Verdict

As a sovereign local coder on one 5090, **Ornith-1.0-35B is worse at real bug-fixing than the base it was trained from** when both are driven by the same strict, neutral harness — and the gap is tool-call *robustness*, not reasoning. Its claimed gains assume the lenient OpenHands scaffold it was tuned against; the "scaffold baked into the weights" did not travel. **Worth it only if** you can run it inside a forgiving harness that tolerates loose tool-call formatting; **not worth it** as a drop-in base replacement under a strict tool protocol, where Qwen3.5-35B-A3B resolves more with fewer give-ups. The real anchor caught what the 75.6 headline hides — again.

---

*Generation: `lib/agentic/native/{tools_repo,run_swebench,agent_loop,client}.py` (40-step loop, temp 0, `--reasoning off`) in notwitcheer/llm-bench-rig; this commit also hardens the loop to recover from llama-server 500s (unparseable tool-call JSON / context overflow) instead of crashing the instance. Grading = official `swebench` harness. Base = the 12-id subset of the banked Qwen3.5-35B-A3B 30-bug anchor run (per-instance independent). Per-bug grid: `reports/ornith-anchor.png`.*
