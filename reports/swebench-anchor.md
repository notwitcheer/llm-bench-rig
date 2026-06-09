# Reality anchor: the synthetic Agentic Score predicts real SWE-bench coding (r = 0.96)

**Rig:** one RTX 5090 32GB · llama.cpp b9562 · native OpenAI tool-calling (`--jinja`) · temp 0
**Setup:** 4 models drove our native tool-calling loop with **real repo tools** (read / list / search /
edit / run-bash, executing via `docker exec` inside each instance's container) against **12 SWE-bench
Verified bugs** — true agentic exploration (the model navigates the repo itself; it never sees the hidden
tests). Patches graded by the **official SWE-bench harness** (apply → run FAIL_TO_PASS + PASS_TO_PASS).

## The result

| model | synthetic Agentic Score | real resolve | avg steps | empty patches |
|---|---|---|---|---|
| Qwen3.6-27B | 98.6 (#1) | **8/12** (67%) | 33 | 3 |
| Qwopus-GLM-18B | 97.1 (#3) | **8/12** (67%) | 36 | 3 |
| Kimi-Linear-48B | 92.9 (#5) | 5/12 (42%) | 39 | 3 (1 a serving error) |
| Granite-4.1-30b | 92.0 (#6) | 3/12 (25%) | 16 | 4 |

**The synthetic score predicts real coding: Pearson r = 0.96.** The 40-task synthetic Agentic Score — built
from deterministic mock-tool tasks — orders these four models the same way real SWE-bench Verified bugs do.
The leaderboard isn't just measuring a synthetic artifact.

## The two headline findings survived contact with reality

**Small models punch up — confirmed on real code.** Qwopus-GLM-18B, an 18B community merge, **resolved 8/12,
tying Qwen3.6-27B** (and beating the 48B and both 30B-class models). Its synthetic punch-up wasn't an artifact
of easy mock tasks; it fixes real bugs at the level of a model 50% larger.

**Granite's brittleness predicted real failure — confirmed.** Granite-4.1-30b, the synthetic "lean but
brittle" model (2/6 on error-recovery, leanest tokens), resolved the **fewest — 3/12**. The telemetry shows
*why*, and it's the same fingerprint: it used the **fewest steps (16 avg)** and produced the **most empty
patches (4 — it gave up)**. The terse model that couldn't recover from a tool error in the synthetic suite
is the terse model that bails on a hard real bug.

**Kimi's verbosity showed up too.** It used the **most steps (39, near the 40 cap)** — matching its low
synthetic tool-efficiency — and lost one instance to a real weakness the synthetic suite hinted at: it
emitted a giant regex as a tool argument that was **malformed JSON**, which llama-server's tool parser
rejected (a 500). A model that can't reliably format tool calls will stall in any real agent loop.

## Honest caveats

- **n = 12, directional.** These are the 12 smallest-patch Verified instances (one per repo) — easier than
  full Verified, so absolute resolve rates run high. The anchor checks the **shape of the correlation across
  models**, not a precise resolve rate. A larger slice would tighten it.
- **A harness fix this forced:** one model's serving 500 originally voided its whole run; `solve_instance`
  now guards per-instance (records the failure, keeps any partial diff) — so a single bad request can't
  destroy a model's score.

## Verdict

A synthetic, deterministic, 5-hour-to-build agentic benchmark on one 5090 ranks local models the same way
real SWE-bench Verified does (r = 0.96). The Agentic Score is a **valid, cheap proxy** for real agentic
coding ability — and the specific findings it produced (small models punch up; lean ≠ robust) hold on real
bugs.

---

*Harness: `lib/agentic/native/{tools_repo,run_swebench}.py` in notwitcheer/llm-bench-rig. Generation = our
agent loop in Docker; grading = official `swebench` harness. Chart: `reports/swebench-anchor.png`.*
