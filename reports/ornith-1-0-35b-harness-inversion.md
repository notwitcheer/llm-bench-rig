# Leg A's Ornith "regression" was the harness: swap it and the gap to the base vanishes

**Rig:** one RTX 5090 32GB · llama.cpp b9653 · Qwen3.5-35B-A3B-arch (`qwen35moe`) · Q4_K_M · think-off · same 12 SWE-bench Verified bugs · official SWE-bench grader
**Subject:** Ornith-1.0-35B (DeepReinforce, MIT) and its base Qwen3.5-35B-A3B, run through **two** agent harnesses each: the rig's strict native tool-calling loop, and omp (a third-party CLI coding agent) as a lenient harness. A full model x harness 2x2.

## A correction first

A few days ago I posted the Leg A result: under the rig's strict native loop, Ornith-1.0-35B resolved **5/12** real bugs against its base's **7/12** — a strict subset, two bugs lost to tool-call JSON fragility, and I called it "worse at real bug-fixing than the base it was trained from." That conclusion was true. It was also a property of **one harness**. Leg B holds the model fixed and changes the loop around it, and the picture changes.

## The 2x2

Same 12 bugs, same Q4_K_M quant, same think-off, same llama.cpp build (b9653) for every cell. Only the model and the harness vary.

| resolved / 12 | strict native loop | lenient omp harness |
|---|---|---|
| **base** Qwen3.5-35B-A3B | **7** | **7** |
| **Ornith-1.0-35B** | **5** | **8** |

Under the strict loop the base leads by two. Under omp that deficit is gone: Leg A's regression does not survive a harness swap.

## What actually moved

Hold Ornith fixed and change only the harness, strict -> omp: it goes **5 -> 8**, a strict superset. The exact two bugs Leg A pinned on format fragility — `astropy-12907` and `xarray-3677`, where Ornith emitted multi-line bash with unescaped newlines and the strict `--jinja` parser 500'd it into a give-up — both come back. So does `matplotlib-23314`, which Ornith gave up on under the strict loop. No losses. The fragility Leg A measured was real, and it was the strict harness punishing it.

Now the honest part, because the counts flatter Ornith by one. Under omp the two models resolve an **identical 7-bug core**: `astropy, django, matplotlib, flask, xarray, scikit-learn, sympy`. Ornith's nominal eighth is `pytest-6202` — and the base did not lose pytest to a worse fix, it lost it to omp's **450s wall-clock cap**: its trajectory was still going at the timeout (rc=124, a 170-line patch in flight), whereas it solved pytest cleanly in the step-budgeted native loop. So the robust reading of the omp column is **parity**, not an Ornith win. The "ranking inversion" the slopegraph draws is real on the official grades (8 vs 7), but the one-bug margin is a budget casualty for the base, not a capability edge.

Either way the load-bearing claim holds: **the strict-harness deficit (-2) closes to a wash under a lenient harness.**

## The mechanism: one model is harness-robust, the other is harness-coupled

Look at the two rows, not the four numbers.

- **The base is harness-robust.** 7 -> 7. It drives clean tool calls under either loop; omp's persistence buys it `matplotlib` (a strict-loop give-up), the wall-clock cap costs it `pytest`, and it nets flat. Strict or lenient, it resolves the same kind of set.
- **Ornith is harness-coupled.** 5 -> 8, every delta a recovery. Its RL-tuned output emits tool-call JSON that a strict parser rejects and a forgiving harness tolerates. Its capability on these bugs is real, but it is only *expressed* under a harness that absorbs its formatting.

That is the precise shape of Ornith's own thesis — "bake the agent scaffold into the weights" — seen from the measurement side. The training did not produce a harness-independent uplift over the base. It produced a model whose real-bug performance is **conditional on a forgiving harness**: give it one and it matches its base, take it away and it falls two behind. The orchestration didn't travel; the *dependence on* an orchestration did.

## This is what the 75.6 sits on

DeepReinforce measured Ornith-35B's 75.6 SWE-bench Verified in OpenHands — temp 1.0, 256K context, the model's own scaffold, and a tolerant tool-call path. That is the lenient corner of this 2x2. The number is not fake; it is harness-specific. The rig now shows the two things the headline hides: the score does not transfer to a strict neutral loop (5/12, two format give-ups), and even in a lenient harness Ornith only reaches its base's level on these twelve, it does not clear it. An agentic-coding number is a property of the model *and* the harness, strongly enough that a two-bug "regression" and a two-bug "parity" are the same model measured two ways.

## Honest caveats

- **n = 12, single seed.** Every delta here is 1-3 bugs, inside the noise band. The signal is the *direction and mechanism* (a strict-harness deficit that a lenient harness erases, driven by tool-call format tolerance), not a resolve rate.
- **omp "leniency" is a bundle, not a knob.** Swapping the harness changes tool-call parsing tolerance *and* the stopping rule (450s wall-clock vs 40 native steps) *and* the prompt scaffold at once. The base's lone omp loss (`pytest`, to the wall-clock cap) is the stopping-rule half of that bundle, not the parsing half — which is exactly why I read the omp column as parity rather than an Ornith win.
- **This does not reproduce or refute the 75.6.** Different harness again (omp, not OpenHands), different temperature (the runs here are think-off, temp-0-ish via the native protocol), no 256K context, not Ornith's own scaffold. It measures whether Leg A's strict-loop regression was harness-dependent. It was.
- **think-off throughout**, matched across all four cells. Ornith is an RL reasoning coder; a think-on pass is a separate axis (the t081 pattern).
- **No build drift this time.** Unlike Leg A (base on b9562, Ornith on b9653), all four cells here ran on b9653. The 2x2 is internally clean.

## Verdict

Leg A's "Ornith is worse than its base at real bugs" was a true statement about a strict tool-calling harness, not about the model. Swap to a forgiving harness and the two-bug deficit vanishes: Ornith recovers exactly the bugs the strict parser cost it and lands level with its base on the same seven-bug core. The RL didn't make Ornith a better bug-fixer than its base; it made it one that **needs a lenient harness to show parity** — which is the harness its 75.6 was measured in. **Worth running** inside an OpenHands-like stack that tolerates loose tool-call formatting, where it performs like its base. **Not worth it** as a drop-in base replacement under a strict tool protocol, where it resolves two fewer and the base gives up nothing. The model is the same either way; what you measure is the harness you measure it in.

---

*Generation: native loop = `lib/agentic/native/{tools_repo,run_swebench,agent_loop,client}.py` (40-step, temp 0, `--reasoning off`); omp = omp v16.1.14, headless `omp -p --auto-approve` inside each SWE-bench container (`--network host` to the local llama-server), 450s/bug; both models served identically (`--jinja -c 32768 --reasoning off`, Q4_K_M, b9653). Grading = official `swebench` harness. Banked reports: `results/swebench/{ornith-35b,qwen3-5-35b-base,ornith-35b-omp,qwen3-5-35b-base-omp}.report.json`. Chart (`reports/ornith-legb-inversion.png`): left = the strict->lenient slopegraph; right = the per-bug 2x2 grid, Ornith's three recoveries highlighted. Companion: `reports/ornith-1-0-35b-anchor.md` (Leg A, the strict-harness chapter).*
