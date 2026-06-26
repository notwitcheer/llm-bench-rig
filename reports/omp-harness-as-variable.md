# Swap the agent harness, keep the model: one bug in twelve moves — and it moves on *persistence*, not reasoning

**Rig:** one RTX 5090 32GB · llama.cpp · Qwen3.6-27B-Q6_K @ `:8090` · think-off · temp 0
**Question:** how much of an agentic-coding score is the *model* and how much is the *scaffold* wrapped around it? Hold the model fixed, change only the harness, grade on real bugs.

## Setup

The same local model (Qwen3.6-27B-Q6_K, served by one llama-server on `:8090`) drove **two different agent harnesses** against the **same 12 SWE-bench Verified bugs**, graded by the **official SWE-bench harness** (apply the patch, run FAIL_TO_PASS + PASS_TO_PASS):

- **rig-native loop** — our own OpenAI tool-calling agent (`read` / `list` / `search` / `edit` / `run-bash` via `docker exec`), budget **40 tool-calling steps** per bug.
- **omp v16.1.14** — a dependency-free third-party CLI coding agent, run headless (`omp -p --auto-approve`) inside each bug's container, pointed at the *same* `:8090` endpoint via its `models.yml`. Budget **450s wall-clock** per bug.

Same weights, same context window, same decode settings, same bugs. The only thing that changes is the loop around the model: its prompt strategy, its tool wiring, and — by construction — its stopping rule.

## Result: a strict superset, +1

| harness | resolved | the delta |
|---|---|---|
| rig-native (40-step) | **8/12** | — |
| omp v16.1.14 (450s) | **9/12** | +`sphinx-8621` |

omp resolves a **strict superset**: every bug the native loop fixed, plus one — `sphinx-doc__sphinx-8621`. Nothing is traded away. And both harnesses **miss the same hard bugs**.

## The per-bug grid is the whole story

| bug | rig-native | omp |
|---|---|---|
| astropy-12907, django-16082, matplotlib-23314, flask-5014, xarray-3677, pytest-6202, scikit-learn-14141, sympy-22914 | ✅ resolved | ✅ resolved |
| **sphinx-8621** | **gave up (empty patch)** | **✅ resolved** |
| pylint-7080 | gave up (empty patch) | wrong patch |
| requests-1921 | wrong patch | wrong patch |
| seaborn-3187 | gave up (empty patch) | gave up (empty patch) |

Eight bugs the model resolves no matter which harness drives it. Three hard bugs neither harness resolves under either scaffold. One bug in the middle flips — and *how* it flips is the finding.

## The mechanism: the scaffold buys persistence, not intelligence

Look at the **miss types**, not the counts. On the four hard bugs, the native loop produced **3 empty patches** (it explored, then terminated without committing any edit) and 1 wrong patch. omp produced **1 empty patch** and 3 attempts:

- `sphinx-8621`: native **gave up** → omp committed a patch that **passed**. The resolve.
- `pylint-7080`: native **gave up** → omp committed a patch that **failed**. Same persistence, unlucky bug.
- `requests-1921`: both committed wrong patches.
- `seaborn-3187`: both gave up.

The scaffold didn't make the model smarter — both harnesses hit the *same* ceiling on the *same* bugs (`seaborn`, `requests`, `pylint` resolve under neither). What omp's loop changed is the **give-up rate**: it keeps the model working long enough to commit a patch where the native loop quits empty-handed. On these four bugs, empty patches fell from **3 → 1**. One of those converted give-ups happened to land a passing fix. That is the entire +1/12.

This is the rig's recurring tell, seen from the other side: across the reality-anchor work, **empty-patch rate is the give-up fingerprint** that separates models. Here it separates *harnesses* on a fixed model — and the lever it pulls is persistence-under-ambiguity, the exact axis a synthetic tool-fluency score never measures.

## Honest caveats

- **n = 12, single seed.** +1/12 is inside the noise band. The claim is *not* "omp is 12% better" — it's the **direction plus the mechanism**: same ceiling, fewer give-ups, a strict superset. A single-seed +1 is a signpost, not a measurement.
- **The budgets differ by construction (40 steps vs 450s).** That is not a confound to apologize for — it *is* the scaffold variable. "Harness" means prompt strategy + tool wiring + stopping policy, bundled; the stopping policy is exactly where the persistence delta lives. Equalizing them would be measuring a different, dis-assembled thing.
- **The native baseline is the 12-bug subset of the rig's 30-bug run** (Qwen3.6-27B, 19/30). SWE-bench grades each instance independently in its own container, so the subset is identical to having run those 12 alone.
- **Bounded scout A/B.** Base model, 12 bugs, one scaffold pair. The full version — a coding-tuned model, 30 bugs, a third harness (Hermes) — is the >3h follow-on, parked under the rig's task-time cap.

## Verdict

On a fixed local model, the agent harness is a **secondary lever**: it moved one bug in twelve, and it did so by quitting less, not by reasoning better. **Worth swapping in** a more persistent scaffold if your model already clears the easy bugs and your losses are give-ups (empty patches) rather than wrong fixes — that's the regime omp helps. **Not worth it** if you're hoping the harness will crack bugs the model fundamentally can't: both loops died on the same three. The model is the ceiling; the scaffold decides how often you stop short of it.

---

*Generation: rig-native = `lib/agentic/native/{tools_repo,run_swebench}.py` (40-step loop, temp 0) in notwitcheer/llm-bench-rig; omp = omp v16.1.14, headless `omp -p` in each container, same `:8090` model via `models.yml`, 450s/bug. Grading = official `swebench` harness. Both report JSONs and the per-bug grid: `reports/omp-harness-as-variable.png`.*
