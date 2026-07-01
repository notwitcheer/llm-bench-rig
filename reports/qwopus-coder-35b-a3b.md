# Qwopus-3.6-35B-A3B-Coder-MTP: the coder curriculum doesn't break the regression pattern, and a lenient harness only closes half the gap

**Rig:** one RTX 5090 32GB (sm_120) · Qwopus-3.6-35B-A3B-Coder-MTP (Jackrong, Q4_K_M) vs its base Qwen3.6-35B-A3B (Q4_K_M) and Ornith-1.0-35B (banked) · llama.cpp b9653, `--reasoning off` (forced — this template defaults to thinking-ON), temp 0 · same bounded 12-bug SWE-bench Verified subset used for the t082 Ornith harness study · two harnesses: the rig's native strict tool-calling loop, and Hermes Agent (Nous Research) as a lenient alternative to omp
**Question:** the model card claims Qwopus-Coder "goes toe-to-toe with the new Ornith 35B MoE across a huge eval suite... edging it on the coding trajectories" and reports 62.4% on 300 SWE-bench cases, think-disabled. Two prior Qwopus tunes at 27B synthetically scored well but regressed on the rig's real SWE-bench anchor — does the coder curriculum break that pattern at 35B-A3B, and does the model actually beat Ornith under a matched, controlled protocol?

## The numbers

| resolved / 12 | native-strict | Hermes-as-harness |
|---|---|---|
| **base** (Qwen3.6-35B-A3B) | 7 | 8 |
| **Qwopus-3.6-35B-A3B-Coder-MTP** | 5 | 7 |
| **Ornith-1.0-35B** (banked, different base family) | 5 | — |

## Finding 1 — under matched protocol, Qwopus ties Ornith, it doesn't edge it

The vendor's own comparison pits Qwopus think-disabled against Ornith **think-enabled** — a protocol mismatch this rig's own t080 finding flags as invalid (a think-mode confound, not a fair A/B). Forcing both to the same think-off protocol on the same 12 bugs: Qwopus **5/12**, Ornith **5/12** (banked, native-strict). Identical count. Not "edging it" — a tie, on a *different set* of bugs (2 unique each way: Qwopus alone gets `astropy-12907` and `xarray-3677`; Ornith alone gets `django-16082` and `pytest-6202`). The marketing claim doesn't survive a matched protocol.

An engineering note that mattered here: unlike the Qwen3.5-based base/Ornith pair (t081), this Qwen3.6-based coder tune's chat template **defaults to thinking-ON** with no flags at all — confirmed by smoke test (`chat template, thinking = 1` in the server log) before a single bug was run. Getting the protocol match required an explicit `--reasoning off`, and a prior banked "base" number for this exact model turned out to have been generated thinking-ON (caught via the server log for that historical run, `/tmp/sweb_qwen3-6-35b-base.log`) — reran it think-off before drawing any conclusion.

## Finding 2 — Qwopus is a strict subset of its true base, under *either* harness

The clean comparison isn't against Ornith or against a different Qwen3.5-based family — it's against Qwopus's own base, Qwen3.6-35B-A3B, same architecture, same everything except the coder-curriculum SFT:

- **native-strict:** base 7/12, Qwopus 5/12 — Qwopus resolves *nothing* the base doesn't (a strict subset).
- **Hermes-as-harness:** base 8/12, Qwopus 7/12 — same pattern, still a strict subset (Qwopus's 7 are all inside the base's 8; the base's lone extra is `pytest-6202`).

This is the third instance of the pattern this rig keeps finding in Qwopus-branded coder tunes (after two at 27B): a coder-curriculum fine-tune that improves nothing on the real anchor and gives some of it back, regardless of harness leniency.

## Finding 3 — the harness narrows the gap, but doesn't erase it (unlike Ornith)

Built a Hermes Agent (Nous Research) harness leg this session — same role as omp played for Ornith's t082 Leg B (a lenient third-party agent driving the same Docker-per-instance SWE-bench setup, `--network host` to the local llama-server), but Hermes instead of omp, since Hermes is the harness worth standardizing on here. Getting it running required real plumbing: Hermes ships as an editable pip install (venv + full source tree + its own uv-managed Python build, ~1.3GB), not a single static binary like omp, so each instance's ephemeral container needs the whole thing `docker cp`'d in (confirmed cheap: ~4-5s per container).

Both models gain under Hermes and lose nothing:

- base: native-strict 7 → Hermes 8 (+1, gains `django-16082`)
- Qwopus: native-strict 5 → Hermes 7 (+2, gains `django-16082` and `matplotlib-23314`)

Both harness bumps are strict supersets — real recoveries, not reshuffles, and both cost the model nothing. `matplotlib-23314` is now the second bug this rig has seen recovered under *any* lenient harness (Ornith recovered it under omp too) — a candidate for a general "hard for strict tool-call parsers, easy once the harness tolerates loose formatting" bug, not model-specific.

But the deficit doesn't close. Qwopus gains one more bug than the base (+2 vs +1), narrowing the gap from -2 to -1 — it does **not** flip to parity or invert, the way Ornith's entire -2 deficit vanished under omp. Two structurally similar 2x2s, two different mechanisms: Ornith's regression was ~100% harness artifact (tool-call JSON format fragility); Qwopus's is roughly half harness-sensitive, half real.

## Finding 4 — the MTP drafter works, modestly

The embedded MTP head is real and functions standalone (no separate draft checkpoint needed, matching how Donald's own Qwen3.6-27B-MTP is served): **1.24-1.27x decode speedup** (baseline 284-285 tok/s → 352-362 tok/s across two workloads) at **75-79% draft acceptance**. That's short of pi-tune's preserved 2.0-2.4x at 27B, in the range of the other 27B coder tunes whose drafters degraded from the vanilla baseline's 1.8-2.2x. No same-family base-MTP checkpoint was available to run a controlled preserved-vs-degraded comparison at 35B-A3B specifically — this number stands on its own, not as a delta.

## Honest caveats

- **n=12, single seed**, the bounded subset shared with the t082 Ornith study, chosen for direct comparability and to fit inside a session's budget.
- **Hermes-as-harness is a first run of new infra**, built this session — its leniency profile (prompt scaffold, `terminal`+`file` toolsets only, 90-turn cap by default here bounded to 40, 600s wall-clock) is not identical to omp's, so "Hermes vs omp" isn't a controlled comparison; each is its own harness-as-variable instrument.
- **Qwopus vs Ornith is a same-count tie on different bugs**, not a resolved-set match — treat "ties Ornith" as the honest headline, not "beats" or "loses to."
- **MTP acceptance has no same-family baseline** at 35B-A3B (no vanilla base-MTP checkpoint tested) — reported as a standalone measurement, contextualized against the 27B pattern from memory, not a fresh controlled delta.

## Verdict

The coder curriculum that produced Qwopus-3.6-35B-A3B-Coder-MTP does not break the pattern this rig has now measured three times: a Qwopus-branded coding tune that gives back real-bug resolve rate relative to its own base, even after controlling for the one confound (harness leniency) that fully explained a structurally similar regression in Ornith. The MTP drafter is genuine but modest. Worth it if you specifically want the marginal persistence gains a lenient harness like Hermes buys any model in this family; not worth it as a coder-tuned upgrade over the plain base, which resolves more bugs than its derivative under every protocol tested here.

---

*Generation: native loop = `lib/agentic/native/{tools_repo,run_swebench,agent_loop,client}.py` (40-step, temp 0), server flags `--jinja -c 32768 --reasoning off`, Q4_K_M/Q5_K_M, b9653. Hermes loop = Hermes Agent (`hermes chat -q ... --provider custom -t terminal,file --ignore-rules --yolo -Q --max-turns 40`), run per-instance inside the SWE-bench Docker image (`--network host`), 600s wall-clock cap, same server config. Grading = official `swebench` harness. Banked reports: `results/swebench/{qwopus-coder-35b-a3b,qwen3-6-35b-base-think-off,qwopus-coder-35b-a3b-hermes,qwen3-6-35b-base-hermes}.report.json`. MTP measurement: `--spec-type draft-mtp --spec-draft-n-max 2`, self-speculative (embedded drafter, no separate checkpoint), fib/lru workload prompts (n_predict=256, temp 0). Chart (`reports/qwopus-coder-35b-a3b.png`): left = the native-strict→Hermes slopegraph (base and Qwopus both rise, gap narrows, doesn't close); right = the per-bug 4-column grid. Companion: `reports/ornith-1-0-35b-harness-inversion.md` (the structurally similar 2x2 where the harness fully explained the deficit, for contrast).*
