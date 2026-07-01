# Thinking mode isn't hiding Qwen-AgentWorld's transfer either: think-ON is bug-for-bug identical to think-OFF

**Rig:** one RTX 5090 32GB (sm_120) · Qwen-AgentWorld-35B-A3B (UD-Q4_K_M) vs base Qwen3.5-35B-A3B (Q4_K_M) · llama.cpp b9653, `--reasoning on`, native tool-calling, temp 0 · same bounded 12-bug SWE-bench Verified subset used for the t082 Ornith harness study, official swebench grader
**Question:** [The original AgentWorld finding](agentworld-lwm-transfer.md) ran think-OFF to match the base's banked protocol and explicitly scoped out think-ON as "the real falsification" — AgentWorld is designed to reason about environment transitions, so if the claimed transfer needs reasoning to surface, forcing thinking on should reveal it. Does it?

## The numbers

| resolved / 12 | think-OFF (banked) | think-ON |
|---|---|---|
| **Qwen3.5-35B-A3B** (base) | 7 | 7 |
| **Qwen-AgentWorld-35B-A3B** (LWM-warmed) | 8 | 8 |

Same count both ways for both models. But the count hides the real result — look at the per-bug outcomes.

## Finding 1 — AgentWorld is bug-for-bug identical, not just count-identical

Every one of the 12 bugs gets the exact same classification (resolved / wrong patch / gave up empty) whether thinking is on or off. The 8 resolved are the same 8. The 3 give-ups are the same 3. The 1 wrong-patch is the same 1. Twelve-for-twelve, zero movement. Thinking mode is not suppressing some latent capability here — there is nothing to unlock. Mean steps did tick up slightly with reasoning on (34.4 vs the native loop's typical ~33/40) but it changed nothing about which bugs got fixed.

## Finding 2 — the base moves by exactly one wash

The base isn't perfectly static: `astropy-12907` flips resolved→give-up and `matplotlib-23314` flips give-up→resolved when reasoning is turned on (`seaborn-3187` also shifts from wrong-patch to give-up, without changing the resolve count). Net zero, and with n=12 this is noise-band movement, not a signal — but it's worth naming precisely rather than waving at "flat."

## What this closes

The original report explicitly left the door open: *"AgentWorld is designed to reason about environment transitions and recommends thinking mode... a think-ON A/B is the real falsification leg."* This is that leg, and it closes clean: **turning reasoning on does not change AgentWorld's real-bug performance at all.** The −2/30 (or, on this smaller 12-bug slice, the +1/12 edge — see caveats) isn't an artifact of an unfair think-OFF protocol suppressing the LWM warm-up's benefit. The model's behavior on these bugs simply doesn't respond to the think toggle, full stop.

## Honest caveats

- **n=12, not the original 30.** This reuses the bounded subset already banked for the t082 Ornith harness-as-variable study, chosen for direct comparability and to fit inside a session's time budget. Treat the 8/12 and 7/12 counts as noisy on their own — the load-bearing result is the **per-bug identity** for AgentWorld (12/12 unchanged classifications), which is a stronger, count-independent signal than a resolve-rate delta.
- **This 12-bug subset flatters AgentWorld relative to the full 30** (8-7 here vs 14-16 over on the full anchor) — expected sampling variance on a 12-bug slice of a 30-bug set, not a contradiction. The 30-bug think-OFF number remains the authoritative anchor result; this run's job was only to test the think toggle, not to re-litigate the resolve rate.
- **Engineering note, in case it saves someone the same debugging:** there is no literal `--think` flag on this llama.cpp build (b9653) — the switch is `--reasoning on|off|auto` (`--reasoning-format` controls how `<think>` gets extracted into `message.reasoning_content`). Smoke-tested before the real run: `reasoning_content` separates cleanly from `content`, tool-calls still parse correctly alongside it, and round-tripping the full assistant message (content + reasoning_content) back into multi-turn history doesn't break the template or need any harness code changes. Zero-code-change wiring, confirmed empirically rather than assumed.

## Verdict

Reasoning was the last plausible explanation for why the LWM warm-up's claimed +3.4–12.8% transfer doesn't show up on real bugs. It isn't the explanation — the model performs identically bug-for-bug with thinking on or off. Between the original think-OFF result and this think-ON check, the claim has now been tested on the two axes that mattered (synthetic vs real, thinking off vs on) and shows no transfer on any of them for real coding.

---

*Generation: native loop = `lib/agentic/native/{tools_repo,run_swebench,agent_loop,client}.py` (40-step, temp 0), server flags `--jinja -c 32768 --reasoning on`, Q4_K_M / UD-Q4_K_M, b9653. Grading = official `swebench` harness. Banked reports: `results/swebench/{agentworld-35b-a3b-think,qwen3-5-35b-base-think}.report.json` (think-ON); `results/swebench/{agentworld-35b-a3b,qwen3-5-35b-base}.report.json` (think-OFF, filtered to the 12-bug subset for this comparison). Chart (`reports/agentworld-think-on-null.png`): left = the think-OFF→think-ON slopegraph (AgentWorld flat, base flat); right = the per-bug grid, the base's one swap highlighted. Companion: `reports/agentworld-lwm-transfer.md` (the original think-OFF finding this closes).*
