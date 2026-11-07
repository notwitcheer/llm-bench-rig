# PREREG: item 97-arete-msm (frozen before any counted take)

question: does Nous Research's fine-tune of Qwen3.8-27B (Arete-27B) change what the model can BUILD? Same series prompt, same harness, same card, base vs fine-tune, both Q6_K.

## why a fresh base-model leg

The older Qwen3.8-27B Mont-Saint-Michel rows ran harness h2 with MTP drafting. Arete has no MTP head and today's harness is h3, so a comparison against those rows would measure the harness and the engine, not the fine-tune (voxelbench rule: compare runs only on the same harness version). Both models therefore run now, on the same h3 harness, both without drafting. The old h2 rows stay as context only.

## regime (identical for both models; quoted from the plan)

prompt `prompts/msm-v2.txt` (md5 30a4b777e510) plus the exact-path line, harness h3 (VX_TMPFIX=1 VX_GATE_STRICT=1 VX_STALL_NUDGE=1, in-run gate quick_gate_msm.py), 60 min wall, 20 goal turns, ctx 131072, q8_0 KV, --parallel 1, flash-attn, llama.cpp-donald 839a572a4, no --spec flags, RB 16384 plus the series budget message, sampling `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0` (both GGUFs carry exactly temp 1.0 / top-p 0.95 / top-k 20 in general.sampling.*), thinking on (series default). Final gate `gate.py --video-seconds 20`, orbit clip per passing take (orbit_pick.py).

takes: 2 per model, interleaved Q1, A1, Q2, A2.

files: Qwen3.8-27B-Q6_K.gguf (sha256 562fbf760503...) and NousResearch_Arete-27B-Q6_K.gguf (sha256 5b4588c250b9...). Different cutters and imatrix files: a labelled confound, not removable.

## verdict rule (verbatim from the plan)

> per take = gate PASS/FAIL + code KB (page minus inlined libs, L26) + 8-element read. per model = passes of 2. LEVEL if both models pass the same number of takes and code KB is within 1.5x; else name the model ahead and by what. a 1-take difference with n=2 is reported as noise-level, never "better".

### operational definitions (fix the wording of the rule above, add no criterion; applied by `arete97_summary.py`)

- gate PASS = the series `gate.py` five checks (no console errors, no page errors, live WebGL context, frame not blank, frames differ), unchanged. A take with no html is a FAIL.
- code KB = bytes of the page's lines of at most 50,000 characters, divided by 1000 (an inlined library sits on one very long line).
- a model's code KB = the mean over its PASSING takes. The ratio is the larger mean over the smaller.
- both models 0 of 2: LEVEL, there is no page to compare.
- equal passes (above 0) and ratio at most 1.5: LEVEL. Equal passes and ratio above 1.5: the model with more code is named "ahead on code size only"; size is not quality, the 8-element read decides what is said.
- passes differ by 1: reported as NOISE-LEVEL (n=2), never "better". Passes differ by 2: that model is AHEAD (2 of 2 vs 0 of 2).
- the 8-element read is a manual look (gate shot at 6 s plus an orbit contact sheet) and is reported next to the verdict; it does not change the label above.
- per take also recorded, not part of the label: API calls, tool calls (sum of tool results by name), wall, server tokens, goal turns used, VRAM peak.

## clip (pre-registered pair, verbatim from the plan)

> take 1 of each; if a take 1 fails the gate, that model's first passing take, labelled.

Landscape two-pane clip, 1920x1080, from each model's orbit clip, labels "Qwen3.8-27B Q6_K on my RTX 5090" | "Arete-27B Q6_K (Nous Research) on my RTX 5090", code KB and the same settings line on both panes.

## as built (the staged item, fixed before any take)

- one server rung only (131072): a smaller fallback context would make the two arms differ, so a failed load at 131072 stops the item instead.
- both models are served under the same alias (`Qwen3.8-27B-duel-Q6_K.gguf`) so Hermes builds the identical system prompt for both: it selects some prompt blocks by model-name substring, and an alias containing the maker's name would give one model blocks the other lacks.
- slugs `msm2-h3-t<take>-qwen38-27b-q6k` and `msm2-h3-t<take>-arete27b-q6k`; a take that died mid-run is retried under `t<take>b`, `t<take>c`.
- no take starts after Sat 2026-11-07 22:00 CET and GPU work ends by 23:40 CET (the next queued item is time-gated to Sun 00:00); a take that cannot finish exits 3 and resumes in the next block.
