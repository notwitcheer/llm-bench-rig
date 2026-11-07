# arete97: Mont-Saint-Michel voxel duel, Qwen3.8-27B vs Arete-27B (item 97-arete-msm)

Question: does Nous Research's fine-tune of Qwen3.8-27B (Arete-27B) change what the model can build? Same series prompt (`voxelbench/prompts/msm-v2.txt`), same harness (h3), same card (one RTX 5090), both models as Q6_K GGUF, no speculative drafting for either, two interleaved takes each (Q1, A1, Q2, A2), one hour of Hermes Agent `/goal` per take.

Files
- `97-arete-msm.sh`: the queue item (night-lib contract: PLAN phase, `DRYRUN=1`, rule-3 card check, Donald drain with an EXIT-trap restore, resumable per take, exit 3 when out of window).
- `PREREG.md` + `FROZEN.sha256`: the verdict rule, frozen before any take. The hashes cover the series prompt, the in-run and final gates, the driver, the stall detector, the orbit scripts, the summary script, the clip builder and PREREG itself. The hash list names the voxelbench harness files at the paths they have in the voxelbench checkout (`voxelbench/scripts/...`); the stage-box paths in the file are the ones the run used.
- `code/arete97_summary.py`: applies the verdict rule to the four takes and writes `summary.json` / `pair.json`.
- `code/vs-arete97-build.sh`: the two-pane 1920x1080 clip (pre-registered pair).
- `cpu-real-probe.sh`: CPU-only probe of both real GGUFs with the exact server flags (sampling defaults, no drafting context, template render).

Regime (identical for both arms): ctx 131072, q8_0 KV, `--parallel 1`, flash-attn, llama.cpp build 839a572a4, sampling `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`, reasoning budget 16384, thinking on, 60 min wall, 20 goal turns, one shared model alias.

Confounds, stated rather than removed: the two GGUFs come from different quantisers (different imatrix files), and the chat templates differ in small ways (a probe conversation renders byte-identically on both).

Status: ran Sat 2026-11-07 13:00-16:14 CET.

## result

verdict on the pre-registered rule (`code/arete97_summary.py`): **NOISE-LEVEL** (a one-take gap at two takes each is never called "better").

| take | model | in-run + final gate | code | tool calls | wall |
|---|---|---|---|---|---|
| t1 | Qwen3.8-27B Q6_K | PASS | 46.1 KB | 85 | 2,839 s |
| t1 | Arete-27B Q6_K | PASS | 86.3 KB | 104 | 2,785 s |
| t2 | Qwen3.8-27B Q6_K | FAIL (the page crashed Chromium on load, one harness retry, same) | 53.1 KB | 51 | 3,635 s |
| t2 | Arete-27B Q6_K | PASS | 49.0 KB | 68 | 1,878 s |

gate passes: Arete 2 of 2, Qwen 1 of 2. code-size ratio on passing takes 1.47 (under the 1.5 threshold). both served under one model alias, so Hermes Agent gave both the same system prompt.
