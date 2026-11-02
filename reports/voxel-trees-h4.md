# A local model turns the 2009 "Self-organizing tree models" paper into a growing 3D tree, and checks the paper's lambda claim itself

**The one line:** Given one prompt and a 60-minute wall in Hermes Agent goal mode, Qwen3.8-27B Q6_K running on one RTX 5090 wrote a single-page three.js implementation of Palubicki et al., SIGGRAPH 2009, in both of two takes, and both pass the pre-registered 11-check final gate. In the better-looking take (take b, seed 1, 30 seasons), lambda 0.46 grows a tree 70.3 wide and 71.8 tall (width/height 0.9798, 6,496 metamers); lambda 0.56 grows one 51.7 wide and 136.0 tall (ratio 0.3805, 4,753 metamers): higher lambda, taller and narrower, the direction the paper reports. **This is not a blind reproduction.** The prompt states the paper's lambda claim, tells the model to check it on its own simulation before finishing, and the in-run quality gate checks it. n = 2 takes of one model.

**Rig:** RTX 5090 32GB (capsule), llama.cpp, one model at a time · **Window:** 2026-11-02, cheap-electricity block, take a 01:10 to 02:08 CET, take b 02:08 to 02:39 CET · **Harness:** [llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig) `voxelbench/` (harness h4) · **Paper:** [Self-organizing tree models for image synthesis](https://www.algorithmicbotany.org/papers/selforg.sig2009.html), Palubicki, Horel, Longay, Runions, Lane, Mech, Prusinkiewicz, SIGGRAPH 2009

Clip: 36 s, lambda 0.46 left and lambda 0.56 right, same camera scale in both halves (cut from take b, see [the clip](#the-clip)). Not stored in this repo.

## TL;DR

1. **Both takes pass the final gate, 11 of 11 checks.** No console or page errors, WebGL context, not blank, animated, main thread answers, `treeStats()` hook present, metamers grow with seasons, non-trivial trees at both lambdas, lambda effect in the paper's direction, deterministic (same seed and lambda give the same tree).
2. **The lambda effect holds on all three seeds in both takes.** Seed 1 is the gating one; seeds 2 and 3 are reported, not gating, and are the check that the model did not tune to seed 1. 3 of 3 seeds hold in both takes.
3. **The numbers (seed 1, 30 seasons):**

| take | lambda | crown width | height | width / height | live metamers |
|---|---|---:|---:|---:|---:|
| b | 0.46 | 70.3 | 71.8 | 0.9798 | 6,496 |
| b | 0.56 | 51.7 | 136.0 | 0.3805 | 4,753 |
| a | 0.46 | 63.0 | 78.8 | 0.7998 | 5,779 |
| a | 0.56 | 51.6 | 139.8 | 0.369 | 4,429 |

   Units are internodes, crown width is the larger of the x extent and the z extent of live metamer end points.
4. **No runaway growth.** 6,496 and 5,779 live metamers at lambda 0.46 against 4,515 in an independent reference simulation (1.4x and 1.3x); the runaway flag (above 25,000) is false in both. The previous passing page (71-r2, harness h2) grew 140,486 metamers.
5. **Before this, 0 of 3 passed on harness h3** (Qwen3.8-27B, Ornith 1.5 35B, Gemma 4 31B QAT). The causes and the fixes are below, including the part that was our spec, not the models.

## What the model was given

- **Model and server:** Qwen3.8-27B Q6_K, llama.cpp build 11777 (commit `839a572a4`), context 131,072, q8_0 KV cache, MTP draft (`--spec-type draft-mtp --spec-draft-n-max 2`), sampling from the maker's card (temperature 1.0, top-p 0.95, top-k 20, min-p 0.05), thinking capped at 16,384 tokens per request. Row in [`voxelbench/model-params.md`](../voxelbench/model-params.md).
- **Agent:** Hermes Agent in `/goal` mode, which keeps a standing objective across turns until a judge says it is met or the turn budget runs out ([goals docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/goals), fetched 2026-11-02). 20 goal turns, 60-minute wall, and a quality gate registered before the first turn: a shell command that must exit 0 before the goal can complete.
- **Prompt:** [`voxelbench/prompts/trees-v2.txt`](../voxelbench/prompts/trees-v2.txt), 7,456 bytes (md5 prefix `a9e08c32`). One page, no source files, three.js from a CDN, ten numbered requirements: metamers and buds, shadow propagation on a 128 x 160 x 128 voxel grid, the extended Borchert-Honda resource split with apical control lambda, shoot growth, shedding, pipe-model girth, foliage and HUD, seasons 1 to 30, and a fixed set of test hooks (`?seed=&lambda=&seasons=&autoplay=&orbit=&instant=`, `window.setSeason(n)`, `window.treeStats()`). The prompt gives starting values for a, b, qmax, alpha, the split formula and the shedding threshold. The paper does not give a, b or C.
- **What the prompt tells the model about the claim:** lambda above 0.5 favours the main axis (tall, narrow crown), below 0.5 favours the laterals (wide crown). Its last paragraph asks the model to load seed 1 at 30 seasons with lambda 0.46 and 0.56, compare crownWidth / height from `treeStats()`, require 0.46 to give the larger ratio, keep every load under 25,000 metamers, check other seeds, and never special-case lambda values or seeds.

## The harness (h4)

- **In-run gate** [`voxelbench/scripts/quick_gate_trees.py`](../voxelbench/scripts/quick_gate_trees.py): the series' five base checks, the `treeStats()` hook, growth, then two loads at 30 seasons (lambda 0.46 and 0.56). `lambda_stable`: neither load crashes or freezes, and each holds 100 to 25,000 live metamers. `lambda_effect`: ratio at 0.46 above ratio at 0.56 (seed 1). Failures come back to the model in plain words (a crash hint names the grid edge). It also closes the autoplaying default page before the instant loads, because a 644 KB page at 0 to 1 fps under software rendering starved a 5-season load past 150 s. Tested on CPU (SwiftShader) before the runs: a stub page passes; a stub with a wrong-direction lambda fails `lambda_effect`; the h3 Qwen page fails `lambda_stable` (crash at 0.56); the h3 Ornith page fails `lambda_effect`; the h2 Qwen page fails `lambda_stable` (140,486 and 467,146 metamers).
- **Final gate** [`voxelbench/final-gate/gate_trees.py`](../voxelbench/final-gate/gate_trees.py), pre-registered 2026-10-26 before any run of this prompt family, **unchanged** by h4. It lives outside the agent's working tree. Its other check names are scanned for in every path the model can read before a run starts; the lambda criterion is visible by design in h4.
- [`voxelbench/HARDENING.md`](../voxelbench/HARDENING.md) rows L40 and L41 for trees (the queue item headers say L39-L40: they were written before the rows were renumbered to sit after the msm row L39).

## Results

### Take b (`trees-h4b-qwen38-27b-q6k`)

- Final gate PASS, 11 of 11. 24,617-byte `index.html`, 60 fps, longest frame gap 177 ms.
- Lambda test: see the table above. Metamers at seasons 5 / 15 / 30: 37 / 924 / 6,496. Determinism: 6,496 = 6,496.
- Extra seeds: seed 2 ratio 1.1295 vs 0.3984, seed 3 0.8408 vs 0.3621.
- Run: ended `done` after 1,668 s (27.8 min), goal done at 2 of 20 turns, no stall nudges, 53 API calls, 95,375 output tokens, 2 tool errors flagged.

### Take a (`trees-h4a-qwen38-27b-q6k`)

- Final gate PASS, 11 of 11. 22,435-byte `index.html`, 60 fps, longest frame gap 417 ms.
- Metamers at seasons 5 / 15 / 30: 39 / 739 / 5,779. Determinism: 5,779 = 5,779.
- Extra seeds: seed 2 ratio 0.8056 vs 0.3813, seed 3 0.9594 vs 0.3383.
- Run: ended `done` after 3,351 s (55.9 min), goal done at 2 of 20 turns, no stall nudges, 127 API calls, 122,665 output tokens, 9 tool errors flagged.

Neither take wrote a path-miss file (the earlier path-retype failure did not recur).

### The clip

The two clips cut straight after the take b run used the page's own default camera (a fixed orbit camera at radius 95 around a point 16 units up; take a's camera was not checked). That camera fits a 72-tall tree and cuts the top off the 136-tall one, so the two clips could not be put side by side as they were. The recut drives the unmodified page from outside: it loads `?seed=1&lambda=X&seasons=30&autoplay=0&orbit=0` at 960 x 1080, sets the page's own camera object to the same values for both lambdas (look-at point 66 units up, distance 230, elevation 0.46 rad, 50 degree vertical field of view, azimuth 0.7 rad), steps `window.setSeason(n)` for n = 2 to 30 on an absolute 0.8 s schedule, then turns the page's own auto-orbit on for 9 s (about 62 degrees, same rate in both). The two renders are stacked left and right with a small lambda label per half; the page's HUD (season, metamers, buds, height, crown width, shed, lambda, seed) stays in. Scripts: [`voxelbench/scripts/trees-fixedcam-clip.py`](../voxelbench/scripts/trees-fixedcam-clip.py), [`voxelbench/scripts/trees-fixedcam-compose.py`](../voxelbench/scripts/trees-fixedcam-compose.py). The numbers in the clip's HUD match the gate: 6,496 / 71.8 / 70.3 and 4,753 / 136.0 / 51.7 at season 30.

## Why h3 failed, and what we fixed

Harness h3 (2026-10-29 to 2026-11-01) ran the same task with prompt trees-v1 on three models and none passed the final gate:

| model | failure | cause |
|---|---|---|
| Qwen3.8-27B Q6_K | page crashed at lambda 0.56 | our spec: no rule for a bud outside the 128 x 160 x 128 shadow grid |
| Ornith 1.5 35B-A3B Q4_K_M | lambda effect reversed (ratio 1.148 at 0.46, 1.586 at 0.56) | our spec: the V tie-break, below |
| Gemma 4 31B QAT | page never loaded (invalid URL error, no WebGL canvas, no stats hook) | the model |

A CPU probe of the two in-range pages (Qwen, Ornith), and then an independent reference simulation written from the paper, found two spec gaps in trees-v1:

1. **The V tie-break.** The prompt said a bud's optimal direction V points to the neighbouring voxel with the lowest shadow value in its perception cone, with no rule for ties. In open space every voxel in the cone ties. Both models (and the reference sim run the same way) took the first tied voxel in loop order, which for an upright bud is offset (-2, +2, -2): every shoot is pulled the same way, the whole crown leans, and because crown width is an extent, a taller lambda 0.56 tree leans further and reads wider. With that scan-order rule the reference sim got the lambda direction right on 0 of 20 seeds; with a neutral tie rule (V is the normalised sum of the unit vectors to all tied voxels, the bud's own direction if that sums to zero) it gets it right on 20 of 20. trees-v2 states the neutral rule and says why.
2. **The grid edge.** A bud outside the grid got zero shadow and cast none, so a crown past x or z = plus or minus 64 had outer buds with full light and nothing shading them: lambda 0.56 seed 2 went from 97.8 wide at season 25 to 124.3 at season 30 with 570,312 metamers, and seed 1 crashed the tab. trees-v2 now says: voxel index = round(coordinate), cells centred on integers, and a bud outside the grid casts no shadow and gets Q = 0.

The cap on new metamers per season (n at most 6) was checked and rejected as a cause. Alpha stays 3 (alpha 4 grows the lambda 0.56 tree to about 162 internodes, past the 160-cell grid height).

The h3 in-run gate also only ever loaded lambda 0.46, so the model never saw the 0.56 blow-up. h4 loads both.

## What this does and does not show

- **Shows:** a local 27B model, in one hour or less of Hermes Agent goal mode, produced a tree simulation that a pre-registered gate accepts, with the paper's lambda direction on three seeds and no runaway growth, twice in a row.
- **Does not show a blind reproduction.** The model was told the paper's claim, was told to verify it, and had a gate that checked it. The final gate's lambda check is the same one the in-run gate runs for seed 1. Seeds 2 and 3 are the evidence the effect comes from the model's mechanism and not from seed 1, not a proof.
- **The prompt carries the fix.** trees-v2 contains the tie rule and the grid-edge rule that the reference-sim check found. A fair reading is "the paper's method, specified well enough, is reproducible by a local 27B model", not "the model found the method's subtleties alone".
- **Two lambdas only.** The gate compares 0.46 with 0.56. No lambda sweep, and the reference sim shows the lambda-to-height curve is not monotone around 0.5, so nothing here claims one.
- **n = 2 takes of one model.** h3 versus h4 is not a clean comparison: prompt and harness both changed after h3. The model's tree ratio at 0.46 (0.98 in take b) is wider than the reference sim's seed-1 value (0.58); the gate checks direction, not magnitude.
- **A tree-shaped result, not a faithful figure.** Crown geometry, foliage and girth are the model's own; nothing here compares them with the paper's renders.
- **Reference simulation:** an independent numpy implementation written from the paper for this check; it informed the prompt, and it is not a copy of the paper's code.
- **Harness oddity, no effect on either verdict:** a second copy of take a's page was found under a hyphen-stripped slug directory (same md5 as the canonical page), and take b has the same stray directory, empty. The gate read the canonical path in both. Cause not established.

## Reproduce

```
# in-run gate (needs Chromium via playwright; SwiftShader works)
python voxelbench/scripts/quick_gate_trees.py <path to index.html>
# fixed-camera clip, one lambda at a time, then the side-by-side
python voxelbench/scripts/trees-fixedcam-clip.py <index.html> out/l0.46 --lambda 0.46 --orbit-seconds 9
python voxelbench/scripts/trees-fixedcam-clip.py <index.html> out/l0.56 --lambda 0.56 --orbit-seconds 9
python voxelbench/scripts/trees-fixedcam-compose.py out out/trees-lambda-fixedcam.mp4 --crf 27
```

The final gate is run from outside the agent's working tree by the queue items [`72fa`](../voxelbench/queue/72fa-voxel-trees-h4-qwen.sh) and [`72fb`](../voxelbench/queue/72fb-voxel-trees-h4-qwen.sh).
