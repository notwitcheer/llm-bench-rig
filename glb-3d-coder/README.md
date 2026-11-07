# glb-3d-coder: 4B GLB-3D-Coder vs two larger local models (ran 2026-11-06/07)

capsule queue item `91-glb-3d-coder.sh` plus its harness. 12 absurd prompts x 2 seeds, three models, same system prompt, script checks only (page loads, scene not empty, Export GLB returns a parseable GLB), then a CPU turntable render. Pre-registration and verdict rule: `PREREG.md` (hashes in `FROZEN.sha256`). Vendored Three.js r170 and the network cache are not copied here (the item fetches them on capsule).

## result (4B + 27B legs Fri 2026-11-06 05:11-05:50 CET, 35B leg Sat 2026-11-07 01:07-01:17 CET, one RTX 5090)

verdict on the pre-registered rule (`score.py`): **MIXED**, level with the 27B (9 vs 10, inside the +-3 band), ahead of the 35B (9 vs 1). harness faults 0 on all three models.

| model | file | working GLB (of 24) | median tokens per try |
|---|---|---|---|
| GLB-3D-Coder 4B | Q8_0 | 9 | 19,601 |
| Qwen3.8-27B | UD-Q4_K_M | 10 | 10,178 |
| Ornith-1.5-35B | Q4_K_M | 1 | 9,029 |

"working" = the page loads, draws a non-empty scene and its Export GLB button downloads a parseable GLB (script checks, no aesthetics). no model passed lighthouse-keeper (the card's own prompt), teapot-tower, snail-bakery or robot-grandma.

caveats: the 4B ran thinking on (its template default), the two comparators thinking off (with it on they spent the whole budget thinking); one sampler and system prompt for all three, the card's, tuned for the 4B; two seeds per prompt. per-cell numbers: `run/aggregate.json`, `run/readout.md`.
