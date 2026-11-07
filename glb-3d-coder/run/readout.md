# GLB-3D-Coder readout (script output, not edited)

overall: **MIXED** (qwen38-27b-q4km: LEVEL (d=-1); ornith15-35b-q4km: 4B AHEAD (d=+8))

line 1: on 12 absurd prompts a 4B model's Three.js page loaded, drew a scene and exported a valid GLB in 9/24 tries; Qwen3.8-27B 10/24, Ornith-1.5-35B 1/24 (same system prompt, script checks only).

| model | scored | html | load_ok | nonempty | page_valid | glb_valid | full_pass | trunc | tok median | tok total |
|---|---|---|---|---|---|---|---|---|---|---|
| GLB-3D-Coder 4B Q8_0 | 24/24 | 20 | 12 | 17 | 12 | 9 | **9** | 4 | 19601 | 482505 |
| Qwen3.8-27B UD-Q4_K_M | 24/24 | 24 | 17 | 20 | 17 | 10 | **10** | 0 | 10178 | 249307 |
| Ornith-1.5-35B Q4_K_M | 24/24 | 24 | 12 | 10 | 8 | 1 | **1** | 2 | 9029 | 313498 |

## per prompt (full_pass per seed; 1 pass, 0 fail, . not run)

| prompt | 4B | 27B | 35B |
|---|---|---|---|
| lighthouse-keeper | 00 | 00 | 00 |
| teapot-tower | 00 | 00 | 00 |
| pancake-ship | 01 | 01 | 00 |
| toaster-knight | 11 | 11 | 00 |
| snail-bakery | 00 | 00 | 00 |
| duck-submarine | 10 | 11 | 00 |
| haunted-vending | 10 | 10 | 00 |
| cactus-sombrero | 00 | 10 | 00 |
| robot-grandma | 00 | 00 | 00 |
| whale-island | 01 | 01 | 00 |
| cone-dragon | 10 | 11 | 00 |
| spatula-windmill | 11 | 00 | 10 |

## top failure reasons
- GLB-3D-Coder 4B Q8_0: page error x8; no page extracted x4; no download: Timeout 20000ms exceeded while waiting for event "download" x3
- Qwen3.8-27B UD-Q4_K_M: page error x7; no download: Timeout 20000ms exceeded while waiting for event "download" x6; glb invalid: too short x1
- Ornith-1.5-35B Q4_K_M: page error x12; no download: Timeout 20000ms exceeded while waiting for event "download" x5; empty scene x4; glb invalid: bad magic x1; glb invalid: too short x1
