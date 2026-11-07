# GLB-3D-Coder 4B vs Qwen3.8-27B and Ornith-1.5-35B: 12 absurd prompts, script-checked (PRE-REGISTRATION)

written and frozen 2026-11-05 (CET evening) BEFORE any model output exists for this item. `FROZEN.sha256` holds the sha256 of this file, `prompts.json`, `system.txt` and `config.json`; the item's DRYRUN re-checks it and refuses to run if any of the four changed. a deliberate change = a new version of this file with the reason written at the bottom, never an edit in place.

## question

a 4B model fine-tuned to write one self-contained Three.js page that builds a 3D asset and exports a GLB (ccore/Qwen3.5-4B-GLB-3D-Coder, final release 2026-11-05, apache-2.0): on 12 absurd prompts it has not seen, how often does its page work, compared with two much larger general local models we already hold, under the SAME system prompt?

## sources (primary, read 2026-11-05)

- model card, prompt format, system prompt, sampler: https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder/raw/main/README.md (hub api sha 57dced645abb, lastModified 2026-11-05T16:25Z, 12,206 downloads, 167 likes at fetch). the card's chat template is the Qwen3.5 one (`chat_template.jinja`, thinking on by default); the card says its recorded evaluation used the Q8_0 file with thinking enabled.
- card sampler: temperature 0.8, top_k 40, top_p 0.95, min_p 0.0, repeat penalty 1.0. we use the card sampler for ALL three models (one regime; a label, not an equal: the two larger models were not tuned for it).
- card system prompt: stored verbatim in `system.txt`.
- the card compares the 4B to DeepSeek V4.1 Flash 1017 on its own showcase prompts. we do NOT repeat that comparison; this item compares only what we run ourselves.
- r/LocalLLaMA thread 1xlh0zw (interest only, no number used).

## models (all llama.cpp, build 11777 / commit 839a572a4, the llama.cpp-donald binary read-only, own port 8196, never :8090)

| key | file | bytes |
|---|---|---|
| glb4b-q8 | Qwen3.5-4B-GLB-3D-Coder-Q8_0.gguf (hub sha256 68a66e25...) | 4,482,402,848 |
| qwen38-27b-q4km | Qwen3.8-27B-UD-Q4_K_M.gguf | 16,464,440,224 |
| ornith15-35b-q4km | Ornith-1.5-35B-Q4_K_M.gguf | 21,713,462,848 |

the 27B is the UD-Q4_K_M file we hold (the task allowed Q4_K_M or Q6_K; Q4 chosen so the three legs fit the card with 4 parallel slots). server: `--jinja --parallel 4 --ctx-size 139264 (34,816 per slot) --flash-attn on --n-gpu-layers 99`, f16 KV, each model's own embedded chat template, thinking: the 4B at its template default (on, as the card's evaluation); the two comparators with thinking OFF (`chat_template_kwargs {enable_thinking: false}`), see the v2 changelog line. a label, not an equal: each model is asked the way it is normally used for a one-shot page. `max_tokens` 32,768 per answer. a cell that ends on the token cap (finish_reason length) is scored as it stands (its extracted HTML, if any, is checked; a missing page is a fail). the cap is part of the regime and is said in any post.

## prompts and seeds

12 prompts in `prompts.json` (absurd by design, plus the card's own lighthouse-keeper prompt replayed on purpose, one prompt that asks for animation). seeds: 20261105 and 20261106. cell = (model, prompt, seed): 24 per model, 72 in all. llama.cpp seeds the sampler per request; with 4 parallel slots the batching can still change the numerics, so a seed is a label, not a replay guarantee.

## checks (script only, no aesthetic scoring, nothing is judged by eye in the verdict)

every cell's extracted HTML (first fenced html block, else the first `<!doctype`/`<html` to the end) is loaded in headless chromium (playwright 1.62, chromium 1234, software GL, viewport 960x540) from a fake origin, network replay cached per url (config.json `cdn_note`). per cell:

1. `html`: a page could be extracted.
2. `load_ok`: the page loads and runs 4 s with no uncaught page error and no console `error` (except the generic "Failed to load resource"; failed requests are logged, not counted).
3. `nonempty`: the screenshot after 4 s has grayscale std >= 5.0 AND >= 2% of pixels differ by > 24 grey levels from the median colour of four 16x16 corner patches AND the page has a canvas with a webgl context.
4. `glb_valid`: clicking the Export GLB control (config.json `export_click`) fires a download within 20 s and the file passes `glb_validate` (magic glTF, version 2, header length == file size, JSON chunk parses, asset.version 2.0, >= 1 mesh with >= 1 primitive, POSITION accessor count > 0, every bufferView inside the BIN chunk, POSITION min/max finite with an extent > 0 on at least 2 axes).
5. `page_valid` = html AND load_ok AND nonempty.  `full_pass` = page_valid AND glb_valid.

## verdict rule (applied by `score.py`, branches proven by `selftest`)

- **BROKEN** (declared invalid, no post, vault note only): the 4B server does not come up, or the 4B has 0 `page_valid` cells once all its cells are scored. a harness fault (a checker crash on more than 20% of a model's cells) is also INVALID, never a model result.
- **PROVISIONAL** while any model has fewer than 24 scored cells (the item exits 3 and resumes).
- otherwise per comparator X in {27B, 35B}: d = full_pass(4B) - full_pass(X) over the 24 cells. d >= 3 = **4B AHEAD**, d <= -3 = **4B BEHIND**, |d| < 3 = **LEVEL** (3 cells = 12.5 points, an action threshold, not significance).
- overall label: AHEAD OF BOTH / BEHIND BOTH / LEVEL WITH BOTH / MIXED (any other combination). all of it is descriptive: 24 cells per model, one author's prompts.
- reported next to it, no verdict weight: per-check counts, median and total completion tokens, truncations, aggregate tokens per second per leg, per-prompt matrix.

## line-1 variants (filled from the data by `score.py`, nothing worded by hand)

`on 12 absurd prompts a 4B model's Three.js page loaded, drew a scene and exported a valid GLB in <a>/24 tries; Qwen3.8-27B <b>/24, Ornith-1.5-35B <c>/24 (same system prompt, script checks only).` the AHEAD/BEHIND/LEVEL label is appended, never a quality claim. BROKEN: no post.

## picture (planned here, CPU only on capsule after the card is released)

for each prompt: a 1920x1080 16:9 contact PNG and a 4 s rotating turntable mp4, the three models side by side, each labelled with its size and its completion-token count. the turntable is the model's own EXPORTED GLB loaded in one fixed shared viewer (same camera, same lights, same fit-to-frame for all), so the picture shows the asset the button produced, not the page chrome; a failed cell shows its own page screenshot (or a text card) with the failure reason. cell shown per (model, prompt) = the lowest seed with a valid GLB, else the lowest seed with a screenshot. subject moving from frame 1; prompts are cut in the order of most models with a GLB first. final card = the tally. a one-sheet contact of all 36 shown cells.

## honest limits (to be repeated in any post)

24 cells per model on one author's absurd prompts; script checks say "ran and exported", not "good"; all three models get the card's sampler and system prompt (tuned for the 4B only); 4B thinking on, comparators thinking off, 32,768-token cap (a long answer can be cut off, and the cut-off counts as a fail; a comparator with thinking on is NOT measured here); the 27B and 35B are Q4_K_M files, the 4B is Q8_0; software-GL rendering, so animations run slow in the page check (only the export is judged); no Hermes involved; the card's dataset release is still pending, so the card may change.

## frozen files (sha256, written before any model output)

- `prompts.json` 9d884890900b2d0872c207f066feaf8455bd902e553a2ee904761b9b02a5e12a
- `system.txt` 8b6290edbf45d093a41ae0e620ef81cf8700d3cac45ecb762b2837dd0dca727b
- `config.json` 7aa62ae7a1c8e90e3636b2bf4404206a1e182f6164b56711efdda30fd14c8409 (v3)
- this file's own hash is in `FROZEN.sha256` (a file cannot list its own hash); the item's DRYRUN runs `sha256sum -c FROZEN.sha256` over all four.

## changelog

v1 2026-11-05: written before any output.
v2 2026-11-05 (still before any scored output): a 6-cell plumbing smoke on 2 prompts with thinking on for all three (cells are never data, not scored) showed both comparators using the full 20,480-token cap inside their reasoning and returning no page at all (27B 2/2, 35B 1/2 with an empty page), which would have turned the item into a token-cap test. the comparators now run with thinking off; the 4B keeps the card's thinking-on regime. the smoke log is kept as `smoke-v1.log` in the item folder.
v3 2026-11-05 (still before any scored output): token cap raised from 20,480 to 32,768 and ctx per slot from 22,528 to 34,816. a 4-cell thinking-off smoke showed Ornith-1.5-35B writing its working-out into the page body and hitting 20,480 on 2/2 cells, so a 20k cap would have measured the cap, not the model.
