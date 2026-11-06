# PREREG: Astra-Stop replay on Q4_K_M (idle-queue item 90-astra-stop-q4)

Frozen 2026-11-05 (CET), before any counted output. The code files and this file are listed in `FROZEN.sha256`; the item's PLAN phase refuses to run if any of them changed. Smoke runs use the slug `astra90-smoke` and are never counted.

## the claim under test (primary sources, fetched 2026-11-05)

- HF model repo `jarrodwatts/Qwen3.8-27B-Astra-Stop`, created 2026-11-04T06:15Z, apache-2.0, revision 27c08370eb1d95c90229345915c44b068785dc41, files `apply.py`, `edit.json`, `u.npy` (20,608 bytes). https://huggingface.co/jarrodwatts/Qwen3.8-27B-Astra-Stop (README raw read 2026-11-05).
- the edit: `W' = W - alpha * outer(u, u @ W)`, u unit length, alpha = -1, on `model.language_model.layers.63.mlp.down_proj.weight` (89.1M weights, 0.32% of the model, relative change 1.4%), base revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. u has cosine 0.995 with the `</think>` row of the output head (card claim, not re-checked by us).
- card results, "reasoning effort xhigh", original vs edited: pelican on a bicycle, 16 tries each, mean thinking tokens 28,474 vs 17,201 (-40%), median 27,346 vs 17,156 (-37%), tries under 5,000 thinking tokens 0/16 vs 5/16, complete SVG 16/16 vs 16/16; 694 unseen maths/code problems mean thinking tokens 4,011 vs 3,886 (-3%), accuracy 95.2% vs 95.2%. The card itself says 16 pelicans per side is a small sample and the win is on open-ended generation.
- author post: https://x.com/jarrodwatts/status/2117808859691811216 (api.fxtwitter.com read 2026-11-05: 94,987 views, 1,643 likes, 745 bookmarks, posted 2026-11-04 02:39 UTC). The attached image shows one pair, 30,699 vs 2,352 tokens (a single try, not the mean).
- idea origin: https://rikiyat.github.io/blog/weight-space-programming/ (MIT CSAIL, 2026-11-02). Sceptical read: https://honnibal.dev/blog/astra-read-write-misleading. We replay the edit's effect, not its discovery.
- base model: https://huggingface.co/Qwen/Qwen3.8-27B, weights unchanged since 2026-08-13 (newest commit 2026-11-05 is a model-card colour fix).

## what we run

- both arms from the SAME base safetensors (rev 1d4bf0f). Edited arm = the author's `apply.py` unmodified (sha256 of the written shard 44ad2e65ea6f5eabf6399f6ddcb18fd156cb65a8a08030bec81df9f0807689af, equal to the card's published hash; original shard beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8 equals `edit.json`).
- identical recipe per arm: `convert_hf_to_gguf.py --outtype bf16 --no-nextn` then `llama-quantize ... Q4_K_M 12`, llama.cpp commit fac3a894f (build 12075). bf16 intermediates deleted after quantising. Q6_K was not built (disk floor).
- the edit survives quantisation, checked on the files: the norm of u^T W for blk.63.ffn_down.weight is 1.5664 unedited and 3.1312 edited (ratio 1.999, the edit doubles the push as designed). The only tensor that differs between the two GGUFs is that one (850 of 851 tensors byte-identical). Per-element the Q6_K-quantised tensor changed in 61% of elements, 2.4% relative norm.
- quantised files (sha256 recorded 2026-11-05):
  - unedited `base-Q4_K_M.gguf` 16,547,400,256 bytes, sha256 `9297d44ba5960f893ed122bb8aaf20b2138eec40613b0d4a1fc99cfc6beffe05`
  - edited `edited-Q4_K_M.gguf` 16,547,400,256 bytes, sha256 `c4c7acdc716ea09c4c4ed448855ec7a7d75030ac368349c9bc61f0d21d56e8e3`
- server: `llama-server` from the same llama.cpp commit, `-ngl 99 -fa on --jinja -c 262144 -np 4 (65,536 per slot) -ctk q8_0 -ctv q8_0 -cram 0 --reasoning-format deepseek`, own port 8265, never :8090. Donald drained and restored only through the night-lib helpers.
- prompt (user turn only, no system prompt): `Generate an SVG of a pelican riding a bicycle`. UNVERIFIED: the card does not print the author's exact wording.
- reasoning effort: the Qwen3.8 chat template (rev 1d4bf0f) takes `reasoning_effort` in {xhigh (default), medium, low}; xhigh injects the system line "Reasoning effort is set to xhigh. Please think carefully ..." and thinking is on by default. We pass `chat_template_kwargs {enable_thinking: true, reasoning_effort: xhigh}` explicitly; a template smoke before every arm checks that xhigh renders a longer prompt than low and that `reasoning_content` comes back. That is the honest mapping of the author's "xhigh" (the card names the setting, not the API call: UNVERIFIED that they used the same field).
- sampler (ours; the card states none, UNVERIFIED): temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repeat penalty 1 (Qwen's own thinking-mode recommendation, model card). Seeds 1 to 32, the same list for both arms (sampling on the GPU is not bit-reproducible across arms, the seeds only fix the sampler stream). `max_tokens` 60,000 per try; a try that ends on the cap counts at its length and is flagged capped, never dropped.
- n = 32 tries per arm, four slots per server, four half-legs run in this order: seeds 1-16 unedited, 1-16 edited, 17-32 unedited, 17-32 edited (a cut block leaves a balanced partial).
- reported only (never gates a verdict): 24 fixed MATH-500 problems (random.Random(20261105) sample of 500 row ids, string-match on the boxed answer, cap 30,000 tokens) per arm, run after the second pelican half-leg of each arm when at least 25 min remain.

## metrics and rule

- headline: thinking tokens per try = the streamed `reasoning_content`, re-tokenised with the model's own tokenizer through the server's `/tokenize`. Two-sided Mann-Whitney U, edited vs unedited, alpha 0.05; medians and means with a 10,000-draw bootstrap 95% CI of the ratio.
- SVG complete = `finish_reason == stop` AND the last `<svg>...</svg>` in the answer closes, parses as XML (lxml, strict) and renders (resvg) to an image with at least 4 colours and at least 3% non-white pixels. Fisher exact on completeness is reported.
- verdict, applied by `astra_summary.py` in this order, nothing worded by hand:
  1. BROKEN: the edited arm did not load or failed the template smoke, OR its SVG completeness is more than 25 points under the unedited arm.
  2. INVALID: the unedited arm did not load or failed the smoke.
  3. REVERSED: p < 0.05 and the edited median is higher.
  4. REPRODUCED: p < 0.05 and the edited median thinking tokens are down at least 25% (the card claims -37% median).
  5. PARTIAL: p < 0.05, edited median lower, but down less than 25%.
  6. NO DIFFERENCE: otherwise. This is a valid result and is worded as "no difference detected on a quantised pair at n = 32 per arm"; it does not say the edit fails in bf16.
  7. INCOMPLETE: fewer than 32 tries on an arm; exit 3, resumes next block, no summary.json.
- line 1 comes from `LINE1` in the summary output (counts, medians, p), not from prose.

## limits to print with any number

Q4_K_M pair, the author tested bf16; our sampler and prompt wording may differ from theirs; one prompt; n = 32; thinking tokens counted with our tokenizer; the arms run one after the other on one card (a video that shows them side by side says so, and its playback speed is shared and is not a speed claim); tool-using or "an LLM read the weights" wording is never used.

## render

CPU only on capsule (resvg + PIL + the static ffmpeg), 1920x1080 16:9: five paired pelicans (seeds 1-5, first in order, never picked) drawn side by side with a 150 px thinking-token counter per side, a 4x4 gallery per arm (seeds 1-16), a result card from `summary.json`. Mock data is stamped MOCK on every frame.
