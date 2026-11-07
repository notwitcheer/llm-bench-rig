# item 93 pre-registration: Basalt v0.3.0 against llama.cpp on the identical Qwen3.8-Flash-Next Q2_0 weights, one RTX 5090 + 59 GB RAM

frozen 2026-11-06 (Europe/Paris), before any counted output. GPU use so far: staging smokes only (slug `-smoke`, never counted), listed in the addendum.

## question

Same quantiser, same weights (ISTA-DASLab GSQ-RCO Q2_0 of Qwen3.8-Flash-Next): does the Basalt v0.3.0 engine decode faster than mainline llama.cpp on this box, with no drafting on either side, and by how much? Then the same for each engine's own drafting, and the same Hermes Agent task typed by each engine in separate recorded takes.
Tested claim (a community guide, r/LocalLLaMA 1xomy0i, score 55 at fetch 2026-11-06, RTX 5080 16 GB + 64 GB RAM, other quant): "about 3.5x llama.cpp's decode without drafting and ~5x with it". Our box is a 32 GB card with 59 GB RAM (the project tests on 128 GB), so this is a replay of the shape, not of the numbers. Our own earlier number (post 2026-09-09, llama.cpp, UD-Q2_K_XL 79.6 GB, n-cpu-moe 22): 59.7 tok/s; it is a different file and is shown as context only, never as a ratio denominator.

## arms (all frozen)

- files: the GGUF pair `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf` (37,623,740,192 + 28,800,138,432 bytes, ISTA-DASLab) and the Basalt pack `Qwen3.8-FN-GSQ-RCO-Q2_0.basalt` (70,396,904,412 bytes, jesdga; embeds the same quantiser's experts plus an MTP layer the GGUF does not carry).
- llama.cpp: mainline master at 18ab62a (built 2026-11-06, CUDA arch 120, gcc-14, nvcc 12.8), `-ngl 99 --n-cpu-moe N -fa on --jinja -c 65536 -ctk q8_0 -ctv q8_0 --parallel 1`. N = the best of the ladder 12, 13, 14, 16 (three short requests each: conv, essay, lru; 300 tokens; seed 5; token-weighted decode tok/s; ties go to the larger N). The ladder is run once, before the llama rows.
- Basalt: v0.3.0 (1bc0ba79 + docs commits; built from the v0.3.0 tag, CUDA 13.3), `gpu 0, kv int8, max-context 65536, resident-experts, vram-reserve-mib 700`, otherwise engine defaults.
- arms: `llama_plain` (no draft), `llama_ngram` (same N, `--spec-type ngram-mod`: this GGUF has no MTP layer, so n-gram is the only drafter mainline can use; stated on the card), `basalt_spec0` (`spec 0`), `basalt_default` (the pack's own MTP drafter, window 8).

## rows (one harness for both engines: oai_bench.py, streamed OpenAI requests, the engine's own `timings`)

- request fields pinned on every row: temperature as stated, top_p 0.95, top_k 20, min_p 0.05 (sampled rows), seed, thinking off, max_tokens as stated (the same fields reach both engines).
- decode: conv ("Explain how a B-tree differs from an LSM tree...", temp 0.6), essay ("Write an essay on the Odyssey. About 500 words.", temp 0.6), lru ("Write a Python class implementing an LRU cache...", temp 0), 3 repetitions each (seeds 101-103), max 700 tokens, thinking off. Arm decode tok/s = sum(predicted_n) / sum(predicted_n / predicted_per_second) over the clean runs (token-weighted). A run is clean when no error, timings present, predicted_n >= 100.
- prefill (plain arms only: llama_plain, basalt_spec0): one cold ~16K-token prompt and one cold ~48K-token prompt (the first 67,000 and 197,000 characters of the Hermes docs text, llms-full of 2026-11-06, byte-identical for both engines; a random session line first; 4.12 characters per token measured on this text), task "In one sentence, what is this document?", max 16 tokens, temp 0; tok/s = timings.prompt_per_second, prompt_n reported.
- `basalt-bench --community` (project preset community-1, int8 KV, engine defaults) is run once; if it refuses (SSD spill on 59 GB RAM) the plain `basalt-bench -t conv,struct,prose -p 16384` row replaces it and the card says so. Smoke 2026-11-06: it printed (see addendum).
- Hermes takes: the item-89 task (fix-shopcart, 4 planted bugs, 5 tests, tests/ untouched), real Hermes Agent sessions, own HERMES_HOME, thinking on, extra_body temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0.05 / seed 721, wall cap 10 min, 2 takes per engine, NO drafting on either side (basalt_spec0 server, llama_plain server).

## ratios and verdict words (applied as written by b93_summary.py; `--selftest` proves the branches)

- R0 = basalt_spec0 decode / llama_plain decode. THE headline, reported first.
- R1 = basalt_default decode / max(llama_plain, llama_ngram) decode (each engine's own best drafting).
- verdict on R0: BASALT AHEAD (R0 >= 1.25) | LEVEL (0.80 <= R0 < 1.25) | LLAMA.CPP AHEAD (R0 < 0.80) | BROKEN (fewer than 6 of 9 clean decode runs on llama_plain or basalt_spec0) | INCOMPLETE (fewer than 2 clean Hermes takes on an engine; provisional, exit 3, resumes).
- line 1 is built by the rule in b93_summary.py (one sentence per verdict word, numbers read from the arms). A loss or a tie is a valid result and is framed as "one rig, one file": LLAMA.CPP AHEAD reads "llama.cpp decodes X, Basalt Y". Hermes Agent is never the subject of a failure; if a take fails, the subject is the engine or the model.
- A Hermes take PASSES when CLEAN (ended done or wall, >= 3 tool calls, request errors <= 20%) and the pristine tests go 5/5 with tests/ untouched and the agent never named the harness. Passes are reported, they do not gate the verdict.

## reported, never gated

prefill tok/s at 16K and 48K, draft acceptance, VRAM peak, GPU watts, time to done and tool calls of each take, the community row (or its fallback), the guide's 3.5x / 5x beside ours, our 59.7 as context.

## clip plan

~8 s result card first (title = the verdict in words, one column per engine with its GB, big numbers with a grey "how it was measured" line under each row, footer = the claim tested and the limits), then the two Hermes takes side by side at real speed (left llama.cpp, right Basalt, start synced on each take's Enter, server-token counters from each engine's own stream), cut 6 s after the faster pane is done. No path, hostname or account on screen (leak mask). A timelapse badge is not a speed claim.

## limits

59 GB RAM (half of the project's test box), one card, one file; n=2 takes per engine; ngram on llama.cpp is not the same drafter as Basalt's MTP; thinking-on Hermes sessions sample, so the take order is not a speed test; llama.cpp's N is tuned on three short requests, Basalt is run at its defaults; the pack and GGUF share a quantiser, not bytes (Basalt re-packs the experts and adds an MTP layer).

## frozen hashes

FROZEN.sha256 (same folder) lists the sha256 of this file, b93_summary.py (verdict rule + line 1), b93_gate.py (pass test), oai_bench.py (rows harness), prompts/fix-shopcart.txt and every file of seed/. The item refuses to run when any differs.
