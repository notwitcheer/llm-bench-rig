# voxel series: maker parameters per model (identify BEFORE the run)

rule (operator 2026-10-22): the "right parameters" are identified before a model's first voxel run, never after a weak result.
how: `scripts/card-params.py <hf repo> [--gguf <file>]` (reads the card + its base_model card + the GGUF's general.sampling keys), then one row here with the source url and fetch date. day0-voxel.sh refuses without `PARAMS_SRC`, and refuses when the GGUF carries no sampling keys and no `SAMPLING_FLAGS` are given. queue items copy the same row.
what counts: sampling (temp / top-p / top-k / min-p / penalties), output or reasoning budget (our default RB 16384 is Qwen-tuned; a card budget wins), reasoning effort/strength knobs and their default, chat-template notes (preserve reasoning_content, stop tokens). when the card names no min-p, keep llama.cpp's 0.05 (parity with the Qwen reference, whose GGUF carries no min-p) unless the card or a stated reason says otherwise.

## Qwen3.8-27B Q6_K (series reference, Donald)
- source: GGUF general.sampling keys (temp 1.0, top_p 0.95, top_k 20), read 2026-10-22 from ~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf; card not re-read for this table
- flags: none needed (keys apply as llama.cpp defaults); min-p 0.05 (llama.cpp default); RB 16384 + budget message
- runs: msm-v1 66w, msm-v2 66x (h1 harness); h2 harness: 66zz1 + 67e1 + 67e2 (msm-v2 repeats, HARDENING L30), 66zz2 (msm-v3), 67f (msm-v3b)

## Muse Glimmer 30B Q4_K_M
- source: https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF README, fetched 2026-10-22: "--temp 1.0 --top-p 0.95 --top-k 64", "Sampling defaults. temperature 1.0, top_p 0.95, top_k 64"; reasoning cannot be switched off, strength via `reasoning_strength` (low / medium / high / xhigh, default high; card: "Use high or xhigh for complex problem solving, coding, and agentic tasks"); "To hard-cap thinking tokens use --reasoning-budget N"; no output budget number
- GGUF: NO general.sampling keys, so flags are mandatory
- flags: `--temp 1.0 --top-p 0.95 --top-k 64` (min-p default 0.05), strength left at the default high, RB 16384
- runs: 66y and 66zz on llama.cpp DEFAULTS (identified late), 66zy with the card values

## Flint (RecursiveSI, unsloth GGUF)
- source: https://huggingface.co/RecursiveSI/flint and https://huggingface.co/unsloth/flint-GGUF, fetched 2026-10-22 (sibling session): "Temperature 1.0 | Top-p / top-k 0.95 / 20 | Output budget for hard problems: At least 32768 tokens, including reasoning", "preserve returned reasoning_content in conversation history"
- GGUF: carries temp 1.0 / top_k 20 / top_p 0.95, no min-p
- flags (66z pass 2): `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`, RB 32768

## Kimi K3.1 Pro (hosted, Nous Portal slug `moonshotai/kimi-k3.1-pro`)
- source: https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart, fetched 2026-10-24: "`temperature=1.0`, `top_p=0.95`, `n=1`, `presence_penalty=0`, and `frequency_penalty=0` are fixed; omit them from requests", "Reasoning effort is configured with the top-level `reasoning_effort` request field and supports `low`, `high`, and `max` (default `max`); K3.1 Pro always has thinking mode enabled", "`max_completion_tokens` defaults to 131072", "Return the complete assistant message unchanged in multi-turn conversations and tool calls". open weights "by Nov 4" (same page), no HF card yet.
- price (live catalogue https://inference-api.nousresearch.com/v1/models, 2026-10-24): $2.00/M input, $0.20/M cache read, $10.00/M output; context 1,048,576.
- what Hermes sends on our custom route: no sampling fields (so the fixed values apply) and `reasoning_effort: "medium"`, which the maker doc does not list (the Portal accepted it with HTTP 200 in a relay probe 2026-10-24, mapping unknown). maker default = field omitted = `max`: the relay runs with `--reasoning-effort omit`.
- harness: `scripts/hosted-voxel.sh` + `scripts/portal_relay.py` (relay holds the credential, logs usage + cost, hard ceiling, SIGSTOP pid for the gate setup). ctx 131072 in the Hermes profile for parity with the local series (compaction behaviour), not the model's 1M.
- relay probe 2026-10-24 (3 tiny requests): streaming passthrough + usage parse ok; Portal's `usage.cost` matched our catalogue maths (0.000866 vs 0.00087); a tool-call turn 2 without `reasoning_content` echoed back still returned 200.
- runs: kimi-k31-pro-msm-v2-h2 (2026-10-24, hosted-voxel.sh, effort omit = max): gate PASS, end wall_limit 3640 s, $3.38.

## Ornith 1.5 35B-A3B Q4_K_M (vendor first-party GGUF; trees item 72a)
- source: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF README (`scripts/card-params.py` run on capsule), fetched 2026-10-28: "Recommended sampling parameters: For general tasks: temperature=0.6, top_p=0.95, top_k=20. To reproduce the reported benchmarks: temperature=1.0" (the maker's agentic-coding evals name temp 1.0 with top_p 0.95 or 1.0). The card names no min-p, no output budget and no reasoning-effort knob (its own benchmark runs used max_new_tokens up to 131072 and 48K output for NL2Repo, not a recommendation for us).
- GGUF `general.sampling.*` keys (read 2026-10-28 from ~/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf): temp 1.0, top_k 20, top_p 0.95. So the file carries the benchmark-reproduction temperature, not the general-tasks one.
- flags (item 72a): `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05`, i.e. the card's general-tasks values override the file's temp 1.0; min-p 0.05 stays the llama.cpp default (card names none, parity with the Qwen reference, L6); RB 16384 (no card budget, floor). Judgement call: the other defensible reading is temp 1.0 (the GGUF key and the maker's agentic evals); flipping it is one flag. The earlier Ornith bench rows (board, 122B) ran greedy, so this is the first card-sampling run.
- template: embedded Qwen-style template, thinking on by default (apply-template ends `<think>\n`), `reasoning_content` split verified on a CPU load 2026-10-28 (port 8197, build b11777-839a572a4), preserve-reasoning enabled by default. `supports_reasoning_effort: false`, so Hermes' `reasoning_effort: medium` is inert.
- speculative: the GGUF carries one nextn/MTP block (`nextn_predict_layers 1`), item 72a runs WITHOUT draft flags: the shipped head measured slower than base decode on vLLM (245-254 vs ~300 tok/s, reports/ornith-1-5-35b.md); not re-measured on llama.cpp (UNVERIFIED).
- runs: ornith15-35b-q4km-trees1 (72a, trees-v1 on the h2 harness), conditional on 71-r2 passing.

## Gemma 4 31B QAT Q4_0 (google first-party GGUF; trees item 72b)
- source: https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf README (card-params.py run on capsule; the base and unquantized cards carry the same text), fetched 2026-10-28: "Use the following standardized sampling configuration across all use cases: temperature=1.0, top_p=0.95, top_k=64"; "Thinking is enabled by including the `<|think|>` token at the start of the system prompt. To disable thinking, remove the token"; "In multi-turn conversations, the historical model output should only include the final response" (tool-call turns keep their thinking). The card names no min-p, no output budget, no effort levels.
- GGUF `general.sampling.*` keys (read 2026-10-28 from ~/models/gemma4-qat-q4/gemma-4-31B_q4_0-it.gguf): temp 1.0, top_k 64, top_p 0.95 (same as the card).
- flags (item 72b): `--temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.05` (flags pin the card values; min-p 0.05 default, parity with the Qwen reference, L6); RB 16384 (no card budget, floor).
- thinking: the template renders NO thinking by default (apply-template shows no `<|think|>`), unlike Qwen3.8 and Ornith. Item 72b passes `--chat-template-kwargs '{"enable_thinking":true}'` so all three trees runs are thinking-on; CPU load 2026-10-28 confirmed `<|think|>` in the system turn and a chat reply with `reasoning_content` (build b11777-839a572a4). Judgement call: thinking-off Gemma is the other defensible reading. Hermes' history handling against the "no thinking in history" note is UNVERIFIED.
- speculative: no nextn head in the GGUF; the separate Gemma assistant drafter on disk is not used (one variable at a time). The mmproj file next to the GGUF is not loaded.
- runs: gemma4-31b-qat-trees1 (72b, trees-v1 on the h2 harness), conditional on 71-r2 passing.
