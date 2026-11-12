# item 9a1 pre-registration: a 6.74 GB ternary Qwen3.8-27B (ttf TQ1_P, no adapters) against the stock Qwen3.8-27B Q6_K, the same Hermes Agent task, one RTX 5090

frozen 2026-11-12 (Thu), before any counted output. This is the first freeze of the Hermes-take harness for item 9a1. A previous PREREG.md and FROZEN.sha256 on capsule (staged by an earlier child) were moved aside to ~/voxel-stage/9a1-ttf-ternary/superseded-1208/ and are not used: that item had no Hermes take legs. Engine and sanity numbers in this item are from the handover (UNVERIFIED by this freeze; reproduced by the counted run only): llama-bench tg128 121 t/s, pp512 1655 t/s, 7.2 GB peak on the ttf engine; stock Q6_K at 65,536 context 23.9 GB, tool call OK. None of it is a counted result.

## question

Does the no-adapter ternary Qwen3.8-27B (ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf, 6,741,621,728 bytes, Apache-2.0 weights, sha256 de56c766... matching the HF LFS oid) run the same Hermes Agent coding task as the stock Qwen3.8-27B Q6_K (22,884,408,288 bytes, the file already on the box), on one 32 GB card?

Second question, the author's claim check (not our number): the author's README, table "The 6.74 GB file", GSM8K first 200 problems: 0.880 for this file against 0.895 for full precision (the 8.2 GB file under the same harness: 0.900). HumanEval 0.787 against 0.933, IFEval strict 0.735 against 0.787. Source https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF, README fetched 2026-11-12. Quoted as their claim, with its source, not as ours. Our check: GSM8K on the first 20 problems only; the HumanEval and IFEval rows are not tested here. Author numbers re-fetched live 2026-11-12 from https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF.

## arms (all frozen)

- stock: `~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf` (22,884,408,288 B), llama.cpp commit cf34d3113 (`~/llama.cpp-moecache/build/bin/llama-server`, 0.9.0-dev build 6), CUDA sm_120. all layers on the GPU, `-ngl 99`.
- ttf: `~/models/ttf-27b/ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf` (6,741,621,728 B), ttf-home commit 92d53f02daa8ef4a818eed62eb9050737a96de35, engine llama.cpp commit 4afa91d03 (`~/ttf-home/engine-q38/build-cuda/bin/llama-server`, build 12193), built with `ttf engine build --q38 --backend cuda` (CUDA 12.8, gcc-14, CMAKE_CUDA_ARCHITECTURES=120). all layers on the GPU, `-ngl 99`.
- both arms: `--jinja`, flash-attn on, `-c 65536`, one slot, 6 threads, KV q8_0 (`-ctk q8_0 -ctv q8_0`), `--reasoning-budget 8192` with the message "Thinking budget reached. Act now with your commands.", no speculative decoding, no adapter, no MoE offload. Server port 8294, counting proxy port 8295.
- agent: Hermes Agent pin 9bda188 (`~/.hermes/hermes-agent`), its own HERMES_HOME per take (`mc_mkhome.py`), toolsets terminal + file, approvals off, memory off, max 60 turns, wall cap 10 min per take.
- sampling for the agent takes: temperature 0.6, top-p 0.95, top-k 20, seed 721, identical on both arms (mc_mkhome.py).
- task: `prompts/fix-shopcart.txt` on the seed repo: 4 planted bugs, 5 tests, 1 passing and 4 failing on the untouched seed. same task as items 89 and 99y.
- order and block structure: headline leg = the stock block (takes 1 to 3, one server load), then the ttf block (takes 1 to 3, one server load). extension legs (resumable): stock takes 4 to 8, then ttf takes 4 to 8, same blocks. a take already done is kept on resume; a take with less than 11 min of window left is skipped and the item exits 3.
- per arm, after its takes: GSM8K check (`arm9a1.py gsm8k`): first 20 rows of the openai/gsm8k `main` test split (parquet from the local HF cache, fixed row order), greedy (temperature 0), chat template with thinking OFF (`chat_template_kwargs.enable_thinking=false`), `max_tokens` 512. The user message is the question followed by the line "Give the final answer as a number on the last line." The prediction is the last number in the reply (commas removed, a trailing dot ignored); gold is the number after `####`. correct = equal as floats. errors count as wrong.
- tool-call smoke (per arm, after load): one request with a `get_time` tool, temperature 0.6, `max_tokens` 3000; it must return a `get_time` call with a `city` argument. failure = arm BROKEN.
- probe (per arm, after the smoke): fixed prompt, temperature 0, 256 tokens, `cache_prompt` off, two runs, the second is reported (decode tok/s). never gating.
- reported with every arm, never gating: decode tok/s (llama-server's own timings, token-weighted over the arm's CLEAN takes), file size, peak VRAM, mean GPU W (nvidia-smi at 1 Hz), time to done, tool calls, the GSM8K count.

## frozen hashes

Each hash below is in FROZEN.sha256 (the item refuses to run when any of them differs; the item script pins the sha256 of FROZEN.sha256 itself). The frozen set: mc_summary9a1.py (verdict rule, line 1, take parsing, self-test), facts9a1.py (card numbers), arm9a1.py (tool-call smoke, probe, GSM8K check), rp_gate.py (item 89's pass test, unchanged), mc_take.sh, mc_mkhome.py, rp_proxy.py, rp_record.py, rp_count.py, rp_recount.py, mc_render.py (items 89/99y copies, unchanged), prompts/fix-shopcart.txt, and the seed repo (`find seed -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum`, recorded in SEED.sha256). The item script is the runner and is not frozen; it carries the FROZEN.sha256 pin.

### hashes at freeze (sha256)
- `mc_summary9a1.py`: b0278b3062d5cad44b2dc380d08444bb2b5dd8147d790408fbb39bc2e126f062
- `facts9a1.py`: 99b362030869710d43d2a36cf7a2b917ebf0cdeffffa32f6f214c515a40e3086
- `arm9a1.py`: 030c8a0d482e7f20570e44ac04824b59f8aecb50238b093a4326077959143a0b
- `rp_gate.py`: 6b8dcbb5194e4c6773622a1fc48254ba8bc597938c8b8e69661ce881d82b2d57
- `mc_take.sh`: 07951fe7b6c7f07804aef55d495dcebbda34367605b2103644adebaca6461e72
- `mc_mkhome.py`: d0a70bd714060fbeb79b2e106ad2f40cb7e8ad0144590e9ecc56e590d55252e0
- `rp_proxy.py`: 0774e23b3735497fc24eb17445c09b7bb212cfa0976c81305aa5b4338338fd2e
- `rp_record.py`: cfc78e0c23be403e203d5c90b6b49d74c567f5f1bae807dde824bafcc92032ee
- `rp_count.py`: 5d24d8bfbf965f3a3637f67888a7a6a77ecc21e2e862eea999c634018b043b51
- `rp_recount.py`: 4cf332831b17a933dbaea9e7f0f60ba7b653249c895f14963d93b0b45914207c
- `mc_render.py`: 2ff68ef17a03cc0decc70e4e2bdf5f86a72f6eac7cb8d7852ca74d870e4d386a
- `prompts/fix-shopcart.txt`: d2e53c7d08ad34e5c974a410b6c5fa22d0fae3d0a5779ec530da94ec0a223afb
- seed repo (`seed/`, find | sort | sha256sum | sha256sum): dc0a36db1200e5203034c020966724b2b2711303fed911b227622c2ee42a5014

## rule (mc_summary9a1.py applies it as written; `mc_summary9a1.py --selftest` proves the branches)

- take CLEAN: ended done or wall, at least 3 tool calls in state.db, request errors at most 20% of requests (and at least one request).
- take PASS: CLEAN, the pristine gate passes (pristine tests 5 passed, 0 failed, 0 errors; no file under the agent's tests/ differs, none added), and the agent's tool calls never named the harness dir (peek regex: voxel-stage, 9a1-ttf-ternary, 9a1-runs, /ref/, rp_gate, PREREG, moecache99a, whichfile99y, ripple89).
- arm BROKEN: the server did not load, or its tool-call smoke failed, or none of its takes was CLEAN.
- verdict, applied in this order:
  - BROKEN (ttf): the ternary arm is BROKEN.
  - BROKEN (baseline): the stock arm is BROKEN (no baseline to hold against).
  - INCOMPLETE: either arm has fewer than 2 CLEAN takes. provisional; the item exits 3 and resumes.
  - HOLDS: ttf PASS count is at least stock PASS count minus 1.
  - FALLS BEHIND: ttf PASS count is at most stock PASS count minus 2.
- GSM8K and speed numbers are reported, never gating. The GSM8K line reads "holds on these 20" when ttf correct is at least stock correct minus 2, "behind on these 20" otherwise. n=20: a reading, not a test.
- line 1 is built by mc_summary9a1.py from the verdict and the counts, nothing typed by hand:
  - HOLDS: "a 27b ternary gguf at 6.7 GB held the same hermes task: [ttf] of [N] passes against [stock] of [N] for the 23 GB file, [X] GB less on disk"
  - FALLS BEHIND: "a 27b ternary gguf at 6.7 GB fell behind the 23 GB file on the same hermes task: [ttf] of [N] passes against [stock] of [N], [X] GB less on disk"
  - X = (stock bytes minus ttf bytes) / 1e9, one decimal (16.1 at the frozen sizes).

## how a null or a loss is framed

HOLDS is a result about one card, one file and one 4-bug task at n=3 takes per arm (n=8 at the extension). FALLS BEHIND is the same, framed as the measurement of this file on this harness, with the subject being the model file and the engine, never Hermes Agent. A GSM8K "behind" on 20 problems is a reading to be checked on more problems, not a retraction of the author's 200-problem table. BROKEN (ttf) means the patched engine or the file did not run on sm_120 for us: a build and load report, not a model verdict. INCOMPLETE is provisional and is never posted as a verdict.

## limits

one card (RTX 5090, 32 GB), one 4-bug task, n=3 takes per arm headline, up to n=8 at the extension, GSM8K on 20 problems. The author's HumanEval (0.787 vs 0.933) and IFEval strict (0.735 vs 0.787) rows are not tested here. The ternary arm's training story (README, fetched 2026-11-12: the 6.74 GB file was recovered with 1210 steps of 16 x 2048 tokens, 39.6M tokens, trits trainable, 0.033% of trits moved; the 8.2 GB file with 2200 steps, 72M tokens, trits frozen) is not tested here and is not claimed. Engine: a single pinned build of ttf-home (`--q38`, patch 0009 per the README); the patches are the author's, not reviewed line by line by us. Licence split: the weights are Apache-2.0; the engine (ttf-home, ttf-ternary, the llama.cpp patches) is PolyForm Noncommercial 1.0.0. A public harness copy may reference the engine, never vendor it.
