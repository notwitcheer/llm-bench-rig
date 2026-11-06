# astra-stop: the Astra-Stop weight edit replayed on Q4_K_M (item 90)

The Astra-Stop edit (https://huggingface.co/jarrodwatts/Qwen3.8-27B-Astra-Stop) changes one matrix of Qwen3.8-27B (`layers.63.mlp.down_proj`, from a 20,608-byte `u.npy`) and reports about 40% fewer thinking tokens on "pelican on a bicycle" at bf16, 16 tries a side. This folder replays it as a pre-registered unedited-vs-edited pair at Q4_K_M on one RTX 5090.

- `PREREG.md`: question, arms, sampler, verdict rule, frozen before any counted output (`FROZEN.sha256`).
- `build.sh`: the author's `apply.py` on the bf16 weights, then GGUF bf16 and `llama-quantize Q4_K_M` for both arms (same recipe).
- `run_legs.py` (4 slots, resumable per try), `astra_summary.py` (applies the rule), `render_astra.py` (CPU split-screen render), `make_mock_results.py` (selftest data, not output), `90-astra-stop-q4.sh` (the idle-queue item, capsule paths).

Result (run 2026-11-06 01:43-05:11 CET, llama.cpp fac3a894f, KV q8_0, temp 1.0 top-p 0.95 top-k 20, reasoning effort xhigh, seeds 1-32, cap 60,000, none capped): pre-registered verdict REPRODUCED. Median thinking tokens 26,254 unedited vs 16,930 edited (-35.5%), mean -45.9%, Mann-Whitney p = 7.5e-06, n = 32 per arm. Tries under 5,000 thinking tokens 0 vs 10. Complete SVGs 32/32 vs 31/32. Maths slice (24 problems, reported only) 22/24 both arms. The patched matrix still reads about twice as strong after quantisation (1.999x); every other tensor is byte-identical. In 11 of 32 edited answers a literal `</think>` appears again inside the visible answer (0 of 32 unedited); the one incomplete SVG is the try where it landed inside the SVG. Summary: `data/astra-stop/`.

A quantised pair is a different test from the author's bf16 one; nothing here is a claim about bf16.
