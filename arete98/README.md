# arete98: toolcall-v1 on Arete-27B Q6_K against the base Qwen3.8-27B Q6_K row

Queue item `98-arete-toolcall.sh`: the frozen toolcall-v1 slice (BFCL v4 raw mode, see `../toolcall/`), legs st-on (S1-S4, single-turn, 899 requests) and s5-on (100 multi-turn entries), on NousResearch Arete-27B Q6_K (bartowski GGUF). Same build (llama.cpp b10371 5d16e81dd), parser, ids, greedy sampling, context and token caps as the existing `qwen3-8-27b-q6-k` row (st-on 799 correct = 88.2, s5-on 81/100).

The Arete GGUF embeds the official Qwen3.8 chat template byte for byte; raw mode renders client-side with that template and checks `/tokenize` parity per leg.

Pre-registered verdict (paired exact McNemar on the same frozen ids): primary s5-on, p < 0.05 decides ARETE AHEAD or ARETE BELOW, otherwise LEVEL. st-on is reported the same way. Confounds: one greedy run each, different GGUF makers (Unsloth vs bartowski), the base row ran three weeks earlier on the same build.

Status: running 2026-11-07 from 18:45 CET. The result is added here after the run.
