# replay-cache readout (replaycache96)

VERDICT DOES NOT HOLD, TRAPS | holds=False trap=True | entries 20/20 validated
LINE1 NVIDIA says a CPU can replay a big model's reasoning from a cache. I rebuilt it on my 5090 and it does not hold here: 83% against 96% for the 27B from scratch (a stock 1B alone gets 48%), and on near-miss questions it was confidently wrong 92% of the time.

regime: llama.cpp 839a572a4; A = Qwen3.8-27B Q6_K, 4 slots, ctx 131072, KV q8_0, temp 1.0 top-p 0.95 top-k 20, reasoning budget 16384, max 20000 tokens; B/C = Llama-3.2-1B Q8_0 + EmbeddingGemma-300M Q8_0 on CPU (6 threads, -ngl 0)
