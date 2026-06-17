# GLM-5.2 autopsy: they did the hard math for cheap 1M context, and it still can't run at home

**Rig:** one RTX 5090 32GB + 64GB DDR5 (96GB addressable) + 2TB NVMe — the consumer-sovereignty test
**Model:** [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) — 743B-param / 39B-active MoE (256 routed experts, 8 + 1 shared active), 78 layers, MIT licence, 1M context. Architecture `glm_moe_dsa`: **MLA** (Multi-head Latent Attention, compressed-latent KV) + **DSA** (DeepSeek Sparse Attention). Released 2026-06-16, the largest open-weights drop in months.
**This is an autopsy, not a benchmark.** GLM-5.2 cannot be served on the rig at any quantisation. The point is *why*, by arithmetic from the published `config.json`, and what the architecture genuinely buys.

## Credit first: MLA makes the KV cache ~57x cheaper

GLM-5.2's KV cache is MLA-compressed. Per token, per layer, it stores a 512-wide latent (`kv_lora_rank`) plus a 64-wide decoupled RoPE key (`qk_rope_head_dim`) = **576 values, shared across all 64 heads**. A standard attention with these head dims (K 256 + V 256 per head × 64 heads) would store **32,768 values**. That is a **~57x smaller KV cache** — real engineering, not a press release.

The payoff is concrete (576 × 78 layers × 2 bytes × tokens):

| context | MLA KV cache | naive KV would be |
|---|---|---|
| 64K | ~5.5 GiB | ~312 GiB |
| 256K | ~22 GiB | ~1.22 TiB |
| **1M** | **~88 GiB** | **~4.9 TiB** |

At 64K, MLA makes the KV trivial. The 1M headline is where the arithmetic bites.

## Wall 1 — the weights never fit addressable memory, at any quant

743B params. In bf16 that is **1.49 TB**. To fit 743B into the rig's **96 GB** addressable memory (32 VRAM + 64 RAM) you would need **1.03 bits per weight** — below any coherent quantisation. The smallest usable quant (unsloth Dynamic ~1.58-bit) is **~147 GB**, already **1.5x** the whole box. So GLM-5.2's weights cannot live in RAM+VRAM at *any* quant; the only consumer path is NVMe mmap streaming off the 2TB SSD, at well under 1–3 tok/s.

## Wall 2 — the 1M KV alone eats the whole machine

Grant MLA its 57x win and the 1M cache is still **~88 GiB ≈ the rig's entire 96 GB working memory**, leaving ~8 GB for the model's active compute. So even the desperate NVMe-mmap hack — which can limp the weights at *small* context — is impossible at 1M: the KV cache alone fills the box. Serving GLM-5.2 at its headline context needs **~147 GB weights + ~88 GiB KV ≈ 235 GB resident**, past a single H200 (141 GB) and into multi-GPU datacenter territory.

## Wall 3 — DSA saves compute, not memory

`glm_moe_dsa` adds DeepSeek Sparse Attention: a lightning indexer selects a top-k of tokens to attend, cutting the O(N²) attention compute. But the full MLA KV still has to be resident — you cannot attend to what you have evicted — and reference sparse-attention kernels materialise the indexer's N×N scores (the exact wall measured on the Keye-VL-2.0 autopsy). DSA is a throughput win at datacenter scale, not a memory key that unlocks consumer hardware. And the 39B active params (of 743B) cut compute per token, not the memory floor: every expert must still be loadable, because the next token may route to any of them.

## The verdict

GLM-5.2 is the largest open-weights release in months, and the team did the genuinely hard work — MLA plus DSA — to make 1M context cheap. It is 57x cheaper. It is still ~235 GB of weights and KV to use that context, on a model whose weights cannot fit a consumer box at any quant. **Clever is not the same as consumer.** GLM-5.2 is a datacenter model with an open licence, not a model you run on a 5090. The home-lab move is to wait for a GLM-5.2-Air — a ~100–130B variant would be the first that even NVMe-streams usefully — or rent the iron.

## Method

All figures derived from the published `config.json`: `num_hidden_layers` 78, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `num_key_value_heads` 64, `qk_nope_head_dim` 192, `v_head_dim` 256, `max_position_embeddings` 1048576; plus the published 743B-total / 39B-active counts. KV cache = (kv_lora_rank + qk_rope_head_dim) × layers × 2 bytes × tokens; naive KV = 2 × heads × head_dim × layers × 2 bytes × tokens. Quant floor scaled from unsloth's measured DeepSeek-R1 671B 1.58-bit GGUF (131 GB). No weights were downloaded or served.
