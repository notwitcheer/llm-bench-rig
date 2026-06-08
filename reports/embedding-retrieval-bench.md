# Does a vision-language embedder beat its text sibling on *text* retrieval?

**Rig:** RTX 5090 32GB (capsule) · **Date:** 2026-06-08 · **Task:** local-ai-roadmap t037
**Lineup:** e5-small-v2 (the incumbent) vs Qwen3-Embedding-0.6B (text) vs Qwen3-VL-Embedding-2B (vision-language)
**Ruler:** BEIR / SciFact (5,183 docs, 300 queries, gold qrels) · all arms capped at 512 tokens, cosine retrieval, identical scoring harness

## The myth

A vision-language embedding model is built for images. Point it at plain text and you'd expect it to
do no better than a same-family *text* embedder — the visual machinery is dead weight. And the small,
old incumbent (e5-small) is "good enough" for a tiny knowledge base anyway.

## What SciFact actually says

| model | dim | nDCG@10 | recall@10 | MRR@10 | encode docs/s¹ | VRAM |
|---|---|---|---|---|---|---|
| e5-small-v2 | 384 | 0.6875 | 0.806 | 0.658 | 1256 | 0.79 GB |
| Qwen3-Embedding-0.6B | 1024 | 0.7011 | 0.830 | 0.667 | 270 | 2.4 GB |
| **Qwen3-VL-Embedding-2B** | 2048 | **0.7444** | **0.882** | **0.704** | 129 | 5.0 GB |

The VL model wins on every quality metric: **+0.057 nDCG@10 over e5 (+8.3% relative), and +0.043 over its
own text sibling** — on a pure-text task with not a single pixel in sight. The visual machinery isn't dead
weight; the contrastive multimodal pre-training produces a stronger *text* representation too.

But quality isn't free: e5 encodes ~10x faster and fits in a sixth of the VRAM. This is a clean frontier,
not a free lunch.

~~~

## The Matryoshka twist — you can throw away most of the VL vector and still win

Both Qwen models use Matryoshka representation learning: the leading dimensions carry the most signal, so
you can truncate the vector and renormalize. nDCG@10 as the vector shrinks:

| dim | Qwen3-VL-2B | Qwen3-Embedding-0.6B |
|---|---|---|
| full | 0.7444 (2048) | 0.7011 (1024) |
| 512 | **0.7307** | 0.6978 |
| 256 | 0.7092 | 0.667 |
| 128 | 0.6691 | 0.6239 |

The headline: **VL truncated to 512 dims (0.731) still beats the full-1024-dim text model (0.701) and the
full e5 (0.688).** Even at 256 dims — one eighth of native — it beats both full-size rivals. So the VL
model's storage cost is negotiable: a quarter-size vector keeps 98% of its quality and remains the best
retriever in the field. e5's speed advantage stands; its quality ceiling does not.

~~~

## The vault reality check (and why the corpus is still the wall)

The point of t037 was Donald's memory vault — does any of this actually improve *his* retrieval? Ran the
trio over the real 15-note vault. Two honest observations:

- **It helps at the margins.** On "who is the user", e5 misses `identity.md` entirely; the text model
  surfaces it at rank 2; the VL model nails it at **rank 1**. Genuine semantic routing the old model couldn't do.
- **The corpus is still the bottleneck.** `project-memories.md` dominates nearly every query's top-3 across
  all three models — exactly the cp13 finding (a few notes swamp a tiny index). A better embedder cannot
  out-retrieve a corpus that's too small to discriminate.

So: at 15 notes, the upgrade buys you a handful of better answers, not a transformation. The retriever was
never the limiting reagent here — corpus hygiene and size are. That's the same lesson cp13 taught, now
confirmed from the other direction.

~~~

## Footnote that matters — the VL model needed image *libraries*, not a CUDA toolkit

Unlike t036 (vLLM quantized inference, walled by a missing CUDA ≥12.9 toolkit on this Blackwell box),
Qwen3-VL-Embedding-2B loaded cleanly via sentence-transformers once `pillow` + `torchvision` were
installed — pure Python image deps, no system toolkit. The `tomaarsen/...-vdr` repackage failed (no
recognized `image_processor_type`); the **official `Qwen/Qwen3-VL-Embedding-2B`** worked. One more datapoint
for the map of what runs toolkit-free on this box: quant *inference* needs the toolkit, embedding *and*
training do not.

## Verdict — worth it if / not if

- **Worth swapping e5 for Qwen3-VL-2B if** retrieval quality is the bottleneck and you can spend the encode
  time + VRAM — and truncate to 512 dims to claw most of the cost back.
- **Not worth it if** your corpus is small enough that retrieval already saturates (Donald's vault today), or
  if encode throughput dominates your workload — e5's 10x speed is real.
- **The non-obvious win:** a VL embedder is a legitimately strong *text* retriever. Don't rule it out because
  "it's for images."

---
¹ Encode throughput is the co-resident rate measured *alongside Donald* (llama-server holding ~18.9 GB), at
per-model batch sizes tuned to fit the free VRAM (e5 64 / text 16 / VL 8) — so it reflects deployment-on-a-busy-box,
not each model's isolated peak. Quality metrics are batch-invariant and clean. Harness + raw scores:
`scripts/embed_bench/`, `results/embed-scifact/`.
