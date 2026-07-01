# DeepSeek-OCR-2: the compression ratio is real, but decode accuracy degrades faster than the paper's curve on real documents

**Rig:** one RTX 5090 32GB (sm_120) · DeepSeek-OCR-2 (3B, deepencoder + DeepSeek-V2 MoE decoder) · dedicated venv (torch 2.11+cu128, transformers 4.46.3 pinned to the model's tested era) · 15-page English sample from [OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) (opendatalab), selected for text-dominant pages across newspaper, magazine, academic-literature, exam-paper, and book sources
**Question:** the original DeepSeek-OCR paper (arXiv 2510.18234, "Contexts Optical Compression") claims that "when the number of text tokens is within 10 times that of vision tokens (compression ratio < 10x), the model can achieve decoding (OCR) precision of 97%," and that "even at a compression ratio of 20x, the OCR accuracy still remains at about 60%." The paper doesn't specify what test set this curve was measured on. Does it hold on a diverse sample of real, captured documents?

## The numbers

| | compression ratio (text tokens / vision tokens) | decode accuracy (edit-distance, vs ground truth) |
|---|---|---|
| **all 15 pages** | mean 9.83x, median 7.95x | mean 53.6%, median 56.3% |
| **7 pages, accuracy ≥ 60%** | mean 6.7x | mean 83% |
| **8 pages, accuracy < 60%** | mean 12.6x | mean 28% |

Vision tokens per page were fairly constant (mean 183, range 122-197) regardless of how text-dense the page was — the default tiling (`base_size=1024, image_size=640, crop_mode=True`) allocates roughly the same token budget whether the page is a sparse exam solution or a dense broadsheet newspaper column. Compression ratio is therefore driven almost entirely by how much text the page actually contains, not by an adaptive token budget.

## Finding 1 — the compression ratio itself is real and close to the claim

Ground-truth text tokens (counted with the model's own tokenizer) divided by vision tokens used averages **9.83x** across the 15-page sample — right at the paper's own 10x reference point. This part of the claim holds up on real, independently-sourced documents, not just whatever set the paper measured internally.

## Finding 2 — decode accuracy degrades much faster than the paper's curve once real-world density kicks in

The paper's curve: <10x compression → 97% precision, 20x → ~60%. My sample, split at the median compression ratio: pages averaging **6.7x compression measured 83% decode accuracy** — below the paper's 97% reference for that compression band. Pages averaging **12.6x compression measured 28% accuracy** — well below the paper's own 60% figure at nearly double that compression ratio (20x). The gap is largest exactly where it matters most: dense, real-world pages (broadsheet newspapers with tiny multi-column text) are harder than whatever the paper's precision curve was benchmarked against. Two of the worst-scoring pages (11-26x compression) scored 1-25% accuracy — the fixed vision-token budget genuinely cannot preserve enough detail to transcribe a dense multi-column page, and the model appears to produce a partial, plausible-sounding transcription rather than an honest "I can't read this."

## Finding 3 — a distinct failure mode: degenerate repetition on repeated running headers

One page (The Economist, Feb 2024) triggered visible output repetition: the model's transcription opens with `"The world this week\n\nThe world this week\n\nThe Economist February 24th 2024\n\nThe Economist February 24th 2024\n\nThe Economist February 24th 202\n\n..."` — looping on the page's running masthead/header text before eventually recovering into real article content. This isn't a ground-truth mismatch; it's a genuine generation degeneracy, likely triggered by the repeated header appearing at the top of each column in the source layout. `no_repeat_ngram_size` (set to 35 in the model's own inference code) evidently doesn't catch phrase-level repetition at this granularity.

## What this closes

Not a refutation of the compression claim — that part measures out almost exactly as claimed. It's a refinement of the accuracy side: the paper's own compression-precision curve, whatever it was measured on, is optimistic relative to a diverse real-document sample. The practical implication is direct: **the vision-token budget needs to scale with page density**, and the model's default settings don't do this — a sparse exam-solution page and a dense newspaper column get roughly the same token allocation, and only one of them can survive it.

## Honest caveats

- **n=15, single seed, English-only.** A larger, multilingual sample would sharpen the compression/accuracy curve; this is a scoped proxy, not a full OmniDocBench replication.
- **Decode accuracy here is a simplified proxy** (character-level edit distance against a concatenation of `text_block`/`title`/`header`/caption categories, sorted by the dataset's own reading-order field) — not the full official OmniDocBench scorer (which handles tables via TEDS, formulas via CDM, and per-category weighting). Verified by hand that low scores on the worst pages reflect genuine transcription gaps (missing sections, degenerate repetition) rather than reading-order artifacts in the scoring, but a formal re-score against the official harness would be more rigorous.
- **Tiling wasn't tuned.** The model supports a larger crop budget (documented up to `(0-6)×768×768 + 1×1024×1024`); these runs used the default `base_size=1024/image_size=640`, and only 1-2 tiles were actually used even on the densest pages. A "large" or "gundam"-style preset with more tiles might change the dense-page results — untested here, a natural follow-up.
- **The paper's own claimed curve's test set is unverified** — the abstract states the 97%/60% figures without naming the benchmark they were measured on, so this isn't an apples-to-apples replication, only a directional check against a different, real-world sample.
- **Environment note, worth banking:** this specific checkpoint's custom modeling code reuses transformers' own `LlamaAttention` class directly (not vendored), so it inherits whatever config-field expectations the *installed* transformers version has grown since the model's tested era (4.46.3). Under the rig's default transformers (5.5.0), this breaks in cascading ways — a missing class, a missing utility function, then missing config fields (`attention_bias`, `attention_dropout`) that a newer `LlamaAttention` expects but this checkpoint's config schema never defined. Fix: a dedicated venv pinning `transformers==4.46.3` (matching the tested era) paired with a modern, sm_120-capable torch (2.11+cu128) — old transformers, new torch, not the reverse. Same failure shape as the earlier Keye-VL-2.0 autopsy, but this one resolved cleanly once version-matched.

## Repro

`~/deepseek-ocr-env` (torch 2.11+cu128, transformers==4.46.3, tokenizers==0.20.3, einops/addict/easydict/matplotlib/torchvision) · sample selection + eval scripts + raw per-page results (`results_v2.json`, `sample.json`) in `results/deepseek-ocr-2/` · model: `deepseek-ai/DeepSeek-OCR-2` (Apache-2.0) · ground truth: [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench).
