# One-Shot Entropy Minimization on a 5090: the entropy fell, the accuracy didn't

**Rig:** one RTX 5090 32GB (sm_120) · Qwen2.5-Math-7B · greedy pass@1 · the authors' Qwen2.5-Math grader, reused verbatim
**Claim under test:** One-Shot EM (arXiv 2505.20282) — minimize the model's output entropy on *one* unlabeled example for ~10 steps, reported **+24.7 avg** on Qwen2.5-Math-7B (MATH500 53.0→78.8, AMC23 44.1→70.3, greedy pass@1).
**What I could actually run:** the paper's recipe is **full-parameter** bf16 with optimizer offload, trained on multiple GPUs. On 32GB that OOMs (weights 14GB + bf16 grads 14GB + activations > 32GB, measured), so the only consumer-feasible version is **LoRA** (r=16, 40M trainable params, 0.53%), batch 16, 15 steps. That is a real deviation — read the caveats before quoting this.

## The numbers (Qwen2.5-Math-7B, RTX 5090)

| | MATH500 | AMC23 |
|---|---|---|
| base, paper protocol (ours) | **53.4** | 45.0 |
| base, fair protocol (max-tokens 4096) | 54.6 | 45.0 |
| **EM @ step 10 (ours)** | **55.4** | **42.5** |
| EM @ step 15 (ours) | 53.0 | 42.5 |
| EM (paper's claim) | 78.8 | 70.3 |

Two things happened, and the gap between them is the whole report.

## Finding 1 — the base reproduces, the gain does not

Our base lands at **53.4 on MATH500** against the paper's reported 53.0 — the harness, the qwen25-math-cot template, and the vendored sympy grader all agree with the paper to within a point. So the baseline is honest (raising max-tokens to 4096 moves it +1.2, not the +12 a truncation artifact would need).

Then one-shot EM adds **+2.0 on MATH500** at its best step and **−2.5 on AMC23** — against a claimed +25.8 and +26.2. By step 15 MATH500 has fallen back to the base (53.0): the peak-then-collapse the paper itself describes. The +24.7 headline does not survive on a single consumer card.

## Finding 2 — the entropy fell exactly as designed; the accuracy ignored it

This is the part worth keeping. The training worked: mean per-token entropy on the pi1 rollouts dropped **0.098 → 0.035** over 15 steps, steepest in the last five. The model became measurably more confident. Accuracy did not follow it up — +2.0 at most, then down. Output length and repetition barely moved at the peak (MATH500: 1191→1195 tokens, repetition 0.332→0.338), so this isn't a generation-degeneracy story at step 10; it's simply that **sharpening a distribution is not the same as making it more correct.** The paper says as much about its own method — "a distribution-shaping tool rather than a learning method" — and on this hardware that is all we measured it to be. It sits in the same place as the Spurious Rewards result for this exact base model (Qwen2.5-Math-7B gains ~+21 on MATH500 from *random* rewards): the signal is doing less than the headline implies.

## Caveats (these matter more than usual here)

- **This is not a refutation of the paper's full-param result.** The full-parameter recipe needs more than 32GB and did not run; LoRA + batch 16 is a weaker training setup and could be why the gain is absent. The honest claim is narrow: *on a 32GB consumer GPU, the recipe that fits does not reproduce +24.7, and the entropy objective trained fine while accuracy stayed flat.*
- **The format-elicitation control backfired and is inconclusive.** A 2-shot scaffold meant to isolate "the model already knows the format" instead *degraded* Qwen2.5-Math (MATH500 14.2, repetition 0.52) — the model's strength is its native zero-shot CoT, and few-shot examples disrupt it. So I can't cleanly separate format-elicitation from capability here; I'm reporting the raw result rather than a clean three-way split.
- **AMC23 is 40 problems** — high variance; the −2.5 is two questions. MATH500 (500q) is the number to weight.
- Greedy pass@1, single seed, the authors' grader. Base-reproduction (53.4 vs 53.0) is the evidence the pipeline is sound.

## Worth it if / not if

- **Not a free consumer win.** If you have one 32GB card, there is no one-example +24.7 waiting for you: the full-param recipe doesn't fit, and the LoRA version that does fit moves MATH500 by +2 then gives it back.
- **The mechanism is the takeaway.** Entropy minimization reliably makes a math model more confident; on this hardware it did not make it more correct. Treat "unsupervised, one example, huge gain" claims on Qwen-Math bases as elicitation-shaped until a controlled run on a *non-Qwen-Math* base (the real falsification) says otherwise — that is the follow-up worth doing.

## Repro

- EM loss + collapse metrics + prompt builders (dep-free, tested): `lib/em.py`. Vendored Qwen2.5-Math grader (the `math_equal` oracle, not reimplemented): `lib/qwen_math_grader/`. Data: `scripts/fetch_em_data.py` → `dataset/em/{math500,amc23,pi1}.jsonl` (from the one-shot-em repo).
- Train: `scripts/train_em.py` (batched rollouts → entropy backward; `--lora` for the 32GB memory gate). Eval: `scripts/eval_math.py` (vLLM greedy gen — set `VLLM_USE_FLASHINFER_SAMPLER=0` on sm_120) + `scripts/grade_em.py` (grader-env). Chart: `scripts/chart_em.py`.
- Grader-env note: the PyPI `latex2sympy2` is broken on modern Python; install the repo's **vendored** latex2sympy (1.9.0) pinned to `antlr4-python3-runtime==4.11.1` + `sympy==1.12`.
