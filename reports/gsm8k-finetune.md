# Fine-tuning a 1B for a measured GSM8K gain (and the eval bug that hid it)

**Model:** `unsloth/Llama-3.2-1B-Instruct` + QLoRA adapter (r=32) · **Data:** MetaMathQA (30k subset, format-aligned)
**Hardware:** RTX 5090 32GB · trained in `~/unsloth-env` (Unsloth 2026.6.1, bnb 4-bit), alongside a live Donald — no drain
**Eval:** in-process GSM8K test, N=300, greedy, identical prompt + extraction for base and tuned

## Result

| | GSM8K (n=300) |
|---|--:|
| Base (Llama-3.2-1B-Instruct) | **39.33%** |
| + QLoRA (MetaMathQA, format-aligned) | **44.00%** |
| **Gain** | **+4.67 pts** |

Train loss 1.41 → 0.47 over 700 steps (~5 min). A small, honest, *measured* lift — and the path to it is the real story.

## The path (three acts)

**Act 1 — the eval lied.** First run scored base 31.7% → tuned **11.7%** (−20). A −20pt "regression" from a clean QLoRA is a red flag, not a result. The tuned model's outputs were correct (`The answer is: 18`) but the extractor returned `None` — it mishandled MetaMathQA's `The answer is:` format, the `<<calc=x>>` annotations, and the model's post-answer rambling. **A capable model scoring absurdly low is almost always the harness** (the same lesson this rig learned twice on HumanEval). Fixed: take the *first* committed answer, strip `<<>>`, handle `answer is X` and `#### X`; and capture raw generations so any future re-score is offline.

**Act 2 — honestly flat.** With the fixed harness: base 39.3% → tuned **38.0%**. The −20 was an artifact; the real result was *flat*. The naive QLoRA had taught the model MetaMathQA's *style*, not math *capability*.

**Act 3 — alignment unlocked the gain.** The base was trained on raw MetaMathQA queries but *evaluated* with a structured "end with `#### <answer>`" prompt — a train/eval mismatch. Aligning them (train on the eval's prompt; reformat every answer to end `#### X`) + more steps/rank turned flat into **+4.67**.

## Before / after (same questions)

- gold **12** — base answered 2 (misread the structure); tuned solved it step-by-step → 12 ✓
- gold **32** — base just divided by 2 → 80; tuned set up the algebra → 32 ✓

## Worth it if / not if

- **The lesson > the number.** +4.67 on a 1B is modest; the value is the loop: *measure honestly, suspect your harness, align train to eval.* That loop generalizes to every fine-tune.
- **Not a capability miracle.** A small instruct model + a quick QLoRA buys you format-alignment and a few points — not a new tier. The gains come from rigor, not magic.

## Honest scope

In-process eval (internally consistent: base vs tuned, identical harness) — not the leaderboard's server harness, so the absolute % isn't directly comparable to the model board. The *delta* is what's claimed.

## Reproduce

```bash
~/unsloth-env/bin/python gsm8k_finetune.py    # base + MetaMathQA (aligned) -> adapter
~/unsloth-env/bin/python eval_gsm8k.py --out eval_base.json                                  # base
~/unsloth-env/bin/python eval_gsm8k.py --adapter ~/unsloth-runs/gsm8k-01/adapter --out eval_tuned.json
```
Scripts: `scripts/train/{gsm8k_finetune,eval_gsm8k,gsm8k_extract}.py`. Adapter: `witcheer/llama-3.2-1b-gsm8k-lora`.
