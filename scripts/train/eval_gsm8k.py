#!/usr/bin/env python3
"""In-process GSM8K test eval (base or base+adapter), rig-consistent extraction.

Usage: eval_gsm8k.py --out eval_base.json [--base <id>] [--adapter <path>] [--n 300]
Same prompt / N / seed for base and tuned so the delta is real.
"""
import argparse
import json

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

try:
    from scripts.train.gsm8k_extract import extract_gsm8k_answer
except ImportError:  # capsule flat layout (~/train/)
    from gsm8k_extract import extract_gsm8k_answer

BASE = "unsloth/Llama-3.2-1B-Instruct"
PROMPT = ("Solve the math problem. Show brief reasoning, then end with '#### <answer>'.\n\n"
          "Problem: {q}\nSolution:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=2048, load_in_4bit=True, dtype=None)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    FastLanguageModel.for_inference(model)

    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(a.n))
    correct, rows = 0, []
    for ex in ds:
        gold = extract_gsm8k_answer(ex["answer"])
        msgs = [{"role": "user", "content": PROMPT.format(q=ex["question"])}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        out = model.generate(input_ids=ids, max_new_tokens=512, do_sample=False)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = extract_gsm8k_answer(gen)
        ok = bool(pred is not None and pred == gold)
        correct += ok
        rows.append({"gold": gold, "pred": pred, "ok": ok, "gen": gen[:700]})

    acc = round(100 * correct / len(ds), 2)
    json.dump({"base": a.base, "adapter": a.adapter, "n": len(ds), "accuracy": acc,
               "rows": rows}, open(a.out, "w"), indent=2)
    print(f"GSM8K acc = {acc}%  ({correct}/{len(ds)})  adapter={a.adapter}")


if __name__ == "__main__":
    main()
