#!/usr/bin/env python3
"""QLoRA fine-tune a small model on MetaMathQA to lift GSM8K. Saves adapter + train log.

Iteration 2: train/eval FORMAT-ALIGNED — the training prompt is identical to eval_gsm8k.PROMPT,
and each MetaMathQA response is reformatted to end with '#### <answer>' (the eval's expected
ending). Aligning train and eval is the lever that turns format-learning into a measured gain.
"""
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

try:
    from scripts.train.gsm8k_extract import extract_gsm8k_answer
except ImportError:  # capsule flat layout (~/train/)
    from gsm8k_extract import extract_gsm8k_answer

BASE = "unsloth/Llama-3.2-1B-Instruct"
# MUST stay identical to eval_gsm8k.PROMPT — train/eval alignment is the point.
PROMPT = ("Solve the math problem. Show brief reasoning, then end with '#### <answer>'.\n\n"
          "Problem: {q}\nSolution:")
OUT = Path.home() / "unsloth-runs" / "gsm8k-01"
OUT.mkdir(parents=True, exist_ok=True)
MAXLEN = 2048

model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE, max_seq_length=MAXLEN, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=42).select(range(30000))


def fmt(ex):
    ans = extract_gsm8k_answer(ex["response"])           # MetaMathQA ends "The answer is: X"
    resp = ex["response"].strip() + (f"\n#### {ans}" if ans else "")
    msgs = [{"role": "user", "content": PROMPT.format(q=ex["query"])},
            {"role": "assistant", "content": resp}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt)

FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text", max_seq_length=MAXLEN,
    args=SFTConfig(max_steps=700, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                   warmup_steps=20, learning_rate=2e-4, logging_steps=20, optim="adamw_8bit",
                   seed=42, output_dir=str(OUT), report_to="none"))
trainer.train()
model.save_pretrained(str(OUT / "adapter"))
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
json.dump({"base": BASE, "steps": 700, "r": 32, "format_aligned": True,
           "loss_first": losses[0] if losses else None,
           "loss_last": losses[-1] if losses else None},
          open(OUT / "train_log.json", "w"), indent=2)
print("TRAIN DONE loss", losses[0] if losses else None, "->", losses[-1] if losses else None)
