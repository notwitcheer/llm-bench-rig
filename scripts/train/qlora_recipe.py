#!/usr/bin/env python3
"""Reproduce a small Unsloth QLoRA recipe end-to-end on the 5090: prove the stack + show the adapter learned.

Runs in ~/unsloth-env (py3.12) on capsule. ~60 steps on a small instruct slice; captures a before/after
generation and the loss curve. Pass --no-4bit to use 16-bit LoRA (only if the bnb 4-bit gate failed).
"""
import json
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

LOAD_4BIT = "--no-4bit" not in sys.argv
OUT = Path.home() / "unsloth-runs" / "recipe-01"
OUT.mkdir(parents=True, exist_ok=True)
MAXLEN = 2048
PROMPT = "Explain what a binary search tree is, in one sentence."

model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit" if LOAD_4BIT else "unsloth/Qwen3-4B",
    max_seq_length=MAXLEN, load_in_4bit=LOAD_4BIT, dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42,
)


def gen(p):
    FastLanguageModel.for_inference(model)
    ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


before = gen(PROMPT)

ds = load_dataset("mlabonne/FineTome-100k", split="train[:2000]")


def fmt(ex):
    out_msgs = []
    for m in ex["conversations"]:
        role = "user" if m["from"] in ("human", "user") else "assistant"
        out_msgs.append({"role": role, "content": m["value"]})
    return {"text": tok.apply_chat_template(out_msgs, tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt)

FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    max_seq_length=MAXLEN,
    args=SFTConfig(max_steps=60, per_device_train_batch_size=2, gradient_accumulation_steps=4,
                   warmup_steps=5, learning_rate=2e-4, logging_steps=1, optim="adamw_8bit",
                   seed=42, output_dir=str(OUT), report_to="none"),
)
trainer.train()
after = gen(PROMPT)
model.save_pretrained(str(OUT / "adapter"))
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
json.dump({"load_4bit": LOAD_4BIT, "losses": losses,
           "loss_first": losses[0] if losses else None,
           "loss_last": losses[-1] if losses else None,
           "before": before, "after": after},
          open(OUT / "train_log.json", "w"), indent=2)
print("DONE loss", losses[0] if losses else None, "->", losses[-1] if losses else None)
