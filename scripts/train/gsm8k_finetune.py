#!/usr/bin/env python3
"""QLoRA fine-tune a small model on MetaMathQA to lift GSM8K. Saves adapter + train log."""
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

BASE = "unsloth/Llama-3.2-1B-Instruct"
OUT = Path.home() / "unsloth-runs" / "gsm8k-01"
OUT.mkdir(parents=True, exist_ok=True)
MAXLEN = 2048

model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE, max_seq_length=MAXLEN, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=42).select(range(20000))


def fmt(ex):
    msgs = [{"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": ex["response"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt)

FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text", max_seq_length=MAXLEN,
    args=SFTConfig(max_steps=400, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                   warmup_steps=10, learning_rate=2e-4, logging_steps=10, optim="adamw_8bit",
                   seed=42, output_dir=str(OUT), report_to="none"))
trainer.train()
model.save_pretrained(str(OUT / "adapter"))
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
json.dump({"base": BASE, "steps": 400,
           "loss_first": losses[0] if losses else None,
           "loss_last": losses[-1] if losses else None},
          open(OUT / "train_log.json", "w"), indent=2)
print("TRAIN DONE loss", losses[0] if losses else None, "->", losses[-1] if losses else None)
