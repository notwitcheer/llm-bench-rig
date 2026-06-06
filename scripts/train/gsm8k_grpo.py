#!/usr/bin/env python3
"""GRPO on Qwen3-4B against a GSM8K verifiable reward. Saves adapter + reward-curve log.

Prompt is identical to eval_gsm8k.PROMPT (train<->eval alignment — the sub-2 lever).
--fp8 enables Unsloth FP8 RL (load_in_fp8=True, confirmed Phase-0); default is BF16.
Rollouts go through vLLM (fast_inference). Run with Donald drained (full 32GB), from ~/grpo-env.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")  # FP8-RL memory opt (Phase-0 finding)

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

try:
    from scripts.train.gsm8k_extract import extract_gsm8k_answer
    from scripts.train.grpo_rewards import correctness_reward, format_reward
except ImportError:  # capsule flat layout (~/train/)
    from gsm8k_extract import extract_gsm8k_answer
    from grpo_rewards import correctness_reward, format_reward

BASE = "unsloth/Qwen3-4B"  # confirmed Phase-0
# MUST stay identical to eval_gsm8k.PROMPT.
PROMPT = ("Solve the math problem. Show brief reasoning, then end with '#### <answer>'.\n\n"
          "Problem: {q}\nSolution:")
MAXPROMPT, MAXGEN = 256, 512


def build_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)

    def to_grpo(ex):
        return {"prompt": [{"role": "user", "content": PROMPT.format(q=ex["question"])}],
                "answer": extract_gsm8k_answer(ex["answer"])}

    return ds.map(to_grpo, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", action="store_true", help="enable Unsloth FP8 RL")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path.home() / "unsloth-runs" / (
        "gsm8k-grpo-fp8" if a.fp8 else "gsm8k-grpo-bf16")
    out.mkdir(parents=True, exist_ok=True)

    load_kwargs = dict(model_name=BASE, max_seq_length=MAXPROMPT + MAXGEN, load_in_4bit=True,
                       fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.85)
    if a.fp8:
        # FP8 enablement — confirmed Phase-0 (Unsloth FP8-RL guide): load_in_fp8=True (row-wise);
        # use "block" for block-FP8. Requires UNSLOTH_VLLM_STANDBY=1 (set at top).
        load_kwargs["load_in_fp8"] = True
    model, tok = FastLanguageModel.from_pretrained(**load_kwargs)
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)

    ds = build_dataset()
    trainer = GRPOTrainer(
        model=model, processing_class=tok,
        reward_funcs=[correctness_reward, format_reward],
        train_dataset=ds,
        args=GRPOConfig(
            output_dir=str(out), learning_rate=5e-6, optim="adamw_8bit",
            per_device_train_batch_size=a.num_generations, gradient_accumulation_steps=1,
            num_generations=a.num_generations, max_prompt_length=MAXPROMPT,
            max_completion_length=MAXGEN, max_steps=a.steps, warmup_ratio=0.1,
            logging_steps=5, save_steps=a.steps, seed=42, report_to="none", use_vllm=True))
    trainer.train()
    model.save_pretrained(str(out / "adapter"))

    rewards = [(h.get("step"), h["reward"]) for h in trainer.state.log_history if "reward" in h]
    json.dump({"base": BASE, "fp8": a.fp8, "steps": a.steps,
               "num_generations": a.num_generations,
               "reward_first": rewards[0][1] if rewards else None,
               "reward_last": rewards[-1][1] if rewards else None,
               "reward_curve": rewards},
              open(out / "train_log.json", "w"), indent=2)
    print(f"GRPO DONE fp8={a.fp8} reward",
          rewards[0][1] if rewards else None, "->", rewards[-1][1] if rewards else None)


if __name__ == "__main__":
    main()
