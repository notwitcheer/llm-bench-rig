#!/usr/bin/env python3
"""GRPO on Qwen3-4B against a GSM8K verifiable reward. Saves adapter + reward-curve log.

Prompt is identical to eval_gsm8k.PROMPT (train<->eval alignment — the sub-2 lever).
Rollouts use HF generation (use_vllm=False) on a 4-bit QLoRA policy — the portable path.

NOTE (2026-06-06): the FP8-RL / vLLM-rollout path is BLOCKED on capsule — Unsloth FP8 RL needs
vLLM fast_inference, whose flashinfer FP8 kernels require CUDA>=12.9 (capsule has 12.8 -> the t036
wall), and the unsloth/vLLM/torch versions conflict on Blackwell (vLLM 0.21 needs torch 2.11+cu130,
unsloth needs torch 2.10). So --fp8 is intentionally disabled; see ~/unsloth-runs/backend_finding.md.
Run with Donald drained, from ~/grpo-env.
"""
import argparse
import json
from pathlib import Path

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
    ap.add_argument("--fp8", action="store_true",
                    help="(BLOCKED on capsule) FP8 RL needs vLLM fast_inference; flashinfer FP8 "
                         "kernels require CUDA>=12.9 and unsloth/vLLM/torch conflict on Blackwell.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=MAXGEN)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.fp8:
        raise SystemExit(
            "FP8 RL is blocked on capsule (CUDA 12.8 < flashinfer's 12.9 + unsloth/vLLM/torch "
            "version conflict). Use the default BF16/QLoRA HF-generation path. "
            "See ~/unsloth-runs/backend_finding.md.")

    out = Path(a.out) if a.out else Path.home() / "unsloth-runs" / "gsm8k-grpo"
    out.mkdir(parents=True, exist_ok=True)

    # 4-bit QLoRA policy; HF generation for rollouts (use_vllm=False) — sidesteps the vLLM/flashinfer
    # FP8 wall and the unsloth<->vLLM LoRA version conflict. Donald drained => full 32GB available.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=MAXPROMPT + MAXGEN, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)

    ds = build_dataset()
    FastLanguageModel.for_training(model)
    trainer = GRPOTrainer(
        model=model, processing_class=tok,
        reward_funcs=[correctness_reward, format_reward],
        train_dataset=ds,
        args=GRPOConfig(
            output_dir=str(out), learning_rate=5e-6, optim="adamw_8bit",
            per_device_train_batch_size=a.num_generations, gradient_accumulation_steps=1,
            num_generations=a.num_generations, max_prompt_length=MAXPROMPT,
            max_completion_length=a.max_completion, max_steps=a.steps, warmup_ratio=0.1,
            logging_steps=5, save_steps=a.steps, seed=42, report_to="none", use_vllm=False))
    trainer.train()
    model.save_pretrained(str(out / "adapter"))

    rewards = [(h.get("step"), h["reward"]) for h in trainer.state.log_history if "reward" in h]
    json.dump({"base": BASE, "fp8": False, "use_vllm": False, "steps": a.steps,
               "num_generations": a.num_generations,
               "reward_first": rewards[0][1] if rewards else None,
               "reward_last": rewards[-1][1] if rewards else None,
               "reward_curve": rewards},
              open(out / "train_log.json", "w"), indent=2)
    print(f"GRPO DONE (use_vllm=False) reward",
          rewards[0][1] if rewards else None, "->", rewards[-1][1] if rewards else None)


if __name__ == "__main__":
    main()
