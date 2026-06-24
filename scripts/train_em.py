"""Reimplemented One-Shot Entropy Minimization (arXiv 2505.20282 train.py) on the modern stack.

Per step: sample a BATCH of rollouts from the single pi1 prompt (no grad, fast batched generate),
then for each rollout teacher-force prompt+rollout WITH grad and minimize the mean per-generated-
token entropy (lib.em.entropy_loss). Full-param bf16 + paged-8bit-AdamW + grad-checkpointing.

Single-5090 adaptation (documented deviation): the paper used effective batch 64 on multi-GPU;
here batch defaults to 16 rollouts/step over the one example, generation BATCHED (not 64 sequential
HF generate calls) to fit the time cap. The mechanism (entropy-min on pi1 rollouts) is unchanged.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.em import build_prompt, entropy_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--example", default="dataset/em/pi1.jsonl")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=15)          # save each; eval step_10 + a late one
    ap.add_argument("--batch", type=int, default=16)          # rollouts/step (paper 64, multi-GPU)
    ap.add_argument("--max-new", type=int, default=256)       # rollout length (paper 512)
    ap.add_argument("--temp", type=float, default=0.5)        # softmax-sharpen temp in the loss
    ap.add_argument("--sample-temp", type=float, default=0.5)
    ap.add_argument("--lora", action="store_true")           # memory-gate fallback
    ap.add_argument("--out", default=os.path.expanduser("~/em-runs/qwen2.5-math-7b"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    import bitsandbytes as bnb
    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    problem = json.loads(open(a.example).read().splitlines()[0])["problem"]
    pin = tok(build_prompt(problem, "paper"), return_tensors="pt").input_ids.to("cuda")
    P = pin.shape[1]
    eos = tok.eos_token_id

    for step in range(1, a.steps + 1):
        # 1) batched rollout — no grad, KV cache on for speed
        model.config.use_cache = True
        with torch.no_grad():
            gen = model.generate(pin.repeat(a.batch, 1),
                                 do_sample=True, temperature=a.sample_temp, top_p=0.95,
                                 repetition_penalty=1.15, max_new_tokens=a.max_new,
                                 pad_token_id=tok.pad_token_id)
        model.config.use_cache = False

        # 2) grad-accumulate entropy over each rollout (micro-batch 1)
        opt.zero_grad()
        step_loss = 0.0
        for i in range(a.batch):
            cont = gen[i, P:].tolist()
            if eos in cont:                                  # trim everything after first EOS
                cont = cont[:cont.index(eos) + 1]
            if not cont:
                continue
            seq = torch.tensor([pin[0].tolist() + cont], device="cuda")
            logits = model(seq).logits[0]                    # [T,V] with grad
            gen_mask = torch.zeros(seq.shape[1], device="cuda")
            gen_mask[P:] = 1                                  # generated positions only
            # entropy at position t predicts token t+1 -> align mask to predictions
            loss = entropy_loss(logits[:-1], gen_mask[1:], temp=a.temp) / a.batch
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        # save only the checkpoints we evaluate (full 7B each ~14GB) — step 5 early ref,
        # step 10 headline, final step for the collapse read. LoRA saves the tiny adapter
        # (T7 merges it before eval if the memory gate forced the LoRA fallback).
        keep = step in (5, 10) or step == a.steps
        msg = f"step {step}: entropy~{step_loss:.4f}"
        if keep:
            d = f"{a.out}/step_{step}"
            os.makedirs(d, exist_ok=True)
            model.save_pretrained(d)                          # full model, or adapter if --lora
            tok.save_pretrained(d)
            msg += f"  saved {d}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
