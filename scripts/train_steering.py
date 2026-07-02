"""Bounded RLOO trainer for bias-only steering (t074 — authors' stack is sm_120-blocked,
so this reimplements their recipe on the rig's stack; also the timing instrument).

Recipe pinned from corl-team/steering-reasoning configs (qwen2.5-math-7b/deepscaler):
steering = trainable bias on EVERY layer's mlp.down_proj (steering.yml: lr 1e-3,
steering_at_layer null); lora = THEIR baseline config verbatim (lora.yml +
policy_model.py:422: r=4, alpha=4, dropout 0, target ["down_proj"] only, lr 1e-4);
both: RLOO reward processing, temp 1.0, top_p 1.0, max_grad_norm 2, qwen_math template
(= lib.em build_prompt "paper" variant). Bounded deviations (disclosed in the report):
prompts/step + generations/prompt + max-new below their 16/16/4096; steps << 1 epoch.
Gradient checkpointing stays OFF for steering (their config: "steering will not work
with True") -> small teacher-force microbatch instead; step-1 grad-norm print is the
is-it-training guard. --reward random = Spurious-Rewards protocol (coin-flip rewards).

Emits STEER-TIMING lines (lib.steering.parse_timing schema).

Run (capsule, Donald drained, tmux):
  HF_HUB_OFFLINE=1 PYTHONPATH=$PWD ~/unsloth-env/bin/python scripts/train_steering.py \
    --model ~/models/Qwen2.5-Math-7B --adapter steering --steps 20
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.em import build_prompt
from lib.qwen_math_grader import extract_answer, grade


def attach_down_bias(model):
    biases = []
    for layer in model.model.layers:
        dp = layer.mlp.down_proj
        b = torch.nn.Parameter(torch.zeros(dp.out_features, dtype=dp.weight.dtype,
                                           device=dp.weight.device))
        dp.bias = b
        biases.append(b)
    for p in model.parameters():
        p.requires_grad_(False)
    for b in biases:
        b.requires_grad_(True)
    return biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", choices=["steering", "lora"], default="steering")
    ap.add_argument("--reward", choices=["grader", "random"], default="grader")
    ap.add_argument("--data", default="dataset/steering/train512.jsonl")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--prompts", type=int, default=8)        # P (theirs 16; bounded)
    ap.add_argument("--k", type=int, default=8)              # RLOO gens/prompt (theirs 16)
    ap.add_argument("--max-new", type=int, default=1024)     # theirs 4096; bounded
    ap.add_argument("--lr", type=float, default=None)        # default per adapter (theirs)
    ap.add_argument("--micro", type=int, default=2)          # teacher-force microbatch
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/steering-runs/run"))
    a = ap.parse_args()
    if a.lr is None:
        a.lr = 1e-3 if a.adapter == "steering" else 1e-4     # their steering.yml / lora.yml
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng_reward = random.Random(a.seed + 1)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if a.adapter == "steering":
        params = attach_down_bias(model)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=4, lora_alpha=4, lora_dropout=0.0, target_modules=["down_proj"],
            init_lora_weights=True))                          # their policy_model.py:422
        model.print_trainable_parameters()
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    rows = [json.loads(l) for l in open(a.data)]
    os.makedirs(a.out, exist_ok=True)
    log = []
    for step in range(1, a.steps + 1):
        lo = ((step - 1) * a.prompts) % len(rows)
        batch = rows[lo:lo + a.prompts]
        prompts = [build_prompt(r["problem"], "paper") for r in batch]

        t0 = time.time()                                  # --- rollout ---
        model.config.use_cache = True
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                 num_return_sequences=a.k, max_new_tokens=a.max_new,
                                 pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
        t1 = time.time()                                  # --- grade ---
        rewards = []
        for i, txt in enumerate(texts):
            gold = batch[i // a.k]["answer"]
            if a.reward == "random":
                rewards.append(float(rng_reward.random() < 0.5))
            else:
                rewards.append(float(grade(extract_answer(txt), gold)))
        t2 = time.time()                                  # --- update (RLOO) ---
        model.config.use_cache = False
        adv = []
        for p_i in range(len(batch)):
            grp = rewards[p_i * a.k:(p_i + 1) * a.k]
            for j, r in enumerate(grp):
                others = (sum(grp) - r) / (a.k - 1)
                adv.append(r - others)                     # leave-one-out baseline
        opt.zero_grad()
        total = 0.0
        for s in range(0, gen.shape[0], a.micro):
            ids = gen[s:s + a.micro]
            att = (ids != tok.pad_token_id).long()
            out = model(input_ids=ids, attention_mask=att)
            lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            genmask = att[:, 1:].clone()
            genmask[:, : plen - 1] = 0
            seq_lp = (tok_lp * genmask).sum(-1)
            a_t = torch.tensor(adv[s:s + a.micro], device="cuda")
            loss = -(a_t * seq_lp).sum() / gen.shape[0]
            loss.backward()
            total += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)  # their max_grad_norm
        opt.step()
        t3 = time.time()
        acc = sum(rewards) / len(rewards)
        print(f"STEER-TIMING step={step} rollout_s={t1-t0:.1f} grade_s={t2-t1:.1f} "
              f"update_s={t3-t2:.1f}", flush=True)
        print(f"step {step}: mean_reward={acc:.3f} loss={total:.4f} "
              f"grad_norm={float(gnorm):.4f}", flush=True)
        log.append({"step": step, "mean_reward": acc, "loss": total,
                    "grad_norm": float(gnorm), "rollout_s": t1 - t0,
                    "grade_s": t2 - t1, "update_s": t3 - t2})
        if a.adapter == "steering":
            torch.save({i: b.detach().cpu() for i, b in enumerate(params)},
                       f"{a.out}/vectors.pt")
        else:
            model.save_pretrained(f"{a.out}/adapter")
        json.dump(log, open(f"{a.out}/train_log.json", "w"))


if __name__ == "__main__":
    main()
