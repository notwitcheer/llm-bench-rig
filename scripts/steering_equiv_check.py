"""Gate: re-badged-Llama checkpoint must match surgery-patched Qwen2 logits (t074).
4 prompts, last-position logits: argmax must MATCH on all; max|diff| reported. bf16
threshold 0.5: cross-arch-class kernel dispatch (Qwen2Attention vs LlamaAttention)
reaches ~0.22 on one shape while fp32 on the SAME prompt is exactly 0.0 (verified
2026-07-02) — bf16 reduction-order noise, not math (t072 lesson). If this gate fails,
re-verify in fp32 before concluding anything.

Run (capsule): PYTHONPATH=$PWD ~/unsloth-env/bin/python scripts/steering_equiv_check.py \
  --base ~/models/Qwen2.5-Math-7B --vectors ~/steering-runs/steer/vectors.pt \
  --rebadged ~/models/steer-rebadged
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["What is 17*23?", "Solve x^2-5x+6=0.",
           "A train travels 60km in 45 minutes. Speed in km/h?", "Compute 2^10."]


def last_logits(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        return model(**ids).logits[0, -1].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--rebadged", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    ref = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                               device_map="cuda")
    vecs = torch.load(a.vectors)
    for i, layer in enumerate(ref.model.layers):
        layer.mlp.down_proj.bias = torch.nn.Parameter(
            vecs[i].to(torch.bfloat16).to("cuda"))
    refs = [last_logits(ref, tok, p) for p in PROMPTS]
    del ref
    torch.cuda.empty_cache()
    reb = AutoModelForCausalLM.from_pretrained(a.rebadged, dtype=torch.bfloat16,
                                               device_map="cuda")
    ok = True
    for p, r in zip(PROMPTS, refs):
        l = last_logits(reb, tok, p)
        same = (l.argmax() == r.argmax()).item()
        diff = (l - r).abs().max().item()
        ok &= same and diff < 0.5
        print(f"argmax_match={same} max_abs_diff={diff:.4f}  {p[:40]}")
    print("EQUIV-GATE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
