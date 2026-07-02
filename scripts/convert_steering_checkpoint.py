"""Bake down_proj steering vectors into a stock-loadable checkpoint (t074).
Module surgery adds the learned down biases + zero o/gate/up biases, saves the state
dict (Qwen2/Llama key names are identical), then swaps config.json for the re-badged
LlamaForCausalLM config (lib.steering.rebadge_config). Tokenizer files copied verbatim.

Run (capsule): PYTHONPATH=$PWD ~/unsloth-env/bin/python scripts/convert_steering_checkpoint.py \
  --base ~/models/Qwen2.5-Math-7B --vectors ~/steering-runs/steer/vectors.pt \
  --out ~/models/steer-rebadged
"""
import argparse
import glob
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM

from lib.steering import rebadge_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    vecs = torch.load(a.vectors)
    h = model.config.hidden_size
    inter = model.config.intermediate_size
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.bias = torch.nn.Parameter(vecs[i].to(torch.bfloat16))
        layer.mlp.gate_proj.bias = torch.nn.Parameter(torch.zeros(inter, dtype=torch.bfloat16))
        layer.mlp.up_proj.bias = torch.nn.Parameter(torch.zeros(inter, dtype=torch.bfloat16))
        layer.self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros(h, dtype=torch.bfloat16))
    model.save_pretrained(a.out)
    cfg = json.load(open(os.path.join(a.base, "config.json")))
    json.dump(rebadge_config(cfg), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in glob.glob(os.path.join(a.base, "tokenizer*")) + \
             glob.glob(os.path.join(a.base, "*.jinja")) + \
             [os.path.join(a.base, "generation_config.json")]:
        if os.path.exists(f):
            shutil.copy(f, a.out)
    print(f"re-badged -> {a.out}")


if __name__ == "__main__":
    main()
