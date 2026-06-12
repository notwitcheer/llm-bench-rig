"""t045 diagnostic: localize the garbage. One forward pass, hook every decoder
layer, print hidden-state norms + top-5 next-token predictions.
Run DRAINED: ~/keye-env/bin/python scripts/keye_diag.py --model ~/keye-local"""
import argparse
import torch
from keye_probe import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/witcheer/keye-local")
args = ap.parse_args()

import keye_probe
keye_probe.MODEL = args.model
model, proc = load("bf16")

tok = proc.tokenizer
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

norms = {}


def mk_hook(i):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        norms[i] = (h.float().norm(dim=-1).mean().item(),
                    h.float().abs().max().item())
    return hook


for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(mk_hook(i))

with torch.no_grad():
    out = model(input_ids=ids)

for i in sorted(norms):
    n, m = norms[i]
    flag = " <<<" if (m > 1e3 or n != n or m != m) else ""
    print(f"layer {i:2d}: mean_norm={n:10.2f} absmax={m:10.2f}{flag}", flush=True)

logits = out.logits[0, -1].float()
top = torch.topk(logits, 5)
print("\ntop-5 next tokens:")
for v, ix in zip(top.values, top.indices):
    print(f"  {v.item():8.3f}  {tok.decode([ix])!r}")
