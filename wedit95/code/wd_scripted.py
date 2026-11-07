#!/usr/bin/env python3
"""wd_scripted.py: arm B, the scripted pipeline (no language model in the loop). Written from the replication repo's stage-2 'ours' diagnosis (KidIkaros/wsp-reproduction src/driver2.py, read 2026-11-06):
  d_act = mean MLP intermediate over looping contexts - mean over productive contexts, at layers 15 and 19 (the layers are the replication's, taken from the paper's released edit: the editor is NOT given them);
  d_res = W_down @ d_act ; d = d_res/|d_res| ; neurons = top 64 by |d_act| ; strength 6 (the form of the published edit).
It reads the same loop_mean/prod_mean tables the editor's tool serves, so both arms see the same evidence. Writes spec.json (obeying the sandbox limits) for wd_edit.
usage: wd_scripted.py --tables DIR --out spec.json [--layers 15,19] [--top 64] [--strength 6]
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wd_edit as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="15,19")
ap.add_argument("--top", type=int, default=64)
ap.add_argument("--strength", type=float, default=6.0)
ap.add_argument("--base", default=E.BASE_GGUF)
a = ap.parse_args()

sys.path.insert(0, os.path.expanduser("~/llama.cpp-qwen4/gguf-py"))
import gguf  # noqa: E402

loop = np.load(os.path.join(a.tables, "loop_mean.npy")).astype(np.float64)
prod = np.load(os.path.join(a.tables, "prod_mean.npy")).astype(np.float64)
r = gguf.GGUFReader(a.base)
tens = {t.name: t for t in r.tensors}
edits, info = [], []
for L in [int(x) for x in a.layers.split(",")]:
    W = E.bf16_to_f32(np.asarray(tens["blk.%d.ffn_down.weight" % L].data).view(np.uint16).reshape(E.D_MODEL, E.D_FF)).astype(np.float64)
    d_act = loop[L] - prod[L]
    d_res = W @ d_act
    d = d_res / np.linalg.norm(d_res)
    neurons = np.argsort(-np.abs(d_act))[:a.top]
    edits.append(dict(layer=L, direction=[float(x) for x in d], neurons=[int(x) for x in neurons], strength=a.strength))
    info.append(dict(layer=L, d_act_norm=float(np.linalg.norm(d_act)), top_neurons=[int(x) for x in neurons[:8]]))
json.dump(dict(edits=edits, note="arm B scripted pipeline", info=info), open(a.out, "w"))
print("WD_SCRIPTED ok", json.dumps(info))
