#!/usr/bin/env python3
"""wd_acts.py: forward passes of Qwen3.5-0.8B (HF transformers, base weights) capturing the INPUT of every layer's mlp.down_proj (the 3584-wide MLP intermediate).
usage: wd_acts.py tables --dev-rows rows.json --out DIR [--device cuda|cpu] [--loop-tail 2500] [--prod-head 1500] [--max-ctx 4096]
          -> DIR/loop_mean.npy, DIR/prod_mean.npy  (24 x 3584 float32: mean activation over the TAIL of looping thinking vs the HEAD of every thinking), DIR/tables.json
       wd_acts.py text --file f.txt --layer L --out out.npy [--device cpu]   (ad hoc, <= 1500 tokens, mean over positions of the layer-L intermediate; used by the sandbox tool)
Contexts are the chat prompt + the model's own thinking text (re-tokenised), the same way the replication's diagnosis fed transcripts back in.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE_DIR = os.environ.get("WD_BASE_DIR", os.path.expanduser("~/voxel-stage/wedit95/base"))
N_LAYERS = 24


def load(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    m = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    m.eval().to(device)
    return tok, m


def hooks(m):
    cap = {}
    hs = []
    mods = {n: mod for n, mod in m.named_modules() if n.endswith("mlp.down_proj")}
    assert len(mods) == N_LAYERS, len(mods)
    for n, mod in mods.items():
        L = int(n.split(".")[2])
        def mk(L):
            def h(module, args, out):
                cap[L] = args[0].detach().float().sum((0, 1)).cpu().numpy()
                cap[("n", L)] = args[0].shape[1]
            return h
        hs.append(mod.register_forward_hook(mk(L)))
    return cap, hs


def ids_for(tok, problem_text, think_text, device, window, tail):
    from wd_common import SUFFIX
    prompt = tok.apply_chat_template([{"role": "user", "content": problem_text + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(prompt + think_text).input_ids
    ids = ids[-window:] if tail else ids[:window]
    import torch
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


def pick_rows(rows, n_loop=6, n_prod=4):
    """THE evidence selection, shared by the tool service (transcripts) and the scripted arm (tables): the first n_loop rows (problem order) whose thinking never closed, and the first n_prod rows with any thinking text."""
    loops = [r for r in rows if r["think"].strip() and not r["closed"]][:n_loop]
    prods = [r for r in rows if r["think"].strip()][:n_prod]
    return loops, prods


def tables(rows_path, outdir, device, loop_tail, prod_head, max_ctx):
    import torch
    import wd_common as C
    data = json.load(open(rows_path))
    probs = {p["id"]: p for p in C.problems("dev")}
    loops, prods = pick_rows(data["rows"])
    tok, m = load(device)
    cap, hs = hooks(m)
    sums = {"loop": np.zeros((N_LAYERS, C.D_FF)), "prod": np.zeros((N_LAYERS, C.D_FF))}
    cnt = {"loop": 0, "prod": 0}
    used = {"loop": [r["id"] for r in loops], "prod": [r["id"] for r in prods]}
    t0 = time.time()
    for kind, sel, window, tail in (("loop", loops, loop_tail, True), ("prod", prods, prod_head, False)):
        for r in sel:
            ids = ids_for(tok, probs[r["id"]]["problem"], r["think"], device, min(window + 400, max_ctx), tail)
            with torch.no_grad():
                m(ids)
            n = cap[("n", 0)]
            for L in range(N_LAYERS):
                sums[kind][L] += cap[L]
            cnt[kind] += n
    for h in hs:
        h.remove()
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "loop_mean.npy"), (sums["loop"] / max(cnt["loop"], 1)).astype(np.float32))
    np.save(os.path.join(outdir, "prod_mean.npy"), (sums["prod"] / max(cnt["prod"], 1)).astype(np.float32))
    json.dump(dict(tokens=cnt, problems=used, seconds=round(time.time() - t0, 1), device=device, loop_tail=loop_tail, prod_head=prod_head), open(os.path.join(outdir, "tables.json"), "w"), indent=1)
    print("WD_ACTS tables ok: loop tokens %d (%d texts), prod tokens %d (%d texts), %.0f s on %s" % (cnt["loop"], len(loops), cnt["prod"], len(prods), time.time() - t0, device))


def text_mode(path, layer, out, device):
    """mean MLP intermediate per layer over the positions of a text (<= 1500 tokens). layer -1 = all 24 layers -> (24, 3584)."""
    import torch
    tok, m = load(device)
    cap, hs = hooks(m)
    ids = tok(open(path).read()).input_ids[:1500]
    with torch.no_grad():
        m(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    if layer < 0:
        arr = np.stack([cap[L] / cap[("n", L)] for L in range(N_LAYERS)]).astype(np.float32)
    else:
        arr = (cap[layer] / cap[("n", layer)]).astype(np.float32)
    np.save(out, arr)
    print("WD_ACTS text ok: %d tokens, saved %s" % (len(ids), "x".join(str(x) for x in arr.shape)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tables", "text"])
    ap.add_argument("--dev-rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loop-tail", type=int, default=2500)
    ap.add_argument("--prod-head", type=int, default=1500)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--file")
    ap.add_argument("--layer", type=int, default=-1)
    a = ap.parse_args()
    if a.mode == "tables":
        tables(a.dev_rows, a.out, a.device, a.loop_tail, a.prod_head, a.max_ctx)
    else:
        text_mode(a.file, a.layer, a.out, a.device)
