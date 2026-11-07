#!/usr/bin/env python3
"""wd_edit.py: apply a projection edit spec to the 0.8B bf16 GGUF (a COPY; the base file is never touched).
spec json: {"edits": [{"layer": 15, "direction": [1024 floats] | "direction_file": "d.npy", "neurons": [ids < 3584], "strength": 6.0}, ...]}
rule (the form of the published edit, recovered from its weights by the replication repo): for each listed neuron j of layer L's down_proj:  W[:, j] <- W[:, j] - strength * (W[:, j] . d) d,   d = direction / |direction|.
limits (the sandbox enforces them; the scripted arm obeys the same): <= 2 distinct layers, <= 128 neurons per layer, |strength| <= 12, layers 0..23.
bf16 rounding is round-to-nearest-even from float32, as the authors' st_numpy. usage: wd_edit.py <spec.json> <out.gguf> [--base gguf] ; as a module: apply_spec(spec, out_path, base_path) -> report dict.
"""
import json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/llama.cpp-qwen4/gguf-py"))
BASE_GGUF = os.environ.get("WD_BASE_GGUF", os.path.expanduser("~/voxel-stage/wedit95/gguf/base-bf16.gguf"))
MAX_LAYERS, MAX_NEURONS, MAX_STRENGTH, N_LAYERS, D_FF, D_MODEL = 2, 128, 12.0, 24, 3584, 1024


def bf16_bits(x32):
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def validate(spec, specdir="."):
    ed = spec.get("edits")
    if not isinstance(ed, list) or not ed:
        raise ValueError("spec needs a non-empty 'edits' list")
    layers = {int(e["layer"]) for e in ed}
    if len(layers) > MAX_LAYERS:
        raise ValueError("at most %d distinct layers per candidate (got %d)" % (MAX_LAYERS, len(layers)))
    out = []
    for e in ed:
        L = int(e["layer"])
        if not 0 <= L < N_LAYERS:
            raise ValueError("layer %d outside 0..%d" % (L, N_LAYERS - 1))
        nr = sorted({int(j) for j in e["neurons"]})
        if not nr or nr[0] < 0 or nr[-1] >= D_FF:
            raise ValueError("neuron ids must be in 0..%d" % (D_FF - 1))
        if len(nr) > MAX_NEURONS:
            raise ValueError("at most %d neurons per layer (got %d)" % (MAX_NEURONS, len(nr)))
        s = float(e["strength"])
        if abs(s) > MAX_STRENGTH:
            raise ValueError("|strength| <= %g (got %g)" % (MAX_STRENGTH, s))
        if "direction" in e:
            d = np.asarray(e["direction"], dtype=np.float64)
        else:
            p = e["direction_file"]
            p = p if os.path.isabs(p) else os.path.join(specdir, p)
            d = np.load(p).astype(np.float64).reshape(-1)
        if d.shape != (D_MODEL,):
            raise ValueError("direction must have %d entries (got %s)" % (D_MODEL, d.shape))
        nrm = float(np.linalg.norm(d))
        if not np.isfinite(nrm) or nrm < 1e-9:
            raise ValueError("direction has zero or non-finite norm")
        out.append((L, nr, s, d / nrm))
    return out


def apply_spec(spec, out_path, base_path=BASE_GGUF, specdir="."):
    import gguf
    plan = validate(spec, specdir)
    shutil.copyfile(base_path, out_path)
    r = gguf.GGUFReader(out_path, "r+")
    tens = {t.name: t for t in r.tensors}
    rep = []
    for L, nr, s, d in plan:
        t = tens["blk.%d.ffn_down.weight" % L]
        assert t.tensor_type.name == "BF16", t.tensor_type
        bits = np.asarray(t.data).view(np.uint16).reshape(D_MODEL, D_FF)      # same bytes as the safetensors (1024, 3584) tensor: checked at staging
        W = bf16_to_f32(bits).astype(np.float64)
        before = W.copy()
        for j in nr:
            col = W[:, j]
            W[:, j] = col - s * float(col @ d) * d
        newb = bf16_bits(W.astype(np.float32))
        changed = int((newb != bits).sum())
        bits[:] = newb
        rep.append(dict(layer=L, neurons=len(nr), strength=s, changed_entries=changed, rel_change=float(np.linalg.norm(W - before) / np.linalg.norm(before))))
    r.data.flush() if hasattr(r.data, "flush") else None
    del r
    return rep


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else BASE_GGUF
    print(json.dumps(apply_spec(spec, sys.argv[2], base, os.path.dirname(os.path.abspath(sys.argv[1])))))
