#!/usr/bin/env python3
"""wd_ref.py <edit.npz> <manifest.json> <out.gguf> [--base gguf]: apply the PUBLISHED edit (MIT CSAIL weight-space-programming, p50c: sparse bf16 entries in down_proj of layers 15 and 19; files from KidIkaros/wsp-reproduction materials/)
to a copy of the base GGUF. Each rebuilt tensor is hashed in the safetensors layout (rows x cols, bf16 bytes) and compared with the manifest's edited_sha256 before the GGUF is written: the reference arm is bit-exact or it does not exist."""
import hashlib, json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/llama.cpp-qwen4/gguf-py"))
import wd_edit as E  # noqa: E402


def apply_published(npz, manifest, out, base=E.BASE_GGUF):
    import gguf
    man = json.load(open(manifest))
    z = np.load(npz)
    shutil.copyfile(base, out)
    r = gguf.GGUFReader(out, "r+")
    tens = {t.name: t for t in r.tensors}
    rep = []
    for i, t in enumerate(man["tensors"]):
        L = int(t["name"].split(".layers.")[1].split(".")[0])
        g = tens["blk.%d.ffn_down.weight" % L]
        bits = np.asarray(g.data).view(np.uint16).reshape(-1)
        h0 = hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()
        assert h0 == t["base_sha256"], "layer %d: base tensor differs from the one the edit was made on" % L
        bits[z["t%d_idx" % i]] = z["t%d_val" % i]
        h1 = hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()
        assert h1 == t["edited_sha256"], "layer %d: rebuilt tensor does not match the published edited hash" % L
        rep.append(dict(layer=L, changed=int(len(z["t%d_idx" % i])), sha_ok=True))
    del r
    return rep


if __name__ == "__main__":
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else E.BASE_GGUF
    print(json.dumps(apply_published(sys.argv[1], sys.argv[2], sys.argv[3], base)))
