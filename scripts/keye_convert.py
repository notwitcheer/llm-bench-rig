"""t045: re-layout Keye-VL-2.0 fused-expert tensors for transformers 5.0.
The checkpoint stores experts as (E, in, out) [pre-5.0 layout]; 5.0's
Qwen3VLMoeTextExperts wants (E, out, in) and applies F.linear per expert.
Streams the 13 shards, transposes mlp.experts.{gate_up_proj,down_proj},
copies everything else verbatim to the output dir.

  ~/keye-env/bin/python scripts/keye_convert.py ~/keye-local
"""
import glob, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Kwai-Keye--Keye-VL-2.0-30B-A3B/snapshots/*")
dst = os.path.expanduser(sys.argv[1])
src = sorted(glob.glob(SRC_GLOB))[-1]
os.makedirs(dst, exist_ok=True)

for name in os.listdir(src):
    p = os.path.join(src, name)
    if name.endswith(".safetensors"):
        continue
    real = os.path.realpath(p)
    if os.path.isdir(real):
        shutil.copytree(real, os.path.join(dst, name), dirs_exist_ok=True)
    else:
        shutil.copy(real, os.path.join(dst, name))
    print("copied", name, flush=True)

for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    out = os.path.join(dst, os.path.basename(shard))
    tensors, n_t = {}, 0
    with safe_open(os.path.realpath(shard), framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if key.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
                t = t.transpose(-1, -2).contiguous()
                n_t += 1
            tensors[key] = t
    save_file(tensors, out, metadata={"format": "pt"})
    print(f"{os.path.basename(shard)}: {n_t} transposed", flush=True)

print("CONVERT_DONE", dst)
