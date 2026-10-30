"""nerve-gate smoke: load gpt-oss-20b via the paper's loader (cape.models.load), capture last-token residuals at 5 depths,
time a ~1.6k-token prefill (the paper's p50 prompt is 1583 tokens, results/cost/bench_agent_gptoss20b.json)."""
import json, os, sys, time, traceback
os.environ["HF_HUB_OFFLINE"] = "0"          # their loader setdefaults it to 1; the MXFP4 kernel is fetched from the hub
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
sys.path.insert(0, os.path.expanduser("~/esoteric/Honesty/src"))
import torch, transformers
from transformers.utils import is_kernels_available

OUT = os.path.expanduser("~/esoteric/nerve-gate/smoke.json")
r = {"transformers": transformers.__version__, "torch": torch.__version__, "kernels_available": is_kernels_available(),
     "gpu": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0))}
def dump():
    json.dump(r, open(OUT, "w"), indent=1); print(json.dumps(r, indent=1))
try:
    from cape.models.load import load_hf_model, discover_transformer_blocks
    cfg = {"local_path": os.path.expanduser("~/models/gpt-oss-20b")}
    t = time.time()
    for attn in ("sdpa", "eager"):
        try:
            model = load_hf_model(cfg, attn_implementation=attn); r["attn"] = attn; break
        except Exception as e:
            r[f"load_error_{attn}"] = repr(e)[:400]
    else:
        raise RuntimeError("model did not load with sdpa or eager")
    torch.cuda.synchronize(); r["load_s"] = round(time.time() - t, 1)
    hq = getattr(model, "hf_quantizer", None)
    r["dequantized_to_bf16"] = bool(hq and hq.quantization_config.dequantize)
    r["vram_alloc_gb_after_load"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    blocks, types = discover_transformer_blocks(model, 24)
    r["expert_module"] = type(getattr(blocks[0].mlp, "experts", blocks[0].mlp)).__name__
    layers = [4, 8, 12, 16, 20]; cap = {}
    def mk(l):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[l] = h[:, -1, :].float().cpu()
        return fn
    hs = [blocks[l].register_forward_hook(mk(l)) for l in layers]
    tok = transformers.AutoTokenizer.from_pretrained(cfg["local_path"])
    text = ("The agent may only call transfer_funds after the account owner confirms in writing. " * 200)
    ids = tok(text, return_tensors="pt").input_ids[:, :1583].to("cuda:0")
    with torch.no_grad():
        model(ids); torch.cuda.synchronize()            # warmup
        ts = []
        for _ in range(3):
            t = time.time(); model(ids); torch.cuda.synchronize(); ts.append(time.time() - t)
    r["prefill_1583_s"] = [round(x, 4) for x in ts]
    r["captured"] = {str(l): [list(v.shape), bool(torch.isfinite(v).all())] for l, v in cap.items()}
    feat = torch.cat([cap[l] for l in layers], dim=-1).numpy()
    import numpy as np
    w = np.random.randn(feat.shape[-1]).astype(np.float32)
    t = time.perf_counter()
    for _ in range(1000): float((feat @ w)[0])
    r["readout_cpu_s"] = (time.perf_counter() - t) / 1000
    r["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    for h in hs: h.remove()
    r["verdict"] = "FITS" if not r["dequantized_to_bf16"] else "LOADED_BUT_DEQUANTISED"
except Exception as e:
    r["error"] = repr(e)[:800]; r["trace"] = traceback.format_exc()[-1500:]; r["verdict"] = "FAILED"
dump()
