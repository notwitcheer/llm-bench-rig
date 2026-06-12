"""t045 Keye-VL-2.0-30B-A3B probe. Runs on capsule in ~/keye-env, DRAINED.

  quantcheck — do bnb-4bit Linear swaps reach the FUSED experts (3D nn.Parameter)?
               This decides whether ANY consumer quant path exists. Reports the
               expert module class/dtype + total footprint.
  smoke      — load (bnb4 if quantcheck passed, else bf16 GPU+CPU offload),
               text-only gen, then a synthetic 60s video QA with known answer.

  ~/keye-env/bin/python scripts/keye_probe.py quantcheck
  ~/keye-env/bin/python scripts/keye_probe.py smoke --load bnb4|bf16
"""
import argparse, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL = "Kwai-Keye/Keye-VL-2.0-30B-A3B"
SKIP = ["visual", "mlp_AR", "lm_head", "sa_indexer", "mlp.gate"]


def _default_rope(config=None, device=None, seq_len=None, **kw):
    base = float(getattr(config, "rope_theta", 10000.0))
    dim = (getattr(config, "head_dim", None)
           or config.hidden_size // config.num_attention_heads)
    dim = int(dim * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                            device=device) / dim))
    return inv_freq, 1.0


def _shims():
    # Keye's modeling file targets a transformers git snapshot between 4.57 and
    # 5.0 (needs 5.0's factory check_model_inputs AND 4.x's SlidingWindowCache /
    # ROPE_INIT_FUNCTIONS['default'] / config.pad_token_id / rotary classes
    # without 5.0's compute_default_rope_parameters). Pin 5.0.0 + these.
    import transformers.cache_utils as cu
    from transformers import modeling_rope_utils as mru
    for name in ("SlidingWindowCache", "StaticCache"):
        if not hasattr(cu, name):
            setattr(cu, name, type(name, (), {}))
    mru.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)


def _patch_rotary_classes():
    # 5.0's _init_weights calls module.compute_default_rope_parameters(config)
    # on every *RotaryEmbedding* module; Keye's 4.x-style classes lack it.
    import inspect, sys
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    get_class_from_dynamic_module(
        "modeling_keye_topk_mask_30ba3b.KeyeVL2MoeForConditionalGeneration", MODEL)

    def _cdrp(self, config=None, device=None, seq_len=None, **kw):
        cfg = config if config is not None else getattr(self, "config", None)
        return _default_rope(cfg, device, seq_len)

    for name, mod in list(sys.modules.items()):
        if "transformers_modules" in name and "eye" in name:
            for cname, c in inspect.getmembers(mod, inspect.isclass):
                if ("RotaryEmbedding" in cname
                        and not hasattr(c, "compute_default_rope_parameters")):
                    c.compute_default_rope_parameters = _cdrp


def load(mode):
    _shims()
    _patch_rotary_classes()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    for c in [cfg] + [getattr(cfg, n) for n in ("text_config", "vision_config")
                      if hasattr(cfg, n)]:
        if not hasattr(c, "pad_token_id"):
            c.pad_token_id = None
    kw = dict(config=cfg, trust_remote_code=True, low_cpu_mem_usage=True,
              attn_implementation="sdpa",
              torch_dtype=torch.bfloat16, device_map="auto",
              max_memory={0: "28GiB", "cpu": "42GiB"})
    if mode == "bnb4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=SKIP)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    return model, proc


def quant_verdict(model):
    layer = model.model.layers[0]
    experts = layer.mlp.experts
    gup = getattr(experts, "gate_up_proj", None)
    qproj = layer.self_attn.q_proj
    print(json.dumps({
        "experts_class": type(experts).__name__,
        "gate_up_proj_type": type(gup).__name__,
        "gate_up_proj_dtype": str(getattr(gup, "dtype", None)),
        "gate_up_proj_shape": list(getattr(gup, "shape", [])),
        "q_proj_class": type(qproj).__name__,  # Linear4bit = bnb reached Linears
        "footprint_gb": round(model.get_memory_footprint() / 2**30, 1),
        "device_map_sample": {k: str(v) for k, v in
                              list(model.hf_device_map.items())[:5]}
        if hasattr(model, "hf_device_map") else None,
    }, indent=2), flush=True)


def make_video(seconds=60, fps=2, w=448, h=448):
    """Synthetic video: colored blocks, big text label changes every 10s."""
    import cv2
    colors = [(220, 60, 60), (60, 220, 60), (60, 60, 220),
              (220, 220, 60), (220, 60, 220), (60, 220, 220)]
    labels = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]
    frames, ts = [], []
    for i in range(int(seconds * fps)):
        sec = i / fps
        k = int(sec // 10) % 6
        img = np.full((h, w, 3), colors[k], dtype=np.uint8)
        cv2.putText(img, labels[k], (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, (255, 255, 255), 6)
        cv2.putText(img, f"t={sec:.0f}s", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 3)
        frames.append(img)
        ts.append(sec)
    return np.stack(frames), ts


def gen(model, proc, messages, videos=None, max_new=128, **vkw):
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], videos=videos, return_tensors="pt", **vkw)
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_in = inputs["input_ids"].shape[1]
    n_new = out.shape[1] - n_in
    txt = proc.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
    return txt, n_in, n_new, dt


def smoke(model, proc):
    msgs = [{"role": "user", "content": [{"type": "text",
             "text": "In one sentence: what does multi-token prediction speed up?"}]}]
    txt, n_in, n_new, dt = gen(model, proc, msgs, max_new=64)
    print(f"\n--- TEXT ({n_in} in, {n_new} tok in {dt:.1f}s = {n_new/dt:.1f} tok/s)\n{txt}",
          flush=True)

    # 20s @ 1fps keeps the DSA indexer's O(N^2) score matrix inside VRAM
    # (60s @ 2fps -> ~32K tokens -> a single 30.65GiB allocation; measured).
    video, ts = make_video(seconds=20, fps=1)
    msgs = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": "What word is shown on screen at around 15 seconds, "
                                 "and what color is the background then?"}]}]
    txt, n_in, n_new, dt = gen(model, proc, msgs, videos=[video], max_new=128,
                               fps=1.0, timestamps=[ts])
    print(f"\n--- VIDEO 20s/{len(ts)}f ({n_in} in, {n_new} tok in {dt:.1f}s)\n{txt}",
          flush=True)
    print(f"expected: BRAVO on green  | peak VRAM "
          f"{torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["quantcheck", "smoke"])
    ap.add_argument("--load", default="bnb4", choices=["bnb4", "bf16"])
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model
    if args.task == "quantcheck":
        model, _ = load("bnb4")
        quant_verdict(model)
    else:
        model, proc = load(args.load)
        quant_verdict(model)
        smoke(model, proc)
