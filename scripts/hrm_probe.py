"""t047 HRM-Text-1B probe: correct-protocol generation, decode tok/s, KV/VRAM,
and the missing-token_type_ids ablation (the PrefixLM harness trap).
Runs on capsule in ~/hrm-env (transformers>=5.9) — NOT the repo .venv. Standalone.

  ~/hrm-env/bin/python scripts/hrm_probe.py smoke
  ~/hrm-env/bin/python scripts/hrm_probe.py bench
  ~/hrm-env/bin/python scripts/hrm_probe.py ablate
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "sapientinc/HRM-Text-1B"

# condition tokens (model card): output-style selectors prefixed inside im_start
COND = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
    "synth+cot": "<|quad_end|><|object_ref_end|>",  # official snippet default
}

PROMPTS = [
    "Explain why the sky is blue.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "List three reasons the Roman Empire declined.",
]


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def build_inputs(tok, text, cond, prefix_mask=True):
    prompt = f"<|im_start|>{COND[cond]}{text}<|im_end|>"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    if prefix_mask:  # mark the whole prompt as bidirectional PrefixLM prefix
        inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])
    return inputs


def gen(model, tok, inputs, max_new=256, force_len=False):
    kw = dict(max_new_tokens=max_new, do_sample=False)
    if force_len:
        kw["min_new_tokens"] = max_new
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_prompt = inputs["input_ids"].shape[1]
    n_new = out.shape[1] - n_prompt
    text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
    return text, n_new, dt


def smoke(tok, model, conds):
    for p in PROMPTS:
        print(f"\n{'='*70}\nPROMPT: {p}")
        for c in conds:
            text, n, dt = gen(model, tok, build_inputs(tok, p, c), max_new=256)
            print(f"\n--- [{c}] {n} tok in {dt:.1f}s ---\n{text[:700]}")


def bench(tok, model):
    inputs = build_inputs(tok, PROMPTS[0], "synth+cot")
    gen(model, tok, inputs, max_new=32, force_len=True)  # warmup (cuda graphs/JIT)
    torch.cuda.reset_peak_memory_stats()
    runs = []
    for _ in range(3):
        _, n1, t1 = gen(model, tok, inputs, max_new=1, force_len=True)
        _, n256, t256 = gen(model, tok, inputs, max_new=256, force_len=True)
        runs.append((n256 - n1) / (t256 - t1))
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps({
        "decode_tok_s": [round(r, 1) for r in runs],
        "decode_tok_s_best": round(max(runs), 1),
        "weights_gb": round(weights_gb, 2),
        "peak_vram_gb": round(peak_gb, 2),
    }, indent=2))


def kv_probe(tok, model):
    # bytes/token of KV cache: forward a 2048-tok dummy prefix, measure the cache
    ids = torch.full((1, 2048), 100, dtype=torch.long, device="cuda")
    tti = torch.ones_like(ids)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = model(input_ids=ids, token_type_ids=tti, use_cache=True)
    torch.cuda.synchronize()
    delta_mb_per_tok = (torch.cuda.memory_allocated() - base) / 2048 / 2**20
    n_slots = None
    try:
        pkv = out.past_key_values
        n_slots = len(pkv)
    except Exception:
        pass
    print(json.dumps({"kv_mb_per_token_approx": round(delta_mb_per_tok, 3),
                      "cache_slots": n_slots}, indent=2))


def ablate(tok, model):
    # the harness trap: same prompt, PrefixLM mask on vs off
    for p in PROMPTS:
        print(f"\n{'='*70}\nPROMPT: {p}")
        for label, mask in [("WITH token_type_ids", True), ("WITHOUT (causal-only)", False)]:
            text, n, dt = gen(model, tok, build_inputs(tok, p, "synth+cot", prefix_mask=mask),
                              max_new=256)
            print(f"\n--- {label}: {n} tok ---\n{text[:500]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["smoke", "bench", "kv", "ablate", "all"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--conds", default="direct,cot,synth+cot")
    args = ap.parse_args()
    tok, model = load(args.model)
    if args.task in ("smoke", "all"):
        smoke(tok, model, args.conds.split(","))
    if args.task in ("bench", "all"):
        bench(tok, model)
    if args.task in ("kv", "all"):
        kv_probe(tok, model)
    if args.task in ("ablate", "all"):
        ablate(tok, model)
