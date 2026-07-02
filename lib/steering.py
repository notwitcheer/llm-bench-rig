"""Bias-only steering primitives (t074, arXiv 2505.18706). Pure logic, tested on the Mac;
GPU drivers only orchestrate (ADR-0003). rebadge_config turns a Qwen2.5 config into a
LlamaForCausalLM config with mlp_bias=True so a checkpoint carrying learned down_proj
biases loads in STOCK transformers/vLLM: Qwen2.5 is Llama-shaped, q/k/v biases exist in
both (attention_bias=True), and the extra o/gate/up biases are zero-filled — identical math."""
from __future__ import annotations

_KEEP = ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers",
         "num_key_value_heads", "rms_norm_eps", "rope_theta", "vocab_size",
         "max_position_embeddings", "tie_word_embeddings", "torch_dtype", "hidden_act",
         "bos_token_id", "eos_token_id", "initializer_range")


def pick_budget(step_seconds: float, cap_minutes: float = 40.0,
                floor: int = 5, ceil: int = 30) -> int:
    """Steps that fit the per-arm wall-clock cap, clamped to [floor, ceil]."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    return max(floor, min(ceil, int((cap_minutes * 60) // step_seconds)))


def rebadge_config(qwen_cfg: dict) -> dict:
    """Qwen2 config dict -> stock-Llama config dict (mlp_bias holds the steering vector)."""
    cfg = {k: qwen_cfg[k] for k in _KEEP}   # KeyError on a missing field is the guard
    cfg.update({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "attention_bias": True, "mlp_bias": True, "attention_dropout": 0.0,
        "pretraining_tp": 1, "rope_scaling": None, "use_cache": True,
    })
    return cfg


def parse_timing(lines: list[str]) -> list[dict]:
    """Parse 'STEER-TIMING k=v ...' lines emitted by scripts/train_steering.py."""
    out = []
    for ln in lines:
        if not ln.startswith("STEER-TIMING "):
            continue
        rec = {}
        for kv in ln.split()[1:]:
            k, v = kv.split("=")
            rec[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        out.append(rec)
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def timing_summary(recs: list[dict]) -> dict:
    """Median per-phase seconds + wall-clock shares (the '34s' decomposition)."""
    r = _median([x["rollout_s"] for x in recs])
    g = _median([x["grade_s"] for x in recs])
    u = _median([x["update_s"] for x in recs])
    tot = r + g + u
    return {"rollout_s": r, "grade_s": g, "update_s": u,
            "rollout_share": r / tot, "grade_share": g / tot, "update_share": u / tot}
