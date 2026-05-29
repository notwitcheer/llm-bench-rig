"""Aggregate the four agentic evaluator scores into the Hermes Pairing Score."""

def pairing_score(evals: dict, weights: dict) -> float:
    total = 0.0
    for name, w in weights.items():
        s = evals.get(name, {}).get("score", 0) or 0
        total += w * s
    return round(total, 2)

def build_summary(model_slug: str, evals: dict, weights: dict, speed: dict | None = None) -> dict:
    return {
        "model": model_slug,
        "pairing_score": pairing_score(evals, weights),
        "evals": {k: v.get("score", 0) for k, v in evals.items()},
        "instruction_delta": evals.get("instruction", {}).get("delta"),
        "longcontext_by_depth": evals.get("longcontext", {}).get("by_depth"),
        "speed": speed or {},
    }
