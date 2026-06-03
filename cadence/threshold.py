"""Deterministic gate decision from the slop-judge's JSON output. No model in this layer."""
import json
import re

REQUIRED = ["ev_plus", "sourced", "finding", "voice", "meta"]
THRESHOLD = 0.7


def parse_judge_output(text: str) -> dict:
    """Extract the judge's JSON from a model response (handles ```json fences / surrounding prose).
    Returns {} if no valid JSON object is found (caller degrades open)."""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        s, e = text.find("{"), text.rfind("}")
        candidate = text[s:e + 1] if s != -1 and e > s else None
    if not candidate:
        return {}
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def decide(judge_json: dict, threshold: float = THRESHOLD) -> dict:
    """Apply the gate rule. pass = every required criterion >= threshold (meta included, so it is a
    hard gate). Missing/malformed criteria -> not valid (caller degrades open)."""
    crits = (judge_json or {}).get("criteria", {})
    for c in REQUIRED:
        if not isinstance(crits.get(c), dict) or "score" not in crits[c]:
            return {"pass": False, "overall": 0.0, "fail_reasons": [f"missing criterion: {c}"], "valid": False}
    fail_reasons = [f"{c} {crits[c]['score']:.2f}: {crits[c].get('reason', '')}"
                    for c in REQUIRED if crits[c]["score"] < threshold]
    overall = round(sum(crits[c]["score"] for c in REQUIRED) / len(REQUIRED), 3)
    return {"pass": len(fail_reasons) == 0, "overall": overall, "fail_reasons": fail_reasons, "valid": True}
