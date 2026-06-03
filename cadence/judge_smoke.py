#!/usr/bin/env python3
"""Smoke-test the slop-judge against the LIVE Hermes server (Qwen3.6-27B, read-only inference).
A gold post must PASS (>=0.7 all, meta hard gate); a hollow/hype draft must FAIL.
Run on capsule from ~/.hermes/skills/content/slop-judge/ (imports the co-located threshold.py)."""
import json
import sys
import urllib.request

sys.path.insert(0, ".")
from threshold import parse_judge_output, decide  # noqa: E402

SKILL = open("SKILL.md").read()
GOLD = open("/home/witcheer/.hermes/vault/cadence/gold-standard-posts.md").read()
SYSTEM = SKILL + "\n\n=== GOLD STANDARD (calibration) ===\n" + GOLD

# A HELD-OUT good draft (NOT a gold post — gold posts trip the judge's duplicate/non-novelty check).
# Fresh, on-brand: EV+ mechanism, a measured number, a genuine finding, in-register.
GOOD_DRAFT = (
    "quick one on llama.cpp's --n-cpu-moe: it's not all-or-nothing.\n\n"
    "i swept it on gpt-oss-120B on the 5090. at --n-cpu-moe 20 it sits at ~30GB of 32GB VRAM and "
    "47 tok/s generation. pull more experts back onto the GPU and VRAM climbs toward the 32GB ceiling, "
    "but generation stays ~47 - the active path (5.1B params) is already on-card; what's in RAM is the "
    "idle experts.\n\n"
    "so the knob trades VRAM headroom, not tok/s. tune it to fill the card, not to chase speed."
)

SLOP_DRAFT = ("Big news! The AI landscape is evolving fast and local models are more powerful than "
              "ever. This model is a true game-changer that pushes boundaries and unlocks incredible "
              "new possibilities for everyone. The future of local AI is brighter than ever!")


def judge(draft: str) -> dict:
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Score this draft:\n\n" + draft}],
        "temperature": 0, "max_tokens": 1500,
        # judge scores against an explicit rubric — no extended reasoning needed; thinking-on
        # blows the token budget before the JSON is emitted (content empty, finish_reason=length).
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    content = resp["choices"][0]["message"].get("content") or ""
    return decide(parse_judge_output(content))


print("=== GOOD (gold REAP post) — expect pass=True ===")
g = judge(GOOD_DRAFT)
print(g)
print("\n=== SLOP (hype, hollow) — expect pass=False ===")
s = judge(SLOP_DRAFT)
print(s)

ok = (g.get("pass") is True) and (s.get("pass") is False)
print("\n[smoke]", "PASS — judge discriminates" if ok else "FAIL — judge miscalibrated (tune SKILL.md rubric)")
sys.exit(0 if ok else 1)
