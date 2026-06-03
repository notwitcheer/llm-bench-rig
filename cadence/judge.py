#!/usr/bin/env python3
"""Production slop-judge tool. Donald calls this on a draft; it scores via the live model
(thinking OFF, so the JSON always lands) and applies the deterministic threshold.

Usage:  python3 judge.py <draft-file>        # or pipe the draft on stdin
Prints the gate decision JSON. Exit: 0 = pass, 1 = fail, 2 = unscored (judge unavailable → degrade open)."""
import json
import sys
import urllib.request

HERE = "/home/witcheer/.hermes/skills/content/slop-judge"
sys.path.insert(0, HERE)
from threshold import parse_judge_output, decide  # noqa: E402

SKILL = open(f"{HERE}/SKILL.md").read()
GOLD = open("/home/witcheer/.hermes/vault/cadence/gold-standard-posts.md").read()
# Our verified numbers — the judge needs these in-context to catch ungrounded/contradicting claims.
try:
    FINDINGS = open("/home/witcheer/.hermes/vault/findings/5090-treatments.md").read()
except OSError:
    FINDINGS = "(no findings file found)"
SYSTEM = (SKILL + "\n\n=== GOLD STANDARD (calibration) ===\n" + GOLD
          + "\n\n=== OUR VERIFIED FINDINGS (every number in the draft MUST match these) ===\n" + FINDINGS)


def judge(draft: str) -> dict:
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Score this draft:\n\n" + draft}],
        "temperature": 0, "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False},  # rubric-scoring needs no reasoning
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    content = resp["choices"][0]["message"].get("content") or ""
    parsed = parse_judge_output(content)
    d = decide(parsed)
    d["criteria"] = parsed.get("criteria", {})  # include per-criterion detail for rework
    return d


def main():
    draft = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    try:
        d = judge(draft)
    except Exception as e:  # model/server unavailable → degrade open
        print(json.dumps({"pass": False, "valid": False, "unscored": True, "error": str(e)[:200]}))
        sys.exit(2)
    print(json.dumps(d, indent=2))
    sys.exit(0 if d.get("pass") else (2 if not d.get("valid") else 1))


if __name__ == "__main__":
    main()
