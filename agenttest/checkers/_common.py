"""shared helpers for the agenttest checkers. every checker: check(work) -> (passed: bool, reason: str).
expected values are derived from the frozen seed_data/, never from files the agent can edit."""
import csv
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import gen_seeds  # noqa: E402

SEED = gen_seeds.SEED


def seed_text(rel):
    return open(os.path.join(SEED, rel), encoding="utf-8").read()


def read(work, rel):
    p = os.path.join(work, rel)
    if not os.path.isfile(p):
        return None
    return open(p, encoding="utf-8", errors="replace").read()


def one_line(s):
    """content of a file that must hold exactly one line: allow a single trailing newline only."""
    if s is None:
        return None
    if s.endswith("\r\n"):
        s = s[:-2]
    elif s.endswith("\n"):
        s = s[:-1]
    if "\n" in s:
        return None
    return s


def main(check):
    import json
    work = sys.argv[1] if len(sys.argv) > 1 else gen_seeds.default_work()
    ok, why = check(work)
    print(json.dumps({"pass": ok, "reason": why}))
    sys.exit(0 if ok else 1)
