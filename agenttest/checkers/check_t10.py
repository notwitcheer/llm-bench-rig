"""t10 summary: exactly 3 lines, each starting "- ", no blank lines, and it names the launch date
(14 november), the budget (48,000 euros) and the migration owner (priya)."""
import re

from _common import main, read

DATE = re.compile(r"(14(st|th)?\s*(of\s*)?nov(ember)?\b)|(nov(ember)?\.?\s*14\b)|(\b14[/.-]11\b)|(2026-11-14)", re.I)
BUDGET = re.compile(r"(48[,. ]?000)|(\b48\s*k\b)|(\u20ac\s*48\s*k)", re.I)
OWNER = re.compile(r"\bpriya\b", re.I)


def check(work):
    s = read(work, "t10/summary.txt")
    if s is None:
        return False, "summary.txt missing"
    body = s[:-1] if s.endswith("\n") else s
    lines = body.split("\n")
    if len(lines) != 3:
        return False, f"{len(lines)} lines"
    for l in lines:
        if not l.startswith("- ") or not l[2:].strip():
            return False, f"bad bullet line {l!r}"
    if not DATE.search(body):
        return False, "launch date (14 november) missing"
    if not BUDGET.search(body):
        return False, "budget (48,000) missing"
    if not OWNER.search(body):
        return False, "migration owner (priya) missing"
    return True, "3 bullets with date, budget, owner"


if __name__ == "__main__":
    main(check)
