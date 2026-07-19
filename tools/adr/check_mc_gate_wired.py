"""ADR-0004 guard: every MC eval must feed the completion-length gate.

Degenerate serving parses to letters; pace is the tell. Any module that calls
parse_choice after client.chat must also call .observe(...) on a gate.
See adr/ADR-0004-mc-evals-gate-completion-length.md.
"""
import ast
from pathlib import Path

_EVALS = Path(__file__).resolve().parents[2] / "lib" / "evals"
TARGETS = [_EVALS / "mmlu.py", _EVALS / "arc.py", _EVALS / "hellaswag.py"]
REF = "See adr/ADR-0004-mc-evals-gate-completion-length.md"


def find_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    uses_parse_choice = any(
        isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) == "parse_choice"
             or getattr(n.func, "attr", None) == "parse_choice")
        for n in ast.walk(tree)
    )
    if not uses_parse_choice:
        return []  # not an MC eval — out of ADR-0004's scope
    observes = any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "observe"
        for n in ast.walk(tree)
    )
    if not observes:
        return [
            f"ADR-0004 VIOLATION: {filename} parses MC choices but never calls "
            f"gate.observe() — degenerate serving could bank a score. {REF}"
        ]
    return []


def main() -> int:
    failed = []
    for target in TARGETS:
        failed += find_violations(target.read_text(), str(target))
    for v in failed:
        print(v)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
