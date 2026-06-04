"""ADR-0001 guard: the code-eval harness must never pass `stop=` to a model call.

Reasoning models reason inline and emit code only after long reasoning; an API
stop sequence fires mid-reasoning and truncates the response before any code is
produced (the cp8 false-low bug). See adr/ADR-0001-no-stop-sequences-code-eval.md.
"""
import ast
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "lib" / "evals" / "humaneval.py"


def find_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "stop":
                    violations.append(
                        f"ADR-0001 VIOLATION: stop= passed to a call at "
                        f"{filename}:{node.lineno} — reasoning models reason inline; "
                        f"stops truncate code before it is emitted. "
                        f"See adr/ADR-0001-no-stop-sequences-code-eval.md"
                    )
    return violations


def main() -> int:
    violations = find_violations(TARGET.read_text(), str(TARGET))
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
