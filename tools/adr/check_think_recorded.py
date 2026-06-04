"""ADR-0002 guard: bench.run_benchmark must record meta["think"] before saving.

Think-on vs think-off quality is non-comparable; the split leaderboard depends on
every result being labeled. See adr/ADR-0002-results-record-think-mode.md.
"""
import ast
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "bench.py"
FUNC = "run_benchmark"
REF = "See adr/ADR-0002-results-record-think-mode.md"


def _slice_str(slice_node) -> str | None:
    if isinstance(slice_node, ast.Constant):
        return slice_node.value
    return None


def find_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    func = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == FUNC),
        None,
    )
    if func is None:
        return [f"ADR-0002 VIOLATION: {FUNC}() not found in {filename}. {REF}"]

    think_line = None
    save_line = None
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "meta"
                        and _slice_str(tgt.slice) == "think"):
                    think_line = node.lineno if think_line is None else min(think_line, node.lineno)
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "save_metadata":
                save_line = node.lineno if save_line is None else min(save_line, node.lineno)

    if think_line is None:
        return [f"ADR-0002 VIOLATION: {FUNC}() never sets meta['think']. {REF}"]
    if save_line is None:
        return [f"ADR-0002 VIOLATION: {FUNC}() never calls save_metadata(). {REF}"]
    if think_line > save_line:
        return [
            f"ADR-0002 VIOLATION: meta['think'] set at line {think_line}, after "
            f"save_metadata at line {save_line}. Record think mode BEFORE saving. {REF}"
        ]
    return []


def main() -> int:
    violations = find_violations(TARGET.read_text(), str(TARGET))
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
