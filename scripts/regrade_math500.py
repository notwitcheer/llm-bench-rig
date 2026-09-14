"""Regrade finished MATH-500 rows in place with the current is_equivalent().

Why: rows graded on a box where the vendored Qwen2.5-Math grader could not import
(latex2sympy2's antlr4 runtime breaks on Python >= 3.13) silently fell back to
normalised string equality and under-counted (2026-09-14: 12 to 67 items per row).
Reads <dir>/math500_progress.json, rewrites `correct`, then rewrites the score
fields of math500_detail.json when it exists. Token fields are left alone.

Usage: python scripts/regrade_math500.py results/<slug>-thinkon [...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.evals.math500 import is_equivalent  # noqa: E402


def regrade(rd: Path) -> None:
    prog_p = rd / "math500_progress.json"
    if not prog_p.exists():
        print(f"{rd}: no math500_progress.json, skipped")
        return
    prog = json.loads(prog_p.read_text())
    done = prog["completed"]
    before = sum(1 for v in done.values() if v.get("correct"))
    for v in done.values():
        if v.get("error_request"):
            continue
        v["correct"] = v.get("predicted") is not None and is_equivalent(v["predicted"], v["expected"])
    after = sum(1 for v in done.values() if v.get("correct"))
    prog_p.write_text(json.dumps(prog, indent=2))
    line = f"{rd.name}: correct {before} -> {after} of {len(done)}"
    det_p = rd / "math500_detail.json"
    if det_p.exists():
        det = json.loads(det_p.read_text())
        n = det.get("total") or len(done)
        det["correct"] = after
        det["score"] = round(100 * after / n, 2)
        if det.get("completion_tokens_total") is not None and after:
            det["tokens_per_correct"] = round(det["completion_tokens_total"] / after, 1)
        det["regraded"] = "2026-09-14 qwen-math grader (latex2sympy2_extended on py>=3.13)"
        det_p.write_text(json.dumps(det, indent=2))
        line += f", detail score {det['score']}"
    print(line)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        regrade(Path(arg))
