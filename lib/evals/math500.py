"""MATH-500 benchmark — generative evaluation via chat completions.

The 500-problem subset of Hendrycks MATH selected by Lightman et al. 2023
("Let's Verify Step by Step"), as released by HuggingFaceH4. Zero-shot; the
prompt asks for the final answer in \\boxed{}; the LAST balanced \\boxed{...}
is extracted and graded with the vendored Qwen2.5-Math grader
(lib/qwen_math_grader.grade: numeric + sympy symbolic equivalence), falling
back to normalised string equality when the grader says no.

Dataset: HuggingFaceH4/MATH-500 on HuggingFace, split `test`, fields (verified
2026-09-10 via the datasets-server rows API):
  problem    str   the problem statement (LaTeX)
  solution   str   reference worked solution (unused for grading)
  answer     str   gold final answer (LaTeX), e.g. "\\left( 3, \\frac{\\pi}{2} \\right)"
  subject    str   e.g. "Precalculus"
  level      int   1..5 (arrives as a string in the datasets-server preview)
  unique_id  str   e.g. "test/precalculus/807.json"

Second-tier task (like gpqa): reported as its own row, never folded into q_avg.
Per-item completion_tokens and a `capped` flag are recorded so a run whose
answers were truncated at the budget is visible in the detail json.
"""

import re
import time
from pathlib import Path

from .base import chat_or_error, drop_errored, LLMClient, ProgressFile
from .gpqa import CAP_SLACK, token_summary

DEFAULT_MAX_TOKENS = 2048
DATASET = "HuggingFaceH4/MATH-500"

_SYSTEM = (
    "Solve the following math problem. Reason step by step, then give the "
    "final answer on the last line as \\boxed{answer}."
)


def _build_messages(problem: str) -> list[dict]:
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": problem}]


def extract_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} (or \\fbox{...}) with balanced braces.

    Returns None when no boxed answer is present or the braces never close
    (a truncated completion), so the caller can count it as a parse failure
    rather than grading a fragment.
    """
    if not text:
        return None
    tag = "</think>"
    if tag in text:
        text = text[text.rfind(tag) + len(tag):]
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth, i = 1, start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def normalize_answer(s: str) -> str:
    """Cheap normalisation for the string-equality fallback."""
    s = s.strip()
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "")
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s


def is_equivalent(pred: str | None, gold: str) -> bool:
    """Vendored Qwen2.5-Math grader first, normalised string equality second."""
    if pred is None:
        return False
    try:
        from lib.qwen_math_grader import grade
        if grade(pred, gold):
            return True
    except Exception:
        pass  # grader raised on exotic LaTeX: fall through to the string check
    return normalize_answer(pred) == normalize_answer(gold)


class Math500Eval:
    """Zero-shot MATH-500. Resumable via math500_progress.json (gpqa.py layout)."""

    def __init__(
        self,
        client: LLMClient,
        limit: int | None = None,
        results_dir: Path | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.client = client
        self.limit = limit
        self.max_tokens = max_tokens
        self.results_dir = Path(results_dir) if results_dir else None
        self.progress = ProgressFile(results_dir, "math500")

    def load_items(self) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(DATASET, split="test")
        items = [{"problem": raw["problem"], "answer": str(raw["answer"]),
                  "subject": raw.get("subject"), "level": raw.get("level"),
                  "unique_id": raw.get("unique_id")} for raw in ds]
        if self.limit:
            items = items[: self.limit]
        return items

    def evaluate(self) -> dict:
        items = self.load_items()
        done = drop_errored(self.progress.load())
        correct = sum(1 for r in done.values() if r["correct"])
        parse_failures = sum(1 for r in done.values() if r.get("predicted") is None)
        fb0 = getattr(self.client, "reasoning_fallback_count", 0)
        t0 = time.time()

        for i, item in enumerate(items):
            key = str(i)
            if key in done:
                continue
            response, err = chat_or_error(self.client, _build_messages(item["problem"]),
                                          max_tokens=self.max_tokens)
            if err is not None:
                print(f"[math500] item {i} request error, recorded and retried on resume: {err}", flush=True)
                done[key] = {"correct": False, "completion_tokens": None, "capped": False, "error_request": err}
                self.progress.save(done)
                continue
            tokens = getattr(self.client, "last_completion_tokens", None)
            predicted = extract_boxed(response)
            ok = is_equivalent(predicted, item["answer"])
            if ok:
                correct += 1
            elif predicted is None:
                parse_failures += 1
            capped = isinstance(tokens, int) and tokens >= self.max_tokens - CAP_SLACK
            done[key] = {"correct": ok, "predicted": predicted, "expected": item["answer"],
                         "subject": item["subject"], "level": item["level"],
                         "completion_tokens": tokens, "capped": capped}
            self.progress.append_token_row({"idx": i, "completion_tokens": tokens,
                                            "correct": ok, "capped": capped})
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                self.progress.save(done)
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"[math500] {i+1}/{len(items)} acc so far "
                      f"{correct/len(done):.1%} ({rate:.1f} q/s)")

        n = len(done)
        acc = correct / n if n else 0
        tok = token_summary(done.values(), self.max_tokens)
        print(f"\n[math500] Overall: {acc:.1%} ({correct}/{n}), {parse_failures} unboxed")
        results = {
            "score": round(acc * 100, 2),
            "metric": "acc",
            "correct": correct,
            "total": n,
            "parse_failures": parse_failures,
            "request_errors": sum(1 for r in done.values() if r.get("error_request")),
            "reasoning_fallback_count": getattr(self.client, "reasoning_fallback_count", 0) - fb0,
            "n_shot": 0,
            **tok,
            "caveat": ("500-item set: one item = 0.2 pts; capped_count > 0 means answers "
                       "hit the token budget, raise max_tokens before comparing rows"),
        }
        self.progress.write_detail(results)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MATH-500 against an OpenAI-compatible server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--think", action="store_true", help="leave reasoning on (default off)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    with LLMClient(args.api_base, args.model, think=args.think) as client:
        Math500Eval(client, limit=args.limit,
                    results_dir=Path(args.results_dir) if args.results_dir else None,
                    max_tokens=args.max_tokens).evaluate()
