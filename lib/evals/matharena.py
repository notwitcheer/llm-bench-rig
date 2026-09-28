"""MathArena final-answer sets (ArXivMath monthly, AIME, HMMT): generative eval via chat completions.

Built 2026-09-28 for the "fresh maths vs the old test" study: the same model, the same regime, on problem
sets published BEFORE and AFTER the model's release. MathArena (ETH SRI, matharena.ai) curates every set;
ArXivMath is a monthly release of research-level questions derived from arXiv papers submitted that month.

Datasets on the HF Hub (fields verified 2026-09-28 via the Hub API):
  MathArena/arxivmath-MMYY   problem_idx int, answer str (LaTeX), problem str, source (arXiv id), title, authors
  MathArena/aime_2026        problem_idx int, answer str (integer 0..999), problem str
  MathArena/hmmt_feb_2026    problem_idx int, answer str, problem_type, problem str
Licences: arxivmath cc-by-sa-4.0; aime/hmmt cc-by-nc-sa-4.0 (non-commercial; we publish scores, not problems).

Scoring: same zero-shot prompt, LAST balanced \\boxed{} extraction and grader as MATH-500 (vendored Qwen2.5-Math
grader, then normalised string equality). MathArena itself grades ArXivMath with an LLM judge for mathematical
equivalence, so a strict grader can under-count research-level answers: every detail json records `grader`, and
the agreement of this grader with MathArena's own judge labels is measured on their published outputs
(scripts/matharena_grader_agreement.py) before any score is quoted. Each item keeps predicted/expected so an
offline re-grade (an LLM judge pass) can rewrite `correct` later without a GPU rerun.

Task names: ma_<set> with the set id minus the MathArena prefix, e.g. ma_arxivmath-0826, ma_aime_2026.
"""

import time
from pathlib import Path

from .base import chat_or_error, drop_errored, LLMClient, ProgressFile
from .gpqa import CAP_SLACK, token_summary
from .math500 import _SYSTEM, extract_boxed, is_equivalent, normalize_answer, UNBOXED_TAIL_CHARS

DEFAULT_MAX_TOKENS = 32768
PREFIX = "ma_"

# published-at dates of each set (Hub createdAt), recorded so a report can place every set
# before or after each model's release without another lookup
SET_PUBLISHED = {
    "arxivmath-1225": "2026-02-16", "arxivmath-0126": "2026-02-16", "arxivmath-0226": "2026-03-02",
    "arxivmath-0326": "2026-04-03", "arxivmath-0826": "2026-09-13",
    "aime_2026": "2026-02-13", "hmmt_feb_2026": "2026-02-19",
}


def task_to_dataset(task: str) -> str:
    assert task.startswith(PREFIX), task
    return "MathArena/" + task[len(PREFIX):]


GRADE_TIMEOUT_S = 20


def _grade_child(pred, gold, q):
    try:
        q.put(bool(is_equivalent(pred, gold)))
    except Exception:
        q.put(False)


def grade_with_timeout(pred: str | None, gold: str, secs: int = GRADE_TIMEOUT_S) -> tuple[bool, bool]:
    """(correct, timed_out). The vendored grader calls sympy with no timeout (math_equal's default), and
    research-level LaTeX can make simplify() run for minutes: found 2026-09-28 when a grader-agreement pass over
    MathArena's published outputs stalled past 10 min. Grade in a forked child with a hard limit; on timeout fall
    back to normalised string equality and flag the item so an offline re-grade can look at it."""
    if pred is None:
        return False, False
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_grade_child, args=(pred, gold, q), daemon=True)
    p.start()
    p.join(secs)
    if p.is_alive():
        p.kill()
        p.join()
        return normalize_answer(pred) == normalize_answer(gold), True
    try:
        return q.get_nowait(), False
    except Exception:
        return False, False


def _build_messages(problem: str) -> list[dict]:
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": problem}]


class MathArenaEval:
    """Zero-shot MathArena set. Resumable via <task>_progress.json (math500 layout)."""

    def __init__(self, client: LLMClient, task: str, limit: int | None = None,
                 results_dir: Path | None = None, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.client = client
        self.task = task
        self.dataset = task_to_dataset(task)
        self.limit = limit
        self.max_tokens = max_tokens
        self.results_dir = Path(results_dir) if results_dir else None
        self.progress = ProgressFile(results_dir, task)

    def load_items(self) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(self.dataset, split="train")
        items = [{"problem": r["problem"], "answer": str(r["answer"]),
                  "problem_idx": r.get("problem_idx"), "source": r.get("source")} for r in ds]
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
        tag = f"[{self.task}]"

        for i, item in enumerate(items):
            key = str(i)
            if key in done:
                continue
            t_item = time.time()
            response, err = chat_or_error(self.client, _build_messages(item["problem"]),
                                          max_tokens=self.max_tokens)
            if err is not None:
                print(f"{tag} item {i} request error, recorded and retried on resume: {err}", flush=True)
                done[key] = {"correct": False, "completion_tokens": None, "capped": False, "error_request": err}
                self.progress.save(done)
                continue
            tokens = getattr(self.client, "last_completion_tokens", None)
            predicted = extract_boxed(response)
            ok, grade_to = grade_with_timeout(predicted, item["answer"])
            if ok:
                correct += 1
            elif predicted is None:
                parse_failures += 1
            capped = isinstance(tokens, int) and tokens >= self.max_tokens - CAP_SLACK
            done[key] = {"correct": ok, "predicted": predicted, "expected": item["answer"],
                         "problem_idx": item["problem_idx"], "source": item["source"],
                         "completion_tokens": tokens, "capped": capped,
                         "grade_timeout": grade_to,
                         "seconds": round(time.time() - t_item, 1)}
            if predicted is None:
                done[key]["response_tail"] = (response or "")[-UNBOXED_TAIL_CHARS:]
            self.progress.append_token_row({"idx": i, "completion_tokens": tokens,
                                            "correct": ok, "capped": capped})
            # small sets: save after every item so a hard stop loses at most the in-flight one
            self.progress.save(done)
            print(f"{tag} {len(done)}/{len(items)} correct {correct} tokens {tokens} "
                  f"capped {capped} {time.time() - t_item:.0f}s", flush=True)

        n = len(done)
        acc = correct / n if n else 0
        tok = token_summary(done.values(), self.max_tokens)
        print(f"\n{tag} Overall: {acc:.1%} ({correct}/{n}), {parse_failures} unboxed")
        results = {
            "score": round(acc * 100, 2),
            "metric": "acc",
            "correct": correct,
            "total": n,
            "parse_failures": parse_failures,
            "request_errors": sum(1 for r in done.values() if r.get("error_request")),
            "grade_timeouts": sum(1 for r in done.values() if r.get("grade_timeout")),
            "reasoning_fallback_count": getattr(self.client, "reasoning_fallback_count", 0) - fb0,
            "n_shot": 0,
            "dataset": self.dataset,
            "set_published": SET_PUBLISHED.get(self.task[len(PREFIX):]),
            "grader": "qwen2.5-math grade + normalised string equality (strict; MathArena uses an LLM judge)",
            **tok,
            "caveat": (f"{n}-item set: one item = {100 / max(n, 1):.1f} pts; greedy single pass; "
                       "capped_count > 0 means answers hit the token budget"),
        }
        self.progress.write_detail(results)
        return results
