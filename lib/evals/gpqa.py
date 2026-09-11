"""GPQA-Diamond benchmark — generative evaluation via chat completions.

Graduate-level Google-proof QA (Rein et al. 2023), diamond subset: 198
questions, 4 options (1 correct + 3 incorrect, shuffled deterministically
per item). Zero-shot, letter extraction, same discipline as the MMLU eval.

Standing second-tier metric since 2026-08-17: the five-task board saturates
at 92-97 on current ~27B models; GPQA-diamond sits in the 40-60 band where
quant ladders can actually separate. Reported with an explicit small-sample
caveat (198 items => one item ~ 0.5 pts; treat gaps < ~3 pts as noise).

Dataset: Idavidrein/gpqa on HuggingFace (GATED: the account must have
accepted the terms on the dataset page once; auto-granted).
"""

import json
import random
import statistics
import time
from pathlib import Path

from .base import CompletionLengthGate, LLMClient, chat_or_error, drop_errored, parse_choice

LETTERS = "ABCD"
SHUFFLE_SEED = 42  # per-item option order derived from this + item index
DEFAULT_MAX_TOKENS = 2048
# An item whose completion lands within this many tokens of the budget is
# treated as capped (the server's usage count can sit a few tokens under the
# cap when the stop sequence or an EOS is charged differently).
CAP_SLACK = 8


def token_summary(entries, max_tokens: int) -> dict:
    """Aggregate per-item completion-token counts into the standing metrics.

    `entries` is the `completed` mapping from gpqa_progress.json. Entries that
    predate token recording (no `completion_tokens`, or None) are excluded;
    `tokens_recorded` says how many items the numbers actually cover so partial
    coverage of a resumed run is visible. `tokens_per_correct` is total tokens
    over the correct count WITHIN the recorded subset (None when nothing there is
    correct) so the ratio is not diluted by unrecorded items.
    """
    recorded = [e for e in entries if isinstance(e.get("completion_tokens"), int)]
    tokens = [e["completion_tokens"] for e in recorded]
    total = sum(tokens)
    correct = sum(1 for e in recorded if e.get("correct"))
    capped = sum(1 for e in recorded
                 if e.get("capped", e["completion_tokens"] >= max_tokens - CAP_SLACK))
    n = len(recorded)
    return {
        "max_tokens": max_tokens,
        "tokens_recorded": n,
        # counts are 0 when nothing is recorded; ratios are None (undefined)
        "completion_tokens_total": total,
        "completion_tokens_median": statistics.median(tokens) if n else None,
        "tokens_per_correct": round(total / correct, 1) if correct else None,
        "capped_count": capped,
        "capped_rate": round(capped / n, 4) if n else None,
    }


def _build_item(raw: dict, idx: int) -> dict:
    """Shuffle the 4 options deterministically; return question, choices, answer idx."""
    options = [
        raw["Correct Answer"].strip(),
        raw["Incorrect Answer 1"].strip(),
        raw["Incorrect Answer 2"].strip(),
        raw["Incorrect Answer 3"].strip(),
    ]
    order = list(range(4))
    random.Random(f"{SHUFFLE_SEED}-{idx}").shuffle(order)
    choices = [options[i] for i in order]
    answer = order.index(0)  # where the correct answer landed
    return {"question": raw["Question"].strip(), "choices": choices, "answer": answer}


def _build_messages(item: dict) -> list[dict]:
    lines = [item["question"], ""]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{LETTERS[i]}. {choice}")
    return [
        {
            "role": "system",
            "content": (
                "Answer the following multiple choice question. "
                "Respond with just the letter (A, B, C, or D). No explanation."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


class GPQAEval:
    """Zero-shot GPQA-diamond. Resumable via gpqa_progress.json like MMLUEval."""

    def __init__(
        self,
        client: LLMClient,
        limit: int | None = None,
        results_dir: Path | None = None,
        gate: CompletionLengthGate | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.client = client
        self.limit = limit
        self.gate = gate
        self.max_tokens = max_tokens
        self.results_dir = Path(results_dir) if results_dir else None
        self._progress_path = (
            self.results_dir / "gpqa_progress.json" if self.results_dir else None
        )
        # per-item token sidecar, one line per NEW item (append; resumes keep old rows)
        self._tokens_path = (
            self.results_dir / "gpqa_tokens.jsonl" if self.results_dir else None
        )

    def evaluate(self) -> dict:
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = [_build_item(raw, i) for i, raw in enumerate(ds)]
        if self.limit:
            items = items[: self.limit]

        done = drop_errored(self._load_progress())
        correct = sum(1 for r in done.values() if r["correct"])
        parse_failures = sum(1 for r in done.values() if r.get("predicted") is None)
        fb0 = getattr(self.client, "reasoning_fallback_count", 0)
        t0 = time.time()

        for i, item in enumerate(items):
            key = str(i)
            if key in done:
                continue
            response, err = chat_or_error(self.client, _build_messages(item), max_tokens=self.max_tokens)
            if err is not None:
                print(f"[gpqa] item {i} request error, recorded and retried on resume: {err}", flush=True)
                done[key] = {"correct": False, "completion_tokens": None, "capped": False, "predicted": None, "error_request": err}
                self.progress.save(done)
                continue
            tokens = getattr(self.client, "last_completion_tokens", None)
            if self.gate is not None:
                self.gate.observe(tokens)
            predicted = parse_choice(response)
            expected = LETTERS[item["answer"]]
            ok = predicted == expected
            if ok:
                correct += 1
            elif predicted is None:
                parse_failures += 1
            capped = (isinstance(tokens, int)
                      and tokens >= self.max_tokens - CAP_SLACK)
            done[key] = {"correct": ok, "predicted": predicted, "expected": expected,
                         "completion_tokens": tokens, "capped": capped}
            self._append_token_row({"idx": i, "completion_tokens": tokens,
                                    "correct": ok, "capped": capped})
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                self._save_progress(done)
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"[gpqa] {i+1}/{len(items)} acc so far "
                      f"{correct/len(done):.1%} ({rate:.1f} q/s)")

        n = len(done)
        acc = correct / n if n else 0
        tok = token_summary(done.values(), self.max_tokens)
        print(f"\n[gpqa] Overall: {acc:.1%} ({correct}/{n}), "
              f"{parse_failures} unparsed")
        if tok["tokens_recorded"]:
            print(f"[gpqa] tokens: median {tok['completion_tokens_median']:.0f}, "
                  f"per correct {tok['tokens_per_correct']}, "
                  f"capped {tok['capped_count']}/{tok['tokens_recorded']} "
                  f"(budget {self.max_tokens}, recorded {tok['tokens_recorded']}/{n})")
        return {
            "score": round(acc * 100, 2),
            "metric": "acc",
            "correct": correct,
            "total": n,
            "parse_failures": parse_failures,
            "request_errors": sum(1 for r in done.values() if r.get("error_request")),
            "reasoning_fallback_count": getattr(self.client, "reasoning_fallback_count", 0) - fb0,
            "n_shot": 0,
            "shuffle_seed": SHUFFLE_SEED,
            "completion_tokens_mean": (
                round(self.gate.mean, 1)
                if self.gate is not None and self.gate.mean is not None else None
            ),
            **tok,
            "caveat": "198-item set: one item ~ 0.5 pts; gaps under ~3 pts are noise",
        }

    def _append_token_row(self, row: dict):
        if not self._tokens_path:
            return
        self._tokens_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._tokens_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _load_progress(self) -> dict:
        if self._progress_path and self._progress_path.exists():
            with open(self._progress_path) as f:
                return json.load(f).get("completed", {})
        return {}

    def _save_progress(self, completed: dict):
        if not self._progress_path:
            return
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._progress_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {"completed": completed,
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        tmp.rename(self._progress_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GPQA-diamond against an OpenAI-compatible server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--think", action="store_true", help="leave reasoning on (board default is off)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="completion budget per item (think-on runs want 16384)")
    args = parser.parse_args()

    client = LLMClient(args.api_base, args.model, think=args.think)
    evaluator = GPQAEval(
        client=client,
        limit=args.limit,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        max_tokens=args.max_tokens,
    )
    try:
        results = evaluator.evaluate()
    finally:
        client.close()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")
