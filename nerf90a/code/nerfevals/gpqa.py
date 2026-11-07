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
import time
from pathlib import Path

from .base import CompletionLengthGate, LLMClient, parse_choice

LETTERS = "ABCD"
SHUFFLE_SEED = 42  # per-item option order derived from this + item index


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
    ):
        self.client = client
        self.limit = limit
        self.gate = gate
        self.results_dir = Path(results_dir) if results_dir else None
        self._progress_path = (
            self.results_dir / "gpqa_progress.json" if self.results_dir else None
        )

    def evaluate(self) -> dict:
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = [_build_item(raw, i) for i, raw in enumerate(ds)]
        if self.limit:
            items = items[: self.limit]

        done = self._load_progress()
        correct = sum(1 for r in done.values() if r["correct"])
        parse_failures = sum(1 for r in done.values() if r.get("predicted") is None)
        t0 = time.time()

        for i, item in enumerate(items):
            key = str(i)
            if key in done:
                continue
            response = self.client.chat(_build_messages(item), max_tokens=2048)
            if self.gate is not None:
                self.gate.observe(self.client.last_completion_tokens)
            predicted = parse_choice(response)
            expected = LETTERS[item["answer"]]
            ok = predicted == expected
            if ok:
                correct += 1
            elif predicted is None:
                parse_failures += 1
            done[key] = {"correct": ok, "predicted": predicted, "expected": expected}
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                self._save_progress(done)
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"[gpqa] {i+1}/{len(items)} acc so far "
                      f"{correct/len(done):.1%} ({rate:.1f} q/s)")

        n = len(done)
        acc = correct / n if n else 0
        print(f"\n[gpqa] Overall: {acc:.1%} ({correct}/{n}), "
              f"{parse_failures} unparsed")
        return {
            "score": round(acc * 100, 2),
            "metric": "acc",
            "correct": correct,
            "total": n,
            "parse_failures": parse_failures,
            "n_shot": 0,
            "shuffle_seed": SHUFFLE_SEED,
            "completion_tokens_mean": (
                round(self.gate.mean, 1)
                if self.gate is not None and self.gate.mean is not None else None
            ),
            "caveat": "198-item set: one item ~ 0.5 pts; gaps under ~3 pts are noise",
        }

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
    args = parser.parse_args()

    client = LLMClient(args.api_base, args.model, think=args.think)
    evaluator = GPQAEval(
        client=client,
        limit=args.limit,
        results_dir=Path(args.results_dir) if args.results_dir else None,
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
