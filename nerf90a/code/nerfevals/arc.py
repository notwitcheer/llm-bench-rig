"""ARC-Challenge benchmark — generative evaluation via chat completions.

Dataset: allenai/ai2_arc (ARC-Challenge config), 1,172 test questions.
Standard: 25-shot from train split.
"""

import json
import time
from pathlib import Path

from .base import (CompletionLengthGate, LLMClient, parse_choice,
                   save_checkpoint, load_checkpoint)

LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def _normalize_label(label: str) -> str:
    return LABEL_MAP.get(label, label)


def _format_question(item: dict) -> str:
    texts = item["choices"]["text"]
    labels = item["choices"]["label"]
    lines = [item["question"]]
    for label, text in zip(labels, texts):
        lines.append(f"{_normalize_label(label)}. {text}")
    return "\n".join(lines)


def _valid_letters(item: dict) -> str:
    return "".join(_normalize_label(l) for l in item["choices"]["label"])


def _build_messages(question: dict, few_shot: list[dict]) -> list[dict]:
    messages = [{
        "role": "system",
        "content": (
            "Answer the following science question by choosing the correct option. "
            "Respond with just the letter. No explanation."
        ),
    }]
    for ex in few_shot:
        messages.append({"role": "user", "content": _format_question(ex)})
        messages.append({"role": "assistant", "content": _normalize_label(ex["answerKey"])})
    messages.append({"role": "user", "content": _format_question(question)})
    return messages


class ARCEval:

    def __init__(
        self,
        client: LLMClient,
        n_shot: int = 25,
        limit: int | None = None,
        results_dir: Path | None = None,
        gate: CompletionLengthGate | None = None,
    ):
        self.client = client
        self.n_shot = n_shot
        self.limit = limit
        self.gate = gate
        self.results_dir = Path(results_dir) if results_dir else None
        self._ckpt = self.results_dir / "arc_checkpoint.json" if self.results_dir else None

    def evaluate(self) -> dict:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        few_shot = list(ds["train"])[:self.n_shot]
        test_items = list(ds["test"])
        if self.limit:
            test_items = test_items[:self.limit]

        ckpt = load_checkpoint(self._ckpt)
        start = ckpt["idx"] if ckpt else 0
        correct = ckpt["correct"] if ckpt else 0
        parse_failures = ckpt["parse_failures"] if ckpt else 0

        t0 = time.time()
        n = len(test_items)

        for i in range(start, n):
            item = test_items[i]
            messages = _build_messages(item, few_shot)
            response = self.client.chat(messages)
            if self.gate is not None:
                self.gate.observe(self.client.last_completion_tokens)
            predicted = parse_choice(response, valid=_valid_letters(item))
            expected = _normalize_label(item["answerKey"])

            if predicted == expected:
                correct += 1
            elif predicted is None:
                parse_failures += 1

            done = i + 1
            if done % 100 == 0 or done == n:
                acc = correct / done
                rate = done / (time.time() - t0) if time.time() > t0 else 0
                fail_str = f", {parse_failures} unparsed" if parse_failures else ""
                print(f"[arc] {acc:.1%} ({correct}/{done}) [{done}/{n}] {rate:.1f} q/s{fail_str}",
                      flush=True)
                save_checkpoint(self._ckpt, {"idx": done, "correct": correct,
                                             "parse_failures": parse_failures})

        if self._ckpt and self._ckpt.exists():
            self._ckpt.unlink()

        elapsed = time.time() - t0
        acc = correct / n if n else 0

        print(f"\n[arc] Final: {acc:.1%} ({correct}/{n})")
        return {
            "score": round(acc * 100, 2),
            "metric": "acc",
            "correct": correct,
            "total": n,
            "parse_failures": parse_failures,
            "completion_tokens_mean": (
                round(self.gate.mean, 1)
                if self.gate is not None and self.gate.mean is not None else None
            ),
            "elapsed_s": round(elapsed, 1),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ARC-Challenge benchmark")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-shot", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    with LLMClient(args.api_base, args.model) as client:
        ev = ARCEval(client, n_shot=args.n_shot, limit=args.limit,
                     results_dir=Path(args.results_dir) if args.results_dir else None)
        results = ev.evaluate()

    if args.results_dir:
        out = Path(args.results_dir) / "arc_detail.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out}")
