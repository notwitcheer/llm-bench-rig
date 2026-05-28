"""HellaSwag benchmark — generative evaluation via chat completions.

Dataset: Rowan/hellaswag, 10,042 validation questions (test split has no labels).
Standard: 10-shot from train split.
"""

import json
import time
from pathlib import Path

from .base import LLMClient, parse_choice, save_checkpoint, load_checkpoint

LETTERS = "ABCD"


def _format_question(item: dict) -> str:
    lines = [item["ctx"]]
    for i, ending in enumerate(item["endings"]):
        lines.append(f"{LETTERS[i]}. {ending}")
    return "\n".join(lines)


def _build_messages(question: dict, few_shot: list[dict]) -> list[dict]:
    messages = [{
        "role": "system",
        "content": (
            "Pick the most plausible continuation for the given context. "
            "Respond with just the letter (A, B, C, or D). No explanation."
        ),
    }]
    for ex in few_shot:
        messages.append({"role": "user", "content": _format_question(ex)})
        messages.append({"role": "assistant", "content": LETTERS[int(ex["label"])]})
    messages.append({"role": "user", "content": _format_question(question)})
    return messages


class HellaSwagEval:

    def __init__(
        self,
        client: LLMClient,
        n_shot: int = 10,
        limit: int | None = None,
        results_dir: Path | None = None,
    ):
        self.client = client
        self.n_shot = n_shot
        self.limit = limit
        self.results_dir = Path(results_dir) if results_dir else None
        self._ckpt = self.results_dir / "hellaswag_checkpoint.json" if self.results_dir else None

    def evaluate(self) -> dict:
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag")
        few_shot = list(ds["train"])[:self.n_shot]
        test_items = list(ds["validation"])
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
            predicted = parse_choice(response)
            expected = LETTERS[int(item["label"])]

            if predicted == expected:
                correct += 1
            elif predicted is None:
                parse_failures += 1

            done = i + 1
            if done % 200 == 0 or done == n:
                acc = correct / done
                rate = done / (time.time() - t0) if time.time() > t0 else 0
                fail_str = f", {parse_failures} unparsed" if parse_failures else ""
                print(f"[hellaswag] {acc:.1%} ({correct}/{done}) [{done}/{n}] {rate:.1f} q/s{fail_str}",
                      flush=True)
                save_checkpoint(self._ckpt, {"idx": done, "correct": correct,
                                             "parse_failures": parse_failures})

        if self._ckpt and self._ckpt.exists():
            self._ckpt.unlink()

        elapsed = time.time() - t0
        acc = correct / n if n else 0

        print(f"\n[hellaswag] Final: {acc:.1%} ({correct}/{n})")
        return {
            "score": round(acc * 100, 2),
            "metric": "acc",
            "correct": correct,
            "total": n,
            "parse_failures": parse_failures,
            "elapsed_s": round(elapsed, 1),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HellaSwag benchmark")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-shot", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    with LLMClient(args.api_base, args.model) as client:
        ev = HellaSwagEval(client, n_shot=args.n_shot, limit=args.limit,
                           results_dir=Path(args.results_dir) if args.results_dir else None)
        results = ev.evaluate()

    if args.results_dir:
        out = Path(args.results_dir) / "hellaswag_detail.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out}")
