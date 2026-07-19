"""MMLU benchmark — generative evaluation via chat completions.

Replicates the standard 5-shot MMLU methodology but uses generative
letter extraction instead of loglikelihood scoring. Results are internally
consistent for model comparison, though absolute scores may differ from
logprob-based evaluations by 5-15%.

Dataset: cais/mmlu on HuggingFace (auto-downloaded and cached on first run).
"""

import json
import random
import sys
import time
from pathlib import Path

from .base import CompletionLengthGate, LLMClient, parse_choice

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

CATEGORIES = {
    "stem": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "world_religions",
    ],
    "social_sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "marketing", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting",
        "professional_law", "professional_medicine", "virology",
    ],
}

LETTERS = "ABCD"


def _format_question(item: dict) -> str:
    lines = [item["question"]]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{LETTERS[i]}. {choice}")
    return "\n".join(lines)


def _build_messages(subject: str, question: dict, few_shot: list[dict]) -> list[dict]:
    subject_display = subject.replace("_", " ")
    messages = [{
        "role": "system",
        "content": (
            f"Answer the following multiple choice questions about {subject_display}. "
            "Respond with just the letter (A, B, C, or D). No explanation."
        ),
    }]
    for ex in few_shot:
        messages.append({"role": "user", "content": _format_question(ex)})
        messages.append({"role": "assistant", "content": LETTERS[ex["answer"]]})
    messages.append({"role": "user", "content": _format_question(question)})
    return messages


class MMLUEval:

    def __init__(
        self,
        client: LLMClient,
        n_shot: int = 5,
        limit: int | None = None,
        sample: float | None = None,
        results_dir: Path | None = None,
        gate: CompletionLengthGate | None = None,
    ):
        self.client = client
        self.n_shot = n_shot
        self.limit = limit
        self.sample = sample
        self.gate = gate
        self.results_dir = Path(results_dir) if results_dir else None
        self._progress_path = self.results_dir / "mmlu_progress.json" if self.results_dir else None

    def evaluate(self, subjects: list[str] | None = None) -> dict:
        from datasets import load_dataset

        subjects = subjects or SUBJECTS
        completed = self._load_progress()
        total_correct = 0
        total_count = 0

        for idx, subject in enumerate(subjects, 1):
            if subject in completed:
                r = completed[subject]
                total_correct += r["correct"]
                total_count += r["total"]
                print(f"[mmlu] {subject}: {r['accuracy']:.1%} ({r['correct']}/{r['total']}) "
                      f"[{idx}/{len(subjects)}] (cached)")
                continue

            ds = load_dataset("cais/mmlu", subject)
            few_shot = list(ds["dev"])[:self.n_shot]
            test_items = list(ds["test"])
            if self.limit:
                test_items = test_items[:self.limit]
            if self.sample and self.sample < 1.0:
                rng = random.Random(42)
                k = max(1, int(len(test_items) * self.sample))
                indices = sorted(rng.sample(range(len(test_items)), k))
                test_items = [test_items[i] for i in indices]

            correct = 0
            parse_failures = 0
            t0 = time.time()

            for item in test_items:
                messages = _build_messages(subject, item, few_shot)
                response = self.client.chat(messages)
                if self.gate is not None:
                    self.gate.observe(self.client.last_completion_tokens)
                predicted = parse_choice(response)
                expected = LETTERS[item["answer"]]
                if predicted == expected:
                    correct += 1
                elif predicted is None:
                    parse_failures += 1

            elapsed = time.time() - t0
            n = len(test_items)
            acc = correct / n if n else 0

            completed[subject] = {
                "correct": correct,
                "total": n,
                "accuracy": round(acc, 4),
                "parse_failures": parse_failures,
                "elapsed_s": round(elapsed, 1),
            }
            self._save_progress(completed)

            total_correct += correct
            total_count += n

            rate = n / elapsed if elapsed > 0 else 0
            fail_str = f", {parse_failures} unparsed" if parse_failures else ""
            print(f"[mmlu] {subject}: {acc:.1%} ({correct}/{n}) "
                  f"[{idx}/{len(subjects)}] {rate:.1f} q/s{fail_str}")

        overall_acc = total_correct / total_count if total_count else 0

        category_scores = {}
        for cat, cat_subjects in CATEGORIES.items():
            cat_correct = sum(completed.get(s, {}).get("correct", 0)
                             for s in cat_subjects if s in completed)
            cat_total = sum(completed.get(s, {}).get("total", 0)
                           for s in cat_subjects if s in completed)
            if cat_total:
                category_scores[cat] = {
                    "score": round(cat_correct / cat_total * 100, 2),
                    "correct": cat_correct,
                    "total": cat_total,
                }

        sample_str = f" (sampled {self.sample:.0%})" if self.sample and self.sample < 1.0 else ""
        print(f"\n[mmlu] Overall: {overall_acc:.1%} ({total_correct}/{total_count}){sample_str}")
        for cat, sc in category_scores.items():
            print(f"[mmlu]   {cat}: {sc['score']:.1f}%")

        result = {
            "score": round(overall_acc * 100, 2),
            "metric": "acc",
            "correct": total_correct,
            "total": total_count,
            "completion_tokens_mean": (
                round(self.gate.mean, 1)
                if self.gate is not None and self.gate.mean is not None else None
            ),
            "categories": category_scores,
            "per_subject": completed,
        }
        if self.sample and self.sample < 1.0:
            result["sample"] = self.sample
            result["sample_seed"] = 42
        return result

    def _load_progress(self) -> dict:
        if self._progress_path and self._progress_path.exists():
            with open(self._progress_path) as f:
                data = json.load(f)
            return data.get("completed", {})
        return {}

    def _save_progress(self, completed: dict):
        if not self._progress_path:
            return
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._progress_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"completed": completed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        tmp.rename(self._progress_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MMLU benchmark against llama-server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True, help="Model name (for API request)")
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Max questions per subject (for testing)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to evaluate")
    parser.add_argument("--results-dir", default=None, help="Directory for progress/resume file")
    parser.add_argument("--output", default=None, help="Save final results JSON to this path")
    args = parser.parse_args()

    client = LLMClient(args.api_base, args.model)
    results_dir = Path(args.results_dir) if args.results_dir else None

    evaluator = MMLUEval(
        client=client,
        n_shot=args.n_shot,
        limit=args.limit,
        results_dir=results_dir,
    )

    try:
        results = evaluator.evaluate(subjects=args.subjects)
    finally:
        client.close()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")
