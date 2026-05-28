"""HumanEval benchmark — code generation + execution via chat completions.

Dataset: openai/openai_humaneval, 164 tasks.
Standard: 0-shot, greedy decoding, pass@1.
Requires code execution — runs generated code in a subprocess with timeout.
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .base import LLMClient


def _build_messages(prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Complete the following Python function. "
                "Output ONLY the function body code, properly indented. "
                "Do not repeat the function signature. No explanation."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _extract_completion(response: str, entry_point: str) -> str:
    """Extract function body from model response."""
    text = response.strip()

    # Strip thinking tags
    tag = "</think>"
    if tag in text:
        text = text[text.rfind(tag) + len(tag):].strip()

    # Extract from code block if present
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1)

    # If model re-output the full function, strip everything up to and
    # including the def line
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and entry_point in stripped and stripped.endswith(":"):
            text = "\n".join(lines[i + 1:])
            break

    # Ensure body is indented (concat with prompt expects indented body)
    lines = text.split("\n")
    first_nonblank = next((l for l in lines if l.strip()), "")
    if first_nonblank and not first_nonblank.startswith((" ", "\t")):
        text = "\n".join(("    " + line if line.strip() else line) for line in lines)

    return text


_STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__"]


def _truncate_at_stop(text: str) -> str:
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0:
            text = text[:idx]
    return text


def _execute(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run code in a subprocess. Returns (passed, error_message)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[-500:] if result.stderr else "non-zero exit"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            Path(f.name).unlink(missing_ok=True)


class HumanEvalEval:

    def __init__(
        self,
        client: LLMClient,
        limit: int | None = None,
        results_dir: Path | None = None,
        exec_timeout: int = 10,
    ):
        self.client = client
        self.limit = limit
        self.results_dir = Path(results_dir) if results_dir else None
        self.exec_timeout = exec_timeout

    def evaluate(self) -> dict:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval")
        test_items = list(ds["test"])
        if self.limit:
            test_items = test_items[:self.limit]

        passed = 0
        errors = []
        t0 = time.time()
        n = len(test_items)

        for i, item in enumerate(test_items):
            task_id = item["task_id"]
            prompt = item["prompt"]
            test_code = item["test"]
            entry_point = item["entry_point"]

            messages = _build_messages(prompt)
            response = self.client.chat(
                messages, max_tokens=1024,
                stop=["\ndef ", "\nclass ", "\nif __name__"],
            )

            completion = _extract_completion(response, entry_point)
            completion = _truncate_at_stop(completion)

            full_code = prompt + completion + "\n\n" + test_code + f"\ncheck({entry_point})\n"

            ok, err = _execute(full_code, timeout=self.exec_timeout)
            if ok:
                passed += 1
            else:
                errors.append({"task_id": task_id, "error": err})

            done = i + 1
            if done % 20 == 0 or done == n:
                rate = done / (time.time() - t0) if time.time() > t0 else 0
                print(f"[humaneval] {passed}/{done} passed [{done}/{n}] {rate:.1f} q/s",
                      flush=True)

        elapsed = time.time() - t0
        score = passed / n * 100 if n else 0

        print(f"\n[humaneval] Final: pass@1 = {score:.1f}% ({passed}/{n})")

        results = {
            "score": round(score, 2),
            "metric": "pass@1",
            "passed": passed,
            "total": n,
            "errors": errors[:50],
            "elapsed_s": round(elapsed, 1),
        }

        if self.results_dir:
            with open(self.results_dir / "humaneval_detail.json", "w") as f:
                json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HumanEval benchmark")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--exec-timeout", type=int, default=10)
    args = parser.parse_args()

    with LLMClient(args.api_base, args.model) as client:
        ev = HumanEvalEval(client, limit=args.limit,
                           results_dir=Path(args.results_dir) if args.results_dir else None,
                           exec_timeout=args.exec_timeout)
        results = ev.evaluate()
