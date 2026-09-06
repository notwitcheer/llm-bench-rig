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
import textwrap
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


def _strip_blank_lines(text: str) -> str:
    """Strip leading and trailing blank lines while preserving per-line indentation.

    Unlike str.strip(), this does NOT remove leading spaces from content lines.
    """
    lines = text.split("\n")
    # Drop leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    # Drop trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extract_completion(response: str, entry_point: str) -> str:
    """Extract function body from model response.

    Returns the raw extracted text (may be a full function def or just a body).
    Does NOT strip the def line or re-indent — callers must handle both cases.

    Critically, we preserve per-line leading indentation.  Only blank lines at
    the very start/end of the response are removed (not leading spaces on
    content lines) so that an already-correctly-indented body is not mangled.
    """
    # Strip only leading/trailing blank lines — NOT per-line leading whitespace.
    text = _strip_blank_lines(response)

    # Strip thinking tags (content after the closing tag may itself be indented)
    tag = "</think>"
    if tag in text:
        text = _strip_blank_lines(text[text.rfind(tag) + len(tag):])

    # Extract from code block if present
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1)
        # Only strip blank boundary lines inside the fence, not content indentation
        text = _strip_blank_lines(text)

    return text


_STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__"]


def _truncate_at_stop(text: str) -> str:
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0:
            text = text[:idx]
    return text


def _is_full_function(text: str, entry_point: str) -> bool:
    """Return True if *text* contains a complete definition of *entry_point*.

    A complete definition means: a ``def <entry_point>(...):`` line (at any
    indentation level, though typically at column 0) followed by at least one
    indented body line.
    """
    lines = text.split("\n")
    found_def = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and entry_point in stripped and stripped.endswith(":"):
            # Check that there is at least one following indented line
            for body_line in lines[i + 1:]:
                if body_line.strip():  # first non-blank line after def
                    if body_line.startswith((" ", "\t")):
                        return True
                    # Non-indented non-blank line right after def → not a proper body
                    break
    return False


def _strip_main_block(text: str) -> str:
    """Strip a trailing ``if __name__ == '__main__':`` block from *text*."""
    # Find the last occurrence of a top-level if __name__ block
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("if __name__") and "__main__" in stripped:
            # Only strip if it's at the top level (no leading indent)
            if not line.startswith((" ", "\t")):
                return "\n".join(lines[:i]).rstrip()
    return text


def _extract_full_function(text: str, entry_point: str) -> str:
    """Extract the complete function definition starting at *entry_point* from *text*.

    Finds the ``def <entry_point>`` line and returns from there to the end of
    the indented block (i.e. until a non-indented, non-blank line that is NOT
    part of the function, or end of file).
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and entry_point in stripped and stripped.endswith(":"):
            start = i
            break
    if start is None:
        return text

    # Collect lines belonging to this function: the def line plus all
    # continuation lines (indented lines, blank lines, decorator lines).
    result = [lines[start]]
    for line in lines[start + 1:]:
        # Stop at the next top-level definition or statement (non-blank,
        # non-indented) — but not at blank lines.
        if line and not line.startswith((" ", "\t")):
            break
        result.append(line)

    return "\n".join(result).rstrip()


def _compiles(code: str) -> bool:
    """Return True if *code* compiles. Uses compile() rather than ast.parse so
    a misplaced ``return`` at module scope (not just IndentationError) is also
    rejected."""
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def build_executable_program(
    prompt: str, raw_response: str, test_code: str, entry_point: str
) -> str:
    """Assemble the full Python program to execute for one HumanEval task.

    Models emit the solution in several indentation conventions: a full
    function, an already-indented body, a relative flush-left body, or an
    absolute body whose first line lost its indent (e.g. Nemotron-3, which
    drops the leading 4 spaces on only the first statement while the rest stay
    at absolute indentation).  Rather than guess the convention, generate each
    plausible assembly and return the first that COMPILES.
    """
    extracted = _extract_completion(raw_response, entry_point)
    tail = "\n\n" + test_code + "\ncheck(" + entry_point + ")\n"

    candidates: list[str] = []

    # Full-function output (reasoning / chat models): use verbatim, prepend the
    # prompt imports, never re-indent.
    if _is_full_function(extracted, entry_point):
        func_code = _strip_main_block(_extract_full_function(extracted, entry_point))
        import_lines = [
            line for line in prompt.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        header = "\n".join(import_lines)
        func_block = (header + "\n\n" + func_code) if header else func_code
        candidates.append(func_block + tail)

    body = _truncate_at_stop(extracted)

    # (a) Already at correct absolute indentation: verbatim.
    candidates.append(prompt + body + tail)

    # (b) Relative flush-left: every line relative to column 0 -> indent +4.
    dedented = textwrap.dedent(body)
    relative = "\n".join(
        ("    " + line if line.strip() else line)
        for line in dedented.split("\n")
    )
    candidates.append(prompt + relative + tail)

    # (c) Absolute, first line dropped (Nemotron-3): later lines already at
    #     absolute indentation; bump only lines below the 4-space body base up
    #     to it, preserving deeper nesting.
    absolute = "\n".join(
        ("    " + line.lstrip()
         if line.strip() and (len(line) - len(line.lstrip())) < 4
         else line)
        for line in body.split("\n")
    )
    candidates.append(prompt + absolute + tail)

    for candidate in candidates:
        if _compiles(candidate):
            return candidate

    # Nothing compiled — return the relative assembly so the executor surfaces a
    # real error instead of this function raising.
    return candidates[-1]

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
        fb0 = getattr(self.client, "reasoning_fallback_count", 0)
        t0 = time.time()
        n = len(test_items)

        for i, item in enumerate(test_items):
            task_id = item["task_id"]
            prompt = item["prompt"]
            test_code = item["test"]
            entry_point = item["entry_point"]

            messages = _build_messages(prompt)
            response = self.client.chat(
                # No stop sequences: reasoning models reason INLINE and mention
                # "def "/"class " mid-reasoning, so "\ndef " stops fire before the
                # code is emitted (empty/truncated content). build_executable_program
                # already trims trailing extras. 4096 covers long reasoning + code.
                messages, max_tokens=4096,
                preserve_indent=True,
            )

            full_code = build_executable_program(prompt, response, test_code, entry_point)

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
            "reasoning_fallback_count": getattr(self.client, "reasoning_fallback_count", 0) - fb0,
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
