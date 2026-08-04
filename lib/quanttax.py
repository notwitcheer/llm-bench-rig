"""t090 shared bits: prompts, llama-server client, graders.

Prompt construction and grading live here rather than in the runners so that a
change to either is impossible to apply to one rung of the ladder and not another.
"""

import json
import re
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

# ---------------------------------------------------------------- prompts

GSM8K_INSTRUCTION = (
    "Solve the problem. Reason step by step, then give the final numeric answer "
    "on its own last line in the form: #### <answer>"
)

MMLU_INSTRUCTION = (
    "Answer the multiple-choice question. Reply with exactly one letter: A, B, C or D. "
    "No explanation."
)


def build_prompt(row):
    """One prompt per row, identical for every quant on the ladder."""
    task = row["task"]
    if task == "gsm8k":
        return f"{GSM8K_INSTRUCTION}\n\nProblem: {row['question']}"
    if task == "mmlu":
        opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", row["choices"]))
        return f"{MMLU_INSTRUCTION}\n\nQuestion: {row['question']}\n{opts}\nAnswer:"
    if task == "humaneval":
        return (
            "Complete the Python function. Output only the full function definition "
            "in a single ```python code block, with no commentary.\n\n"
            f"```python\n{row['prompt']}```"
        )
    raise ValueError(f"unknown task: {task}")


def max_tokens_for(task):
    """Held constant across the ladder; generous enough that truncation is not the tax.

    MMLU asks for one letter but is not budgeted at one letter: a thinking model spends
    tokens in its reasoning block first, and llama-server strips that block out of
    `message.content`, so a tight budget returns an empty string that looks like a
    quality collapse and is really a budget bug.
    """
    return {"gsm8k": 1024, "mmlu": 64, "humaneval": 768}[task]


# ---------------------------------------------------------------- server client


def server_health(base_url, timeout=5):
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def complete(base_url, prompt, max_tokens, seed, timeout=600):
    """Raw completion. Used for the SPEED workload only.

    Speed wants one fixed token stream that is byte-identical across the ladder, so
    it deliberately bypasses the chat template. Quality must NOT use this: feeding an
    instruct model raw text puts it outside the format it was tuned for, which is a
    protocol mismatch that would show up as a quality delta having nothing to do with
    the quant.

    Returns (text, timings) where timings is llama-server's own timing block.
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "seed": seed,
            "cache_prompt": False,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/completion", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d.get("content", ""), d.get("timings", {})


# Stops keep a model that has already answered from rambling to max_tokens. Without
# them a lower rung's different rambling changes how much text the grader has to
# search, which is a quant-correlated grading artefact rather than a quant tax.
STOPS = {
    "gsm8k": ["\nProblem:", "\nQuestion:"],
    "mmlu": ["\nQuestion:"],
    "humaneval": ["\n```\n```", "\nProblem:"],
}

# Thinking is a recorded experiment parameter, not a default that drifts. It is OFF for
# the ladder: reasoning length varies run to run and would add variance on the same axis
# the quant tax is measured on, and it multiplies decode tokens across five rungs. A
# thinking-on ladder is a separate run, not a silent difference between rungs.
ENABLE_THINKING = False


def chat(base_url, prompt, max_tokens, seed, task, timeout=600):
    """Chat completion with the model's own template applied server-side.

    This is the path for every quality task. Greedy and seeded; the server applies
    the chat template from the GGUF, so the model sees the format it was tuned for.
    """
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "seed": seed,
            "stream": False,
            "stop": STOPS.get(task, []),
            "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    choice = d["choices"][0]
    return (
        choice["message"]["content"] or "",
        {"finish_reason": choice.get("finish_reason"), "usage": d.get("usage", {})},
    )


# ---------------------------------------------------------------- graders

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
# The marker is often wrapped in whatever tags the model likes (<answer>, **, backticks),
# so skip a bounded run of non-numeric junk rather than requiring a digit immediately.
_MARKER = re.compile(r"####[^\d\-]{0,24}(-?[\d,]*\.?\d+)")


def grade_gsm8k(text, gold):
    """Returns (correct, how). `how` records which path graded it.

    The fallback path is reported rather than hidden: if a lower rung of the ladder
    stops emitting the marker and silently shifts onto last-number extraction, that
    is a grading-method change correlated with the quant, and it would read as a
    quality cliff that is not one.
    """
    hits = _MARKER.findall(text)
    if hits:
        cand, how = hits[-1], "marker"
    else:
        nums = _NUM.findall(text)
        if not nums:
            return False, "no_number"
        cand, how = nums[-1], "last_number_fallback"
    try:
        ok = abs(float(cand.replace(",", "")) - float(gold.replace(",", ""))) < 1e-4
    except ValueError:
        return False, "unparseable"
    return ok, how


def grade_mmlu(text, gold):
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    if not m:
        return False, "no_letter"
    return m.group(1) == gold, "letter"


_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text, row):
    """Pull the function body out of a fenced block, else treat the raw text as code."""
    m = _CODE_BLOCK.search(text)
    code = m.group(1) if m else text
    if f"def {row['entry_point']}" not in code:
        # model continued the signature instead of restating it
        code = row["prompt"] + code
    return code


def grade_humaneval(text, row, timeout=10):
    """Execute the candidate against the reference tests in a throwaway subprocess.

    Isolation is a subprocess with a hard timeout, not a sandbox. The inputs are the
    published HumanEval tests, so this is only safe for that fixed corpus.
    """
    code = extract_code(text, row)
    # HumanEval prompts import from typing; a completion that drops the import fails
    # for a reason unrelated to the quant, so supply the usual names.
    prog = (
        "from typing import List, Tuple, Dict, Optional, Any\n"
        "import math\n"
        f"{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        p = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout, text=True
        )
        return p.returncode == 0, ("pass" if p.returncode == 0 else "test_fail")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, "exec_error"
    finally:
        os.unlink(path)


def grade(row, text):
    """Returns (correct, how). `how` is the grading path, kept for auditing."""
    task = row["task"]
    if task == "gsm8k":
        return grade_gsm8k(text, row["gold"])
    if task == "mmlu":
        return grade_mmlu(text, row["gold"])
    if task == "humaneval":
        return grade_humaneval(text, row)
    raise ValueError(f"unknown task: {task}")
