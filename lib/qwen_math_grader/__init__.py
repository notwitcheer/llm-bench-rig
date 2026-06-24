"""Qwen2.5-Math grader, reused as the trusted oracle (NOT reimplemented — math grading is
exactly where the rig's measurement bugs live). `grade` = the authors' `math_equal` from the
vendored grader.py (verbatim from QwenLM/Qwen2.5-Math via zitian-gao/one-shot-em). `extract_answer`
is the standard last-`\\boxed{}` balanced-brace extractor (the authors' `find_box`), with a
last-number fallback. Needs the grader-env: sympy 1.12 + latex2sympy2 + antlr4 4.11.1 + regex."""
import re

from .grader import math_equal


def find_box(pred_str: str) -> str:
    """Authors' boxed-answer extractor: content of the LAST \\boxed{...} (balanced braces)."""
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def extract_answer(text: str) -> str:
    """Last \\boxed{} if present, else the last number in the text (use_last_number fallback)."""
    if "boxed" in text:
        return find_box(text).strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else ""


def grade(pred: str, gold: str) -> bool:
    """True iff pred and gold are numerically/symbolically equal (the authors' math_equal)."""
    return bool(math_equal(pred, gold))
