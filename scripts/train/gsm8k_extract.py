"""GSM8K answer extraction — robust to gold (#### N), MetaMathQA ("The answer is: N"),
GSM calculator annotations (<<a*b=N>>), and post-answer rambling (no-EOS models).

Takes the model's FIRST committed answer, not the last number, so trailing rambling
(a model that keeps generating new problems) doesn't poison the score. Same rule for
base and tuned so the delta is real.
"""
import re

_NUM = r"-?\$?\d[\d,]*\.?\d*"


def _norm(s):
    return s.replace(",", "").replace("$", "").rstrip(".")


def extract_gsm8k_answer(text):
    if text is None:
        return None
    text = re.sub(r"<<[^>]*>>", "", text)          # strip GSM calculator annotations
    # 1) explicit '####' marker — take the FIRST one's number (gold, or model echoing the format)
    if "####" in text:
        m = re.search(_NUM, text.split("####", 1)[1])
        if m:
            return _norm(m.group(0))
    # 2) MetaMathQA / instruct: 'The answer is: N' — first occurrence
    m = re.search(r"answer\s*(?:is)?\s*:?\s*\$?(" + _NUM + r")", text, re.I)
    if m:
        return _norm(m.group(1))
    # 3) fallback: last number in the text
    nums = re.findall(_NUM, text)
    return _norm(nums[-1]) if nums else None
