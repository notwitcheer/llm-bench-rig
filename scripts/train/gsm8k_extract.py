"""GSM8K answer extraction — gold (after '####') and predicted (last number), normalized.
Same convention as the rig's GSM8K eval so base/tuned are scored identically."""
import re


def extract_gsm8k_answer(text):
    if text is None:
        return None
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    return nums[-1].replace(",", "").replace("$", "").rstrip(".")
