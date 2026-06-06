"""GRPO reward functions for the GSM8K run. Correctness dominates; format is a small nudge
toward the '#### <answer>' ending (keeps train<->eval prompt identical — the sub-2 lever).
Same extractor as eval_gsm8k so reward and held-out eval score answers identically.
"""
try:
    from scripts.train.gsm8k_extract import extract_gsm8k_answer
except ImportError:  # capsule flat layout (~/train/)
    from gsm8k_extract import extract_gsm8k_answer


def _text(completion):
    # TRL conversational completion = [{"role": "assistant", "content": "..."}]; else a str.
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def correctness_reward(completions, answer, **kwargs):
    """+2.0 when the extracted answer equals gold (gold non-None), else 0.0."""
    out = []
    for comp, gold in zip(completions, answer):
        pred = extract_gsm8k_answer(_text(comp))
        out.append(2.0 if (gold is not None and pred == gold) else 0.0)
    return out


def format_reward(completions, **kwargs):
    """+0.5 when the completion contains a '####' answer marker."""
    return [0.5 if "####" in _text(comp) else 0.0 for comp in completions]
