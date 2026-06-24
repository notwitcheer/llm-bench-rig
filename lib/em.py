"""One-Shot Entropy Minimization primitives (ADR-0003: tested pure logic; the GPU
driver scripts/train_em.py only orchestrates). The loss is the mean per-generated-token
entropy of softmax(logits/temp); minimizing it sharpens the model's own distribution
(see arXiv 2505.20282 train.py). entropy_loss_ref is a torch-free reference for tests;
entropy_loss is the torch version the trainer calls (imported lazily so this module stays
torch-free on the Mac)."""
from __future__ import annotations

import math


def _entropy_row(row: list[float], temp: float) -> float:
    z = [x / temp for x in row]
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    p = [e / s for e in exps]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def entropy_loss_ref(logits: list[list[float]], gen_mask: list[int], temp: float = 0.5) -> float:
    """Pure-Python reference: mean entropy over masked (generated) positions."""
    num = 0.0
    den = 0.0
    for row, m in zip(logits, gen_mask):
        if m:
            num += _entropy_row(row, temp)
            den += 1
    return num / den if den else 0.0


def repetition_rate(token_ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams that are repeats (1 - unique/total). Collapse tell."""
    if len(token_ids) < n + 1:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def mean_len(list_of_token_lists) -> float:
    if not list_of_token_lists:
        return 0.0
    return sum(len(x) for x in list_of_token_lists) / len(list_of_token_lists)


def entropy_loss(logits, gen_mask, temp: float = 0.5):
    """Torch version used by the trainer. logits [T,V], gen_mask [T]. Mean entropy over
    generated positions. torch imported lazily so the module stays torch-free for tests."""
    import torch.nn.functional as F
    logp = F.log_softmax(logits / temp, dim=-1)
    p = logp.exp()
    h = -(p * logp).sum(dim=-1)                       # [T] per-position entropy
    mask = gen_mask.to(h.dtype)
    return (h * mask).sum() / mask.sum().clamp(min=1)


# --- prompt protocols (the eval variants differ by how the base is prompted) ---
QWEN_MATH_COT = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}"
)

FEWSHOT_EXEMPLARS = [
    ("What is 7 times 6?",
     "We multiply: 7 times 6 equals 42. The final answer is \\boxed{42}."),
    ("If x + 5 = 12, what is x?",
     "Subtract 5 from both sides: x = 12 - 5 = 7. The final answer is \\boxed{7}."),
]


def build_prompt(problem: str, variant: str) -> str:
    """paper/fair share the qwen25-math-cot prompt (they differ only in generation max-tokens,
    set by the driver); format prepends a fixed few-shot scaffold that elicits the boxed CoT
    format without any weight update."""
    if variant in ("paper", "fair"):
        return QWEN_MATH_COT.format(problem=problem)
    if variant == "format":
        shots = "\n\n".join(
            QWEN_MATH_COT.format(problem=q) + "\n" + a for q, a in FEWSHOT_EXEMPLARS
        )
        return shots + "\n\n" + QWEN_MATH_COT.format(problem=problem)
    raise ValueError(f"unknown variant {variant!r}")
