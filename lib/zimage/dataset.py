"""Prompt-set loading for the image-gen bench (validated — ADR-0003 spirit).

Mirrors ``lib.ace_step.dataset``: fail loudly on a malformed prompt set rather
than let a missing field surface as a confusing runtime error mid-GPU-run.
"""
import json

REQUIRED = ("name", "prompt", "category")


def load_prompts(path: str, limit=None) -> list:
    """Load and validate the image prompt set. Raises ValueError on missing keys."""
    rows = json.load(open(path))
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"prompt {i} ({r.get('name', '?')}) missing keys: {', '.join(missing)}")
    return rows[:limit] if limit else rows
