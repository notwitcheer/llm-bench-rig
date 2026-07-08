"""Prompt-set loading for the audio-video bench (validated — ADR-0003 spirit).

Mirrors ``lib.zimage.dataset``: fail loudly on a malformed prompt set rather
than let a missing field surface as a confusing runtime error mid-GPU-run.
"""
import json

REQUIRED = ("name", "prompt", "category")


def load_prompts(path: str, limit=None) -> list:
    """Load and validate the video prompt set. Raises ValueError on missing keys."""
    rows = json.load(open(path))
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"prompt {i} ({r.get('name', '?')}) missing keys: {', '.join(missing)}")
    return rows[:limit] if limit else rows
