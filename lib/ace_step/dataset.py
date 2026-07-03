"""Fixed prompt set for the music-gen bench — small, reproducible, committed.

A prompt row is ``{name, caption, lyrics, duration_s}``. ``duration_s`` is the
default per row; the run overrides it per matrix cell (30/120/240 s).
"""
import json

_REQUIRED = ("name", "caption", "lyrics", "duration_s")


def load_prompts(path: str, limit=None) -> list:
    """Load + validate the prompt manifest. ``limit`` truncates for smoke runs."""
    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        missing = [k for k in _REQUIRED if k not in r]
        if missing:
            raise ValueError(f"prompt {r.get('name', '?')!r} missing keys: {missing}")
    return rows[:limit] if limit is not None else rows
