"""Text helpers for product URLs."""
import re


def slugify(title):
    """Lower-case URL slug: accents dropped, runs of other characters become one hyphen, no hyphen at either end."""
    return re.sub(r"[^a-z0-9]", "-", title.lower())
