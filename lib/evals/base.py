import re
import time
import httpx


class LLMClient:
    """Sync HTTP client for llama-server chat completions."""

    def __init__(self, api_base: str, model_name: str, timeout: float = 120):
        self.url = f"{api_base.rstrip('/')}/chat/completions"
        self.model = model_name
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], max_tokens: int = 2048,
             temperature: float = 0, stop: list[str] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        for attempt in range(3):
            try:
                resp = self._client.post(self.url, json=payload)
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                text = (msg.get("content") or "").strip()
                if not text:
                    text = (msg.get("reasoning_content") or "").strip()
                return text
            except (httpx.HTTPError, KeyError, IndexError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- Letter choice parsing (MMLU, ARC, HellaSwag) ---

_ANSWER_PATTERN = re.compile(
    r'(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-E])\)?', re.IGNORECASE
)

def parse_choice(text: str, valid: str = "ABCD") -> str | None:
    text = text.strip()
    if not text:
        return None

    tag = "</think>"
    if tag in text:
        text = text[text.rfind(tag) + len(tag):].strip()
        if not text:
            return None

    if len(text) < 20:
        stripped = text.lstrip("*_(\"'`[")
        if stripped and stripped[0] in valid:
            return stripped[0]

    matches = list(_ANSWER_PATTERN.finditer(text))
    if matches:
        last = matches[-1].group(1).upper()
        if last in valid:
            return last

    stripped = text.lstrip("*_(\"'`[")
    if stripped and stripped[0] in valid:
        return stripped[0]

    tail = text[-200:]
    for m in reversed(list(re.finditer(r'\b([' + valid + r'])\b', tail))):
        return m.group(1)

    return None


# --- Numeric answer parsing (GSM8K) ---

def parse_number(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    Returns the answer as a cleaned string for exact-match comparison.
    """
    text = text.strip()
    if not text:
        return None

    tag = "</think>"
    if tag in text:
        text = text[text.rfind(tag) + len(tag):].strip()

    # #### pattern (GSM8K gold format)
    m = re.search(r'####\s*([-−]?\d[\d,]*\.?\d*)', text)
    if m:
        return _clean_num(m.group(1))

    # "the answer is X" pattern
    m = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*([-−]?\d[\d,]*\.?\d*)', text, re.IGNORECASE)
    if m:
        return _clean_num(m.group(1))

    # Last number in text
    numbers = re.findall(r'[-−]?\d[\d,]*\.?\d*', text)
    if numbers:
        return _clean_num(numbers[-1])

    return None


def _clean_num(s: str) -> str:
    s = s.replace(",", "").replace("−", "-")
    try:
        val = float(s)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return s


# --- Checkpoint helpers ---

import json
from pathlib import Path

def save_checkpoint(path: Path | None, data: dict):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(path)


def load_checkpoint(path: Path | None) -> dict | None:
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    return None
