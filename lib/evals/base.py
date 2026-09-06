import re
import time
from collections import deque

import httpx


def _strip_blank_edges(s: str) -> str:
    """Strip leading/trailing blank lines and trailing whitespace per line.

    Unlike str.strip(), this does NOT remove leading spaces from content lines,
    so per-line indentation is preserved.  Used by LLMClient.chat() when
    preserve_indent=True (e.g. HumanEval code bodies).
    """
    lines = s.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(l.rstrip() for l in lines)


class LLMClient:
    """Sync HTTP client for llama-server chat completions."""

    def __init__(self, api_base: str, model_name: str, timeout: float = 300,
                 think: bool = True):
        self.url = f"{api_base.rstrip('/')}/chat/completions"
        self.model = model_name
        self.think = think
        self.last_completion_tokens: int | None = None
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], max_tokens: int = 2048,
             temperature: float = 0, stop: list[str] | None = None,
             preserve_indent: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if not self.think:
            # Suppress reasoning across template families in one payload: Gemma/Qwen read
            # `enable_thinking`, Cohere/North (cohere2moe) read `reasoning`/`reasoning_effort`.
            # Each template uses the key it knows and ignores the rest. Requires the server
            # to run with --jinja (build_server_command) or these kwargs are dropped.
            payload["chat_template_kwargs"] = {
                "enable_thinking": False,
                "reasoning": False,
                "reasoning_effort": "none",
                # Muse Glimmer (muse_glimmer) reads reasoning_strength, default high (2026-08-10)
                "reasoning_strength": "low",
            }
        if stop:
            payload["stop"] = stop
        _clean = _strip_blank_edges if preserve_indent else str.strip
        for attempt in range(3):
            try:
                resp = self._client.post(self.url, json=payload)
                resp.raise_for_status()
                body = resp.json()
                msg = body["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                usage = (body.get("usage") or {}).get("completion_tokens")
                if usage is None:
                    # Server omitted usage: chars≈4/tok estimate over the raw
                    # completion, so the pace gate (t103) can never silently disarm.
                    usage = len(content + reasoning) // 4 + 1
                self.last_completion_tokens = usage
                text = _clean(content)
                if not text:
                    text = _clean(reasoning)
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


# --- MC completion-length instrument gate (t103, ADR-0004) ---

class InstrumentGateError(RuntimeError):
    """The eval instrument (serving stack) is degenerate — scores are meaningless.

    Raised by CompletionLengthGate; distinct from any subject/model failure so a
    poisoned run can never be banked as a score (2026-07-16 incident).
    """

    def __init__(self, mean: float, threshold: float, n_samples: int):
        self.mean = mean
        self.threshold = threshold
        self.n_samples = n_samples
        super().__init__(
            f"mean MC completion length {mean:.0f} tok over last {n_samples} "
            f"answers exceeds gate threshold {threshold:.0f} — degenerate serving, "
            f"aborting suite (think-off MC answers run 1-7 tok)"
        )


class CompletionLengthGate:
    """Rolling-window pace watchdog for multiple-choice evals.

    Degenerate server output can still parse to letters; pace is the mechanical
    tell (~850 tok/answer poisoned vs 1-7 healthy). Armed only for think-off
    runs — think-on completions are legitimately long, so those only log.
    """

    def __init__(self, threshold: float | None, armed: bool,
                 window: int = 25, min_samples: int = 10):
        self.threshold = threshold
        self.armed = armed
        self.min_samples = min_samples
        self._window = deque(maxlen=window)

    @property
    def mean(self) -> float | None:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)

    def observe(self, n_tokens: int) -> None:
        self._window.append(n_tokens)
        if not self.armed or self.threshold is None:
            return
        if len(self._window) >= self.min_samples and self.mean > self.threshold:
            raise InstrumentGateError(self.mean, self.threshold, len(self._window))


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
