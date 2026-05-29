"""Shared helpers for the agentic evaluators."""
import re
from pathlib import Path
from lib.agentic.mock_tools import TOOLS

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def extract_code(text: str) -> str:
    """Pull the Python code action from a model response."""
    t = (text or "").strip()
    tag = "</think>"
    if tag in t:
        t = t[t.rfind(tag) + len(tag):].strip()
    m = _FENCE.search(t)
    if m:
        return m.group(1).strip()
    return t  # assume the whole response is code

def build_tool_doc() -> str:
    lines = ["Available tools (call them as plain Python functions):"]
    for tool in TOOLS:
        lines.append(f"- {tool['signature']}  # {tool['description']}")
    return "\n".join(lines)

_SYNTHETIC_FALLBACK = (
    "You are an autonomous agent. You act by writing Python code inside a "
    "```python``` block that calls the available tools and assigns the final "
    "answer to a variable named `result`.\n\n" + build_tool_doc()
)

def load_system_prompt(path: str) -> str:
    """Load the extracted Hermes system prompt; fall back to a synthetic one
    (so tests/dev work before the prompt fixture has been generated)."""
    p = Path(path).expanduser()
    if p.exists():
        return p.read_text()
    return _SYNTHETIC_FALLBACK


# Light prompt for the CodeAct eval. The real Hermes prompt conditions models to
# emit incremental tool calls in Hermes's own markup (<tool_parsing>...) and await
# observations, which fights the one-shot self-contained-block format CodeAct scores.
# CodeAct measures pure code-orchestration capability under this light prompt; the
# real incremental-agent behavior is what Phase B (real Hermes harness) measures.
CODEACT_SYSTEM_PROMPT = (
    "You are a precise coding assistant operating in a CodeAct loop. For each task, "
    "respond with a SINGLE ```python``` code block that calls the provided tools as "
    "plain Python functions and assigns the final answer to a variable named `result`. "
    "Output only that code block — no prose, and do NOT use any tool-call markup or XML "
    "tags; just write the Python."
)
