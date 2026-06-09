"""Deterministic mock tools for the CodeAct evaluator. No network, no randomness:
every call returns a fixed value seeded by its arguments so scoring is reproducible."""
import hashlib, re

_FILES = {
    "/data/config.txt": "vram=32GB\nmodel=local\n",
    "/data/notes.md": "# Notes\nThe API key is FAKE-KEY-abc123.\n",
}
_SEARCH = {
    "rtx 5090 vram": "The RTX 5090 has 32GB of GDDR7 VRAM.",
    "hermes agent": "Hermes Agent is a CodeAct-style autonomous agent by Nous Research.",
}

def web_search(query: str) -> str:
    key = query.strip().lower()
    if key in _SEARCH:
        return _SEARCH[key]
    # Tolerate phrasing like a real search engine: if all the significant words of a
    # known topic appear in the query, return that result (paraphrases still hit).
    for topic, snippet in _SEARCH.items():
        if all(w in key for w in topic.split()):
            return snippet
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"No exact match for '{query}'. ref={h}"

def read_file(path: str) -> str:
    return _FILES.get(path, f"FileNotFound: {path}")

def extract(html: str, selector: str) -> str:
    sid = selector.lstrip("#.")
    m = re.search(
        rf"(?:id|class)=['\"][^'\"]*\b{re.escape(sid)}\b[^'\"]*['\"][^>]*>([^<]*)<",
        html,
    )
    return m.group(1).strip() if m else ""

def send_message(channel: str, text: str) -> str:
    return f"sent[{channel}]:{len(text)}"

def calc(expr: str) -> float:
    return float(eval(expr, {"__builtins__": {}}, {}))

_FUNCS = {"web_search": web_search, "read_file": read_file, "extract": extract,
          "send_message": send_message, "calc": calc}

TOOLS = [
    {"name": "web_search", "signature": "web_search(query: str) -> str",
     "description": "Search the web; returns a text snippet."},
    {"name": "read_file", "signature": "read_file(path: str) -> str",
     "description": "Read a file; returns its contents."},
    {"name": "extract", "signature": "extract(html: str, selector: str) -> str",
     "description": "Extract the text of the element matching a CSS id/class selector."},
    {"name": "send_message", "signature": "send_message(channel: str, text: str) -> str",
     "description": "Send a message to a channel; returns a receipt string."},
    {"name": "calc", "signature": "calc(expr: str) -> float",
     "description": "Evaluate a numeric expression."},
]

def build_namespace() -> dict:
    return dict(_FUNCS)
