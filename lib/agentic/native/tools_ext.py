"""Native-only extension tools for the hardened agentic bench: big-document reads
(long-context), deterministic decoys, and deterministically-failing tools. Kept OUT
of the shared mock_tools.py so the Hermes suite stays untouched. All deterministic."""
import hashlib

_FILLER = ("the system processed the request and logged the result for later review "
           "each subsystem reported nominal status across the monitored interval and "
           "no anomalies were detected by the watchdog during the sampling window ").split()


def vault_code(seed: int) -> str:
    return hashlib.sha256(f"vault-{seed}".encode()).hexdigest()[:8].upper()


def make_doc(seed: int, target_tokens: int, needle_depth: float) -> str:
    """Deterministic filler document of ~target_tokens (token proxy = word*1.3) with one
    unique needle sentence at needle_depth (0..1). Reproducible for fixed args."""
    needle = f"IMPORTANT: the activation code for vault {seed} is {vault_code(seed)}. "
    target_words = max(1, int(target_tokens / 1.3))
    words = [_FILLER[i % len(_FILLER)] for i in range(target_words)]
    at = min(len(words), int(len(words) * needle_depth))
    return " ".join(words[:at]) + " " + needle + " ".join(words[at:])


# doc_id -> (seed, target_tokens, needle_depth). Two tiers x two depths.
_DOCS = {
    "logs-32k-early":  (101, 32000, 0.25),
    "logs-32k-late":   (102, 32000, 0.75),
    "logs-128k-early": (103, 128000, 0.25),
    "logs-128k-late":  (104, 128000, 0.75),
}


def read_doc(doc_id: str) -> str:
    if doc_id not in _DOCS:
        return f"ERROR: unknown doc_id '{doc_id}'. Known ids: {', '.join(_DOCS)}"
    return make_doc(*_DOCS[doc_id])


# --- error-recovery tools: deterministic failure on the obvious call + a recovery hint ---
_RECORDS = {"007": "status=active; owner=ops", "042": "status=archived; owner=root"}


def read_record(rec_id: str) -> str:
    if not (rec_id.isdigit() and len(rec_id) == 3):
        return f"ERROR: record '{rec_id}' not found; ids are zero-padded to 3 digits (e.g. 007)"
    return _RECORDS.get(rec_id, f"ERROR: no such record {rec_id}")


def fetch(url: str) -> str:
    if url.startswith("http://"):
        return "503 Service Unavailable: insecure transport; use https://"
    if url == "https://api.local/health":
        return "ok: build=9562 status=healthy"
    return f"404 Not Found: {url}"


# --- decoys: plausible names, wrong/empty/lure outputs (distractor axis) ---
def web_search_v2(query: str) -> str:
    return "No results (web_search_v2 is deprecated). Use web_search."


def calc_legacy(expr: str) -> str:
    return "ERR: calc_legacy is disabled; use calc"


def read_cache(path: str) -> str:
    return "stale cache: vram=24GB model=old"          # lure (real value is 32GB)


EXT_SHORT_TOOLS = [
    {"name": "read_record", "signature": "read_record(rec_id: str) -> str",
     "description": "Read a record by its id; returns the record contents."},
    {"name": "fetch", "signature": "fetch(url: str) -> str",
     "description": "HTTP GET a URL; returns the response."},
    {"name": "web_search_v2", "signature": "web_search_v2(query: str) -> str",
     "description": "Search the web (v2)."},
    {"name": "calc_legacy", "signature": "calc_legacy(expr: str) -> str",
     "description": "Evaluate a numeric expression (legacy)."},
    {"name": "read_cache", "signature": "read_cache(path: str) -> str",
     "description": "Read a cached copy of a file."},
]
EXT_LONG_TOOLS = [
    {"name": "read_doc", "signature": "read_doc(doc_id: str) -> str",
     "description": "Read a full document by id; returns its entire text."},
]
EXT_NS = {"read_record": read_record, "fetch": fetch, "web_search_v2": web_search_v2,
          "calc_legacy": calc_legacy, "read_cache": read_cache, "read_doc": read_doc}

# commit axis: apply_fix is executed by a per-run dispatch (lib/agentic/native/commit.py),
# not via EXT_NS — its side effect (world_state) is per-run, not a static function.
COMMIT_TOOLS = [
    {"name": "apply_fix", "signature": "apply_fix(target: str, value: str) -> str",
     "description": "Commit a change: set <target> to <value>. This is the ONLY way to actually "
                    "apply a fix — describing the change in your answer is not enough."},
]
