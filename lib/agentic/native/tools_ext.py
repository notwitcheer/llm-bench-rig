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
