"""Provenance block for a benchmark result.

Every row on the board is a single greedy pass through one llama.cpp build with
one gguf file and one set of quality settings. None of that used to be written
into the result, so a reader could not tell which build, which file or which
config produced a number. `collect_provenance` gathers what a sceptic asks for:

- llama-server `/props` fields (build_info, chat template hash, n_ctx, slots)
- the exact server command line
- sha256 of the gguf file (cached in a `<gguf>.sha256` sidecar when writable)
- harness git sha plus a dirty flag
- python, `datasets` and `httpx` versions
- the resolved `quality.*` config (sample, think, mmlu_limit, mc_gate_tokens, limit)
- a utc timestamp

Collection must never crash a run: `record_provenance` wraps everything and
writes `{"error": ...}` on failure. The pure helpers (template hash, git parsing,
config resolution) are unit tested without a network or a gpu.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Above this size hashing takes long enough to stall a run; record the skip instead.
GGUF_SHA_MAX_BYTES = 60 * 1024 ** 3
QUALITY_CONFIG_KEYS = ("sample", "think", "mmlu_limit", "mc_gate_tokens", "limit")
# Defaults mirror lib.quality.run_quality_bench so the recorded config is what ran.
QUALITY_CONFIG_DEFAULTS = {
    "sample": None,
    "think": True,
    "mmlu_limit": None,
    "mc_gate_tokens": 50,
    "limit": None,
}


# --- pure helpers (tested) ---

def chat_template_sha256(template) -> str | None:
    """sha256 of the chat template string as the server reports it. None if absent."""
    if not isinstance(template, str) or not template:
        return None
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def summarise_props(props: dict | None) -> dict:
    """Keep the /props fields that shape a score; drop the rest of the payload."""
    if not isinstance(props, dict):
        return {"error": "props missing or not an object"}
    gen = props.get("default_generation_settings") or {}
    out = {
        "build_info": props.get("build_info"),
        "version": props.get("version"),
        "chat_template_sha256": chat_template_sha256(props.get("chat_template")),
        "n_ctx": gen.get("n_ctx"),
        "total_slots": props.get("total_slots"),
    }
    model_path = props.get("model_path")
    if model_path:
        out["model_path"] = model_path
    return out


def parse_git_state(rev_parse_output: str, status_output: str) -> dict:
    """Turn `git rev-parse HEAD` and `git status --porcelain` text into {sha, dirty}."""
    sha = (rev_parse_output or "").strip().splitlines()
    sha = sha[0].strip() if sha else ""
    if not sha or len(sha) < 7 or any(c not in "0123456789abcdef" for c in sha):
        return {"sha": "unknown", "dirty": None}
    dirty = any(line.strip() for line in (status_output or "").splitlines())
    return {"sha": sha, "dirty": dirty}


def resolve_quality_config(get) -> dict:
    """Resolve the quality.* keys through a config getter `get(key, default)`."""
    return {k: get(f"quality.{k}", QUALITY_CONFIG_DEFAULTS[k]) for k in QUALITY_CONFIG_KEYS}


def sha256_of_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def gguf_sha256(model_path, max_bytes: int = GGUF_SHA_MAX_BYTES) -> dict:
    """sha256 of a single model file, read from or written to a sidecar when allowed.

    Returns {sha256, size_bytes, cached, skipped, reason}. The sidecar is the
    `sha256sum` text format so `sha256sum -c <file>.sha256` verifies it.
    """
    p = Path(model_path).expanduser()
    out = {"sha256": None, "size_bytes": None, "cached": False, "skipped": False, "reason": None}
    if not p.is_file():
        out.update(skipped=True, reason="not a single file")
        return out
    size = p.stat().st_size
    out["size_bytes"] = size
    sidecar = p.with_name(p.name + ".sha256")
    if sidecar.is_file():
        try:
            first = sidecar.read_text().split()
            if first and len(first[0]) == 64:
                out.update(sha256=first[0], cached=True)
                return out
        except OSError:
            pass
    if size > max_bytes:
        out.update(skipped=True, reason=f"file larger than {max_bytes} bytes, hash skipped")
        return out
    digest = sha256_of_file(p)
    out["sha256"] = digest
    if os.access(p.parent, os.W_OK):
        try:
            sidecar.write_text(f"{digest}  {p.name}\n")
        except OSError:
            pass
    return out


def harness_git_state(repo_root: Path = REPO_ROOT) -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return {"sha": "unknown", "dirty": None, "error": str(e)}
    if rev.returncode != 0:
        return {"sha": "unknown", "dirty": None, "error": rev.stderr.strip()[:200]}
    return parse_git_state(rev.stdout, status.stdout)


def package_versions(names=("datasets", "httpx")) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def fetch_props(api_base: str, timeout: float = 10) -> dict:
    """GET llama-server /props. `api_base` may end in /v1; /props lives at the root."""
    import httpx
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    resp = httpx.get(f"{root}/props", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def collect_provenance(api_base: str | None, model_path: str,
                       server_command: list[str] | None,
                       quality_config: dict | None,
                       repo_root: Path = REPO_ROOT) -> dict:
    """Assemble the block. Each part records its own error instead of raising."""
    prov = {
        "source": "run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": package_versions(),
        "harness_git": harness_git_state(repo_root),
        "server_command": list(server_command) if server_command else None,
        "quality_config": dict(quality_config) if quality_config else None,
    }
    if api_base:
        try:
            prov["server_props"] = summarise_props(fetch_props(api_base))
        except Exception as e:  # noqa: BLE001 - never crash a run
            prov["server_props"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        prov["server_props"] = {"error": "no api_base"}
    try:
        prov["gguf"] = gguf_sha256(model_path)
    except Exception as e:  # noqa: BLE001
        prov["gguf"] = {"error": f"{type(e).__name__}: {e}"}
    return prov


def attach_to_meta(meta_path: Path, provenance: dict) -> None:
    """Add a `provenance` key to an existing meta.json, leaving other keys unchanged."""
    meta_path = Path(meta_path)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text() or "{}")
    meta["provenance"] = provenance
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(meta_path)


def record_provenance(results_dir, api_base: str | None, model_path: str,
                      server_command: list[str] | None,
                      quality_config: dict | None) -> dict:
    """Collect and write meta.json['provenance']; on any failure write {'error': ...}."""
    meta_path = Path(results_dir) / "meta.json"
    try:
        prov = collect_provenance(api_base, model_path, server_command, quality_config)
    except Exception as e:  # noqa: BLE001
        prov = {"error": f"{type(e).__name__}: {e}"}
    try:
        attach_to_meta(meta_path, prov)
    except Exception as e:  # noqa: BLE001
        print(f"[provenance] could not write {meta_path}: {e}", file=sys.stderr)
    return prov
