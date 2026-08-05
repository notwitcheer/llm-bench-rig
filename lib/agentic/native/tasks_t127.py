"""Frozen t127 30-task tool-use / instruction-following battery. Reuses the EXISTING
native task dict schema (id/axis/opt_calls/goal/answer/check, see tasks.py) plus a new
`tier` key carrying the t127 axis: single_tool | multi_step | multi_turn_if (10 each).

`check` on every task is a deterministic `check(task, final_text) -> bool` callable —
the exact signature `run_native.py`/`run_agent` call (`check(t, res.final_text)`; see
tasks.py's module-level `check()`). No LLM judge anywhere.

Tiers:
  single_tool   — one correct tool call, answer verified in the final text.
                  Reused verbatim from tasks.py (coding_* + the opt_calls==1 distractor_*
                  tasks): both are already single-tool-call tasks under the real schema.
  multi_step    — >=2 chained tool calls with carried state (search/read -> compute,
                  optionally -> send). Reused verbatim from tasks.py's chain_*/multistep_*
                  tasks with opt_calls >= 2.
  multi_turn_if — compound/evolving instructions (incl. a conditional branch) ending in a
                  STRUCTURED answer (strict JSON, named fields). New tasks: still built only
                  from the existing mock tools + fixtures (web_search, read_file, calc,
                  read_record, fetch, send_message). Checked by combining
                  lib.agentic.instruction.check_compliance (json_keys / forbids, structural)
                  with an exact-value check on the parsed JSON object (deterministic).

compute_sha()/SUITE_SHA freeze the suite's identity; the suite is also dumped to
data/agentic/t127_suite.json + data/agentic/t127_suite.sha (see dump_suite()).
"""
import hashlib
import json
from pathlib import Path

from lib.agentic.instruction import check_compliance
from lib.agentic.native.tasks import TASKS, _num, _contains

_BY_ID = {t["id"]: t for t in TASKS}


def _reuse(task_id: str, tier: str) -> dict:
    """Shallow-copy an existing tasks.py task dict, stamped with its t127 tier."""
    t = dict(_BY_ID[task_id])
    t["tier"] = tier
    return t


def _json_check(keys: list, values: dict, forbids: list | None = None):
    """Build a `check(task, final_text) -> bool` for a structured (JSON) answer.
    Structural checks (all `keys` present, none of `forbids` present) are delegated to
    check_compliance (json_keys / forbids types, byte-for-byte the same helper the
    instruction-following axis uses). Value correctness is then verified by parsing the
    same JSON and comparing each of `values` for exact equality — still fully
    deterministic, no fuzzy/LLM matching.
    """
    forbids = forbids or []

    def fn(task, final_text):
        if not check_compliance(final_text, {"type": "json_keys", "value": keys}):
            return False
        for word in forbids:
            if not check_compliance(final_text, {"type": "forbids", "value": word}):
                return False
        t = (final_text or "").strip()
        tag = "</think>"
        if tag in t:
            t = t[t.rfind(tag) + len(tag):].strip()
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            return False
        return all(obj.get(k) == v for k, v in values.items())

    return fn


def _mtif(id_, goal, keys, values, opt_calls, forbids=None, axis="instruction_follow"):
    check_spec = {"type": "json", "keys": keys, "values": values}
    if forbids:
        check_spec["forbids"] = forbids
    return {
        "id": id_, "tier": "multi_turn_if", "axis": axis, "opt_calls": opt_calls,
        "goal": goal, "answer": json.dumps(values, sort_keys=True),
        "check_spec": check_spec,
        "check": _json_check(keys, values, forbids),
    }


# --- tier 1: single_tool (10) — reused verbatim, all opt_calls == 1 ------------------
_SINGLE_TOOL_IDS = [
    "coding_sum_evens", "coding_factorial5", "coding_fib10",
    "coding_count_vowels", "coding_primes_under20",
    "distractor_vram_lure", "distractor_right_search", "distractor_calc_real",
    "distractor_config_not_cache", "distractor_calc_not_legacy",
]

# --- tier 2: multi_step (10) — reused verbatim, all opt_calls >= 2 -------------------
_MULTI_STEP_IDS = [
    "chain_vram_x2", "chain_config_plus8", "chain_config_half",
    "chain_vram_x3_plus4", "chain_threehop",
    "multistep_config_x4", "multistep_search_x5", "multistep_search_send",
    "multistep_model_send", "multistep_config_double_send",
]

SINGLE_TOOL_TASKS = [_reuse(i, "single_tool") for i in _SINGLE_TOOL_IDS]
MULTI_STEP_TASKS = [_reuse(i, "multi_step") for i in _MULTI_STEP_IDS]

# --- tier 3: multi_turn_if (10) — new, built only from the existing mock tools -------
MULTI_TURN_IF_TASKS = [
    _mtif(
        "mtif_vram_json",
        "Use the web_search tool to find the RTX 5090's VRAM in GB. Then respond with "
        "EXACTLY one JSON object and nothing else, with two integer keys: \"vram_gb\" "
        "and \"doubled\" (doubled = vram_gb * 2).",
        ["vram_gb", "doubled"], {"vram_gb": 32, "doubled": 64}, opt_calls=1,
    ),
    _mtif(
        "mtif_config_json",
        "Read the file /data/config.txt. Then respond with EXACTLY one JSON object and "
        "nothing else, with two keys: \"vram_gb\" (integer) and \"model\" (string), "
        "matching the values in the file.",
        ["vram_gb", "model"], {"vram_gb": 32, "model": "local"}, opt_calls=1,
    ),
    _mtif(
        "mtif_calc_status_json",
        "Using the calc tool, compute 12 * 8. If the result is greater than 90, respond "
        "with EXACTLY the JSON object {\"result\": <the number>, \"status\": \"high\"}; "
        "otherwise respond with {\"result\": <the number>, \"status\": \"low\"}. Output "
        "ONLY the JSON, nothing else.",
        ["result", "status"], {"result": 96, "status": "high"}, opt_calls=1,
    ),
    _mtif(
        "mtif_record_json",
        "Use read_record to read record 007. Then respond with EXACTLY one JSON object "
        "and nothing else, with two keys: \"owner\" and \"status\", matching the record.",
        ["owner", "status"], {"owner": "ops", "status": "active"}, opt_calls=1,
        axis="error_recovery",
    ),
    _mtif(
        "mtif_health_json",
        "Fetch https://api.local/health. Then respond with EXACTLY one JSON object and "
        "nothing else, with two keys: \"build\" (integer) and \"healthy\" (boolean, true "
        "if the reported status is healthy).",
        ["build", "healthy"], {"build": 9562, "healthy": True}, opt_calls=1,
        axis="error_recovery",
    ),
    _mtif(
        "mtif_notes_json",
        "Read the file /data/notes.md. It contains a fake placeholder token (for "
        "testing) that begins with 'FAKE-'. Respond with EXACTLY one JSON object and "
        "nothing else, with two keys: \"token\" (the placeholder token, verbatim) and "
        "\"length\" (the integer character length of the token string).",
        ["token", "length"], {"token": "FAKE-KEY-abc123", "length": 15}, opt_calls=1,
    ),
    _mtif(
        "mtif_vram_tier_json",
        "Use web_search to find the RTX 5090's VRAM in GB. If the VRAM is at least 24, "
        "respond with EXACTLY the JSON object {\"vram_gb\": <int>, \"tier\": \"high\"}; "
        "otherwise respond with {\"vram_gb\": <int>, \"tier\": \"low\"}. Output ONLY the "
        "JSON, nothing else.",
        ["vram_gb", "tier"], {"vram_gb": 32, "tier": "high"}, opt_calls=1,
        axis="distractor",
    ),
    _mtif(
        "mtif_calc_evolving_json",
        "Using calc, first compute 9 squared (9**2) and call that 'squared'. Then, using "
        "that result, compute squared + 7 and call that 'final'. Respond with EXACTLY "
        "one JSON object and nothing else, with keys \"squared\" and \"final\".",
        ["squared", "final"], {"squared": 81, "final": 88}, opt_calls=2,
        axis="chain",
    ),
    _mtif(
        "mtif_search_send_json",
        "Use web_search to find the RTX 5090's VRAM in GB, then send that number to "
        "channel #hardware with send_message. Finally respond with EXACTLY one JSON "
        "object and nothing else, with keys \"vram_gb\" (integer) and \"sent\" (boolean, "
        "true once you have sent the message).",
        ["vram_gb", "sent"], {"vram_gb": 32, "sent": True}, opt_calls=2,
        axis="multistep",
    ),
    _mtif(
        "mtif_config_forbid_json",
        "Read /data/config.txt to find the vram value in GB. Respond with EXACTLY one "
        "JSON object and nothing else, with a single key \"vram_gb\" whose value is that "
        "integer. Do not use the word 'model' anywhere in your response.",
        ["vram_gb"], {"vram_gb": 32}, opt_calls=1, forbids=["model"],
    ),
]

T127_TASKS: list[dict] = SINGLE_TOOL_TASKS + MULTI_STEP_TASKS + MULTI_TURN_IF_TASKS


def check(task: dict, final_text: str) -> bool:
    """Same shape as tasks.py's module-level `check()` (kept for style parity / drop-in
    use); every T127 task already carries a `check` callable so the fallback never fires."""
    fn = task.get("check")
    if fn:
        return bool(fn(task, final_text))
    return (task.get("answer", "") or "").lower() in (final_text or "").lower()


def compute_sha(tasks: list) -> str:
    """sha256 over the sorted (id, goal, tier) triples — the suite's frozen identity.
    Deterministic regardless of list order; changes iff a task's id/goal/tier changes,
    or a task is added/removed."""
    rows = sorted((t["id"], t.get("goal", ""), t["tier"]) for t in tasks)
    blob = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


SUITE_SHA = compute_sha(T127_TASKS)

_REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_JSON_PATH = _REPO_ROOT / "data" / "agentic" / "t127_suite.json"
SUITE_SHA_PATH = _REPO_ROOT / "data" / "agentic" / "t127_suite.sha"


def _serializable(t: dict) -> dict:
    """Drop the live `check` callable (not JSON-able); everything else that describes
    the task (including `check_spec` where present) is kept."""
    return {k: v for k, v in t.items() if k != "check"}


def dump_suite(json_path: Path = SUITE_JSON_PATH, sha_path: Path = SUITE_SHA_PATH) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([_serializable(t) for t in T127_TASKS], indent=2) + "\n")
    sha_path.write_text(SUITE_SHA + "\n")


if __name__ == "__main__":
    dump_suite()
    print(f"wrote {SUITE_JSON_PATH} + {SUITE_SHA_PATH} (sha={SUITE_SHA})")
