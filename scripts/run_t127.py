"""t127 orchestration: plan the 5 legs (LFM2.5 reasoning-on/off + the 3 default-regime
anchors), run tasks x replicates per leg through the EXISTING native tool-calling loop,
write result rows as JSONL, and project the full-sweep GPU-hours from a small timing
probe. A thin extension of lib/agentic/native/ (see run_native.py) — no new tools,
dispatch, or scoring: Tasks 1-3 (reasoning toggle + n_reasoning_tokens, T127_TASKS,
mcnemar/run_paired) are reused verbatim.

The offline-testable core is plan_legs/run_leg/write_results/expected_runs/
project_hours — pure functions, no server/network, driven by an injected client
factory (ScriptedClient in tests, LlamaClient for real). GPU serving (starting
llama-server per leg, Donald-safe) lives only in main(), which the night script
(a later task) drives; it is NOT unit-tested.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from lib.config import get, load_config
from lib.agentic.native.agent_loop import run_agent
from lib.agentic.native.client import LlamaClient
from lib.agentic.native.run_native import _short_tools
from lib.agentic.native.tasks_t127 import T127_TASKS
from lib.quality import start_llama_server, stop_llama_server

RESULTS_PATH = Path("results/t127/results.jsonl")


def plan_legs(cfg: dict) -> list[dict]:
    """Exactly 5 legs from the 4 configured models (config.yaml t127.models): the
    LFM2.5 model runs twice (reasoning-on AND reasoning-off — the A/B this bench
    exists to measure); every other model (the two anchors + the same-size control)
    runs once, think=True, regime="default" — they're identical in treatment, so no
    special-casing "control" vs "anchor" is needed. `cfg` is the loaded config dict
    (see lib.config.load_config), passed in so this stays offline-testable."""
    models = ((cfg or {}).get("t127") or {}).get("models") or []
    legs = []
    for m in models:
        if "lfm2.5" in m["slug"].lower():
            legs.append({"slug": m["slug"], "gguf": m["gguf"], "think": True,
                        "regime": "reasoning_on"})
            legs.append({"slug": m["slug"], "gguf": m["gguf"], "think": False,
                        "regime": "reasoning_off"})
        else:
            legs.append({"slug": m["slug"], "gguf": m["gguf"], "think": True,
                        "regime": "default"})
    return legs


def run_leg(leg: dict, tasks: list, reps: int, make_client, tools: list,
           dispatch=None, max_steps: int = 12) -> list[dict]:
    """Run every task x rep through run_agent for one leg, scored by the task's own
    deterministic `check(task, final_text)`. `make_client(leg)` is called fresh for
    every (task, rep) so a real run gets a clean LlamaClient(think=leg["think"]) per
    trajectory (no cross-talk) and tests can inject a factory returning a
    ScriptedClient instead (see tests/test_native_agent_loop.py's pattern)."""
    rows = []
    for task in tasks:
        for rep in range(reps):
            client = make_client(leg)
            result = run_agent(client, task["goal"], tools, max_steps=max_steps,
                              dispatch=dispatch)
            ok = bool(task["check"](task, result.final_text))
            rows.append({
                "slug": leg["slug"], "regime": leg["regime"], "think": leg["think"],
                "task_id": task["id"], "rep": rep, "ok": ok,
                "tool_calls": result.n_tool_calls, "bad_calls": result.bad_calls,
                "gen_tokens": result.n_tokens, "reasoning_tokens": result.n_reasoning_tokens,
                "stalled": result.stalled,
            })
    return rows


def write_results(rows: list, path: Path = RESULTS_PATH) -> None:
    """Append `rows` as JSONL (one result per line), creating the parent dir if
    needed. Append (not overwrite) so a multi-leg night can call this once per leg
    without clobbering earlier legs' rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def expected_runs(n_tasks: int, reps: int, n_legs: int = 5) -> int:
    return n_tasks * reps * n_legs


def project_hours(sec_per_task: float, n_tasks: int, reps: int, n_legs: int = 5) -> float:
    return sec_per_task * expected_runs(n_tasks, reps, n_legs) / 3600


def reasoning_tokens_ok(rows: list) -> bool:
    """Operational-safety check for the LFM2.5 reasoning A/B: True iff at least one
    reasoning_on row actually has reasoning_tokens > 0. If every reasoning_on row
    comes back at 0, llama-server most likely isn't splitting `reasoning_content`
    out of the response (needs `--reasoning-format` or a reasoning-splitting chat
    template), which makes the reasoning-cost axis meaningless -- see the loud
    warning in --probe below."""
    return any(r["reasoning_tokens"] > 0 for r in rows if r["regime"] == "reasoning_on")


def reasoning_off_is_off(rows: list, tol: int = 50) -> bool:
    """Symmetric safety check to reasoning_tokens_ok: True iff the reasoning_off leg
    actually suppressed reasoning (every reasoning_off row has reasoning_tokens <=
    tol). If an off row still carries hundreds of reasoning tokens, the served chat
    template is IGNORING the disable flag (enable_thinking/reasoning/reasoning_effort)
    and BOTH legs reasoned -- the on/off A/B is confounded and its comparison is
    meaningless. This is exactly the LFM2.5 confound (2026-08-06): the shipped GGUF
    template hard-forced `<think>` every turn and read none of the kwargs, so the first
    sweep compared reasoning-on to reasoning-on (p=1.0); a patched template that
    pre-closes the block (--chat-template-file) fixed it and the true result was a
    significant 26.7pt drop. reasoning_tokens_ok did NOT catch this -- it only checks
    that ON reasons, never that OFF stops. See the loud warning in --probe below."""
    off = [r for r in rows if r["regime"] == "reasoning_off"]
    return all(r["reasoning_tokens"] <= tol for r in off) if off else True


def _group_legs_by_gguf(legs: list) -> dict:
    """Legs that share a gguf (LFM2.5's reasoning-on/off pair) share one served
    model — only the per-request `think` flag differs — so they group under one
    server instead of two."""
    groups: dict = {}
    for leg in legs:
        groups.setdefault(leg["gguf"], []).append(leg)
    return groups


def _endpoint_configured(leg: dict) -> bool:
    """Guard for --probe: only attempt a real run if the leg's gguf actually exists
    on disk. 3 of the 4 t127.models entries are placeholder slugs/paths (to be
    finalized before the night) — attempting to serve a placeholder path would just
    crash llama-server, so skip cleanly instead."""
    gguf = leg.get("gguf")
    return bool(gguf) and Path(gguf).expanduser().exists()


def _probe_legs(legs: list) -> list:
    """The LFM2.5 reasoning-on/off legs + one anchor (the first default-regime leg
    encountered) — 3 legs, enough to see both the reasoning toggle's cost and one
    cross-model data point without probing all 5."""
    anchor = next(l for l in legs if l["regime"] == "default")
    return [l for l in legs if l["regime"] != "default"] + [anchor]


def donald(action: str) -> None:
    """Start/stop the resident llama-server.service (mirrors scripts/diff_vs_ar.py's
    `donald()` helper) so the subject model can take port 8090 for its run."""
    subprocess.run(["sudo", "-n", "systemctl", action, "llama-server.service"], check=False)


def _serve_and_run(gguf: str, group_legs: list, tasks: list, reps: int, max_steps: int,
                   tools: list, api_base: str) -> list:
    """Donald-safe: drain the resident server, serve `gguf` on the configured port,
    run every leg in `group_legs` against it (they share this one gguf), then ALWAYS
    stop the subject server and restore Donald, even on error."""
    def make_client(leg):
        return LlamaClient(api_base, leg["slug"], think=leg["think"])

    donald("stop")
    time.sleep(3)
    proc = start_llama_server(str(Path(gguf).expanduser()))
    rows = []
    try:
        for leg in group_legs:
            rows += run_leg(leg, tasks, reps, make_client, tools, max_steps=max_steps)
    finally:
        stop_llama_server(proc)
        donald("start")
    return rows


def main():
    ap = argparse.ArgumentParser(description="t127: plan legs, run the full sweep, or "
                                             "probe its duration.")
    ap.add_argument("--probe", action="store_true",
                    help="run a 3-task x1-rep timing probe (LFM2.5 on/off + one anchor) "
                         "instead of the full sweep")
    args = ap.parse_args()

    cfg = load_config()
    legs = plan_legs(cfg)
    tools = _short_tools()
    port = get("llama_cpp.server_port", 8090)
    api_base = f"http://127.0.0.1:{port}/v1"
    n_tasks = get("t127.n_tasks", len(T127_TASKS))
    reps = get("t127.replicates", 3)
    max_steps = get("t127.max_steps", 12)

    if args.probe:
        probe_legs = _probe_legs(legs)
        if not all(_endpoint_configured(l) for l in probe_legs):
            print("[t127 probe] skipped: a probe leg's gguf isn't on disk yet "
                  "(placeholder in config.yaml t127.models) -- finalize the 4 model "
                  "paths before probing.")
            return
        t0 = time.time()
        rows = []
        for gguf, group in _group_legs_by_gguf(probe_legs).items():
            rows += _serve_and_run(gguf, group, T127_TASKS[:3], reps=1, max_steps=max_steps,
                                   tools=tools, api_base=api_base)
        elapsed = time.time() - t0
        sec_per_task = elapsed / max(1, len(rows))
        hours = project_hours(sec_per_task, n_tasks, reps)
        print(f"[t127 probe] {len(rows)} runs in {elapsed:.1f}s -> {sec_per_task:.2f}s/task "
              f"-> projected full sweep ({expected_runs(n_tasks, reps)} runs): {hours:.2f}h")
        if not reasoning_tokens_ok(rows):
            print("[t127 probe] !!! WARNING !!! every reasoning_on row has "
                  "reasoning_tokens == 0 -- llama-server likely isn't splitting "
                  "reasoning_content out of the response (needs --reasoning-format "
                  "or a reasoning-splitting chat template). The LFM2.5 reasoning "
                  "A/B's cost axis would be MEANINGLESS as configured -- fix this "
                  "before running the full sweep.")
        if not reasoning_off_is_off(rows):
            print("[t127 probe] !!! WARNING !!! reasoning_off rows still carry "
                  "reasoning tokens -- the served chat template is IGNORING the "
                  "disable flag, so BOTH legs reasoned and the on/off A/B is "
                  "CONFOUNDED (the LFM2.5 template-forces-<think> confound). Serve a "
                  "patched template that pre-closes the think block "
                  "(--chat-template-file) and re-run until reasoning_off shows ~0 "
                  "reasoning tokens BEFORE trusting the on/off comparison.")
        return

    for gguf, group in _group_legs_by_gguf(legs).items():
        rows = _serve_and_run(gguf, group, T127_TASKS, reps, max_steps, tools, api_base)
        write_results(rows, RESULTS_PATH)
        print(f"[t127] wrote {len(rows)} rows for {group[0]['slug']} "
              f"({', '.join(l['regime'] for l in group)}) -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
