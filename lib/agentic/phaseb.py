"""Phase B (v2) — run the top finishers through the REAL Hermes Agent on a harder,
multi-step task battery, with richer metrics, to validate the Phase A ranking.

For each (model, task, repeat): start llama-server (64K), point Hermes at it, run
`hermes -z "<prompt>" --yolo`, then run the task's objective FILESYSTEM check.
Metrics per task: completion (pass/fail), turns (model calls, from the llama-server
log), generation tokens, wall-clock. Aggregated per model: completion rate, plus
avg turns / tokens / time over the runs that completed.

Run on capsule:  ~/benchmark-rig/.venv/bin/python -m lib.agentic.phaseb
"""
import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

from lib.agentic.phaseb_tasks import TASKS


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (k passes of n runs),
    returned as percentages. Closed-form, no deps. Unlike the normal approximation
    it stays sane at the extremes — at k==n it gives an upper-bounded interval, not
    a zero-width 100%–100%. This is the interval that decides whether two models'
    completion rates actually differ or are a statistical tie."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, round(100 * (center - half), 1))
    hi = min(100.0, round(100 * (center + half), 1))
    return (lo, hi)

HOME = Path.home()
HERMES = HOME / "hermes-src" / ".venv" / "bin" / "hermes"
CONFIG = HOME / ".hermes" / "config.yaml"
SERVER = HOME / "llama.cpp" / "build" / "bin" / "llama-server"
WORK = Path("/tmp/phaseb")
SRV_LOG = Path("/tmp/phaseb_srv.log")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "phaseb"
PORT = 8090
CTX = 65536  # Hermes requires >= 64K

HUB = HOME / ".cache" / "huggingface" / "hub"
MODELS = [
    {"slug": "qwopus-18b", "model_id": "Qwopus-GLM-18B-Healed-Q6_K.gguf",
     "path": HUB / "models--KyleHessling1--Qwopus-GLM-18B-Merged-GGUF/snapshots/5202561f49e2c558ada57e456a9bf8f7e81d522f/Qwopus-GLM-18B-Healed-Q6_K.gguf"},
    {"slug": "qwen3-6-27b", "model_id": "Qwen3.6-27B-Q6_K.gguf",
     "path": HUB / "models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q6_K.gguf"},
    {"slug": "nemotron-cascade-2-30b", "model_id": "nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf",
     "path": HUB / "models--bartowski--nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/snapshots/931b595fc71b7ca14fb9d935af011f69f7c0434c/nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf"},
]

_REL = re.compile(r"slot\s+release:")
# llama-server prefixes each line with a timestamp, so do NOT anchor at ^.
# Gen tokens are on the "eval time" line; the prompt line ("prompt eval time")
# is skipped by the caller before this matches.
_GEN = re.compile(r"eval time =.*?/\s*(\d+)\s*tokens")


def write_config(model_id: str):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(
        "model:\n"
        f'  default: "{model_id}"\n'
        '  provider: "custom"\n'
        f'  base_url: "http://127.0.0.1:{PORT}/v1"\n'
        '  api_key: "local-key"\n'
        f"  context_length: {CTX}\n"
    )


def start_server(path: Path):
    import httpx
    SRV_LOG.write_text("")  # truncate; cursor resets per model
    fh = open(SRV_LOG, "a")
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(path), "--port", str(PORT), "-ngl", "99",
         "--host", "127.0.0.1", "--ctx-size", str(CTX)],
        stdout=fh, stderr=fh)
    for _ in range(150):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc, fh
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    fh.close()
    raise RuntimeError("llama-server did not become healthy")


def stop_server(proc, fh):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    try:
        fh.close()
    except Exception:
        pass


def _log_lines():
    try:
        return SRV_LOG.read_text(errors="ignore").splitlines()
    except Exception:
        return []


def _parse_slice(lines):
    turns = sum(1 for l in lines if _REL.search(l))
    gen = 0
    for l in lines:
        if "prompt eval time" in l:
            continue
        m = _GEN.search(l)
        if m:
            gen += int(m.group(1))
    return turns, gen


def run_task(task: dict, cursor: int, timeout: int = 300):
    WORK.mkdir(parents=True, exist_ok=True)
    if task.get("setup"):
        subprocess.run(task["setup"], shell=True, timeout=60)
    t0 = time.time()
    timed_out = False
    try:
        subprocess.run(
            [str(HERMES), "-z", task["prompt"], "--yolo"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, cwd=str(WORK))
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = round(time.time() - t0, 1)
    passed = (not timed_out) and subprocess.run(task["check"], shell=True).returncode == 0
    lines = _log_lines()
    turns, gen = _parse_slice(lines[cursor:])
    return ({"id": task["id"], "passed": passed, "timed_out": timed_out,
             "elapsed_s": elapsed, "turns": turns, "gen_tokens": gen}, len(lines))


def run_model(model: dict, tasks: list, repeats: int = 3) -> dict:
    print(f"\n===== {model['slug']} =====", flush=True)
    write_config(model["model_id"])
    proc, fh = start_server(model["path"])
    cursor = len(_log_lines())  # skip startup lines
    rows = []
    try:
        for rep in range(repeats):
            for task in tasks:
                r, cursor = run_task(task, cursor)
                r["repeat"] = rep
                rows.append(r)
                print(f"[{model['slug']}] r{rep} {task['id']:16s} "
                      f"{'PASS' if r['passed'] else 'FAIL'}  {r['turns']}t {r['gen_tokens']}tok "
                      f"({r['elapsed_s']}s)", flush=True)
    finally:
        stop_server(proc, fh)
    n = len(rows)
    passed = [r for r in rows if r["passed"]]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["id"], {"pass": 0, "n": 0})
        by_task[r["id"]]["n"] += 1
        by_task[r["id"]]["pass"] += int(r["passed"])
    ci_lo, ci_hi = wilson_ci(len(passed), n)
    summary = {
        "model": model["slug"],
        "completion_rate": round(100 * len(passed) / n, 1) if n else 0,
        "completion_ci95": [ci_lo, ci_hi],
        "passed": len(passed), "total": n,
        "avg_turns_passed": round(sum(r["turns"] for r in passed) / len(passed), 2) if passed else None,
        "avg_gen_tokens_passed": round(sum(r["gen_tokens"] for r in passed) / len(passed)) if passed else None,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / n, 1) if n else 0,
        "by_task": {k: f"{v['pass']}/{v['n']}" for k, v in by_task.items()},
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{model['slug']}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{model['slug']}] completion {summary['completion_rate']}% "
          f"(95% CI {ci_lo}–{ci_hi}, {len(passed)}/{n})  avg "
          f"{summary['avg_turns_passed']} turns / "
          f"{summary['avg_gen_tokens_passed']} tok on passes", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=None)
    args = ap.parse_args()
    models = [m for m in MODELS if not args.model or m["slug"] == args.model]
    tasks = TASKS[:args.tasks] if args.tasks else TASKS
    t0 = time.time()
    summaries = []
    for m in models:
        try:
            summaries.append(run_model(m, tasks, repeats=args.repeats))
        except Exception as e:
            print(f"[{m['slug']}] FAILED: {type(e).__name__}: {e} — skipping", flush=True)
    summaries.sort(key=lambda s: (s["completion_rate"], -(s["avg_turns_passed"] or 1e9)), reverse=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leaderboard.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n===== PHASE B LEADERBOARD ({(time.time()-t0)/60:.0f} min) =====")
    for rank, s in enumerate(summaries, 1):
        lo, hi = s["completion_ci95"]
        tie = ""
        if rank > 1:
            plo, phi = summaries[rank - 2]["completion_ci95"]
            if lo <= phi and plo <= hi:  # intervals overlap → not a real difference
                tie = "  ← CI overlaps #%d (statistical tie)" % (rank - 1)
        print(f"{rank}. {s['model']:24s} {s['completion_rate']:5.1f}%  "
              f"[95% CI {lo:.0f}–{hi:.0f}]  ({s['passed']}/{s['total']})  "
              f"{s['avg_turns_passed']} turns avg{tie}")


if __name__ == "__main__":
    main()
