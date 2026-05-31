"""Phase B — run the top finishers through the REAL Hermes Agent on a fixed task
battery, to break the Phase A tie under real conditions.

For each (model, task, repeat): start llama-server (64K, Hermes requires >=64K),
point Hermes at it via ~/.hermes/config.yaml, run `hermes -z "<prompt>" --yolo`,
then run the task's objective filesystem check. Primary metric: completion rate.

Run on capsule:  ~/benchmark-rig/.venv/bin/python -m lib.agentic.phaseb
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from lib.agentic.phaseb_tasks import TASKS

HOME = Path.home()
HERMES = HOME / "hermes-src" / ".venv" / "bin" / "hermes"
CONFIG = HOME / ".hermes" / "config.yaml"
SERVER = HOME / "llama.cpp" / "build" / "bin" / "llama-server"
WORK = Path("/tmp/phaseb")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "phaseb"
PORT = 8090
CTX = 65536  # Hermes Agent requires >= 64K context

HUB = HOME / ".cache" / "huggingface" / "hub"
MODELS = [
    {"slug": "qwopus-18b", "model_id": "Qwopus-GLM-18B-Healed-Q6_K.gguf",
     "path": HUB / "models--KyleHessling1--Qwopus-GLM-18B-Merged-GGUF/snapshots/5202561f49e2c558ada57e456a9bf8f7e81d522f/Qwopus-GLM-18B-Healed-Q6_K.gguf"},
    {"slug": "qwen3-6-27b", "model_id": "Qwen3.6-27B-Q6_K.gguf",
     "path": HUB / "models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q6_K.gguf"},
    {"slug": "nemotron-cascade-2-30b", "model_id": "nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf",
     "path": HUB / "models--bartowski--nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/snapshots/931b595fc71b7ca14fb9d935af011f69f7c0434c/nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf"},
]


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
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(path), "--port", str(PORT), "-ngl", "99",
         "--host", "127.0.0.1", "--ctx-size", str(CTX)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(150):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError("llama-server did not become healthy")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()


def run_task(task: dict, timeout: int = 300) -> dict:
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
    return {"id": task["id"], "passed": passed, "timed_out": timed_out, "elapsed_s": elapsed}


def run_model(model: dict, tasks: list, repeats: int = 2) -> dict:
    print(f"\n===== {model['slug']} =====", flush=True)
    write_config(model["model_id"])
    proc = start_server(model["path"])
    rows = []
    try:
        for rep in range(repeats):
            for task in tasks:
                r = run_task(task)
                r["repeat"] = rep
                rows.append(r)
                print(f"[{model['slug']}] r{rep} {task['id']:14s} "
                      f"{'PASS' if r['passed'] else 'FAIL'} ({r['elapsed_s']}s)", flush=True)
    finally:
        stop_server(proc)
    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    by_task = {}
    for r in rows:
        by_task.setdefault(r["id"], []).append(r["passed"])
    summary = {
        "model": model["slug"],
        "completion_rate": round(100 * passed / n, 1) if n else 0,
        "passed": passed, "total": n,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / n, 1) if n else 0,
        "by_task_anypass": {k: any(v) for k, v in by_task.items()},
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{model['slug']}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{model['slug']}] completion {summary['completion_rate']}% ({passed}/{n})", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="run only this slug (default: all)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tasks", type=int, default=None, help="limit to first N tasks (smoke)")
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
    summaries.sort(key=lambda s: s["completion_rate"], reverse=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leaderboard.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n===== PHASE B LEADERBOARD ({(time.time()-t0)/60:.0f} min) =====")
    for rank, s in enumerate(summaries, 1):
        print(f"{rank}. {s['model']:26s} {s['completion_rate']}%  ({s['passed']}/{s['total']})")


if __name__ == "__main__":
    main()
