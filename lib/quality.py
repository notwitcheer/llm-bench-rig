import json
import signal
import subprocess
import time
from pathlib import Path

from lib.config import get

EVAL_REGISTRY = {"mmlu", "arc_challenge", "hellaswag", "gsm8k", "humaneval"}


def start_llama_server(model_path: str, ctx_size: int | None = None) -> subprocess.Popen:
    server_bin = str(Path(get("llama_cpp.server_bin")).expanduser())
    port = get("llama_cpp.server_port", 8090)
    ngl = get("speed.n_gpu_layers", 99)
    cmd = [server_bin, "-m", model_path, "--port", str(port),
           "--n-gpu-layers", str(ngl), "--host", "127.0.0.1"]
    if ctx_size:
        cmd += ["--ctx-size", str(ctx_size)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=180)
    except TimeoutError:
        # Failed/OOM launch (e.g. ctx too large for VRAM) — don't leak the process.
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    return proc


def stop_llama_server(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_for_server(url: str, timeout: int):
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1)
    raise TimeoutError(f"Server at {url} did not start within {timeout}s")


def _run_evals(api_base: str, model_name: str, tasks: list[str],
               results_dir: Path | None, sample: float | None,
               think: bool = True) -> dict:
    from lib.evals import (LLMClient, MMLUEval, ARCEval,
                           HellaSwagEval, GSM8KEval, HumanEvalEval)

    results = {}
    with LLMClient(api_base, model_name, think=think) as client:
        for task in tasks:
            print(f"\n[quality] === {task} ===")
            ev = _make_evaluator(task, client, results_dir, sample)
            task_results = ev.evaluate()
            results[task] = {
                "score": task_results["score"],
                "metric": task_results["metric"],
            }
            if results_dir:
                with open(results_dir / f"{task}_detail.json", "w") as f:
                    json.dump(task_results, f, indent=2)
    return results


def _make_evaluator(task: str, client, results_dir, sample: float | None):
    from lib.evals import (MMLUEval, ARCEval, HellaSwagEval,
                           GSM8KEval, HumanEvalEval)

    if task == "mmlu":
        return MMLUEval(client=client, sample=sample, results_dir=results_dir)
    if task == "arc_challenge":
        return ARCEval(client=client, results_dir=results_dir)
    if task == "hellaswag":
        return HellaSwagEval(client=client, sample=sample, results_dir=results_dir)
    if task == "gsm8k":
        return GSM8KEval(client=client, results_dir=results_dir)
    if task == "humaneval":
        return HumanEvalEval(client=client, results_dir=results_dir)
    raise ValueError(f"Unknown eval task: {task}")


def run_quality_bench(model_path: str, engine: str, results_dir: Path | None = None) -> dict:
    tasks = get("quality.tasks", [])
    sample = get("quality.sample", None)
    think = get("quality.think", True)
    server_proc = None

    if sample:
        print(f"[quality] Sampling {sample:.0%} of MMLU/HellaSwag (seed=42)")
    if not think:
        print(f"[quality] Thinking disabled (/nothink injected)")

    if engine == "llama.cpp":
        server_proc = start_llama_server(model_path)
        port = get("llama_cpp.server_port", 8090)
        api_base = f"http://127.0.0.1:{port}/v1"
        model_name = Path(model_path).stem
    else:
        api_base = get("vllm.api_base")
        model_name = Path(model_path).name

    eval_tasks = [t for t in tasks if t in EVAL_REGISTRY]
    unknown = [t for t in tasks if t not in EVAL_REGISTRY]
    if unknown:
        print(f"[quality] Skipping unknown tasks: {', '.join(unknown)}")

    try:
        results = _run_evals(api_base, model_name, eval_tasks, results_dir, sample, think)
    finally:
        if server_proc:
            stop_llama_server(server_proc)

    return results
