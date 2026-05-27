import subprocess
import json
import signal
import time
from pathlib import Path
from lib.config import get

def start_llama_server(model_path: str) -> subprocess.Popen:
    server_bin = str(Path(get("llama_cpp.server_bin")).expanduser())
    port = get("llama_cpp.server_port", 8090)
    ngl = get("speed.n_gpu_layers", 99)
    proc = subprocess.Popen(
        [server_bin, "-m", model_path, "--port", str(port),
         "--n-gpu-layers", str(ngl), "--host", "127.0.0.1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=120)
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

def run_lm_eval(api_base: str, model_name: str, tasks: list[str]) -> dict:
    task_str = ",".join(tasks)
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", f"model={model_name},base_url={api_base},tokenizer_backend=huggingface",
        "--tasks", task_str,
        "--output_path", "/tmp/lm_eval_results",
        "--log_samples",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed: {result.stderr[:500]}")
    return _parse_lm_eval_results("/tmp/lm_eval_results")

def _parse_lm_eval_results(output_path: str) -> dict:
    output_dir = Path(output_path)
    results = {}
    for results_file in output_dir.rglob("results.json"):
        with open(results_file) as f:
            data = json.load(f)
        if "results" in data:
            for task_name, task_results in data["results"].items():
                score_key = None
                for k in ("acc_norm,none", "acc,none", "pass@1,none", "exact_match,none"):
                    if k in task_results:
                        score_key = k
                        break
                if score_key:
                    results[task_name] = {
                        "score": round(task_results[score_key] * 100, 2),
                        "metric": score_key.split(",")[0],
                    }
    return results

def run_quality_bench(model_path: str, engine: str) -> dict:
    tasks = get("quality.tasks", [])
    server_proc = None

    if engine == "llama.cpp":
        server_proc = start_llama_server(model_path)
        port = get("llama_cpp.server_port", 8090)
        api_base = f"http://127.0.0.1:{port}/v1"
        model_name = Path(model_path).stem
    else:
        api_base = get("vllm.api_base")
        model_name = Path(model_path).name

    try:
        results = run_lm_eval(api_base, model_name, tasks)
    finally:
        if server_proc:
            stop_llama_server(server_proc)

    return results
