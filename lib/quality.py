import json
import signal
import subprocess
import time
from pathlib import Path

from lib.config import get
from lib.evals import (LLMClient, MMLUEval, ARCEval,  # noqa: E402
                       HellaSwagEval, GSM8KEval, HumanEvalEval)
from lib.evals.base import CompletionLengthGate, InstrumentGateError

EVAL_REGISTRY = {"mmlu", "arc_challenge", "hellaswag", "gsm8k", "humaneval"}
MC_TASKS = {"mmlu", "arc_challenge", "hellaswag"}


def _build_mc_gate(think: bool, threshold) -> CompletionLengthGate | None:
    """Fresh gate per MC eval — windows must not blur across evals (t103, ADR-0004)."""
    if threshold is None:
        return None
    return CompletionLengthGate(threshold=threshold, armed=not think)


def build_server_command(model_path: str, port: int, ngl: int,
                         ctx_size: int | None = None,
                         n_cpu_moe: int | None = None) -> list[str]:
    cmd = [str(Path(get("llama_cpp.server_bin")).expanduser()),
           "-m", model_path, "--port", str(port),
           "--n-gpu-layers", str(ngl), "--host", "127.0.0.1",
           # --jinja applies the GGUF's own Jinja chat template (so chat_template_kwargs
           # like reasoning:False are honored) and routes reasoning spans into
           # reasoning_content instead of leaking <think> tokens into content. Without it,
           # reasoning models (cohere2moe/North-Mini) think on every question -> slow + unparsed.
           "--jinja"]
    if ctx_size:
        cmd += ["--ctx-size", str(ctx_size)]
    if n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    return cmd


def start_llama_server(model_path: str, ctx_size: int | None = None,
                       n_cpu_moe: int | None = None,
                       n_gpu_layers: int | None = None) -> subprocess.Popen:
    port = get("llama_cpp.server_port", 8090)
    ngl = n_gpu_layers if n_gpu_layers is not None else get("speed.n_gpu_layers", 99)
    cmd = build_server_command(model_path, port=port, ngl=ngl,
                               ctx_size=ctx_size, n_cpu_moe=n_cpu_moe)
    # 2026-08-08 ling-3 run: PIPE with no drain deadlocks chatty servers (PR26608 build
    # logs per-request; ~64KB pipe fills mid-MMLU -> server blocks on write -> client
    # ReadTimeout). Log to a file instead; also keeps the log inspectable.
    server_log = open(Path.home() / "bench-server.log", "w")
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=300, proc=proc)
    except (TimeoutError, RuntimeError):
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


def _wait_for_server(url: str, timeout: int, proc=None):
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early with code {proc.returncode} before becoming healthy"
            )
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
               think: bool = True, limit: int | None = None,
               mmlu_limit: int | None = None,
               mc_gate_tokens=None) -> dict:
    results = {}
    with LLMClient(api_base, model_name, think=think) as client:
        for task in tasks:
            print(f"\n[quality] === {task} ===")
            gate = _build_mc_gate(think, mc_gate_tokens) if task in MC_TASKS else None
            ev = _make_evaluator(task, client, results_dir, sample,
                                 limit=limit, mmlu_limit=mmlu_limit, gate=gate)
            task_results = ev.evaluate()
            results[task] = {
                "score": task_results["score"],
                "metric": task_results["metric"],
            }
            if results_dir:
                with open(results_dir / f"{task}_detail.json", "w") as f:
                    json.dump(task_results, f, indent=2)
    return results


def _make_evaluator(task: str, client, results_dir, sample: float | None,
                    limit: int | None = None, mmlu_limit: int | None = None,
                    gate: CompletionLengthGate | None = None):
    if task == "mmlu":
        return MMLUEval(client=client, sample=sample, limit=mmlu_limit,
                        results_dir=results_dir, gate=gate)
    if task == "arc_challenge":
        return ARCEval(client=client, limit=limit, results_dir=results_dir, gate=gate)
    if task == "hellaswag":
        return HellaSwagEval(client=client, sample=sample, limit=limit,
                             results_dir=results_dir, gate=gate)
    if task == "gsm8k":
        return GSM8KEval(client=client, limit=limit, results_dir=results_dir)
    if task == "humaneval":
        return HumanEvalEval(client=client, limit=limit, results_dir=results_dir)
    raise ValueError(f"Unknown eval task: {task}")


def run_quality_bench(model_path: str, engine: str, results_dir: Path | None = None,
                      offload: dict | None = None) -> dict:
    tasks = get("quality.tasks", [])
    sample = get("quality.sample", None)
    think = get("quality.think", True)
    limit = get("quality.limit", None)
    mmlu_limit = get("quality.mmlu_limit", None)
    mc_gate_tokens = get("quality.mc_gate_tokens", 50)
    server_proc = None

    if sample:
        print(f"[quality] Sampling {sample:.0%} of MMLU/HellaSwag (seed=42)")
    if not think:
        print(f"[quality] Thinking disabled (chat_template_kwargs: enable_thinking/reasoning off, --jinja)")
    if mc_gate_tokens is None:
        print("[quality] MC completion-length gate: DISABLED (quality.mc_gate_tokens: null)")
    elif think:
        print(f"[quality] MC completion-length gate: log-only (think-on) @ {mc_gate_tokens} tok")
    else:
        print(f"[quality] MC completion-length gate: ARMED @ {mc_gate_tokens} tok (window 25)")

    if engine == "llama.cpp":
        server_proc = start_llama_server(model_path, **(offload or {}))
        port = get("llama_cpp.server_port", 8090)
        api_base = f"http://127.0.0.1:{port}/v1"
        model_name = Path(model_path).stem
    else:
        from lib.speed_vllm import get_vllm_models
        api_base = get("vllm.api_base")
        served = get_vllm_models(api_base)
        model_name = served[0] if served else Path(model_path).name

    eval_tasks = [t for t in tasks if t in EVAL_REGISTRY]
    unknown = [t for t in tasks if t not in EVAL_REGISTRY]
    if unknown:
        print(f"[quality] Skipping unknown tasks: {', '.join(unknown)}")

    try:
        results = _run_evals(api_base, model_name, eval_tasks, results_dir, sample, think,
                             limit=limit, mmlu_limit=mmlu_limit,
                             mc_gate_tokens=mc_gate_tokens)
    except InstrumentGateError as e:
        print(f"\n[quality] INSTRUMENT FAILURE — {e}")
        print("[quality] Scores are NOT bankable. Suite aborted; no results written. "
              "Check the serving stack (see 2026-07-16 sglang GDN incident).")
        raise
    finally:
        if server_proc:
            stop_llama_server(server_proc)

    return results
