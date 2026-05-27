import time
import httpx
from lib.config import get

def _generate_prompt_tokens(n_tokens: int) -> str:
    word = "hello "
    return (word * n_tokens)[:n_tokens * 6]

def measure_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    first_token_time = None

    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{base_url}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
            },
        )
        response.raise_for_status()

        total_tokens = 0
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line.strip() == "data: [DONE]":
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
            total_tokens += 1

    end = time.perf_counter()
    elapsed = end - start
    ttft = (first_token_time - start) * 1000 if first_token_time else None

    return {
        "tokens_generated": total_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
        "ttft_ms": round(ttft, 1) if ttft else None,
    }

def get_vllm_models(base_url: str) -> list[str]:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{base_url}/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json()["data"]]

def run_vllm_bench(model: str = None) -> dict:
    base_url = get("vllm.api_base")
    if model is None:
        models = get_vllm_models(base_url)
        if not models:
            raise RuntimeError("No models loaded in vLLM")
        model = models[0]

    ctx_lengths = get("speed.context_lengths", [128, 512, 2048])
    gen_length = get("speed.generation_length", 128)
    results = {}

    for cl in ctx_lengths:
        prompt = _generate_prompt_tokens(cl)
        measurement = measure_completion(base_url, model, prompt, max_tokens=1)
        results[f"pp{cl}"] = {
            "tokens_per_sec": measurement["tokens_per_sec"],
            "ttft_ms": measurement["ttft_ms"],
        }

    prompt = _generate_prompt_tokens(128)
    measurement = measure_completion(base_url, model, prompt, max_tokens=gen_length)
    results[f"tg{gen_length}"] = {
        "tokens_per_sec": measurement["tokens_per_sec"],
    }
    results["ttft_ms"] = results["pp128"].get("ttft_ms")
    results["model"] = model
    results["backend"] = "vLLM"

    return results
