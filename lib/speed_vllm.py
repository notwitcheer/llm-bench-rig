import json as _json
import time

import httpx

from lib.config import get


def prefill_decode_tps(prompt_tokens, ttft_s, output_tokens, total_s):
    """Prefill tok/s = prompt tokens over time-to-first-token; decode tok/s = output tokens over the rest.
    Matches llama-bench's pp{N}/tg{N} semantics so vLLM and llama.cpp are comparable."""
    decode_s = max(total_s - ttft_s, 1e-6)
    return {
        "prefill_tps": round(prompt_tokens / ttft_s, 2) if ttft_s else None,
        "decode_tps": round(output_tokens / decode_s, 2),
    }


def _root(base_url):
    return base_url.rstrip("/").removesuffix("/v1")


def exact_token_prompt(base_url, model, n_tokens):
    """Build a prompt of exactly n_tokens via vLLM /tokenize+/detokenize (parity with llama-bench pp{N})."""
    seed = "the quick brown fox jumps over the lazy dog " * (n_tokens // 6 + 2)
    root = _root(base_url)
    with httpx.Client(timeout=30) as c:
        toks = c.post(f"{root}/tokenize", json={"model": model, "prompt": seed}).json()["tokens"]
        ids = toks[:n_tokens]
        return c.post(f"{root}/detokenize", json={"model": model, "tokens": ids}).json()["prompt"]


def measure_completion(base_url, model, prompt, max_tokens):
    """Streaming completion; returns prompt_tokens, output_tokens, ttft_s, total_s (usage-accurate)."""
    t0 = time.perf_counter()
    first = None
    prompt_tokens = output_tokens = 0
    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{base_url.rstrip('/')}/completions", json={
            "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                d = _json.loads(payload)
                if d.get("choices") and d["choices"][0].get("text"):
                    if first is None:
                        first = time.perf_counter()
                    output_tokens += 1
                if d.get("usage"):
                    prompt_tokens = d["usage"].get("prompt_tokens", prompt_tokens)
                    output_tokens = d["usage"].get("completion_tokens", output_tokens)
    end = time.perf_counter()
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": (first - t0) if first else (end - t0),
        "total_s": end - t0,
    }


def get_vllm_models(base_url):
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{base_url.rstrip('/')}/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json()["data"]]


def run_vllm_bench(model=None):
    base_url = get("vllm.api_base")
    if model is None:
        models = get_vllm_models(base_url)
        if not models:
            raise RuntimeError("No models loaded in vLLM")
        model = models[0]

    ctx_lengths = get("speed.context_lengths", [128, 512, 2048])
    gen_length = get("speed.generation_length", 128)
    results = {}

    # prefill: exact-token prompt, 1 output token -> prefill_tps = prompt_tokens / ttft
    for cl in ctx_lengths:
        prompt = exact_token_prompt(base_url, model, cl)
        m = measure_completion(base_url, model, prompt, max_tokens=1)
        tps = prefill_decode_tps(m["prompt_tokens"], m["ttft_s"], m["output_tokens"], m["total_s"])
        results[f"pp{cl}"] = {
            "tokens_per_sec": tps["prefill_tps"],
            "prompt_tokens": m["prompt_tokens"],
            "ttft_ms": round(m["ttft_s"] * 1000, 1),
        }

    # decode: generate gen_length tokens -> decode_tps = output / (total - ttft)
    prompt = exact_token_prompt(base_url, model, 128)
    m = measure_completion(base_url, model, prompt, max_tokens=gen_length)
    tps = prefill_decode_tps(m["prompt_tokens"], m["ttft_s"], m["output_tokens"], m["total_s"])
    results[f"tg{gen_length}"] = {"tokens_per_sec": tps["decode_tps"], "output_tokens": m["output_tokens"]}

    results["model"] = model
    results["backend"] = "vLLM"
    return results
