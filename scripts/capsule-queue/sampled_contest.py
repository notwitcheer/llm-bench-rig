#!/usr/bin/env python3
"""Sampled pass@1 on a MathArena contest set (2026-10-20, greedy-vs-sampled pilot).

Why: DeepSeek's DeepInfer eval (github.com/deepseek-ai/DeepInfer eval/README.md, read 2026-10-20) reports AIME 2026
as the mean of 32 samples per problem and GPQA Diamond as the mean of 8. our boards score one greedy pass. this
driver runs K sampled completions per problem on the same prompt, grader, veto and budget as lib/evals/matharena.py,
so the greedy row in results/<slug>-matharena-32k/ and the sampled mean are a paired comparison.

Concurrency: llama-server -np <slots>, one HTTP request per slot in flight. slots are independent sequences
(kv_unified false), so each sample sees the same 40,960-token context as the greedy run. sampling params are sent per
request; a fixed per-(problem, sample) seed is recorded but parallel slots are not batch-invariant in llama.cpp, so
a rerun is not guaranteed to reproduce the same tokens.

Resumable: <out>/<task>_samples.json maps "i:k" -> record, saved after every finished sample under a lock, so a hard
stop loses only the samples in flight. writes <task>_sampled_detail.json when every (i, k) is done.

usage (from ~/benchmark-rig, PYTHONPATH=.):
  python sampled_contest.py --model <gguf> --slug <slug> --task ma_aime_2026 --k 8 --np 3 [--server-bin <bin>]
"""
import argparse
import json
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

import lib.quality as q
from lib.evals.matharena import _build_messages, grade_with_timeout, task_to_dataset, SET_PUBLISHED, PREFIX
from lib.evals.math500 import extract_boxed
from lib.evals.gpqa import CAP_SLACK

PORT = 8090  # overridden by --port


def wait_health(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("llama-server health timeout")


def one(client, model_name, problem, max_tokens, params, seed):
    payload = {"model": model_name, "messages": _build_messages(problem), "max_tokens": max_tokens,
               "temperature": params["temperature"], "top_p": params["top_p"], "top_k": params["top_k"],
               "seed": seed}
    for attempt in range(3):
        try:
            r = client.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload)
            r.raise_for_status()
            body = r.json()
            msg = body["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            tokens = (body.get("usage") or {}).get("completion_tokens")
            fallback = False
            if not content and reasoning:
                content, fallback = reasoning, True
            return content, tokens, fallback, body["choices"][0].get("finish_reason"), None
        except (httpx.HTTPError, KeyError, IndexError) as e:
            if attempt == 2:
                return None, None, False, None, repr(e)[:300]
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--task", default="ma_aime_2026")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--np", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--ctx-slot", type=int, default=40960)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--server-bin", default=None)
    ap.add_argument("--server-label", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (smoke)")
    ap.add_argument("--out-suffix", default=None)
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    global PORT
    PORT = a.port

    params = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
    out = Path("results") / f"{a.slug}-matharena-32k-{a.out_suffix or f's{a.k}'}"
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / f"{a.task}_samples.json"
    done = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    done = {k: v for k, v in done.items() if not v.get("error_request")}  # errored samples re-run

    from datasets import load_dataset
    ds = load_dataset(task_to_dataset(a.task), split="train")
    items = [{"problem": r["problem"], "answer": str(r["answer"]), "problem_idx": r.get("problem_idx")} for r in ds]
    if a.limit:
        items = items[: a.limit]
    jobs = [(i, k) for k in range(a.k) for i in range(len(items)) if f"{i}:{k}" not in done]  # sample-major
    print(f"[sampled] {a.slug} {a.task}: {len(items)} problems x {a.k} = {len(items) * a.k}, "
          f"done {len(done)}, to run {len(jobs)}, np {a.np}, params {params}", flush=True)
    if not jobs:
        write_detail(a, items, done, out, params)
        return

    if a.server_bin:
        _orig = q.get
        q.get = lambda key, *x, **kw: a.server_bin if key == "llama_cpp.server_bin" else _orig(key, *x, **kw)
    cmd = q.build_server_command(a.model, port=PORT, ngl=99, ctx_size=a.ctx_slot * a.np) + ["-np", str(a.np)]
    print("[sampled] server:", " ".join(cmd), flush=True)
    log = open(Path.home() / f"bench-server-sampled-{PORT}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    lock = threading.Lock()
    model_name = Path(a.model).stem
    t_start = time.time()
    try:
        wait_health(proc)
        client = httpx.Client(timeout=14400)

        def work(i, k):
            seed = 1000 * k + i
            t0 = time.time()
            text, tokens, fb, finish, err = one(client, model_name, items[i]["problem"], a.max_tokens, params, seed)
            if err:
                return i, k, {"error_request": err, "seed": seed}
            pred = extract_boxed(text)
            ok, gto, vetoed = grade_with_timeout(pred, items[i]["answer"], veto=True, return_vetoed=True)
            capped = isinstance(tokens, int) and tokens >= a.max_tokens - CAP_SLACK
            return i, k, {"correct": ok, "predicted": pred, "expected": items[i]["answer"], "seed": seed,
                          "completion_tokens": tokens, "capped": capped, "finish_reason": finish,
                          "reasoning_fallback": fb, "grade_timeout": gto, "exact_vetoed": vetoed,
                          "seconds": round(time.time() - t0, 1)}

        with ThreadPoolExecutor(max_workers=a.np) as ex:
            futs = [ex.submit(work, i, k) for i, k in jobs]
            for f in as_completed(futs):
                i, k, rec = f.result()
                with lock:
                    done[f"{i}:{k}"] = rec
                    prog_path.write_text(json.dumps(done))
                    n_ok = sum(1 for v in done.values() if v.get("correct"))
                print(f"[sampled] {len(done)}/{len(items) * a.k} p{i} s{k} correct {rec.get('correct')} "
                      f"tokens {rec.get('completion_tokens')} capped {rec.get('capped')} {rec.get('seconds')}s "
                      f"| running acc {n_ok / len(done):.3f} | {time.time() - t_start:.0f}s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    if all(f"{i}:{k}" in done and not done[f"{i}:{k}"].get("error_request")
           for i in range(len(items)) for k in range(a.k)):
        write_detail(a, items, done, out, params)


def write_detail(a, items, done, out, params):
    n, K = len(items), a.k
    per = []
    for i in range(n):
        recs = [done[f"{i}:{k}"] for k in range(K)]
        per.append({"i": i, "problem_idx": items[i]["problem_idx"], "solved": sum(r["correct"] for r in recs),
                    "capped": sum(r["capped"] for r in recs)})
    sample_acc = [sum(done[f"{i}:{k}"]["correct"] for i in range(n)) / n for k in range(K)]
    toks = [r["completion_tokens"] for r in done.values() if isinstance(r.get("completion_tokens"), int)]
    correct_total = sum(p["solved"] for p in per)
    res = {
        "score_mean_at_k": round(100 * correct_total / (n * K), 2),
        "metric": f"pass@1 as the mean over {K} sampled completions per problem",
        "k": K, "problems": n, "samples": n * K, "correct_samples": correct_total,
        "per_sample_run_score": [round(100 * x, 2) for x in sample_acc],
        "per_sample_run_score_sd": round(100 * statistics.pstdev(sample_acc), 2),
        "pass_at_k_any": sum(1 for p in per if p["solved"] > 0),
        "majority_solved": sum(1 for p in per if p["solved"] * 2 > K),
        "capped_count": sum(p["capped"] for p in per),
        "capped_rate": round(sum(p["capped"] for p in per) / (n * K), 4),
        "parse_failures": sum(1 for r in done.values() if r.get("predicted") is None),
        "exact_vetoes": sum(1 for r in done.values() if r.get("exact_vetoed")),
        "grade_timeouts": sum(1 for r in done.values() if r.get("grade_timeout")),
        "reasoning_fallback_count": sum(1 for r in done.values() if r.get("reasoning_fallback")),
        "completion_tokens_total": sum(toks),
        "completion_tokens_median": statistics.median(toks) if toks else None,
        "tokens_per_correct": round(sum(toks) / correct_total, 1) if correct_total else None,
        "per_problem": per,
        "sampling": params, "max_tokens": a.max_tokens, "ctx_per_slot": a.ctx_slot, "parallel_slots": a.np,
        "regime": f"think-on, max_tokens {a.max_tokens}, ctx {a.ctx_slot} per slot, zero-shot, "
                  f"sampled T={params['temperature']} top_p={params['top_p']} top_k={params['top_k']}, K={K}",
        "dataset": task_to_dataset(a.task), "set_published": SET_PUBLISHED.get(a.task[len(PREFIX):]),
        "grader": "vendored Qwen2.5-Math grader + exact-number veto (same as lib/evals/matharena.py)",
        "server_bin": a.server_label or q.get("llama_cpp.server_bin"),
        "bench": "greedy-vs-sampled pilot (2026-10-20)",
    }
    (out / f"{a.task}_sampled_detail.json").write_text(json.dumps(res, indent=2))
    print(f"[sampled] DONE {a.slug} {a.task}: mean@{K} {res['score_mean_at_k']} "
          f"(per-run {res['per_sample_run_score']}), any {res['pass_at_k_any']}/{n}, "
          f"majority {res['majority_solved']}/{n}, capped {res['capped_rate']}", flush=True)


if __name__ == "__main__":
    main()
