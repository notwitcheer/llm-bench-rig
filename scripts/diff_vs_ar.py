#!/usr/bin/env python3
"""t049 — AR vs block-diffusion A/B on identical training.

gemma-4-26B-A4B-it (autoregressive, llama-cli) vs diffusiongemma-26B-A4B-it (block
diffusion, llama-diffusion-gemma-cli). SAME Q4_K_M (both source-converted via the same
pipeline), SAME llama.cpp build, SAME RTX 5090 — the only variable is AR vs diffusion.

Per prompt we record the answer, answer-token count, generation seconds, and the fair
cross-paradigm metric: EFFECTIVE answer tok/s = answer_tokens / gen_seconds. Diffusion
pays the full canvas cost regardless of answer length, so this exposes the real tradeoff:
short answers should favour AR heavily, long canvas-filling answers are diffusion's best case.
Timings are load-EXCLUDED (each CLI reports its own generation time). Donald drained + restored.

  tmux new -d -s ab "~/benchmark-rig/.venv/bin/python scripts/diff_vs_ar.py 2>&1 | tee ~/ab.log"
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
DIFF_CLI = str(HOME / "llama.cpp-diffgemma/build/bin/llama-diffusion-gemma-cli")
AR_SERVER = str(HOME / "llama.cpp-diffgemma/build/bin/llama-server")
AR_PORT = 8091  # not Donald's 8090 (drained anyway, but keep distinct)
DIFF_GGUF = str(HOME / "models/diffusiongemma-conv/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
AR_GGUF = str(HOME / "models/gemma-4-26B-A4B-it-conv/gemma-4-26B-A4B-it-Q4_K_M.gguf")
OUT = HOME / "benchmark-rig/results/diffusion-ar"
LIMIT = int(os.environ.get("AB_LIMIT", "0"))
CANVAS = 256  # diffusion canvas size == AR max tokens (same output budget)

PROMPTS = [
    ("factual-short", "What is the capital of Australia? Answer in one word."),
    ("factual", "Who painted the Mona Lisa, and in roughly what year?"),
    ("math", "If a car travels 150 km in 2.5 hours, what is its average speed in km/h? Show your reasoning."),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain."),
    ("explain-med", "Explain how photosynthesis works in three or four sentences."),
    ("generate-long", "Write a paragraph of about 6 sentences on the rise and fall of the Roman Empire."),
]


class VramSampler:
    def __init__(self):
        self.peak = 0
        self._stop = False

    def _run(self):
        while not self._stop:
            try:
                self.peak = max(self.peak, int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0]))
            except Exception:
                pass
            time.sleep(0.25)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=2)


def start_ar_server():
    proc = subprocess.Popen(
        [AR_SERVER, "-m", AR_GGUF, "-ngl", "99", "-c", "4096", "--host", "127.0.0.1",
         "--port", str(AR_PORT), "--no-warmup", "--reasoning", "off",
         "--chat-template-kwargs", '{"enable_thinking": false}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):  # wait for /health
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{AR_PORT}/health", timeout=2).status == 200:
                return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("AR server did not come up")


def query_ar(prompt):
    # OpenAI chat endpoint -> clean content + exact timings. AR emits only answer tokens
    # (no canvas waste), so its decode rate IS its effective answer rate.
    # enable_thinking:false -> gemma-4 skips the thought channel and answers directly, so the
    # 256-token budget isn't consumed by reasoning (the cause of empty content on hard prompts).
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "n_predict": CANVAS, "temperature": 0.4, "cache_prompt": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{AR_PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    content = r["choices"][0]["message"]["content"].strip()
    t = r.get("timings", {})
    tps = round(t.get("predicted_per_second", 0.0), 1)
    n = int(t.get("predicted_n", 0))
    return {"answer": content, "answer_tokens": n, "tok_s": tps, "eff_answer_tok_s": tps}


def run_diff(prompt):
    p = subprocess.run([DIFF_CLI, "-m", DIFF_GGUF, "-ngl", "99", "--diffusion-steps", "128",
                        "-n", str(CANVAS), "-p", prompt], capture_output=True, text=True, timeout=400)
    o = p.stdout + "\n" + p.stderr
    ans = ""
    if "=== answer ===" in o:                                       # cut at the next log-timestamp line
        ans = re.split(r"\d+\.\d+\.\d+\.\d+\s+I", o.split("=== answer ===", 1)[1])[0].strip()
    gm = re.search(r"generation:.*?in ([\d.]+) s \(([\d.]+) canvas tok/s.*?answer tokens=(\d+)", o, re.S)
    gen_s = float(gm.group(1)) if gm else 0.0
    canvas_tps = float(gm.group(2)) if gm else 0.0
    atoks = int(gm.group(3)) if gm else len(ans.split())
    return {"answer": ans, "answer_tokens": atoks, "gen_s": round(gen_s, 3),
            "canvas_tok_s": round(canvas_tps, 1),
            "eff_answer_tok_s": round(atoks / gen_s, 1) if gen_s else 0.0}


def donald(a):
    subprocess.run(["sudo", "-n", "systemctl", a, "llama-server.service"], check=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[:LIMIT] if LIMIT else PROMPTS
    donald("stop")
    time.sleep(3)
    results = [{"tag": t, "prompt": p} for t, p in prompts]
    ar_vram = diff_vram = 0
    try:
        print("=== AR: gemma-4-26B-A4B-it (llama-server) ===", flush=True)
        srv = start_ar_server()
        try:
            with VramSampler() as vs:
                for i, (tag, pr) in enumerate(prompts):
                    r = query_ar(pr)
                    results[i]["ar"] = r
                    print(f"[AR] {tag}: {r['answer_tokens']}tok @ {r['tok_s']} tok/s", flush=True)
                ar_vram = vs.peak
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=15)
            except Exception:
                srv.kill()
            time.sleep(2)
        print("=== DIFFUSION: diffusiongemma-26B-A4B-it ===", flush=True)
        with VramSampler() as vs:
            for i, (tag, pr) in enumerate(prompts):
                r = run_diff(pr)
                results[i]["diff"] = r
                print(f"[DIFF] {tag}: {r['answer_tokens']}tok in {r['gen_s']}s | "
                      f"{r['canvas_tok_s']} canvas tok/s | eff {r['eff_answer_tok_s']} tok/s", flush=True)
            diff_vram = vs.peak
    finally:
        donald("start")
    (OUT / "results.json").write_text(json.dumps(
        {"ar_peak_vram_mib": ar_vram, "diff_peak_vram_mib": diff_vram, "canvas": CANVAS, "items": results}, indent=2))
    print(f"AB_DONE  AR_VRAM={ar_vram}MiB  DIFF_VRAM={diff_vram}MiB", flush=True)


if __name__ == "__main__":
    main()
