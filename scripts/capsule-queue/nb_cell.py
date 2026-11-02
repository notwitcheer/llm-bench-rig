#!/usr/bin/env python3
"""One server-based eval cell for item 77 (Nanbeige4.3-33B): board5 (think-off), gpqa_on (think-on 16k), ma_aime_2026 (think-on 32k).

Run from ~/benchmark-rig with its .venv and PYTHONPATH=~/benchmark-rig (same as the 68-ornith122b cells). Same eval classes, same regimes and result
dirs as the Qwen3.8-27B rows; the only server changes are monkey-patched into lib.quality.build_server_command IN THIS PROCESS (no rig file edited):
  --override-kv tokenizer.ggml.add_space_prefix=bool:false   (without it llama.cpp's SPM path puts a "▁" after every special token, so the chat template
                                                             reaches the model as <|im_start|>▁user instead of <|im_start|>user; with it the server's
                                                             /tokenize equals the HF tokenizer on all 1,998 toolcall prompts, probed 2026-10-30)
  --fit off                                                  (an explicit -c must never be shrunk silently)
  q8_0 KV only when --kv q8_0: -fa on --cache-type-k q8_0 --cache-type-v q8_0
usage: nb_cell.py --model M --slug S --cell board5|gpqa_on|ma_aime_2026 --ctx N --kv f16|q8_0 [--max-tokens N] [--build-label L] [--file-label L]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lib.quality as q
from lib.evals.base import LLMClient

OVERRIDE = ["--override-kv", "tokenizer.ggml.add_space_prefix=bool:false", "--fit", "off"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--cell", required=True, choices=["board5", "gpqa_on", "ma_aime_2026"])
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--kv", default="f16", choices=["f16", "q8_0"])
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--build-label", default="")
    ap.add_argument("--file-label", default="")
    ap.add_argument("--port", type=int, default=0, help="test knob: server port (default = config llama_cpp.server_port, 8090)")
    ap.add_argument("--no-guard", action="store_true", help="test knob: skip the Nanbeige tokenizer guard (non-Nanbeige models)")
    ap.add_argument("--limit", type=int, default=0, help="test knob: first N items only (gpqa_on / ma_aime_2026)")
    a = ap.parse_args()
    if a.port:
        from lib.config import load_config as _lc
        _lc().setdefault("llama_cpp", {})["server_port"] = a.port

    orig = q.build_server_command

    def patched(model_path, port, ngl, ctx_size=None, n_cpu_moe=None):
        cmd = orig(model_path, port=port, ngl=ngl, ctx_size=ctx_size, n_cpu_moe=n_cpu_moe) + OVERRIDE
        if a.kv == "q8_0":
            cmd += ["-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        return cmd

    q.build_server_command = patched

    # tokenizer guard: the server must tokenise the chat-template special-token boundaries exactly like the HF tokenizer (ids probed 2026-10-30,
    # HF LlamaTokenizer vs llama.cpp with and without the override); a mismatch means the override did not apply, so stop before any item runs
    GUARD = [("<|im_start|>user\nhi<|im_end|>\n", [166100, 2714, 13, 6687, 166101, 13]),
             ("<|im_start|>assistant\n<think>\nabc\n</think>\n\nhello", [166100, 66354, 13, 166103, 13, 21893, 13, 166104, 13, 13, 20255])]
    orig_start = q.start_llama_server

    def guarded_start(model_path, **kw):
        import urllib.request
        proc = orig_start(model_path, **kw)
        port = a.port or 8090
        for text, want in GUARD:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"content": text, "add_special": False, "parse_special": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            got = json.loads(urllib.request.urlopen(req, timeout=30).read())["tokens"]
            if got != want:
                q.stop_llama_server(proc)
                raise RuntimeError(f"tokenizer guard FAILED: server {got} != HF {want} (override-kv not applied?)")
        print("[guard] server tokenisation of the template boundaries equals the HF tokenizer", flush=True)
        return proc

    if not a.no_guard:
        q.start_llama_server = guarded_start
    slug, path, ctx, cell = a.slug, a.model, a.ctx, a.cell
    base = Path("results")
    kvnote = f"kv {a.kv}, ctx {ctx}, {a.build_label}, tokenizer fix add_space_prefix=false"
    if cell == "board5":
        rd = base / slug
        rd.mkdir(parents=True, exist_ok=True)
        from lib.config import load_config
        _c = load_config()
        _c.setdefault("quality", {})["mc_gate_tokens"] = 400   # armed for true degeneracy, tolerates a hybrid model's short rationale (precedent: 68-ornith122b)
        print(f"[{slug}] board5 server: {' '.join(q.build_server_command(path, port=a.port or 8090, ngl=99, ctx_size=ctx))}", flush=True)
        t0 = time.time()
        r = q.run_quality_bench(path, engine="llama.cpp", results_dir=rd, offload={"ctx_size": ctx})
        (rd / "quality.regime.txt").write_text(f"think-off board (config sample 0.5 seed 42, mc_gate_tokens 400), greedy, {kvnote}\n")
        (rd / "quality.json").write_text(json.dumps(r, indent=2))
        print(f"[{slug}] board5:off: {json.dumps({k: (v.get('score') if isinstance(v, dict) else v) for k, v in r.items()})} in {time.time()-t0:.0f}s", flush=True)
        return
    if cell == "gpqa_on":
        think_max, rd, out = a.max_tokens or 16384, base / f"{slug}-thinkon", "gpqa.json"
    else:
        think_max, rd, out = a.max_tokens or 32768, base / f"{slug}-matharena-32k", "ma_aime_2026_detail.json"
    rd.mkdir(parents=True, exist_ok=True)
    print(f"[{slug}] {cell} server: {' '.join(q.build_server_command(path, port=a.port or 8090, ngl=99, ctx_size=ctx))}", flush=True)
    proc = q.start_llama_server(path, ctx_size=ctx)
    try:
        with LLMClient(f"http://127.0.0.1:{a.port or 8090}/v1", Path(path).stem, think=True, timeout=3600) as client:
            t0 = time.time()
            if cell == "gpqa_on":
                from lib.evals.gpqa import GPQAEval
                ev = GPQAEval(client=client, results_dir=rd, max_tokens=think_max, limit=a.limit or None)
            else:
                from lib.evals.matharena import MathArenaEval
                ev = MathArenaEval(client=client, task="ma_aime_2026", results_dir=rd, exact_veto=True, max_tokens=think_max, limit=a.limit or None)
            r = ev.evaluate()
            r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy, {kvnote}"
            r["bench"] = ("Nanbeige4.3-33B vs the Qwen3.8-27B rows, same eval class and regime (greedy; the Nanbeige card samples at temp 0.6 / top_p 0.95 for reasoning "
                          "and evaluates at an unlimited budget)")
            r["model_file"] = a.file_label
            r["kv"], r["ctx"] = a.kv, ctx
            (rd / out).write_text(json.dumps(r, indent=2))
            print(f"[{slug}] {cell}: {json.dumps({k: r.get(k) for k in ('score','metric','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
    finally:
        q.stop_llama_server(proc)


if __name__ == "__main__":
    sys.exit(main())
