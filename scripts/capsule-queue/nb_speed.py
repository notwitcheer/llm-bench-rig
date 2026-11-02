#!/usr/bin/env python3
"""Speed leg for item 77 (Nanbeige4.3-33B): the board's llama-bench regime with fit fallbacks.

Run from ~/benchmark-rig with its .venv (needs lib.*). Writes results/<slug>/{meta.json,speed.json}:
 - main pass = lib.speed_llama's command (pp128/512/2048/4096/8192/16384 + tg128, -ngl 99, no -fa flag = the board regime), VRAM peak sampled card-wide
   every 0.5 s like bench.py. If the f16-KV pass fails (a 29 GB Q6_K cannot hold a 16k f16 KV on a 32 GB card) it retries with fewer prompt
   sizes, then with q8_0 K+V (-fa 1 -ctk q8_0 -ctv q8_0); the first pass that completes is kept and labelled with its KV type.
 - depth pass = the Qwen rows' bench.depth.json recipe: -fa 1 -p 512 -n 128 -d 0,8192,32768 -r 2 (f16 first, shorter list on failure), and ALWAYS a q8_0
   depth pass beside it, so decode tok/s per KV type from the same build is on file.
usage: nb_speed.py --model M --bench-bin B --results-dir D [--build-label L]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.expanduser("~/benchmark-rig"))
from pathlib import Path  # noqa: E402

from lib.config import get, load_config  # noqa: E402
from lib.gpu import get_vram_used_mib  # noqa: E402
from lib.meta import extract_metadata, save_metadata  # noqa: E402
from lib.speed_llama import parse_llama_bench_output  # noqa: E402


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.ev = 0, threading.Event()

    def run(self):
        while not self.ev.is_set():
            try:
                self.peak = max(self.peak, get_vram_used_mib())
            except Exception:
                pass
            self.ev.wait(0.5)

    def stop(self):
        self.ev.set()
        self.join(timeout=3)
        return self.peak


def run(cmd, timeout):
    s = Sampler()
    s.start()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        s.stop()
        return None, "timeout", s.peak
    return r, (r.stderr or "")[-400:], s.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench-bin", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--build-label", default="")
    ap.add_argument("--pp-list", default="", help="test knob: override the prompt sizes, e.g. '128 512'")
    ap.add_argument("--d-list", default="", help="test knob: override the depth list, e.g. '0,1024'")
    a = ap.parse_args()
    load_config()
    meta = extract_metadata(Path(a.model))
    meta["think"] = False
    out_dir = save_metadata(meta, Path(a.results_dir))
    ctxs = [int(x) for x in a.pp_list.split()] or list(get("speed.context_lengths", [128, 512, 2048, 4096, 8192, 16384]))
    dls = [[int(x) for x in a.d_list.split(",")]] if a.d_list else [[0, 8192, 32768], [0, 8192]]
    gen = get("speed.generation_length", 128)
    vram_before = get_vram_used_mib()
    attempts, main_res, main_kv, main_peak = [], None, None, None
    plan = [("f16", ctxs), ("f16", [c for c in ctxs if c <= 8192]), ("q8_0", ctxs), ("q8_0", [c for c in ctxs if c <= 8192])]
    for kv, cl in plan:
        cmd = [a.bench_bin, "-m", a.model, "-ngl", "99"] + [x for c in cl for x in ("-p", str(c))] + ["-n", str(gen)]
        if kv == "q8_0":
            cmd += ["-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0"]
        r, err, peak = run(cmd, 1800)
        ok = r is not None and r.returncode == 0
        attempts.append({"kv": kv, "pp_list": cl, "ok": ok, "vram_peak_mib": peak, "err": "" if ok else err})
        print(f"[speed] main pass kv {kv} pp {cl}: {'OK' if ok else 'FAIL ' + err[-120:]} (peak {peak} MiB)", flush=True)
        if ok:
            main_res, main_kv, main_peak = parse_llama_bench_output(r.stdout), kv, peak
            main_res["pp_list"] = cl
            break
    if main_res is None:
        raise SystemExit("speed: no main pass completed")
    sp = dict(main_res)
    sp.update({"vram_before_mib": vram_before, "vram_peak_mib": main_peak, "vram_idle_mib": main_peak, "engine": "llama.cpp", "kv": main_kv,
               "build": a.build_label, "bench_attempts": attempts})

    def depth(kv, dl):
        cmd = [a.bench_bin, "-m", a.model, "-ngl", "99", "-fa", "1", "-p", "512", "-n", str(gen), "-d", ",".join(map(str, dl)), "-r", "2", "-o", "json"]
        if kv == "q8_0":
            cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
        r, err, peak = run(cmd, 1800)
        if r is None or r.returncode != 0:
            return None, err, peak
        rows = json.loads(r.stdout)
        out = {}
        for x in rows:
            t = f"pp{x['n_prompt']}" if x["n_prompt"] else f"tg{x['n_gen']}"
            if x.get("n_depth"):
                t += f"@d{x['n_depth']}"
            out[t] = {"tokens_per_sec": round(x["avg_ts"], 2), "stddev": round(x["stddev_ts"], 2)}
        return {"rows": out, "raw_build_commit": rows[0].get("build_commit")}, "", peak

    for kv in ("f16", "q8_0"):
        got = None
        for dl in dls:
            res, err, peak = depth(kv, dl)
            attempts.append({"depth_kv": kv, "d_list": dl, "ok": res is not None, "vram_peak_mib": peak, "err": err})
            print(f"[speed] depth pass kv {kv} d {dl}: {'OK' if res else 'FAIL ' + err[-120:]} (peak {peak} MiB)", flush=True)
            if res:
                got = (res, dl, peak)
                break
        if got:
            res, dl, peak = got
            if kv == "f16":
                for k, v in res["rows"].items():
                    if "@d" in k:
                        sp[k] = v
                sp["depth_kv"] = "f16"
                sp["vram_peak_depth_mib"] = peak
            sp[f"depth_{kv}"] = {"d_list": dl, "rows": res["rows"], "vram_peak_mib": peak}
    sp["model_size_gib"] = sp.get("model_size_gib")
    (out_dir / "speed.json").write_text(json.dumps(sp, indent=2))
    print("[speed] DONE", json.dumps({k: sp[k] for k in ("kv", "vram_peak_mib") if k in sp}), "tg128", sp.get("tg128"), flush=True)


if __name__ == "__main__":
    main()
