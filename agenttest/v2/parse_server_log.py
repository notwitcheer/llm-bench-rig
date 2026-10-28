#!/usr/bin/env python3
"""parse llama-server per-request timings from its log (prompt eval + eval lines) and a gpu sampler log.

usage: [SKIP_FIRST=N] parse_server_log.py <server.log> [<gpu.log>] > server-timings.json
SKIP_FIRST drops the first N timed requests (manual curl probes before the agent run).
decode = the `eval time = ... tokens per second` line of each request (perceived TPS, single user)."""
import json
import re
import statistics
import sys

EVAL = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")
PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)


def main():
    reqs, pend = [], None
    for line in open(sys.argv[1], errors="replace"):
        m = PROMPT.search(line)
        if m:
            pend = {"prompt_ms": float(m[1]), "prompt_n": int(m[2]), "prompt_tps": float(m[4])}
            continue
        m = EVAL.search(line)
        if m:
            r = pend or {}
            r.update({"eval_ms": float(m[1]), "eval_n": int(m[2]), "eval_tps": float(m[4])})
            reqs.append(r)
            pend = None
    import os
    skip = int(os.environ.get("SKIP_FIRST", "0"))
    reqs = reqs[skip:]
    dec = [r["eval_tps"] for r in reqs if r.get("eval_n", 0) >= 32]
    pre = [r["prompt_tps"] for r in reqs if r.get("prompt_n", 0) >= 256]
    out = {
        "requests": len(reqs),
        "skipped_first": skip,
        "decode_tps_requests_ge32tok": {"n": len(dec), "p50": pct(dec, 50), "p90": pct(dec, 90), "p10": pct(dec, 10),
                                        "min": min(dec) if dec else None, "max": max(dec) if dec else None},
        "prefill_tps_requests_ge256tok": {"n": len(pre), "p50": pct(pre, 50), "p90": pct(pre, 90)},
        "total_eval_tokens": sum(r.get("eval_n", 0) for r in reqs),
        "total_prompt_tokens_evaluated": sum(r.get("prompt_n", 0) for r in reqs),
        "eval_n_p50": pct([r.get("eval_n", 0) for r in reqs], 50),
        "eval_n_max": max([r.get("eval_n", 0) for r in reqs], default=None),
    }
    if len(sys.argv) > 2:
        used, foreign = [], set()
        for line in open(sys.argv[2], errors="replace"):
            m = re.search(r" (\d+)MiB,(\d+)% apps=([^ ]*)", line)
            if m:
                used.append(int(m[1]))
                pids = [x.split(",")[0] for x in m[3].split(";") if x]
                if len(pids) > 1:
                    foreign.add(m[3])
        out["gpu"] = {"samples": len(used), "vram_used_mib_max": max(used, default=None),
                      "vram_used_mib_p50": pct(used, 50), "samples_with_more_than_one_compute_app": sorted(foreign)}
    out["requests_detail"] = reqs
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
