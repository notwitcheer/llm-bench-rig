#!/usr/bin/env python3
"""make_mock_results.py <run_dir> <mode>: writes MOCK per-try files for the CPU selftests (summary branches + render). mode = repro | nodiff | broken | partial.
The run dir gets a MOCK marker file so the render stamps MOCK DATA on every frame. Nothing here is ever counted."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astra_common import svg_check

def pel(seed, ok=True):
    r = random.Random(seed); c = lambda: "#%02x%02x%02x" % (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
    s = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><rect width="400" height="300" fill="{c()}"/><circle cx="120" cy="220" r="45" fill="none" stroke="#222" stroke-width="6"/><circle cx="290" cy="220" r="45" fill="none" stroke="#222" stroke-width="6"/><path d="M120 220 L190 150 L290 220" stroke="{c()}" stroke-width="7" fill="none"/><ellipse cx="190" cy="120" rx="{r.randint(28, 45)}" ry="30" fill="{c()}"/><path d="M215 112 L300 128 L215 140Z" fill="#f5a623"/></svg>'
    return s if ok else s[:-30]

def main():
    run, mode = sys.argv[1], sys.argv[2]; rnd = random.Random(5)
    open(os.path.join(run, "MOCK"), "w").write("mock\n")
    for arm in ("base", "edited"):
        d = os.path.join(run, "results", arm); os.makedirs(d, exist_ok=True)
        for s in range(1, 33):
            if mode == "nodiff": mean = 28000
            elif mode == "partial": mean = 28000 if arm == "base" else 24000
            else: mean = 28000 if arm == "base" else 17000
            n = max(900, int(rnd.gauss(mean, 5500)))
            ok = not (mode == "broken" and arm == "edited" and s % 2 == 0)
            think = "Let me think about the pelican. " * 40 + f"\n(mock try {arm} {s})\n" * 5
            ans = "Here is the SVG:\n```svg\n" + pel(s * (1 if arm == "base" else 7), ok) + "\n```\n"
            chk = svg_check(pel(s, True) if ok else None)
            wall = n / 70.0; k = min(n, 400)
            tt = [int(i * wall * 1000 / k) for i in range(1, k + 1)]; ta = [int(wall * 1000 + i * 8) for i in range(60)]
            r = dict(tag=arm, kind="pelican", id=f"{s:02d}", seed=s, finish_reason="stop" if ok else "length", wall_s=round(wall + 2, 1), think_tokens=n, think_chunks=k, answer_tokens=600, answer_chunks=60,
                     completion_tokens=n + 600, prompt_tokens=60, think_end_ms=tt[-1], answer_end_ms=ta[-1], decode_tps_server=70.0, svg=chk, complete=bool(ok and chk["renders"]))
            b = os.path.join(d, f"pelican-{s:02d}"); open(b + ".think.txt", "w").write(think); open(b + ".answer.txt", "w").write(ans)
            json.dump(dict(think_ms=tt, answer_ms=ta), open(b + ".tok.json", "w")); json.dump(r, open(b + ".json", "w"))
        json.dump({"loaded": mode != "loadfail"}, open(os.path.join(run, f"fit-{arm}.json"), "w"))
main()
