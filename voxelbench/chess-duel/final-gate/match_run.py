#!/usr/bin/env python3
"""match_run.py: the engine-vs-engine match and the Elo anchor of the chess duel (CPU only, any hour). Lives with the final gate (capsule ~/.vxfinal/chess/).

  duel   <outdir> --a ENGINE.py --a-name NAME --b ENGINE.py --b-name NAME [--games 20] [--movetime 1000] [--conc 4]
         engine A and B, alternating colours (A white in odd rounds), 10 fixed 4-ply openings x 2 colours, fixed time per move, illegal move / time forfeit / crash = loss.
         writes match.json + match.pgn (SAN with {score depth nodes ms} comments)
  anchor <outdir> --a ENGINE.py --a-name NAME --sf STOCKFISH [--games 10] [--movetime 1000] [--conc 3] [--elo 1320]
         the engine against Stockfish with UCI_LimitStrength + UCI_Elo (1320 is its minimum), Threads 1. writes anchor.json + anchor.pgn. The Elo estimate is
         INDICATIVE only: Stockfish's UCI_Elo scale is calibrated at a much longer time control than 1 s per move, so say "scored s/10 against Stockfish limited to
         1320", never "rated N".
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chessmatch as cm  # noqa: E402

PY = os.environ.get("ENGINE_PY", "python3")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    mode, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    n = int(arg("--games", 20 if mode == "duel" else 10))
    mt = int(arg("--movetime", 1000))
    conc = int(arg("--conc", 4 if mode == "duel" else 3))
    a = {"name": arg("--a-name", "A"), "cmd": [PY, os.path.abspath(arg("--a"))]}
    if mode == "duel":
        b = {"name": arg("--b-name", "B"), "cmd": [PY, os.path.abspath(arg("--b"))]}
        event = "chess duel: %s vs %s" % (a["name"], b["name"])
    else:
        elo = arg("--elo", "1320")
        b = {"name": "Stockfish (UCI_Elo %s)" % elo, "cmd": [os.path.abspath(arg("--sf"))],
             "options": [("UCI_LimitStrength", "true"), ("UCI_Elo", elo), ("Threads", "1"), ("Hash", "16")]}
        event = "elo anchor: %s vs Stockfish limited to %s" % (a["name"], elo)
    t0 = time.time()

    def on_game(g):
        print("game %d: %s vs %s -> %s (%s, %d plies) %.0fs" % (g["round"], g["white"], g["black"], g["result"], g["reason"], g["plies"], time.time() - t0), flush=True)
    res = cm.run_match(a, b, n, mt, concurrency=conc, errdir=out, event=event, on_game=on_game)
    t = res["tally"]
    pts_a = t["a_win"] + 0.5 * t["draw"]
    res["score_a"], res["score_b"], res["elapsed_s"] = pts_a, n - pts_a, round(time.time() - t0, 1)
    if mode == "anchor":
        s = (pts_a + 0.5) / (n + 1)      # Laplace-smoothed so a 10-0 or 0-10 stays finite; reported as a bound
        se = math.sqrt(max(s * (1 - s), 1e-9) / n)
        f = lambda x: float(elo) + 400 * math.log10(max(min(x, 0.995), 0.005) / (1 - max(min(x, 0.995), 0.005)))  # noqa: E731
        res["anchor"] = {"sf_elo_setting": float(elo), "score": pts_a, "games": n, "elo_estimate": round(f(s)), "elo_lo": round(f(s - 1.96 * se)), "elo_hi": round(f(s + 1.96 * se)),
                         "bounded": pts_a in (0, n), "note": "indicative, n = %d, Stockfish UCI_Elo scale calibrated at a longer time control" % n}
    json.dump(res, open(os.path.join(out, "match.json" if mode == "duel" else "anchor.json"), "w"))
    cm.write_pgn(res, os.path.join(out, "match.pgn" if mode == "duel" else "anchor.pgn"))
    print("DONE %s: %s %.1f - %.1f %s | illegal a/b %d/%d, time-or-crash a/b %d/%d | %.0fs" % (mode, a["name"], pts_a, n - pts_a, b["name"], t["a_illegal"], t["b_illegal"], t["a_time_or_crash"], t["b_time_or_crash"], res["elapsed_s"]), flush=True)


if __name__ == "__main__":
    main()
