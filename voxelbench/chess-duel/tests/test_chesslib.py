#!/usr/bin/env python3
"""test_chesslib.py: the referee's rules against (1) the frozen chessprogramming.org perft numbers and (2) optionally python-chess
(a CPU-only throwaway venv at staging; python-chess is a TEST ORACLE here, it is never imported by any engine or by the gate itself).
usage: test_chesslib.py [--deep] [--pychess]   (--deep adds startpos depth 5 and kiwipete depth 3 as gate-size runs)"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FG = next(d for d in (os.path.join(HERE, "..", "final-gate"), os.path.join(HERE, "..")) if os.path.exists(os.path.join(d, "chesslib.py")))
sys.path.insert(0, FG)
import chesslib as cl  # noqa: E402

ref = json.load(open(os.path.join(FG, "perft_ref.json")))
deep = "--deep" in sys.argv
bad = 0
cases = [ref["gating"]] + ref["extras"]
for c in cases:
    for d, want in c["depths"].items():
        d = int(d)
        if not deep and want > 250000:
            continue
        t = time.time()
        got = cl.perft(cl.Position(c["fen"]), d)
        ok = got == want
        bad += 0 if ok else 1
        print("perft %-9s d%d got %-9d want %-9d %s  %.1fs" % (c["name"], d, got, want, "ok" if ok else "MISMATCH", time.time() - t))

# fen round trip, uci/san basics
p = cl.Position()
assert p.fen() == cl.START_FEN, p.fen()
assert cl.Position.uci(p.parse_uci("e2e4")) == "e2e4" and p.parse_uci("e2e5") is None and p.parse_uci("e7e5") is None
print("fen/uci basics ok")

if "--pychess" in sys.argv:
    import chess  # test oracle only
    rnd = random.Random(7)
    games, plies, bad_games = 60, 0, 0
    for g in range(games):
        mine, theirs = cl.Position(), chess.Board()
        san_mine = []
        for ply in range(160):
            a = sorted(cl.Position.uci(m) for m in mine.legal_moves())
            b = sorted(m.uci() for m in theirs.legal_moves)
            if a != b:
                print("LEGAL-MOVE MISMATCH game %d ply %d fen %s\n  mine-theirs %s\n  theirs-mine %s" % (g, ply, theirs.fen(), sorted(set(a) - set(b)), sorted(set(b) - set(a))))
                bad_games += 1
                break
            if not a:
                break
            mv = rnd.choice(b)
            sm = mine.parse_uci(mv)
            s1, s2 = mine.san(sm), theirs.san(chess.Move.from_uci(mv))
            if s1 != s2:
                print("SAN MISMATCH game %d ply %d: mine %s theirs %s" % (g, ply, s1, s2))
                bad_games += 1
                break
            mine.play(sm)
            theirs.push(chess.Move.from_uci(mv))
            plies += 1
            o1, o2 = mine.outcome(), theirs.outcome(claim_draw=True)
            r1 = o1[0] if o1 else None
            r2 = o2.result() if o2 else None
            if r1 != r2:
                # python-chess claim_draw may claim a 3-fold or 50-move draw earlier/later by one ply convention: report, count
                print("OUTCOME DIFF game %d ply %d: mine %s theirs %s fen %s" % (g, ply, o1, o2, theirs.fen()))
                bad_games += 1
                break
            if r1:
                break
    print("random-game oracle: %d games, %d plies, %d disagreements" % (games, plies, bad_games))
    bad += bad_games
print("RESULT", "PASS" if bad == 0 else "FAIL (%d)" % bad)
sys.exit(1 if bad else 0)
