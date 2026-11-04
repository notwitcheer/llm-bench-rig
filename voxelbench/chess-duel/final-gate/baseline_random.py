#!/usr/bin/env python3
"""baseline_random.py: the FIXED WEAK BASELINE of the chess duel gate. A UCI engine that plays a uniformly random legal move; the choice is seeded by the
position (crc32 of the fen) so the same position always gets the same move, and the whole baseline is reproducible. It answers `go` instantly."""
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chesslib as cl  # noqa: E402

pos = cl.Position()
while True:
    line = sys.stdin.readline()
    if not line:
        break
    a = line.split()
    if not a:
        continue
    if a[0] == "uci":
        print("id name baseline-random\nid author harness\nuciok", flush=True)
    elif a[0] == "isready":
        print("readyok", flush=True)
    elif a[0] == "ucinewgame":
        pos = cl.Position()
    elif a[0] == "position":
        if a[1] == "startpos":
            pos, i = cl.Position(), 2
        else:
            j = a.index("moves") if "moves" in a else len(a)
            pos, i = cl.Position(" ".join(a[2:j])), j
        if i < len(a) and a[i] == "moves":
            for mv in a[i + 1:]:
                m = pos.parse_uci(mv)
                if m is None:
                    break
                pos.play(m)
    elif a[0] == "go":
        ms = pos.legal_moves()
        if not ms:
            print("bestmove 0000", flush=True)
        else:
            r = random.Random(zlib.crc32(pos.fen().encode()))
            print("info depth 1 score cp 0 nodes %d" % len(ms), flush=True)
            print("bestmove " + cl.Position.uci(r.choice(ms)), flush=True)
    elif a[0] == "quit":
        break
