#!/usr/bin/env python3
"""build_stubs.py <outdir>: builds the harness test engines as SINGLE files (chesslib inlined + the stub UCI loop), plus deliberately broken variants.
good.py is the known-good stub for the PASS path; every bad_*.py must fail the final gate at the named check (tests/test-chess-stubs.sh asserts the table).

  good            all checks pass
  bad_noep        no en passant: startpos perft depth 5 is off (258 en passant captures missing at depth 5) -> gate FAIL at perft 5
  bad_nocastle    no castling: startpos perft 1-5 still PASS (the start position never castles inside 5 plies) -> gate PASS, coverage extra kiwipete FAILS (the headline case)
  bad_nopromo     no underpromotion, queen only: startpos perft passes, position4 / position5 coverage fail
  bad_illegal     search walks pseudo-legal moves (ignores pins and checks): perft passes, the games vs the baseline hit an illegal move -> gate FAIL
  bad_silent      reads stdin, never answers uci -> gate FAIL at the handshake
  bad_slowperft   answers go perft N after a long sleep -> perft timeout
  bad_overtime    ignores movetime, thinks ~6 s every move -> time forfeit in games
  bad_import      starts with `import chess` -> static scan FAIL (python-chess is forbidden inside an engine)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FG = next(d for d in (os.path.join(HERE, "..", "final-gate"), os.path.join(HERE, "..")) if os.path.exists(os.path.join(d, "chesslib.py")))   # repo layout or the flat capsule layout
lib = open(os.path.join(FG, "chesslib.py")).read()
main = open(os.path.join(HERE, "stub_main.py")).read()
lib = re.sub(r'^""".*?"""\n', "", lib, count=1, flags=re.S)
good = lib + "\n" + main
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def put(name, src):
    open(os.path.join(out, name + ".py"), "w").write(src)


def swap(src, a, b):
    assert a in src, "pattern missing: " + a
    return src.replace(a, b, 1)


put("good", good)
put("bad_noep", swap(good, "elif to == self.ep and tp == 0:", "elif False and to == self.ep and tp == 0:"))
put("bad_nocastle", swap(swap(good, "if us == 0 and s == 4:", "if False and us == 0 and s == 4:"), "elif us == 1 and s == 60:", "elif False and us == 1 and s == 60:"))
put("bad_nopromo", good.replace("for pr in (5, 4, 3, 2):", "for pr in (5,):"))
# illegal-mover: the in-game search uses pseudo-legal moves (the perft path keeps legal_moves)
il = good.replace("moves = pos.legal_moves()\n        if not moves:\n            return -30000 + ply if pos.in_check() else 0", "moves = pos.pseudo_moves()\n        if not moves:\n            return 0")
il = il.replace("    moves = pos.legal_moves()\n    if not moves:\n        return None\n    best_move = moves[0]", "    moves = pos.pseudo_moves()\n    if not moves:\n        return None\n    best_move = moves[0]")
assert il != good
put("bad_illegal", il)
put("bad_silent", "import sys\nfor line in sys.stdin:\n    pass\n")
put("bad_slowperft", good.replace("def perft_divide(pos, depth):\n", "def perft_divide(pos, depth):\n    time.sleep(30)\n", 1))
put("bad_overtime", good.replace("deadline = t0 + movetime_ms / 1000.0 * 0.85", "deadline = t0 + 6.0", 1))
put("bad_import", "import chess\n" + good)
print("built", sorted(os.listdir(out)))
