#!/usr/bin/env python3
"""test_stubs.py: the gate's PASS path and every red path, on stub engines (CPU only, no model, no card).
usage: test_stubs.py [--workdir DIR] [--only name,name]
Builds the stubs (stubs/build_stubs.py), runs chess_final_gate.py on each, and checks the verdict against EXPECT. Exit 1 on any surprise."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FG = next(d for d in (os.path.join(HERE, "..", "final-gate"), os.path.join(HERE, "..")) if os.path.exists(os.path.join(d, "chess_final_gate.py")))
wd = sys.argv[sys.argv.index("--workdir") + 1] if "--workdir" in sys.argv else "/tmp/chess-stubtest"
only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
os.makedirs(wd, exist_ok=True)
subprocess.run([sys.executable, os.path.join(HERE, "..", "stubs", "build_stubs.py"), os.path.join(wd, "stubs")], check=True, stdout=subprocess.DEVNULL)

# name -> (expected pass, failed checks that MUST be present, failed list must be exactly those, coverage failures that must be present, env)
EXPECT = {
    "good": (True, [], True, [], {}),
    "bad_noep": (False, ["perft_5"], True, ["kiwipete", "position3"], {}),
    # no castling at all: the generator cannot even follow an opponent's castling move, so vs the seeded random baseline the engine desyncs and the
    # games leg fails ("illegal move"); if the baseline never castles in the 10 games it passes and only the kiwipete coverage extra shows the gap
    "bad_nocastle": (None, [], False, ["kiwipete"], {}),
    "bad_nopromo": (True, [], True, ["position4", "position5"], {}),
    "bad_illegal": (False, [], False, [], {}),   # caught at go_legal or games_clean, either is the pre-registered illegal-move failure
    "bad_silent": (False, ["uci_handshake"], False, [], {}),
    "bad_slowperft": (False, ["perft_1", "perft_5"], False, [], {"GATE_PERFT_TIMEOUTS": "3,3,3,3,3", "GATE_SKIP_EXTRAS": "1"}),
    "bad_overtime": (False, ["games_clean"], False, [], {}),
    "bad_import": (False, ["source_scan"], False, [], {}),   # the missing module also kills the handshake on a box without python-chess; the first failure must be source_scan
}


def run(name):
    out = os.path.join(wd, "out-" + name)
    subprocess.run(["rm", "-rf", out])
    env = dict(os.environ, **EXPECT[name][4])
    t = time.time()
    subprocess.run([sys.executable, os.path.join(FG, "chess_final_gate.py"), os.path.join(wd, "stubs", name + ".py"), out], env=env,
                   stdout=open(os.path.join(wd, name + ".log"), "w"), stderr=subprocess.STDOUT, timeout=1500)
    g = json.load(open(os.path.join(out, "gate.json")))
    return name, g, time.time() - t


names = [n for n in EXPECT if not only or n in only]
bad = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    for name, g, secs in ex.map(run, names):
        want_pass, must, exact, cov, _ = EXPECT[name]
        ok = (want_pass is None or g["pass"] == want_pass) and (name != "bad_nocastle" or set(g["failed"]) <= {"games_clean"}) \
            and (name != "bad_illegal" or bool({"go_legal", "games_clean"} & set(g["failed"]))) and (name != "bad_import" or g["failed"][:1] == ["source_scan"]) and all(m in g["failed"] for m in must) and (not exact or sorted(g["failed"]) == sorted(must)) \
            and all(c in g["coverage_failures"] for c in cov) and (name != "good" or not g["coverage_failures"])
        bad += 0 if ok else 1
        print("%-14s %s  pass=%-5s failed=%s coverage_fail=%s  %.0fs\n   %s" % (name, "OK  " if ok else "SURPRISE", g["pass"], g["failed"], g["coverage_failures"], secs, g["headline"]), flush=True)
print("RESULT", "all %d verdicts as pre-registered" % len(names) if bad == 0 else "%d SURPRISES" % bad)
sys.exit(1 if bad else 0)
