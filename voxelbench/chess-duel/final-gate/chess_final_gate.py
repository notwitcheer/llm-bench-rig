#!/usr/bin/env python3
"""chess_final_gate.py <engine.py> <outdir>: the PRE-REGISTERED FINAL GATE of the chess duel (queue item 81). Lives OUTSIDE the agent's tree
(capsule ~/.vxfinal/chess/, public mirror voxelbench/chess-duel/final-gate/). Written 2026-11-03 before any agent run of this task.

An engine PASSES iff every one of these holds (the frozen numbers are in perft_ref.json, copied from chessprogramming.org/Perft_Results, fetched 2026-11-03):
  1. source_scan      the file compiles and imports only the standard library: no chess library (python-chess etc.), no subprocess / os.system / sockets / network / ctypes
  2. uci_handshake    `uci` -> `uciok` and `isready` -> `readyok` (15 s each)
  3. go_legal         `go movetime 500` from the start position and from three fixed positions returns a LEGAL move (checked by the referee's own rules, chesslib.py)
  4. perft_1 .. perft_5   `go perft N` from the start position prints `Nodes searched: <total>` equal to the frozen count at depths 1..5
                          (20, 400, 8,902, 197,281, 4,865,609); timeouts 60 / 60 / 60 / 300 / 1,200 s per depth (GATE_PERFT_TIMEOUTS overrides for tests only)
  5. games_clean      10 games against the fixed weak baseline (baseline_random.py: a seeded random legal mover), 5 openings x 2 colours, `go movetime 500` + 1,500 ms grace,
                      with NO illegal move, no crash and no time forfeit by the engine under test. W/D/L is reported, not gating.
REPORTED, NOT GATING: coverage extras (kiwipete d1-3, position3 d1-4, position4 d1-3, position5 d1-3: castling / en passant / promotion), per-depth perft seconds, games W/D/L.
(The start position never castles or promotes within 5 plies: an engine can pass the gate and still not know how to castle. The extras report it.)
Exit code 0 = PASS, 1 = FAIL (gate.json holds `pass`, `checks`, `failed`, `headline`)."""
import ast
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chesslib as cl  # noqa: E402
import chessmatch as cm  # noqa: E402

BANNED_MODULES = {"subprocess", "socket", "ssl", "urllib", "http", "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "xmlrpc", "ctypes", "webbrowser",
                  "pty", "asyncio", "multiprocessing", "concurrent", "importlib", "pip", "ensurepip", "venv", "shutil", "tempfile", "requests", "chess", "stockfish"}
BANNED_CALLS = {"system", "popen", "execv", "execvp", "execl", "execlp", "spawnl", "spawnv", "fork", "forkpty", "__import__", "eval_import"}
ENGINE_PY = os.environ.get("ENGINE_PY", "python3")
PT = [int(x) for x in os.environ.get("GATE_PERFT_TIMEOUTS", "60,60,60,300,1200").split(",")]
N_GAMES = int(os.environ.get("GATE_GAMES", "10"))
MOVETIME = int(os.environ.get("GATE_MOVETIME", "500"))
CONC = int(os.environ.get("GATE_CONC", "4"))
SKIP_EXTRAS = os.environ.get("GATE_SKIP_EXTRAS") == "1"
GO_LEGAL_FENS = [cl.START_FEN, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                 "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"]


def scan_source(path):
    try:
        src = open(path, encoding="utf-8").read()
    except Exception as e:
        return False, "cannot read: %s" % e
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, "syntax error: %s" % e
    std = set(sys.stdlib_module_names)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods = [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods = [(n.module or "").split(".")[0]] if n.level == 0 else []
        else:
            mods = []
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
                if nm in BANNED_CALLS:
                    bad.append("call " + nm)
        for m in mods:
            if m in BANNED_MODULES:
                bad.append("import " + m)
            elif m not in std:
                bad.append("non-stdlib import " + m)
    return (not bad), ("ok" if not bad else "; ".join(sorted(set(bad))))


def perft_run(path, fen, depth, timeout, workdir):
    e = cm.UciEngine([ENGINE_PY, path], cwd=workdir)
    try:
        ok, why = e.handshake()
        if not ok:
            return None, 0.0, "handshake: " + why
        e.send("position fen " + fen if fen != cl.START_FEN else "position startpos")
        t = time.time()
        e.send("go perft %d" % depth)
        lines, m, st = e.read_until(lambda l: re.search(r"Nodes searched:\s*\d+", l) is not None, timeout)
        secs = time.time() - t
        if st != "ok":
            return None, secs, "timeout after %.0f s" % secs if st == "timeout" else "engine closed its output"
        return int(re.search(r"Nodes searched:\s*(\d+)", m).group(1)), secs, "ok"
    finally:
        e.close()


def main():
    path, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ref = json.load(open(os.path.join(HERE, "perft_ref.json")))
    work = os.path.join(out, "cwd")
    os.makedirs(work, exist_ok=True)
    G = {"engine": os.path.basename(path), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": {}, "perft": {}, "extras": {}, "games": None, "detail": {}}
    C, D = G["checks"], G["detail"]
    t_all = time.time()

    ok, why = scan_source(path)
    C["source_scan"], D["source_scan"] = ok, why
    ee = cm.UciEngine([ENGINE_PY, path], cwd=work, stderr_path=os.path.join(out, "engine-stderr.txt"))
    hs, hwhy = ee.handshake()
    ee.close()
    C["uci_handshake"], D["uci_handshake"] = hs, hwhy
    if C["source_scan"] is False and hs is False:
        pass
    if hs:
        # go_legal
        bad, names = [], []
        for fen in GO_LEGAL_FENS:
            e = cm.UciEngine([ENGINE_PY, path], cwd=work)
            try:
                e.handshake()
                e.send("position fen " + fen)
                e.send("go movetime %d" % MOVETIME)
                lines, m, st = e.read_until(lambda l: l.startswith("bestmove"), MOVETIME / 1000.0 + 1.5)
                pos = cl.Position(fen)
                bm = m.split()[1] if m and len(m.split()) > 1 else ""
                if st != "ok" or pos.parse_uci(bm) is None:
                    bad.append("%s -> %s" % (fen.split()[0][:20], bm or st))
            finally:
                e.close()
        C["go_legal"], D["go_legal"] = (not bad), ("ok" if not bad else "; ".join(bad))
        # perft start position
        g = ref["gating"]
        for d in range(1, 6):
            want = g["depths"][str(d)]
            got, secs, st = perft_run(path, g["fen"], d, PT[d - 1], work)
            key = "perft_%d" % d
            C[key] = (got == want)
            G["perft"][str(d)] = {"want": want, "got": got, "secs": round(secs, 1), "status": st}
            D[key] = "ok" if got == want else ("got %s, want %s" % (got, want) if st == "ok" else "got %s, want %s (%s)" % (got, want, st))
            print("%s: %s (%.1fs)" % (key, D[key], secs), flush=True)
        # games vs the fixed weak baseline
        base = {"name": "baseline-random", "cmd": [sys.executable, os.path.join(HERE, "baseline_random.py")]}
        eng = {"name": os.path.basename(os.path.dirname(path)) or "engine", "cmd": [ENGINE_PY, path]}
        res = cm.run_match(eng, base, N_GAMES, MOVETIME, openings=cm.OPENINGS[:max(1, N_GAMES // 2)], concurrency=CONC, errdir=out, event="gate vs baseline")
        cm.write_pgn(res, os.path.join(out, "baseline-games.pgn"))
        t = res["tally"]
        G["games"] = {"engine_w": t["a_win"], "draw": t["draw"], "engine_l": t["b_win"], "illegal": t["a_illegal"], "time_or_crash": t["a_time_or_crash"],
                      "reasons": [(x["round"], x["result"], x["reason"]) for x in res["games"]]}
        clean = t["a_illegal"] == 0 and t["a_time_or_crash"] == 0
        C["games_clean"] = clean
        D["games_clean"] = "ok (%d-%d-%d, no illegal/crash/forfeit)" % (t["a_win"], t["draw"], t["b_win"]) if clean else \
            "engine illegal %d, time/crash %d (W-D-L %d-%d-%d)" % (t["a_illegal"], t["a_time_or_crash"], t["a_win"], t["draw"], t["b_win"])
        print("games_clean: %s" % D["games_clean"], flush=True)
        # coverage extras, reported only
        if not SKIP_EXTRAS:
            for x in ref["extras"]:
                row = {}
                for d, want in x["depths"].items():
                    got, secs, st = perft_run(path, x["fen"], int(d), 120, work)
                    row[d] = {"want": want, "got": got, "ok": got == want, "secs": round(secs, 1)}
                G["extras"][x["name"]] = {"covers": x["covers"], "depths": row, "ok": all(v["ok"] for v in row.values())}
                print("extra %s: %s" % (x["name"], "ok" if G["extras"][x["name"]]["ok"] else "FAIL " + json.dumps(row)), flush=True)
    else:
        for k in ["go_legal", "perft_1", "perft_2", "perft_3", "perft_4", "perft_5", "games_clean"]:
            C[k], D[k] = False, "not run (no UCI handshake)"
    order = ["source_scan", "uci_handshake", "go_legal", "perft_1", "perft_2", "perft_3", "perft_4", "perft_5", "games_clean"]
    failed = [k for k in order if not C.get(k)]
    G["pass"] = not failed
    G["failed"] = failed
    cover = {k: v["ok"] for k, v in G["extras"].items()}
    G["coverage_failures"] = [k for k, v in cover.items() if not v]
    if G["pass"]:
        G["headline"] = "PASS: perft 1-5 exact, legal in all games vs the baseline" + (
            "; coverage gaps: " + ", ".join("%s (%s)" % (k, G["extras"][k]["covers"]) for k in G["coverage_failures"]) if G["coverage_failures"] else "; coverage extras all ok")
    else:
        f0 = failed[0]
        G["headline"] = "FAIL at %s: %s" % (f0, D.get(f0))
    G["elapsed_s"] = round(time.time() - t_all, 1)
    json.dump(G, open(os.path.join(out, "gate.json"), "w"), indent=1)
    print("GATE", "PASS" if G["pass"] else "FAIL", "-", G["headline"], flush=True)
    sys.exit(0 if G["pass"] else 1)


if __name__ == "__main__":
    main()
