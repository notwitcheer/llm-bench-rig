#!/usr/bin/env python3
"""in-run quality gate for the chess-engine task (capsule, CPU). This is the command registered with the Hermes /goal gate; the agent can read this
file and run it itself. exit 0 = pass, 1 = fail (the JSON line says what is wrong), 2 = harness error.

usage: quick_gate_chess.py <engine.py> <outdir>

checks, in order (the task text names every one of them):
  1. engine.py exists at the exact expected path (otherwise it names the path and lists any engine.py found elsewhere)
  2. the file compiles and imports only the standard library (no chess library, subprocess, network)
  3. `uci` -> `uciok`, `isready` -> `readyok`
  4. `go movetime 500` from five fixed positions returns a legal move within the time (the legal moves were frozen at staging; no move generator ships here)
  5. `go perft 1/2/3` from the start position prints `Nodes searched: N` with N = 20 / 400 / 8,902
It is a SMOKE gate. A pass here does not mean the engine is correct: the full check is run after the hour by a referee you cannot see.
"""
import ast
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
exe = os.path.abspath(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
D = json.load(open(os.path.join(HERE, "qgate_data.json")))
BANNED = {"subprocess", "socket", "ssl", "urllib", "http", "ftplib", "ctypes", "multiprocessing", "asyncio", "requests", "chess", "stockfish", "importlib", "shutil", "pip"}
fails = []


def fail(msg):
    fails.append(msg)


def finish():
    res = {"gate": "quick_gate_chess", "engine": exe, "pass": not fails, "failures": fails}
    open(os.path.join(out, "gate.json"), "w").write(json.dumps(res, indent=1))
    print(json.dumps({"pass": not fails, "failures": fails, "engine": exe}))
    sys.exit(0 if not fails else 1)


if not os.path.isfile(exe):
    strays = []
    for root in ("/tmp/chess81", os.path.expanduser("~/voxelbench/runs")):
        for dp, dn, fn in os.walk(root):
            if "engine.py" in fn:
                strays.append(os.path.join(dp, "engine.py"))
    fail("engine file missing: the final file must be exactly %s (copy this path, never retype it). engine.py files found elsewhere: %s" % (exe, strays[:5] or "none"))
    finish()
src = open(exe, encoding="utf-8", errors="replace").read()
try:
    tree = ast.parse(src)
    std = set(sys.stdlib_module_names)
    for n in ast.walk(tree):
        mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else ([(n.module or "").split(".")[0]] if isinstance(n, ast.ImportFrom) and n.level == 0 else [])
        for m in mods:
            if m in BANNED or m not in std:
                fail("import of '%s' is not allowed: the engine may use the Python 3 standard library only, no chess library, no subprocess, no network" % m)
        if isinstance(n, ast.Call):
            nm = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ""
            if nm in ("system", "popen", "fork", "__import__"):
                fail("call to '%s' is not allowed: the engine must not start other programs" % nm)
except SyntaxError as e:
    fail("engine.py has a syntax error: %s" % e)
    finish()


class E:
    def __init__(self):
        self.p = subprocess.Popen(["python3", exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=out, bufsize=0, start_new_session=True)
        self.q = queue.Queue()
        threading.Thread(target=self._r, daemon=True).start()
        self.err = b""
        threading.Thread(target=self._e, daemon=True).start()

    def _r(self):
        for ln in iter(self.p.stdout.readline, b""):
            self.q.put(ln.decode("utf-8", "replace").rstrip("\r\n"))
        self.q.put(None)

    def _e(self):
        for ln in iter(self.p.stderr.readline, b""):
            self.err += ln

    def tx(self, s):
        try:
            self.p.stdin.write((s + "\n").encode())
        except Exception:
            pass

    def until(self, pred, timeout):
        end, lines = time.time() + timeout, []
        while time.time() < end:
            try:
                ln = self.q.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                break
            if ln is None:
                return lines, None, "closed"
            lines.append(ln)
            if pred(ln):
                return lines, ln, "ok"
        return lines, None, "timeout"

    def stop(self):
        try:
            self.tx("quit")
            self.p.wait(timeout=1)
        except Exception:
            pass
        try:
            self.p.kill()
        except Exception:
            pass

    def tail(self):
        t = self.err.decode("utf-8", "replace").strip().splitlines()[-4:]
        return (" | last stderr: " + " / ".join(t)) if t else ""


e = E()
try:
    e.tx("uci")
    _, m, st = e.until(lambda l: l.strip() == "uciok", 10)
    if st != "ok":
        fail("`uci` did not answer `uciok` (%s within 10 s). Print `id name ...`, `id author ...` and `uciok`, and flush stdout.%s" % (st, e.tail()))
        finish()
    e.tx("isready")
    _, m, st = e.until(lambda l: l.strip() == "readyok", 10)
    if st != "ok":
        fail("`isready` did not answer `readyok` (%s).%s" % (st, e.tail()))
        finish()
    for pdat in D["positions"]:
        e.tx("ucinewgame")
        e.tx("position fen " + pdat["fen"])
        t0 = time.time()
        e.tx("go movetime 500")
        _, m, st = e.until(lambda l: l.startswith("bestmove"), 2.5)
        ms = int((time.time() - t0) * 1000)
        if st != "ok":
            fail("position '%s': no `bestmove` line within 2.5 s of `go movetime 500` (%s).%s" % (pdat["name"], st, e.tail()))
            break
        bm = m.split()[1] if len(m.split()) > 1 else ""
        if bm not in pdat["legal"]:
            fail("position '%s' (%s): bestmove '%s' is not legal there. Legal moves are: %s" % (pdat["name"], pdat["fen"], bm, " ".join(pdat["legal"])))
        elif ms > 1500:
            fail("position '%s': the move took %d ms for movetime 500; a late move loses a game" % (pdat["name"], ms))
    e.tx("position startpos")
    for d in ("1", "2", "3"):
        want = D["perft_startpos"][d]
        e.tx("go perft " + d)
        _, m, st = e.until(lambda l: re.search(r"Nodes searched:\s*\d+", l) is not None, 60)
        if st != "ok":
            fail("`go perft %s` printed no `Nodes searched: N` line (%s within 60 s).%s" % (d, st, e.tail()))
            break
        got = int(re.search(r"Nodes searched:\s*(\d+)", m).group(1))
        if got != want:
            fail("`go perft %s` from the start position gave %d, the correct count is %d: the move generator has a bug (re-read task rules 3 and 6; test smaller depths first)" % (d, got, want))
            break
finally:
    e.stop()
finish()
