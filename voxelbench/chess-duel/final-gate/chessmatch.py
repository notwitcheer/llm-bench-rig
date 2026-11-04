"""chessmatch.py: UCI client + referee + match runner (stdlib only; rules from chesslib.py). Lives with the final gate (capsule ~/.vxfinal/chess/).

play_game() is the referee: it applies every move to its own Position, so an illegal move, a late move, a crash or `bestmove 0000` is caught by
the rules engine of the referee, never by trusting the engine. Illegal move, time forfeit (movetime + grace) and crash all LOSE the game.
"""
import datetime
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chesslib as cl  # noqa: E402

# fixed opening book: 4-ply lines from the start position (frozen; both colours of each line are played)
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6", "e2e4 c7c5 g1f3 d7d6", "g1f3 g8f6 c2c4 e7e6", "e2e4 e7e6 d2d4 d7d5",
    "d2d4 g8f6 c2c4 g7g6", "e2e4 c7c6 d2d4 d7d5", "c2c4 e7e5 b1c3 g8f6", "e2e4 d7d6 d2d4 g8f6", "d2d4 d7d5 g1f3 g8f6",
]


class UciEngine:
    def __init__(self, cmd, name=None, options=(), stderr_path=None, cwd=None):
        self.cmd, self.name = cmd, name or os.path.basename(cmd[-1])
        self.options = list(options)
        self.q = queue.Queue()
        self.log = []
        self.err = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, cwd=cwd, bufsize=0, start_new_session=True)
        self.dead = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf = b""
        while True:
            try:
                chunk = os.read(self.p.stdout.fileno(), 65536)
            except Exception:
                chunk = b""
            if not chunk:
                self.q.put(None)
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.q.put(line.decode("utf-8", "replace").rstrip("\r"))

    def send(self, line):
        try:
            self.p.stdin.write((line + "\n").encode())
        except Exception:
            self.dead = True

    def read_until(self, pred, timeout):
        """collect lines until pred(line) is true; returns (lines, matched_line_or_None, status) status in ok|timeout|closed"""
        end = time.time() + timeout
        lines = []
        while True:
            left = end - time.time()
            if left <= 0:
                return lines, None, "timeout"
            try:
                ln = self.q.get(timeout=left)
            except queue.Empty:
                return lines, None, "timeout"
            if ln is None:
                self.dead = True
                return lines, None, "closed"
            lines.append(ln)
            if pred(ln):
                return lines, ln, "ok"

    def handshake(self, timeout=15):
        self.send("uci")
        lines, m, st = self.read_until(lambda l: l.strip() == "uciok", timeout)
        if st != "ok":
            return False, "no uciok (%s)" % st
        for o in self.options:
            self.send("setoption name %s value %s" % o)
        self.send("isready")
        lines2, m, st = self.read_until(lambda l: l.strip() == "readyok", timeout)
        if st != "ok":
            return False, "no readyok (%s)" % st
        nm = [l for l in lines if l.startswith("id name")]
        if nm:
            self.name = nm[0][len("id name"):].strip() or self.name
        return True, "ok"

    def go(self, moves, movetime_ms, grace_ms):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        t0 = time.time()
        self.send("go movetime %d" % movetime_ms)
        lines, m, st = self.read_until(lambda l: l.startswith("bestmove"), movetime_ms / 1000.0 + grace_ms / 1000.0)
        ms = int((time.time() - t0) * 1000)
        info = {}
        for l in lines:
            if l.startswith("info") and " depth " in l:
                d = re.search(r"depth (\d+)", l)
                n = re.search(r"nodes (\d+)", l)
                sc = re.search(r"score (cp|mate) (-?\d+)", l)
                if d:
                    info["depth"] = int(d.group(1))
                if n:
                    info["nodes"] = int(n.group(1))
                if sc:
                    info["score"] = (sc.group(1), int(sc.group(2)))
        if st != "ok":
            return None, ms, info, st
        parts = m.split()
        return (parts[1] if len(parts) > 1 else ""), ms, info, "ok"

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=1.5)
        except Exception:
            pass
        try:
            self.p.kill()  # own child only, by handle
            self.p.wait(timeout=3)
        except Exception:
            pass
        try:
            self.p.stdin.close()
        except Exception:
            pass
        if self.err is not subprocess.DEVNULL:
            try:
                self.err.close()
            except Exception:
                pass


def play_game(white, black, opening, movetime_ms=1000, grace_ms=1500, max_plies=240, errdir=None, tag="g"):
    """white / black: dicts {name, cmd, options}. returns a game record dict."""
    pos = cl.Position()
    moves, rec = [], []
    ens = []
    result, reason, loser = None, None, None
    try:
        for side, spec in ((0, white), (1, black)):
            ep = os.path.join(errdir, "%s-%s.err" % (tag, "wb"[side])) if errdir else None
            e = UciEngine(spec["cmd"], spec["name"], spec.get("options", ()), ep)
            ens.append(e)
        for e in ens:
            ok, why = e.handshake()
            if not ok:
                loser = 0 if e is ens[0] else 1
                result = "0-1" if loser == 0 else "1-0"
                reason = "handshake failed: " + why
                raise StopIteration
            e.send("ucinewgame")
        for mv in opening.split() if opening else []:
            m = pos.parse_uci(mv)
            rec.append({"uci": mv, "san": pos.san(m), "book": True})
            pos.play(m)
            moves.append(mv)
        while True:
            oc = pos.outcome()
            if oc:
                result, reason = oc
                break
            if len(moves) >= max_plies:
                result, reason = "1/2-1/2", "adjudicated draw (max plies %d)" % max_plies
                break
            side = pos.turn
            e = ens[side]
            bm, ms, info, st = e.go(moves, movetime_ms, grace_ms)
            if st != "ok":
                loser = side
                result = "0-1" if side == 0 else "1-0"
                reason = ("time forfeit (no bestmove within %d ms)" % (movetime_ms + grace_ms)) if st == "timeout" else "engine crashed or closed its output"
                break
            if ms > movetime_ms + grace_ms:
                loser, result, reason = side, ("0-1" if side == 0 else "1-0"), "time forfeit (%d ms for movetime %d)" % (ms, movetime_ms)
                break
            m = pos.parse_uci(bm)
            if m is None:
                loser = side
                result = "0-1" if side == 0 else "1-0"
                reason = "illegal move %s" % (bm or "(empty)")
                rec.append({"uci": bm, "san": "illegal", "illegal": True, "ms": ms, **info})
                break
            sc = info.get("score")
            if sc and side == 1:
                sc = (sc[0], -sc[1])  # white's point of view
            rec.append({"uci": bm, "san": pos.san(m), "ms": ms, "nodes": info.get("nodes"), "depth": info.get("depth"), "score": sc})
            pos.play(m)
            moves.append(bm)
    except StopIteration:
        pass
    finally:
        for e in ens:
            e.close()
    return {"white": white["name"], "black": black["name"], "opening": opening, "result": result, "reason": reason, "loser": loser,
            "plies": len(moves), "moves": rec, "final_fen": pos.fen()}


def pgn_text(g, event, round_no, tc, date=None):
    date = date or datetime.date.today().strftime("%Y.%m.%d")
    term = g["reason"]
    h = ['[Event "%s"]' % event, '[Site "capsule"]', '[Date "%s"]' % date, '[Round "%s"]' % round_no, '[White "%s"]' % g["white"], '[Black "%s"]' % g["black"],
         '[Result "%s"]' % g["result"], '[Termination "%s"]' % term, '[TimeControl "%s"]' % tc, '[PlyCount "%d"]' % g["plies"]]
    toks = []
    for i, m in enumerate(g["moves"]):
        if i % 2 == 0:
            toks.append("%d." % (i // 2 + 1))
        t = m["san"]
        c = ""
        if not m.get("book") and m.get("ms") is not None:
            sc = m.get("score")
            s = ("%+.2f" % (sc[1] / 100.0) if sc[0] == "cp" else "#%d" % sc[1]) if sc else "?"
            c = " {%s d%s n=%s %dms}" % (s, m.get("depth", "?"), m.get("nodes", "?"), m["ms"])
        toks.append(t + c)
    toks.append(g["result"])
    lines, cur = [], ""
    for t in toks:
        if len(cur) + len(t) + 1 > 90:
            lines.append(cur)
            cur = t
        else:
            cur = (cur + " " + t).strip()
    lines.append(cur)
    return "\n".join(h) + "\n\n" + "\n".join(lines) + "\n\n"


def run_match(a, b, n_games, movetime_ms, openings=OPENINGS, concurrency=4, grace_ms=1500, max_plies=240, errdir=None, event="match", on_game=None):
    """a plays white in even games, black in odd games; opening = openings[(i // 2) % len]. returns dict with games + tally for a."""
    def one(i):
        o = openings[(i // 2) % len(openings)]
        w, bl = (a, b) if i % 2 == 0 else (b, a)
        g = play_game(w, bl, o, movetime_ms, grace_ms, max_plies, errdir, tag="g%02d" % (i + 1))
        g["round"] = i + 1
        g["a_is_white"] = (i % 2 == 0)
        if on_game:
            on_game(g)
        return g
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        games = list(ex.map(one, range(n_games)))
    tally = {"a_win": 0, "draw": 0, "b_win": 0, "a_illegal": 0, "b_illegal": 0, "a_time_or_crash": 0, "b_time_or_crash": 0}
    for g in games:
        r = g["result"]
        awhite = g["a_is_white"]
        if r == "1/2-1/2":
            tally["draw"] += 1
        elif (r == "1-0") == awhite:
            tally["a_win"] += 1
        else:
            tally["b_win"] += 1
        if g["loser"] is not None:
            loser_is_a = (g["loser"] == 0) == awhite
            key = "a" if loser_is_a else "b"
            if (g["reason"] or "").startswith("illegal"):
                tally[key + "_illegal"] += 1
            elif (g["reason"] or "").startswith(("time", "engine crashed", "handshake")):
                tally[key + "_time_or_crash"] += 1
    return {"a": a["name"], "b": b["name"], "movetime_ms": movetime_ms, "games": games, "tally": tally, "event": event}


def write_pgn(res, path, tc=None):
    tc = tc or "movetime %d ms per move" % res["movetime_ms"]
    with open(path, "w") as f:
        for g in res["games"]:
            f.write(pgn_text(g, res["event"], g["round"], tc))
