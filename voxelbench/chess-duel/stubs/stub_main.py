# ---- stub UCI engine (known-good reference stub for the harness PASS path; inlined after chesslib by build_stubs.py) ----
import sys
import time


def uci_move(m):
    return Position.uci(m)


PV = {0: 0, 1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}
CENTRE = [0, 0, 0, 0, 0, 0, 0, 0]


def evaluate(pos):
    score = 0
    b = pos.board
    for s in range(64):
        p = b[s]
        if p:
            t = p & 7
            v = PV[t]
            f, r = s & 7, s >> 3
            c = min(f, 7 - f) + min(r, 7 - r)  # 0 edge .. 6 centre
            if t in (1, 2, 3):
                v += 4 * c
            if t == 1:
                v += 6 * (r if p & 8 == 0 else 7 - r)
            score += -v if p & 8 else v
    return score if pos.turn == 0 else -score


class Searcher:
    def __init__(self, pos, deadline):
        self.pos, self.deadline, self.nodes, self.stop = pos, deadline, 0, False

    def order(self, moves):
        b = self.pos.board

        def key(m):
            cap = b[m[1]]
            return -(10 * PV[cap & 7] - PV[b[m[0]] & 7] // 10 + (800 if m[2] == 5 else 0)) if cap or m[2] else 0
        return sorted(moves, key=key)

    def negamax(self, depth, alpha, beta, ply):
        self.nodes += 1
        if self.nodes & 255 == 0 and time.time() > self.deadline:
            self.stop = True
        if self.stop:
            return 0
        pos = self.pos
        moves = pos.legal_moves()
        if not moves:
            return -30000 + ply if pos.in_check() else 0
        if depth <= 0:
            return evaluate(pos)
        best = -31000
        for m in self.order(moves):
            pos.push(m)
            sc = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            pos.pop()
            if self.stop:
                return 0
            if sc > best:
                best = sc
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best


def search(pos, movetime_ms, max_depth=64):
    t0 = time.time()
    deadline = t0 + movetime_ms / 1000.0 * 0.85
    moves = pos.legal_moves()
    if not moves:
        return None
    best_move = moves[0]
    s = Searcher(pos, deadline)
    for depth in range(1, max_depth + 1):
        best_score, cur = -31000, None
        for m in s.order(moves):
            pos.push(m)
            sc = -s.negamax(depth - 1, -31000, 31000 if cur is None else -best_score, 1)
            pos.pop()
            if s.stop:
                break
            if cur is None or sc > best_score:
                best_score, cur = sc, m
        if s.stop or cur is None:
            break
        best_move = cur
        ms = int((time.time() - t0) * 1000)
        print("info depth %d score cp %d nodes %d time %d nps %d pv %s" % (depth, best_score, s.nodes, ms, s.nodes * 1000 // max(ms, 1), uci_move(cur)), flush=True)
        if abs(best_score) > 29000:
            break
    return best_move


def perft_divide(pos, depth):
    total = 0
    for m in pos.legal_moves():
        pos.push(m)
        n = perft(pos, depth - 1)
        pos.pop()
        print("%s: %d" % (uci_move(m), n), flush=True)
        total += n
    print("\nNodes searched: %d" % total, flush=True)


def main():
    pos = Position()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        a = line.split()
        if not a:
            continue
        c = a[0]
        if c == "uci":
            print("id name stub-good\nid author harness\nuciok", flush=True)
        elif c == "isready":
            print("readyok", flush=True)
        elif c == "ucinewgame":
            pos = Position()
        elif c == "position":
            i = 1
            if a[1] == "startpos":
                pos = Position()
                i = 2
            elif a[1] == "fen":
                j = a.index("moves") if "moves" in a else len(a)
                pos = Position(" ".join(a[2:j]))
                i = j
            if i < len(a) and a[i] == "moves":
                for mv in a[i + 1:]:
                    m = pos.parse_uci(mv)
                    if m is None:
                        break
                    pos.play(m)
        elif c == "go":
            if "perft" in a:
                perft_divide(pos, int(a[a.index("perft") + 1]))
                continue
            mt, depth = 1000, 64
            if "movetime" in a:
                mt = int(a[a.index("movetime") + 1])
            elif "depth" in a:
                depth, mt = int(a[a.index("depth") + 1]), 10 ** 9
            elif "wtime" in a:
                key = "wtime" if pos.turn == 0 else "btime"
                mt = max(50, int(a[a.index(key) + 1]) // 30)
            bm = search(pos, mt, depth)
            print("bestmove %s" % (uci_move(bm) if bm else "0000"), flush=True)
        elif c == "quit":
            break


main()
