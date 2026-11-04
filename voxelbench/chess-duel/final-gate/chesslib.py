"""chesslib.py: the referee's own chess rules (stdlib only, no python-chess).

Used by the final gate and the match referee (capsule ~/.vxfinal/chess/, never inside the agent's tree). Verified against the frozen perft
numbers in perft_ref.json and, at staging, against python-chess on a CPU-only throwaway venv (tests/test_chesslib_vs_pychess.py).

Board: 64 ints, a1 = 0 .. h8 = 63. Piece = type | (8 if black): 1 P, 2 N, 3 B, 4 R, 5 Q, 6 K. Move = (from, to, promo) with promo 0 or 2..5.
"""
import re

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6
_CH = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
_SYM = ".PNBRQK"


def sq_name(s):
    return "abcdefgh"[s & 7] + str((s >> 3) + 1)


def sq_from(name):
    return (int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])


def _build_tables():
    kn, kg = [[] for _ in range(64)], [[] for _ in range(64)]
    rays = {d: [[] for _ in range(64)] for d in (1, -1, 8, -8, 9, -9, 7, -7)}
    dirs = {1: (1, 0), -1: (-1, 0), 8: (0, 1), -8: (0, -1), 9: (1, 1), -9: (-1, -1), 7: (-1, 1), -7: (1, -1)}
    for s in range(64):
        f, r = s & 7, s >> 3
        for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            if 0 <= f + df < 8 and 0 <= r + dr < 8:
                kn[s].append((r + dr) * 8 + f + df)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if (df or dr) and 0 <= f + df < 8 and 0 <= r + dr < 8:
                    kg[s].append((r + dr) * 8 + f + df)
        for d, (df, dr) in dirs.items():
            ff, rr = f + df, r + dr
            while 0 <= ff < 8 and 0 <= rr < 8:
                rays[d][s].append(rr * 8 + ff)
                ff += df
                rr += dr
    return kn, kg, rays


KN, KG, RAYS = _build_tables()
ROOK_DIRS = (1, -1, 8, -8)
BISHOP_DIRS = (9, -9, 7, -7)
# castling rights lost when a move touches this square
CASTLE_MASK = [15] * 64
CASTLE_MASK[0], CASTLE_MASK[4], CASTLE_MASK[7] = 15 & ~2, 15 & ~3, 15 & ~1
CASTLE_MASK[56], CASTLE_MASK[60], CASTLE_MASK[63] = 15 & ~8, 15 & ~12, 15 & ~4


class Position:
    def __init__(self, fen=START_FEN):
        self.stack = []
        self.set_fen(fen)

    # ---- fen ----
    def set_fen(self, fen):
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError("bad fen: " + fen)
        self.board = [0] * 64
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError("bad fen board")
        for i, row in enumerate(rows):
            r, f = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    p = _CH[ch.upper()] | (8 if ch.islower() else 0)
                    self.board[r * 8 + f] = p
                    f += 1
        self.turn = 0 if parts[1] == "w" else 1
        self.castling = 0
        for ch in parts[2]:
            self.castling |= {"K": 1, "Q": 2, "k": 4, "q": 8}.get(ch, 0)
        self.ep = -1 if parts[3] == "-" else sq_from(parts[3])
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1
        self.stack = []
        self.hist = [self.rep_key()]

    def fen(self):
        rows = []
        for r in range(7, -1, -1):
            row, e = "", 0
            for f in range(8):
                p = self.board[r * 8 + f]
                if p == 0:
                    e += 1
                else:
                    if e:
                        row += str(e)
                        e = 0
                    c = _SYM[p & 7]
                    row += c.lower() if p & 8 else c
            if e:
                row += str(e)
            rows.append(row)
        cs = "".join(c for b, c in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if self.castling & b) or "-"
        ep = sq_name(self.ep) if self.ep >= 0 else "-"
        return "%s %s %s %s %d %d" % ("/".join(rows), "w" if self.turn == 0 else "b", cs, ep, self.halfmove, self.fullmove)

    def rep_key(self):
        # ep square counts only when an ep capture is actually legal (FIDE repetition rule)
        ep = -1
        if self.ep >= 0:
            for m in self.legal_moves():
                if m[1] == self.ep and (self.board[m[0]] & 7) == P:
                    ep = self.ep
                    break
        return (tuple(self.board), self.turn, self.castling, ep)

    # ---- attacks ----
    def attacked(self, s, by):
        b = self.board
        side = 8 if by else 0
        for t in KN[s]:
            if b[t] == N | side:
                return True
        for t in KG[s]:
            if b[t] == K | side:
                return True
        # pawns: a white pawn attacks s from s-7 / s-9 (south), black pawn from s+7 / s+9
        f = s & 7
        if by == 0:
            if f > 0 and s - 9 >= 0 and b[s - 9] == P:
                return True
            if f < 7 and s - 7 >= 0 and b[s - 7] == P:
                return True
        else:
            if f < 7 and s + 9 < 64 and b[s + 9] == P | 8:
                return True
            if f > 0 and s + 7 < 64 and b[s + 7] == P | 8:
                return True
        for d in ROOK_DIRS:
            for t in RAYS[d][s]:
                p = b[t]
                if p:
                    if p == (R | side) or p == (Q | side):
                        return True
                    break
        for d in BISHOP_DIRS:
            for t in RAYS[d][s]:
                p = b[t]
                if p:
                    if p == (B | side) or p == (Q | side):
                        return True
                    break
        return False

    def king_sq(self, color):
        kp = K | (8 if color else 0)
        return self.board.index(kp)

    def in_check(self):
        return self.attacked(self.king_sq(self.turn), self.turn ^ 1)

    # ---- generation ----
    def pseudo_moves(self):
        b, us = self.board, self.turn
        side = 8 if us else 0
        out = []
        for s in range(64):
            p = b[s]
            if not p or (p & 8) != side:
                continue
            t = p & 7
            if t == P:
                fwd = 8 if us == 0 else -8
                r = s >> 3
                promo_rank = 6 if us == 0 else 1
                start_rank = 1 if us == 0 else 6
                to = s + fwd
                if 0 <= to < 64 and b[to] == 0:
                    if r == promo_rank:
                        for pr in (5, 4, 3, 2):
                            out.append((s, to, pr))
                    else:
                        out.append((s, to, 0))
                        if r == start_rank and b[to + fwd] == 0:
                            out.append((s, to + fwd, 0))
                f = s & 7
                for df in (-1, 1):
                    if 0 <= f + df < 8:
                        to = s + fwd + df
                        if 0 <= to < 64:
                            tp = b[to]
                            if tp and (tp & 8) != side:
                                if r == promo_rank:
                                    for pr in (5, 4, 3, 2):
                                        out.append((s, to, pr))
                                else:
                                    out.append((s, to, 0))
                            elif to == self.ep and tp == 0:
                                out.append((s, to, 0))
            elif t == N:
                for to in KN[s]:
                    tp = b[to]
                    if tp == 0 or (tp & 8) != side:
                        out.append((s, to, 0))
            elif t == K:
                for to in KG[s]:
                    tp = b[to]
                    if tp == 0 or (tp & 8) != side:
                        out.append((s, to, 0))
                # castling: squares empty, king not in check, passing square not attacked (landing square is checked by legality)
                if us == 0 and s == 4:
                    if self.castling & 1 and b[5] == 0 and b[6] == 0 and b[7] == R and not self.attacked(4, 1) and not self.attacked(5, 1):
                        out.append((4, 6, 0))
                    if self.castling & 2 and b[3] == 0 and b[2] == 0 and b[1] == 0 and b[0] == R and not self.attacked(4, 1) and not self.attacked(3, 1):
                        out.append((4, 2, 0))
                elif us == 1 and s == 60:
                    if self.castling & 4 and b[61] == 0 and b[62] == 0 and b[63] == R | 8 and not self.attacked(60, 0) and not self.attacked(61, 0):
                        out.append((60, 62, 0))
                    if self.castling & 8 and b[59] == 0 and b[58] == 0 and b[57] == 0 and b[56] == R | 8 and not self.attacked(60, 0) and not self.attacked(59, 0):
                        out.append((60, 58, 0))
            else:
                dirs = BISHOP_DIRS if t == B else ROOK_DIRS if t == R else ROOK_DIRS + BISHOP_DIRS
                for d in dirs:
                    for to in RAYS[d][s]:
                        tp = b[to]
                        if tp == 0:
                            out.append((s, to, 0))
                        else:
                            if (tp & 8) != side:
                                out.append((s, to, 0))
                            break
        return out

    def push(self, m):
        b = self.board
        fr, to, pr = m
        p = b[fr]
        cap = b[to]
        undo = (m, p, cap, self.castling, self.ep, self.halfmove, self.fullmove, None)
        ep_cap = None
        t = p & 7
        if t == P and to == self.ep and cap == 0 and (fr & 7) != (to & 7):
            ep_cap = to - 8 if self.turn == 0 else to + 8
            undo = undo[:7] + ((ep_cap, b[ep_cap]),)
            b[ep_cap] = 0
        b[to] = (pr | (p & 8)) if pr else p
        b[fr] = 0
        if t == K and abs(to - fr) == 2:
            if to > fr:
                b[to - 1], b[to + 1] = b[to + 1], 0
            else:
                b[to + 1], b[to - 2] = b[to - 2], 0
        self.castling &= CASTLE_MASK[fr] & CASTLE_MASK[to]
        self.ep = (fr + to) // 2 if t == P and abs(to - fr) == 16 else -1
        self.halfmove = 0 if (t == P or cap or ep_cap is not None) else self.halfmove + 1
        if self.turn == 1:
            self.fullmove += 1
        self.turn ^= 1
        self.stack.append(undo)

    def pop(self):
        m, p, cap, castling, ep, half, full, epc = self.stack.pop()
        fr, to, pr = m
        b = self.board
        self.turn ^= 1
        b[fr] = p
        b[to] = cap
        if epc:
            b[epc[0]] = epc[1]
        if (p & 7) == K and abs(to - fr) == 2:
            if to > fr:
                b[to + 1], b[to - 1] = b[to - 1], 0
            else:
                b[to - 2], b[to + 1] = b[to + 1], 0
        self.castling, self.ep, self.halfmove, self.fullmove = castling, ep, half, full

    def legal_moves(self):
        us = self.turn
        out = []
        for m in self.pseudo_moves():
            self.push(m)
            if not self.attacked(self.king_sq(us), us ^ 1):
                out.append(m)
            self.pop()
        return out

    # ---- uci strings ----
    @staticmethod
    def uci(m):
        return sq_name(m[0]) + sq_name(m[1]) + ({2: "n", 3: "b", 4: "r", 5: "q"}[m[2]] if m[2] else "")

    def parse_uci(self, s):
        """return the legal move matching the uci string, or None"""
        if not re.fullmatch(r"[a-h][1-8][a-h][1-8][nbrq]?", s or ""):
            return None
        pr = {"": 0, "n": 2, "b": 3, "r": 4, "q": 5}[s[4:]]
        want = (sq_from(s[0:2]), sq_from(s[2:4]), pr)
        for m in self.legal_moves():
            if m == want:
                return m
        return None

    # ---- game state ----
    def play(self, m):
        self.push(m)
        self.hist.append(self.rep_key())

    def insufficient(self):
        pcs = [p for p in self.board if p and (p & 7) != K]
        if not pcs:
            return True
        if all((p & 7) in (N, B) for p in pcs):
            if len(pcs) == 1:
                return True
            if all((p & 7) == B for p in pcs):
                sqs = [s for s in range(64) if self.board[s] and (self.board[s] & 7) == B]
                return len({((s & 7) + (s >> 3)) % 2 for s in sqs}) == 1
        return False

    def outcome(self):
        """None while the game goes on, else (result, reason): result is '1-0', '0-1' or '1/2-1/2'"""
        if not self.legal_moves():
            if self.in_check():
                return ("0-1" if self.turn == 0 else "1-0", "checkmate")
            return ("1/2-1/2", "stalemate")
        if self.insufficient():
            return ("1/2-1/2", "insufficient material")
        if self.halfmove >= 100:
            return ("1/2-1/2", "fifty-move rule")
        if self.hist.count(self.hist[-1]) >= 3:
            return ("1/2-1/2", "threefold repetition")
        return None

    # ---- san ----
    def san(self, m):
        fr, to, pr = m
        b = self.board
        p = b[fr]
        t = p & 7
        if t == K and abs(to - fr) == 2:
            s = "O-O" if to > fr else "O-O-O"
        else:
            cap = b[to] != 0 or (t == P and (fr & 7) != (to & 7))
            if t == P:
                s = (("abcdefgh"[fr & 7] + "x") if cap else "") + sq_name(to)
                if pr:
                    s += "=" + _SYM[pr]
            else:
                s = _SYM[t]
                others = [x for x in self.legal_moves() if x[1] == to and x[0] != fr and (b[x[0]] & 7) == t]
                if others:
                    if not any((x[0] & 7) == (fr & 7) for x in others):
                        s += "abcdefgh"[fr & 7]
                    elif not any((x[0] >> 3) == (fr >> 3) for x in others):
                        s += str((fr >> 3) + 1)
                    else:
                        s += sq_name(fr)
                if cap:
                    s += "x"
                s += sq_name(to)
        self.push(m)
        if self.in_check():
            s += "#" if not self.legal_moves() else "+"
        self.pop()
        return s


def perft(pos, depth):
    if depth == 0:
        return 1
    ms = pos.legal_moves()
    if depth == 1:
        return len(ms)
    n = 0
    for m in ms:
        pos.push(m)
        n += perft(pos, depth - 1)
        pos.pop()
    return n
