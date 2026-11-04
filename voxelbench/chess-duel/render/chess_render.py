#!/usr/bin/env python3
"""chess_render.py: the chess-duel clip, rendered ON CAPSULE (PIL + pyte + the bundled ffmpeg), fixed layout, nothing from the screen is recorded raw.

  replay     <match.json> <out.mp4> --name-a A --name-b B [--game N] [--sub TEXT] [--no-title]
             title card, ONE featured game move by move (board, per-side think-time clock, nodes, depth, eval bar, move list), then a montage of every
             other game's final position with the running score, then the scoreboard card if --scoreboard-json is given
  scoreboard <summary.json> <out.png>      the final scoreboard still (gate verdicts, match score, anchor, agent stats)
  cutaway    <frames-a.jsonl> <frames-b.jsonl> <out.mp4> --name-a A --name-b B [--speed 8] [--windows 3 --window-s 8] [--full]
             the two agents' Hermes panes (pane_rec.py snapshots, pyte-rendered) side by side at 8x; --full renders the whole hour at 8x

LEAK RULES (checked with contact sheets before any clip leaves capsule): every pane line goes through mask() (account name, host name, any absolute path,
tailnet addresses) and the Hermes status bar row (it shows the working directory and a session clock) is replaced by a fixed bar. Fonts: DejaVu (system).
"""
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.dirname(HERE)):
    sys.path.insert(0, d)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import chesslib as cl  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", os.path.expanduser("~/voxelbench/ffmpeg"))
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
W, H = 1920, 1080
FPS = 25
BG, PANEL, INK, DIM, AMBER, GREEN, RED, LIGHT_SQ, DARK_SQ = (14, 17, 22), (24, 29, 38), (236, 238, 242), (168, 176, 190), (245, 183, 66), (96, 200, 132), (232, 96, 96), (204, 195, 168), (104, 132, 96)


def font(name, size):
    return ImageFont.truetype(FONT_DIR + name, size)


F = {}


def fonts():
    if not F:
        F.update(title=font("DejaVuSans-Bold.ttf", 46), h2=font("DejaVuSans-Bold.ttf", 30), body=font("DejaVuSans.ttf", 24), small=font("DejaVuSans.ttf", 19),
                 mono=font("DejaVuSansMono.ttf", 24), monob=font("DejaVuSansMono-Bold.ttf", 28), big=font("DejaVuSans-Bold.ttf", 96), piece=font("DejaVuSans.ttf", 86),
                 coord=font("DejaVuSans.ttf", 20))
    return F


# ---------------------------------------------------------------- masking
def mask(line, user=os.environ.get("USER", "witcheer")):
    line = re.sub(r"/(?:home|opt|usr|root|var|etc|mnt|srv)/[^\s'\"`)\]]*", lambda m: "<path>/" + os.path.basename(m.group(0).rstrip("/")) if "." in os.path.basename(m.group(0)) else "<path>", line)
    line = re.sub(r"/tmp/[^\s'\"`)\]]*", lambda m: "<arena>/" + os.path.basename(m.group(0).rstrip("/")) if "." in os.path.basename(m.group(0)) else "<arena>", line)
    line = re.sub(r"~/[^\s'\"`)\]]*", "~/<path>", line)
    for pat, rep in ((r"Session: [0-9a-f]{6,}", "Session: ········"), (r"witcheer", "user"), (re.escape(user), "user"), (r"\bcapsule(?:-w)?\b", "host"), (r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "x.x.x.x"), (r"[\w.-]+\.ts\.net", "host"),
                     (r"\bhermes-magic\b", "vault"), (r"\bmercury\b", "host")):
        line = re.sub(pat, rep, line, flags=re.I)
    return line


# ---------------------------------------------------------------- ffmpeg pipe
class Pipe:
    def __init__(self, out, crf=21):
        self.p = subprocess.Popen([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.n = 0
        self.last = None

    def put(self, img, hold=1):
        raw = img.tobytes()
        for _ in range(hold):
            self.p.stdin.write(raw)
            self.n += 1

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode()
        self.p.wait()
        if self.p.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + err[-400:])
        return self.n / FPS


# ---------------------------------------------------------------- chess board
GLYPH = {1: "\u265f", 2: "\u265e", 3: "\u265d", 4: "\u265c", 5: "\u265b", 6: "\u265a"}
SQ = 100
BX, BY = 96, 190


def draw_board(d, board, last=None, flip=False):
    f = fonts()
    for s in range(64):
        fl, r = s & 7, s >> 3
        x = BX + fl * SQ
        y = BY + (7 - r) * SQ
        col = DARK_SQ if (fl + r) % 2 == 0 else LIGHT_SQ
        if last and s in last:
            col = (min(255, col[0] + 22), min(255, col[1] + 34), max(0, col[2] - 40))
        d.rectangle([x, y, x + SQ, y + SQ], fill=col)
    for s in range(64):
        p = board[s]
        if not p:
            continue
        fl, r = s & 7, s >> 3
        x = BX + fl * SQ + SQ // 2
        y = BY + (7 - r) * SQ + SQ // 2
        white = (p & 8) == 0
        fill, stroke = ((250, 246, 232), (30, 30, 34)) if white else ((38, 40, 48), (232, 232, 236))
        d.text((x, y + 4), GLYPH[p & 7], font=f["piece"], fill=fill, anchor="mm", stroke_width=2, stroke_fill=stroke)
    for i in range(8):
        d.text((BX + i * SQ + SQ // 2, BY + 8 * SQ + 8), "abcdefgh"[i], font=f["coord"], fill=DIM, anchor="mt")
        d.text((BX - 8, BY + (7 - i) * SQ + SQ // 2), str(i + 1), font=f["coord"], fill=DIM, anchor="rm")


def evalbar(d, score_cp):
    """vertical bar left of the board, white at the bottom; score from white's point of view, clamp +-6 pawns"""
    x0, x1 = BX - 52, BX - 28
    d.rectangle([x0, BY, x1, BY + 8 * SQ], fill=(38, 40, 48))
    if score_cp is None:
        return
    v = max(-600, min(600, score_cp))
    h = int((v + 600) / 1200 * 8 * SQ)
    d.rectangle([x0, BY + 8 * SQ - h, x1, BY + 8 * SQ], fill=(246, 240, 224))
    d.line([x0, BY + 4 * SQ, x1, BY + 4 * SQ], fill=AMBER, width=2)


def fmt_score(sc):
    if not sc:
        return "-"
    return ("%+.2f" % (sc[1] / 100.0)) if sc[0] == "cp" else "#%d" % sc[1]


def cp_of(sc):
    if not sc:
        return None
    return sc[1] if sc[0] == "cp" else (30000 if sc[1] > 0 else -30000)


def fmt_ms(ms):
    return "%d.%02d" % (ms // 1000, (ms % 1000) // 10)


def short(n, k=24):
    return n if len(n) <= k else n[:k - 1] + "\u2026"


class GameView:
    """precomputes the board after every ply and the per-side running stats of one game"""

    def __init__(self, g):
        self.g = g
        pos = cl.Position()
        self.boards, self.lasts = [list(pos.board)], [None]
        for m in g["moves"]:
            mv = pos.parse_uci(m["uci"]) if not m.get("illegal") else None
            if mv is None:
                self.boards.append(list(pos.board))
                self.lasts.append(None)
                continue
            pos.play(mv)
            self.boards.append(list(pos.board))
            self.lasts.append((mv[0], mv[1]))
        self.n = len(g["moves"])

    def stats(self, upto):
        """per side: total ms, last ms, last nodes, total nodes, last depth, last score (white pov)"""
        out = {0: dict(total=0, last=0, nodes=0, tot_nodes=0, depth=None, nps=0), 1: dict(total=0, last=0, nodes=0, tot_nodes=0, depth=None, nps=0)}
        score = None
        for i, m in enumerate(self.g["moves"][:upto]):
            side = i % 2
            if m.get("book"):
                continue
            s = out[side]
            s["last"] = m.get("ms") or 0
            s["total"] += s["last"]
            s["nodes"] = m.get("nodes") or 0
            s["tot_nodes"] += s["nodes"]
            s["depth"] = m.get("depth")
            s["nps"] = int(s["nodes"] * 1000 / max(1, s["last"])) if s["nodes"] else 0
            if m.get("score"):
                score = m["score"]
        return out, score


def card(d, x, y, w, h, name, side_white, st, active, frac_ms, is_book_turn=False):
    f = fonts()
    d.rounded_rectangle([x, y, x + w, y + h], 14, fill=PANEL, outline=AMBER if active else (46, 52, 64), width=3 if active else 1)
    chip = (250, 246, 232) if side_white else (38, 40, 48)
    d.ellipse([x + 20, y + 22, x + 46, y + 48], fill=chip, outline=(200, 200, 206), width=2)
    d.text((x + 62, y + 34), short(name, 26), font=f["h2"], fill=INK, anchor="lm")
    shown = frac_ms if active else st["last"]
    d.text((x + w - 24, y + 34), fmt_ms(int(shown)) + " s", font=f["monob"], fill=AMBER if active else INK, anchor="rm")
    rows = [("total think time", fmt_ms(st["total"]) + " s"), ("nodes, last move", "{:,}".format(st["nodes"]) if st["nodes"] else "-"),
            ("depth, last move", str(st["depth"]) if st["depth"] else "-"), ("nodes per second", "{:,}".format(st["nps"]) if st["nps"] else "-"), ("nodes, whole game", "{:,}".format(st["tot_nodes"]))]
    for i, (a, b) in enumerate(rows):
        yy = y + 78 + i * 30
        d.text((x + 24, yy), a, font=f["small"], fill=DIM, anchor="lm")
        d.text((x + w - 24, yy), b, font=f["mono"], fill=INK, anchor="rm")


def movelist(d, x, y, w, h, g, upto):
    f = fonts()
    d.rounded_rectangle([x, y, x + w, y + h], 14, fill=PANEL)
    moves = g["moves"][:upto]
    rows = []
    for i in range(0, len(moves), 2):
        a = moves[i]["san"]
        b = moves[i + 1]["san"] if i + 1 < len(moves) else ""
        rows.append("%3d. %-9s %s" % (i // 2 + 1, a, b))
    show = rows[-6:]
    if len(rows) > 6:
        show[0] = "  ..."
    for i, r in enumerate(show):
        d.text((x + 24, y + 22 + i * 36), r, font=f["mono"], fill=INK if i == len(show) - 1 else DIM)


def fit_font(text, maxw, name="DejaVuSans-Bold.ttf", start=46):
    sz = start
    while sz > 20:
        ft = font(name, sz)
        if ft.getlength(text) <= maxw:
            return ft
        sz -= 2
    return font(name, 20)


def header(d, title, sub):
    f = fonts()
    d.text((96, 54), title, font=fit_font(title, W - 192), fill=INK, anchor="lm")
    d.text((96, 118), sub, font=f["body"], fill=DIM, anchor="lm")


def chips(d, x, y, results, a_name, k=None):
    """one chip per game: engine A win green, B win red-ish, draw grey; results = list of 'a'|'b'|'d'|None"""
    f = fonts()
    for i, r in enumerate(results):
        cx = x + (i % 10) * 54
        cy = y + (i // 10) * 54
        col = {"a": GREEN, "b": AMBER, "d": (120, 128, 142), None: (44, 50, 62)}[r]
        d.rounded_rectangle([cx, cy, cx + 44, cy + 44], 8, fill=col)
        d.text((cx + 22, cy + 22), str(i + 1), font=f["small"], fill=(10, 12, 16), anchor="mm")


def game_result_key(g):
    r = g["result"]
    if r == "1/2-1/2":
        return "d"
    return "a" if ((r == "1-0") == g["a_is_white"]) else "b"


def tally_pts(results):
    a = sum(1 for r in results if r == "a") + 0.5 * sum(1 for r in results if r == "d")
    n = sum(1 for r in results if r is not None)
    return a, n - a


def base_frame(title, sub):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    header(d, title, sub)
    return img, d


def render_game_frame(gv, ply, frac, name_a, name_b, results, title, sub, banner=None, mt_ms=1000):
    f = fonts()
    g = gv.g
    img, d = base_frame(title, sub)
    st, score = gv.stats(ply)
    draw_board(d, gv.boards[ply], gv.lasts[ply])
    evalbar(d, cp_of(score))
    nxt = g["moves"][ply] if ply < gv.n else None
    side_to_move = ply % 2
    frac_ms = (nxt.get("ms") or 0) * frac if nxt and not nxt.get("book") else 0
    wname, bname = g["white"], g["black"]
    px, pw = 1010, 814
    d.text((px, 172), "Game %d of %d" % (g.get("round", 1), g.get("of", 20)), font=f["h2"], fill=AMBER, anchor="lm")
    d.text((px + pw, 172), "eval %s (white's view)" % fmt_score(score), font=f["mono"], fill=INK, anchor="rm")
    card(d, px, 206, pw, 224, bname, False, st[1], side_to_move == 1 and nxt is not None, frac_ms)
    card(d, px, 450, pw, 224, wname, True, st[0], side_to_move == 0 and nxt is not None, frac_ms)
    movelist(d, px, 694, pw, 250, g, ply)
    a, b = tally_pts(results)
    shown = sum(1 for r in results if r is not None)
    lab = "featured game: the longest decisive one. recap of all games follows" if shown == 0 else "recap %d of %d  (green %s, amber %s, grey draw)" % (shown, len(results), short(name_a, 9), short(name_b, 9))
    d.text((px, 966), lab, font=f["small"], fill=DIM, anchor="lm")
    d.text((px, 996), "%s  %.1f - %.1f  %s" % (short(name_a, 20), a, b, short(name_b, 20)), font=f["h2"], fill=INK, anchor="lm")
    cur = g.get("round", 1) - 1
    for i, r in enumerate(results):
        col = {"a": GREEN, "b": AMBER, "d": (120, 128, 142), None: (44, 50, 62)}[r]
        x0 = px + i * 40
        d.rounded_rectangle([x0, 1022, x0 + 34, 1046], 5, fill=col, outline=INK if i == cur else None, width=3)
    d.text((96, 1036), "%g s per move, fixed. illegal move, late move or crash = loss." % (mt_ms / 1000.0), font=f["small"], fill=DIM, anchor="lm")
    d.text((96, 1058), "referee: its own rules engine, checked against published perft counts.", font=f["small"], fill=DIM, anchor="lm")
    if banner:
        d.rounded_rectangle([px, 694, px + pw, 944], 14, fill=(10, 12, 16), outline=AMBER, width=4)
        d.text((px + pw // 2, 780), banner[0], font=f["h2"], fill=AMBER, anchor="mm")
        d.text((px + pw // 2, 840), banner[1], font=f["body"], fill=INK, anchor="mm")
    return img


def montage_frame(gv, name_a, name_b, results, title, sub, banner):
    return render_game_frame(gv, gv.n, 0, name_a, name_b, results, title, sub, banner)


def title_card(title, lines):
    f = fonts()
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 360), title, font=f["big"] if len(title) < 22 else f["title"], fill=INK, anchor="mm")
    for i, ln in enumerate(lines):
        d.text((W // 2, 520 + i * 54), ln, font=f["h2"] if i == 0 else f["body"], fill=AMBER if i == 0 else DIM, anchor="mm")
    return img


def scoreboard_img(S):
    """S: summary dict (see cpu_phase.sh): names, gate{a,b}, match{...}, anchor{a,b}, agent{a,b}, code{a,b}"""
    f = fonts()
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((96, 70), "Two chess engines. Two local models. One Hermes hour each.", font=fit_font("Two chess engines. Two local models. One Hermes hour each.", W - 192), fill=INK, anchor="lm")
    d.text((96, 128), S.get("sub", ""), font=f["body"], fill=DIM, anchor="lm")
    xs = (96, 1010)
    for k, x in zip(("a", "b"), xs):
        e = S["engines"][k]
        d.rounded_rectangle([x, 180, x + 814, 760], 16, fill=PANEL)
        d.text((x + 30, 232), e["name"], font=f["h2"], fill=INK, anchor="lm")
        d.text((x + 30, 272), e.get("regime", ""), font=f["small"], fill=DIM, anchor="lm")
        ok = e["gate_pass"]
        d.rounded_rectangle([x + 30, 304, x + 784, 372], 12, fill=(26, 70, 46) if ok else (84, 30, 30))
        d.text((x + 50, 338), ("GATE PASS" if ok else "GATE FAIL") + "   " + short(e.get("gate_line", ""), 52), font=f["body"], fill=GREEN if ok else RED, anchor="lm")
        rows = e.get("rows", [])
        for i, (a, b) in enumerate(rows[:10]):
            yy = 410 + i * 34
            d.text((x + 30, yy), a, font=f["small"], fill=DIM, anchor="lm")
            d.text((x + 784, yy), str(b), font=f["mono"], fill=INK, anchor="rm")
    m = S["match"]
    d.rounded_rectangle([96, 790, 1824, 1010], 16, fill=PANEL)
    d.text((960, 852), m["headline"], font=f["h2"], fill=AMBER, anchor="mm")
    d.text((960, 930), m["score"], font=f["big"], fill=INK, anchor="mm")
    d.text((960, 986), m.get("note", ""), font=f["small"], fill=DIM, anchor="mm")
    words, lines, cur = S.get("footer", "").split(), [], ""
    for w_ in words:
        if f["small"].getlength((cur + " " + w_).strip()) > W - 192:
            lines.append(cur)
            cur = w_
        else:
            cur = (cur + " " + w_).strip()
    lines.append(cur)
    for i, ln in enumerate(lines[:2]):
        d.text((96, 1034 + i * 26), ln, font=f["small"], fill=DIM, anchor="lm")
    return img


# ---------------------------------------------------------------- replay
def cmd_replay(a):
    f = fonts()
    M = json.load(open(a["in"]))
    name_a, name_b = a["name-a"], a["name-b"]
    games = M["games"]
    for g in games:   # display names
        g["of"] = len(games)
        g["white"] = name_a if g["white"] == M["a"] else name_b
        g["black"] = name_a if g["black"] == M["a"] else name_b
    mt = int(M.get("movetime_ms", 1000))
    title = a.get("title", "Two chess engines, each written by a local model in one Hermes hour")
    sub = a.get("sub", "%s  vs  %s   ·   %d games   ·   %s s per move   ·   CPU only" % (name_a, name_b, len(games), "%g" % (mt / 1000.0)))
    pipe = Pipe(a["out"])
    if "no-title" not in a:
        pipe.put(title_card("%s  vs  %s" % (short(name_a, 20), short(name_b, 20)), ["each engine written by a local model in one Hermes Agent hour", "now they play: %d games, %g s per move, CPU only" % (len(games), mt / 1000.0)]), int(2.0 * FPS))
    results = [None] * len(games)
    fg = int(a.get("game", "1")) - 1
    gv = GameView(games[fg])
    # featured game timing: book plies fast, the first 20 engine plies at 0.5 s, then faster so the featured game is at most ~30 s
    book = sum(1 for m in gv.g["moves"] if m.get("book"))
    eng_plies = gv.n - book
    slow = min(20, eng_plies)
    fast_budget = max(2.0, 30.0 - slow * 0.5 - book * 0.15)
    fast_each = min(0.5, fast_budget / max(1, eng_plies - slow))
    for ply in range(gv.n):
        m = gv.g["moves"][ply]
        dur = 0.15 if m.get("book") else (0.5 if ply - book < slow else fast_each)
        nfr = max(1, int(dur * FPS))
        for k in range(nfr):
            pipe.put(render_game_frame(gv, ply, k / nfr, name_a, name_b, results, title, sub, None, mt))
    results[fg] = game_result_key(gv.g)
    last_banner = ("%s  %s" % (gv.g["result"], gv.g["reason"]), "%s (white) vs %s (black)" % (gv.g["white"], gv.g["black"]))
    pipe.put(render_game_frame(gv, gv.n, 0, name_a, name_b, results, title, sub, last_banner, mt), int(2.0 * FPS))
    # montage of the other games' final positions
    for i, g in enumerate(games):
        if i == fg:
            continue
        v = GameView(g)
        results[i] = game_result_key(g)
        who = {"a": name_a, "b": name_b, "d": None}[results[i]]
        bn = ("%s  %s" % (g["result"], g["reason"]), ("won by " + who) if who else "drawn")
        bn = (bn[0], "%s   %d plies" % (bn[1], g["plies"]))
        pipe.put(render_game_frame(v, v.n, 0, name_a, name_b, results, title, sub, bn, mt), int(0.7 * FPS))
    if "scoreboard-json" in a:
        S = json.load(open(a["scoreboard-json"]))
        pipe.put(scoreboard_img(S), int(5.0 * FPS))
    secs = pipe.close()
    print("replay %s: %.1f s" % (a["out"], secs))


# ---------------------------------------------------------------- hermes pane cutaway
def pane_images(frames_path, cols=110, rows=34):
    """yield (t, PIL image) per snapshot, rendered from the pyte screen after masking"""
    import pyte
    f = font("DejaVuSansMono.ttf", 13)
    fb = font("DejaVuSansMono-Bold.ttf", 13)
    cw, ch = 8, 21
    have = {}

    def has(c):
        if c not in have:
            try:
                have[c] = f.getmask(c).getbbox() is not None or c == " "
            except Exception:
                have[c] = False
        return have[c]
    from fontTools.ttLib import TTFont
    cmap = TTFont(FONT_DIR + "DejaVuSansMono.ttf").getBestCmap()
    palette = {"default": INK, "black": (60, 64, 72), "red": RED, "green": GREEN, "brown": AMBER, "blue": (110, 160, 240), "magenta": (200, 130, 230), "cyan": (110, 210, 220), "white": INK,
               "brightblack": DIM, "brightred": RED, "brightgreen": GREEN, "brightbrown": AMBER, "brightblue": (130, 180, 255), "brightmagenta": (220, 150, 250), "brightcyan": (130, 230, 240), "brightwhite": INK}

    def colour(name):
        if name in palette:
            return palette[name]
        if re.fullmatch(r"[0-9a-fA-F]{6}", name or ""):
            return tuple(int(name[i:i + 2], 16) for i in (0, 2, 4))
        return INK
    for line in open(frames_path):
        rec = json.loads(line)
        screen = pyte.Screen(cols, rows)
        stream = pyte.Stream(screen)
        stream.feed(rec["text"].replace("\n", "\r\n"))
        img = Image.new("RGB", (cols * cw + 16, rows * ch + 12), (10, 12, 16))
        d = ImageDraw.Draw(img)
        lines = ["".join(screen.buffer[y][x].data for x in range(cols)) for y in range(rows)]
        status_rows = {y for y, l in enumerate(lines) if "\u2624" in l and (" \u00b7 " in l or "ctx" in l or "pinned" in l)}
        for y in range(rows):
            raw = lines[y]
            if y in status_rows:
                d.rectangle([8, 6 + y * ch, 8 + cols * cw, 6 + y * ch + ch], fill=(24, 29, 38))
                d.text((10, 6 + y * ch), "\u2624 hermes agent session", font=f, fill=DIM)
                continue
            masked = mask(raw)
            if masked != raw:     # a masked line is re-drawn flat (colour runs no longer line up)
                for x, c in enumerate(masked[:cols]):
                    if c != " " and ord(c) in cmap:
                        d.text((8 + x * cw, 6 + y * ch), c, font=f, fill=INK)
                continue
            for x in range(cols):
                cell = screen.buffer[y][x]
                c = cell.data
                if cell.bg not in ("default", ""):
                    d.rectangle([8 + x * cw, 6 + y * ch, 8 + x * cw + cw, 6 + y * ch + ch], fill=colour(cell.bg))
                if c.strip() and ord(c[0]) in cmap:
                    d.text((8 + x * cw, 6 + y * ch), c[0], font=fb if cell.bold else f, fill=colour(cell.fg))
        yield rec["t"], img


def cmd_cutaway(a):
    f = fonts()
    speed = float(a.get("speed", 8))
    pa = list(pane_images(a["in-a"]))
    pb = list(pane_images(a["in-b"]))
    if not pa or not pb:
        raise SystemExit("no pane frames")
    pipe = Pipe(a["out"], crf=23)
    na, nb = a["name-a"], a["name-b"]
    ta0, tb0 = pa[0][0], pb[0][0]
    dur_a, dur_b = pa[-1][0] - ta0, pb[-1][0] - tb0

    def at(frames, t):
        lo, hi = 0, len(frames) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if frames[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        return frames[lo][1]

    def comp(ia, ib, ea, eb, label):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((96, 54), "Hermes Agent, two hours, %dx" % speed, font=f["title"], fill=INK, anchor="lm")
        d.text((96, 112), label, font=f["body"], fill=DIM, anchor="lm")
        for i, (im, nm, el) in enumerate(((ia, na, ea), (ib, nb, eb))):
            x = 36 + i * 940
            d.text((x + 8, 168), short(nm, 30), font=f["h2"], fill=AMBER, anchor="lm")
            d.text((x + 888, 168), "%02d:%02d:%02d" % (el // 3600, el % 3600 // 60, el % 60), font=f["monob"], fill=INK, anchor="rm")
            img.paste(im.resize((888, int(im.height * 888 / im.width))), (x, 200))
        return img
    wins = []
    if "full" in a:
        wins = [(0.0, 1.0)]
    else:
        n = int(a.get("windows", 3))
        wsec = float(a.get("window-s", 8)) * speed
        for k in range(n):
            frac = k / max(1, n - 1) * 0.92
            wins.append((frac, wsec))
    for frac, wlen in wins:
        for (da, db, ra, rb) in [(dur_a, dur_b, None, None)]:
            if "full" in a:
                start_a, start_b, length = 0.0, 0.0, max(da, db)
            else:
                start_a, start_b, length = frac * da, frac * db, wlen
            nfr = int(length / speed * FPS)
            for k in range(nfr):
                t = k * speed / FPS
                ea, eb = min(start_a + t, da), min(start_b + t, db)
                pipe.put(comp(at(pa, ta0 + ea), at(pb, tb0 + eb), int(ea), int(eb), "real time, sped up %dx; paths, user and host names are masked" % speed))
    secs = pipe.close()
    print("cutaway %s: %.1f s" % (a["out"], secs))


def parse(argv):
    a, pos, i = {}, [], 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            k = t[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                a[k] = argv[i + 1]
                i += 2
            else:
                a[k] = True
                i += 1
        else:
            pos.append(t)
            i += 1
    return pos, a


def main():
    pos, a = parse(sys.argv[1:])
    cmd = pos[0]
    if cmd == "replay":
        a["in"], a["out"] = pos[1], pos[2]
        cmd_replay(a)
    elif cmd == "scoreboard":
        scoreboard_img(json.load(open(pos[1]))).save(pos[2])
        print("scoreboard", pos[2])
    elif cmd == "cutaway":
        a["in-a"], a["in-b"], a["out"] = pos[1], pos[2], pos[3]
        cmd_cutaway(a)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
