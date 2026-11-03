#!/usr/bin/env python3
"""r78z_render.py: the item-78z clip. TWO Hermes Agent panes (pyte + PIL), left = NVFP4 on vLLM, right = UD-Q4_K_M on llama.cpp, each recorded in its own take, composited
with the START SYNCED on each take's own Enter mark. Capsule only (compositing never runs on mercury). pyte + PIL + fontTools, frames piped to the static ffmpeg.
Derived from four_render.py (item 80): same terminal emulation + leak mask, a two-pane layout, a server-token HUD, and a separate end card.

usage: r78z_render.py race <out.mp4> --left TAKEDIR --right TAKEDIR [--left-label T] [--right-label T] [--stills DIR] [--leakscan-only]
       r78z_render.py card <out.mp4> --facts facts.json --summary race-summary.json [--seconds 8]
race: <TAKEDIR>/rec/events-a.jsonl + <TAKEDIR>/hud/hud.jsonl. The HUD tok/s is the SERVER's own generated-token count (hud row `srv_tokens`: llama-server /slots n_decoded,
      vLLM vllm:generation_tokens_total), the same reading on both panes; proxy stream deltas are only a cross-check (vLLM emits a tool call in a few big deltas).
      The clock of each pane starts at its own Enter (the prompt submit); both clocks sit at 0:00 until then and the typed prompt is on screen for --pre-s before.
      Speed: real speed all the way when the longer take is <= --real-max-s (default 240 s); longer takes keep --open-s and --close-s at real speed and compress the middle to
      about --target-main-s, and the speed pill says so ("Nx speed"). One speed for both panes at any moment.
      Writes <out.mp4> + <out>.summary.json (finish times, tokens by three counts, tok/s mean/peak, leak-mask counts) and <takedir>/leakscan.json.
card: the end card (numbers from item 78 + this race), a still held for --seconds.
Fixed layout 1920x1080. Leaks: every screen state is scanned for the account name, home paths, tailnet addresses, hostnames and e-mail shapes; matching cells are drawn as blocks;
any row carrying Hermes' status bar (session clock, model alias) is blanked wherever it sits. Agent output keeps its own em dashes: flag it, do not edit the footage.
"""
import argparse
import base64
import bisect
import json
import os
import re
import subprocess
import sys

import pyte
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["race", "card"])
ap.add_argument("out")
ap.add_argument("--left")
ap.add_argument("--right")
ap.add_argument("--left-label", default="NVFP4  ·  vLLM 0.32.0")
ap.add_argument("--right-label", default="UD-Q4_K_M  ·  llama.cpp b10371")
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--stills", default=None)
ap.add_argument("--facts", default=None)
ap.add_argument("--summary", default=None)
ap.add_argument("--seconds", type=float, default=8.0)
ap.add_argument("--real-max-s", type=float, default=240.0)
ap.add_argument("--target-main-s", type=float, default=100.0)
ap.add_argument("--open-s", type=float, default=30.0)
ap.add_argument("--close-s", type=float, default=40.0)
ap.add_argument("--hold-s", type=float, default=2.5)
ap.add_argument("--pre-s", type=float, default=4.0)
ap.add_argument("--leakscan-only", action="store_true")
ap.add_argument("--font", type=int, default=20)
ap.add_argument("--crf", type=int, default=21)
ap.add_argument("--fps", type=int, default=25)
a = ap.parse_args()

FD = "/usr/share/fonts/truetype/dejavu/"
W, H, FPS = 1920, 1080, a.fps
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a"
COLS_ = {"L": "#6fd6a8", "R": "#e8b64c"}
CORAL = "#ff6b6b"
DEF_FG, DEF_BG = "#dde4ee", "#0b101a"
NAMES = {"black": "#1b2230", "red": "#ff6b6b", "green": "#6fd6a8", "brown": "#e8b64c", "yellow": "#e8b64c", "blue": "#7aa7e8", "magenta": "#c792ea", "cyan": "#59c3c3",
         "white": "#dde4ee", "brightblack": "#6b7a90", "brightred": "#ff8a8a", "brightgreen": "#8ff0c4", "brightbrown": "#f5cf72", "brightyellow": "#f5cf72",
         "brightblue": "#9bbcf2", "brightmagenta": "#dcb0f7", "brightcyan": "#7fdcdc", "brightwhite": "#ffffff"}
LEAK = re.compile(r"witcheer|/home/\S*|/Users/\S*|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net|\bcapsule\b|\bmercury\b|[\w.+-]+@[\w-]+\.[\w.-]+", re.I)
STATUS = re.compile(r"◷|⏲|~[\d.]+K/[\d.]+K")      # Hermes' status bar: model alias + a session clock that would read as a second timer

_cm = {}


def cmap(fn):
    if fn not in _cm:
        _cm[fn] = set(TTFont(FD + fn).getBestCmap().keys())
    return _cm[fn]


def col(c, default):
    if c == "default":
        return default
    if c in NAMES:
        return NAMES[c]
    if len(c) == 6:
        try:
            int(c, 16)
            return "#" + c
        except ValueError:
            pass
    return default


def F(sz, b=False, mono=False):
    return ImageFont.truetype(FD + ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if b else "") + ".ttf", sz)


def fmt(t):
    t = max(t, 0.0)
    m, s = int(t // 60), t - 60 * int(t // 60)
    return "%d:%04.1f" % (m, s) if m else "%4.1f s" % s


def ffmpeg_pipe(out):
    return subprocess.Popen([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)


# ------------------------------------------------------------------------------------------------------------------------------------------ end card
if a.mode == "card":
    facts = json.load(open(a.facts))
    summ = json.load(open(a.summary)) if a.summary and os.path.exists(a.summary) else {}
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 114], fill=BAND)
    d.rectangle([0, 114, W, 117], fill=GRID)
    d.text((48, 16), facts["title"], font=F(40, True), fill=TEXT)
    d.text((48, 72), facts["subtitle"], font=F(22), fill=MUTE)
    x_l, x_r, x_lab = 760, 1240, 80
    d.text((x_l, 150), "NVFP4", font=F(40, True), fill=COLS_["L"])
    d.text((x_l, 198), "vLLM 0.32.0", font=F(24), fill=MUTE)
    d.text((x_r, 150), "UD-Q4_K_M", font=F(40, True), fill=COLS_["R"])
    d.text((x_r, 198), "llama.cpp b10371", font=F(24), fill=MUTE)
    y = 270
    for lab, l, r, note in facts["rows"]:
        d.rectangle([x_lab - 20, y - 14, W - 60, y - 12], fill=GRID)
        d.text((x_lab, y), lab, font=F(30), fill=TEXT)
        d.text((x_l, y - 8), l, font=F(54, True, True), fill=COLS_["L"])
        d.text((x_r, y - 8), r, font=F(54, True, True), fill=COLS_["R"])
        if note:
            d.text((x_lab, y + 44), note, font=F(21), fill=MUTE)
        y += 128
    d.rectangle([x_lab - 20, y - 14, W - 60, y - 12], fill=GRID)
    for i, line in enumerate(facts["small"]):
        d.text((x_lab, y + 4 + 32 * i), line, font=F(22), fill=MUTE)
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    pr = ffmpeg_pipe(a.out)
    for _ in range(int(a.seconds * FPS)):
        pr.stdin.write(im.tobytes())
    pr.stdin.close()
    pr.wait()
    im.save(a.out + ".png")
    print("card rc", pr.returncode)
    sys.exit(pr.returncode)


# ------------------------------------------------------------------------------------------------------------------------------------------ race
class Term:
    def __init__(s, cols, rows, size):
        s.cols, s.rows = cols, rows
        s.scr = pyte.Screen(cols, rows)
        s.stream = pyte.ByteStream(s.scr)
        s.f, s.fb = F(size, False, True), F(size, True, True)
        s.fs, s.fsb = F(size), F(size, True)
        s.cw, s.ch = round(s.f.getlength("M")), round(size * 1.32)
        s.W, s.H = s.cw * cols, s.ch * rows

    def feed(s, b):
        s.stream.feed(b)

    def line(s, y):
        return "".join((s.scr.buffer[y][x].data or " ") for x in range(s.cols))

    def leak_cells(s):
        hide = set()
        for y in range(s.rows - 1):
            for m in LEAK.finditer(s.line(y)):
                for x in range(m.start(), m.end()):
                    hide.add((x, y))
        return hide

    def image(s):
        im = Image.new("RGB", (s.W, s.H), DEF_BG)
        d = ImageDraw.Draw(im)
        hide = s.leak_cells()
        for y in range(s.rows - 1):                      # the last row is the status bar: never drawn
            if STATUS.search(s.line(y)):                 # ... and neither is a status bar that floats up before the screen fills
                continue
            row = s.scr.buffer[y]
            for x in range(s.cols):
                c = row[x]
                fg, bgc = col(c.fg, DEF_FG), col(c.bg, DEF_BG)
                if c.reverse:
                    fg, bgc = (bgc if c.bg != "default" else DEF_BG), fg
                px, py = x * s.cw, y * s.ch
                if (x, y) in hide:
                    d.rectangle([px, py, px + s.cw - 1, py + s.ch - 1], fill="#2a3446")
                    continue
                if bgc != DEF_BG:
                    d.rectangle([px, py, px + s.cw - 1, py + s.ch - 1], fill=bgc)
                if c.data and c.data != " ":
                    cp = ord(c.data[0])
                    if cp in cmap("DejaVuSansMono.ttf"):
                        fnt = s.fb if c.bold else s.f
                    elif cp in cmap("DejaVuSans.ttf"):
                        fnt = s.fsb if c.bold else s.fs
                    else:
                        continue                       # no glyph anywhere: skip, never tofu
                    d.text((px, py + 2), c.data, font=fnt, fill=fg)
        return im


def load_events(path):
    hdr, items, marks = None, [], {}
    for l in open(path):
        j = json.loads(l)
        if j.get("hdr"):
            hdr = j
        elif "d" in j:
            items.append((j["t"], base64.b64decode(j["d"])))
        elif "mark" in j:
            marks.setdefault(j["mark"], j["t"])
            if j["mark"] == "end":
                marks["end_reason"] = j.get("reason")
    items.sort(key=lambda x: x[0])
    return hdr, items, marks


SIDES = {"L": a.left, "R": a.right}
S = {}
for k, d_ in SIDES.items():
    d_ = d_.rstrip("/")
    hdr, items, marks = load_events(d_ + "/rec/events-a.jsonl")
    hud = [json.loads(l) for l in open(d_ + "/hud/hud.jsonl") if l.strip()]
    enter = marks.get("enter") or float(open(d_ + "/rec/go-epoch.txt").read().strip())
    S[k] = dict(dir=d_, hdr=hdr, items=items, marks=marks, hud=hud, ht=[r["t"] for r in hud], enter=enter)
cols, rows = S["L"]["hdr"]["cols"], S["L"]["hdr"]["rows"]
if (S["R"]["hdr"]["cols"], S["R"]["hdr"]["rows"]) != (cols, rows):
    sys.exit("the two takes have different pane sizes")

fin, state = {}, {}
for k in S:
    mk = S[k]["marks"]
    if mk.get("end_reason") == "done" and "last_change" in mk:
        fin[k] = mk["last_change"] - S[k]["enter"]
        state[k] = "done"
    else:
        fin[k] = mk.get("end", S[k]["enter"]) - S[k]["enter"]
        state[k] = mk.get("end_reason") or "unknown"
T_race = max(fin.values())

# leak scan over every screen state
scan = {}
for k in S:
    T = Term(cols, rows, a.font)
    n = 0
    for t, b in S[k]["items"]:
        T.feed(b)
        if T.leak_cells():
            n += 1
    scan[k] = n
    json.dump({"screen_states_with_a_masked_cell": n}, open(S[k]["dir"] + "/leakscan.json", "w"))
print("leakscan (screen states with a masked cell):", scan, flush=True)
if a.leakscan_only:
    sys.exit(0)


def hud_at(k, t):
    i = bisect.bisect_right(S[k]["ht"], t) - 1
    return S[k]["hud"][max(i, 0)]


def tok(k, t):
    r = hud_at(k, t)
    v = r.get("srv_tokens")
    return v if v is not None else r["agents"]["a"]["chunks"]


def rate(k, tabs, w=6.0):
    r1, r0 = hud_at(k, tabs), hud_at(k, tabs - w)
    dt = max(r1["t"] - r0["t"], 1e-3)
    return max(0.0, (tok(k, tabs) - tok(k, tabs - w)) / dt)


def tokens_since_enter(k, tr):
    return max(0.0, tok(k, S[k]["enter"] + tr) - tok(k, S[k]["enter"]))


# speed plan over race time (seconds from each take's Enter)
if T_race <= a.real_max_s:
    segs = [(-a.pre_s, T_race, 1.0)]
else:
    mid0, mid1 = a.open_s, max(a.open_s + 30, T_race - a.close_s)
    sp_ = max(2.0, (mid1 - mid0) / a.target_main_s)
    segs = [(-a.pre_s, mid0, 1.0), (mid0, mid1, sp_), (mid1, T_race, 1.0)]
frames = []
for r0, r1, sp in segs:
    n = int(round((r1 - r0) / sp * FPS))
    frames += [(r0 + k * sp / FPS, sp) for k in range(n)]
hold_n = int(a.hold_s * FPS)
nvid = len(frames) + hold_n
print("race %.0f s (left %.0f, right %.0f), segments %s, %d frames (%.1f s)" % (T_race, fin["L"], fin["R"], [(round(x, 1), round(y, 1), round(s_, 1)) for x, y, s_ in segs], nvid, nvid / FPS), flush=True)


class Panel:
    def __init__(s, k):
        s.k = k
        s.T = Term(cols, rows, a.font)
        s.items = S[k]["items"]
        s.i = 0
        s.img = None
        s.dirty = True

    def at(s, tabs):
        while s.i < len(s.items) and s.items[s.i][0] <= tabs:
            s.T.feed(s.items[s.i][1])
            s.i += 1
            s.dirty = True
        if s.dirty:
            s.img = s.T.image()
            s.dirty = False
        return s.img


P = {k: Panel(k) for k in S}
TW, TH = P["L"].T.W, P["L"].T.H
LAB = 46
GAPX = 40
X0 = (W - (2 * TW + GAPX)) // 2
POS = {"L": (X0, 128), "R": (X0 + TW + GAPX, 128)}
YH = 128 + LAB + TH + 22                     # HUD block under the panes
LABELS = {"L": a.left_label, "R": a.right_label}

base = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(base)
d.rectangle([0, 0, W, 114], fill=BAND)
d.rectangle([0, 114, W, 117], fill=GRID)
d.text((48, 14), "the same Hermes Agent task on two files of Qwen3.8-27B, one RTX 5090", font=F(36, True), fill=TEXT)
d.text((48, 70), "two takes, start synced: the two servers cannot sit on the card together, so each file ran alone, back to back. real sessions, same prompt, same sampling.", font=F(21), fill=MUTE)
for k in S:
    x, y = POS[k]
    d.rounded_rectangle([x - 8, y + LAB - 6, x + TW + 8, y + LAB + TH + 8], radius=10, outline=COLS_[k], width=3, fill=DEF_BG)
d.text((48, H - 38), "witcheer", font=F(22, True), fill=TEXT)
d.text((180, H - 35), "task: an animated isometric voxel cottage as one HTML file. tok/s = the server's own token count, same reading on both.", font=F(18), fill=MUTE)

# summary numbers
SUM = {}
for k in S:
    last = S[k]["hud"][-1]
    t0, t1 = S[k]["enter"], S[k]["enter"] + fin[k]
    series = [rate(k, t) for t in [t0 + i for i in range(5, int(fin[k]) + 1)]]
    gen = [v for v in series if v > 5]
    srv = tokens_since_enter(k, fin[k])
    SUM[k] = {"finish_s": round(fin[k], 1), "end": state[k], "tokens_server": round(srv),
              "tokens_stream_deltas": last["agents"]["a"]["chunks"], "tokens_exact_from_usage": last["agents"]["a"]["exact_pred"],
              "tok_s_mean_while_generating": round(sum(gen) / len(gen), 1) if gen else None, "tok_s_peak_6s": round(max(series), 1) if series else None,
              "tok_s_mean_over_take": round(srv / fin[k], 1) if fin[k] > 0 else None, "engine": last.get("engine"), "vram_mib_max": max([r.get("vram_mib") or 0 for r in S[k]["hud"]] or [0]),
              "requests": last["agents"]["a"]["reqs"], "request_errors": last["agents"]["a"]["err"], "leak_masked_states": scan[k]}


def frame(kf):
    if kf < len(frames):
        tr, sp = frames[kf]
        hold = False
    else:
        tr, sp, hold = T_race, 1.0, True
    im = base.copy()
    dr = ImageDraw.Draw(im)
    for k in S:
        x, y = POS[k]
        tabs = S[k]["enter"] + tr
        im.paste(P[k].at(tabs), (x, y + LAB))
        done = tr >= fin[k]
        t_show = min(max(tr, 0.0), fin[k])
        c = COLS_[k]
        dr.text((x, y + 2), LABELS[k], font=F(26, True), fill=c)
        tx = fmt(t_show)
        fnt = F(34, True, True)
        wt = dr.textlength(tx, font=fnt)
        dr.text((x + TW - wt, y - 2), tx, font=fnt, fill=(c if done else TEXT))
        # HUD under the pane
        tps = 0.0 if done else rate(k, tabs)
        big = ("%3d" % round(tps)) if not done else "  -"
        dr.text((x, YH), big, font=F(76, True, True), fill=(TEXT if not done else MUTE))
        dr.text((x + 150, YH + 40), "tok/s", font=F(26, True), fill=MUTE)
        ntok = tokens_since_enter(k, t_show)
        dr.text((x + 270, YH + 8), "%s tokens generated" % format(int(round(ntok)), ","), font=F(28, True), fill=TEXT)
        dr.text((x + 270, YH + 46), "server count, since the prompt was sent", font=F(18), fill=MUTE)
        if done:
            chip = "done  %s" % fmt(fin[k]).strip() if state[k] == "done" else "stopped (%s)  %s" % (state[k], fmt(fin[k]).strip())
            cw_ = dr.textlength(chip, font=F(30, True)) + 36
            dr.rounded_rectangle([x + TW - cw_, YH + 4, x + TW, YH + 52], radius=12, fill=c)
            dr.text((x + TW - cw_ + 18, YH + 10), chip, font=F(30, True), fill=BG)
        # sparkline: whole take, shared 0-100 tok/s scale on both panes
        sy0, sy1 = YH + 100, YH + 190
        dr.rectangle([x, sy0, x + TW, sy1], outline=GRID, width=1)
        for g_ in (25, 50, 75):
            yy = sy1 - (sy1 - sy0) * g_ / 100.0
            dr.line([(x, yy), (x + TW, yy)], fill="#171f2e", width=1)
            dr.text((x + TW - 34, yy - 14), str(g_), font=F(13), fill="#6b7a90")
        horizon = max(T_race, 30.0)
        pts = []
        t_end = min(max(tr, 0.0), fin[k])
        n_pts = int(t_end) + 1
        for i in range(0, n_pts + 1, 1):
            ti = min(float(i), t_end)
            pts.append((x + TW * ti / horizon, sy1 - (sy1 - sy0 - 4) * min(rate(k, S[k]["enter"] + ti) / 100.0, 1.0)))
        if len(pts) > 1:
            dr.line(pts, fill=c, width=3)
        dr.text((x + 6, sy0 + 4), "tok/s over the take (0-100)", font=F(14), fill="#6b7a90")
    fast = sp != 1.0 and not hold
    pill = ("%gx speed (middle of the take)" % round(sp, 1)) if fast else "real speed"
    pc = CORAL if fast else COLS_["L"]
    pw = dr.textlength(pill, font=F(22, True)) + 36
    dr.rounded_rectangle([W - 48 - pw, H - 44, W - 48, H - 12], radius=16, outline=pc, width=3, fill=(pc if fast else BG))
    dr.text((W - 48 - pw + 18, H - 41), pill, font=F(22, True), fill=(BG if fast else pc))
    return im


pr = ffmpeg_pipe(a.out)
still_at = set()
if a.stills:
    os.makedirs(a.stills, exist_ok=True)
    for frac in (0.02, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 0.995):
        still_at.add(min(int(nvid * frac), nvid - 1))
for kf in range(nvid):
    im = frame(kf)
    pr.stdin.write(im.tobytes())
    if kf in still_at:
        im.save(os.path.join(a.stills, "f%05d.png" % kf))
pr.stdin.close()
pr.wait()
out = {"clip": a.out, "frames": nvid, "seconds": round(nvid / FPS, 1), "race_s": round(T_race, 1), "segments": [(round(x, 1), round(y, 1), round(s_, 1)) for x, y, s_ in segs],
       "left": SUM["L"], "right": SUM["R"], "ffmpeg_rc": pr.returncode}
json.dump(out, open(a.out + ".summary.json", "w"), indent=1)
print(json.dumps(out), flush=True)
