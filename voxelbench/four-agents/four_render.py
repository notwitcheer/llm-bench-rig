#!/usr/bin/env python3
"""four_render.py: the item-80 clip. Four Hermes panes (pyte + PIL) in a fixed 2x2 grid with a live HUD: per-agent tok/s, aggregate tok/s, clocks, finish chips.
Capsule only (compositing never runs on mercury). pyte + PIL + fontTools, frames piped to the static ffmpeg.

usage: four_render.py <rundir> <out.mp4> [--ffmpeg PATH] [--stills DIR] [--solo-tps N] [--model-label TXT] [--no-timelapse] [--target-main-s 60]
                      [--open-s 20] [--close-s 30] [--hold-s 6] [--pre-s 4] [--max-race-s N] [--leakscan-only] [--font 17]
reads  <rundir>/rec/events-{a,b,c,d}.jsonl  <rundir>/hud/hud.jsonl  <rundir>/rec/go-epoch.txt
writes <out.mp4> (+ <out>.summary.json: finish times, tokens, tok/s, peak, VRAM, leak-mask counts) and <rundir>/leakscan.json

Fixed layout: 1920x1080, title band 0-117 (title left, aggregate tok/s + sparkline right), panels 780x396 (78x18 cells, font 17), a 36 px label strip above each panel,
footer strip at the bottom. The race clock starts at the shared Enter (GO); typing is shown for --pre-s before it.
Timelapse: --open-s of real speed from GO, then a speed S so the middle takes about --target-main-s, then --close-s of real speed ending at the last finish, then
--hold-s on a result card. The speed pill says which. Short races (<= open+close+30 s) stay at real speed.
Leaks: every screen state is scanned for the account name, home paths, tailnet addresses, hostnames and e-mail shapes; matching cells are drawn as blocks (never shown),
the status-bar row is always blanked, and the counts land in the summary (a non-zero count means the model's own terminal output carried it, e.g. `ls -l` owner column).
Agent output keeps its own em dashes: flag it, do not edit the footage.
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
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--stills", default=None)
ap.add_argument("--solo-tps", type=float, default=None)
ap.add_argument("--model-label", default="Qwen3.8-27B Q6_K")
ap.add_argument("--no-timelapse", action="store_true")
ap.add_argument("--target-main-s", type=float, default=60.0)
ap.add_argument("--open-s", type=float, default=20.0)
ap.add_argument("--close-s", type=float, default=30.0)
ap.add_argument("--hold-s", type=float, default=6.0)
ap.add_argument("--pre-s", type=float, default=4.0)
ap.add_argument("--max-race-s", type=float, default=None)
ap.add_argument("--leakscan-only", action="store_true")
ap.add_argument("--font", type=int, default=17)
ap.add_argument("--crf", type=int, default=21)
ap.add_argument("--fps", type=int, default=25)
a = ap.parse_args()

AG = "abcd"
FD = "/usr/share/fonts/truetype/dejavu/"
W, H, FPS = 1920, 1080, a.fps
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a"
COL = {"a": "#6fd6a8", "b": "#e8b64c", "c": "#7aa7e8", "d": "#c792ea"}
CORAL = "#ff6b6b"
DEF_FG, DEF_BG = "#dde4ee", "#0b101a"
NAMES = {"black": "#1b2230", "red": "#ff6b6b", "green": "#6fd6a8", "brown": "#e8b64c", "yellow": "#e8b64c", "blue": "#7aa7e8", "magenta": "#c792ea", "cyan": "#59c3c3",
         "white": "#dde4ee", "brightblack": "#6b7a90", "brightred": "#ff8a8a", "brightgreen": "#8ff0c4", "brightbrown": "#f5cf72", "brightyellow": "#f5cf72",
         "brightblue": "#9bbcf2", "brightmagenta": "#dcb0f7", "brightcyan": "#7fdcdc", "brightwhite": "#ffffff"}
# what must never be on screen (the model's own terminal output can carry any of it)
LEAK = re.compile(r"witcheer|/home/\S*|/Users/\S*|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net|\bcapsule\b|\bmercury\b|[\w.+-]+@[\w-]+\.[\w.-]+", re.I)

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

    def leak_cells(s):
        """cells to hide (every row but the status bar, which is blanked whole)"""
        hide = set()
        for y in range(s.rows - 1):
            line = "".join((s.scr.buffer[y][x].data or " ") for x in range(s.cols))
            for m in LEAK.finditer(line):
                for x in range(m.start(), m.end()):
                    hide.add((x, y))
        return hide

    def image(s):
        im = Image.new("RGB", (s.W, s.H), DEF_BG)
        d = ImageDraw.Draw(im)
        hide = s.leak_cells()
        for y in range(s.rows - 1):
            line = s.scr.buffer[y]
            for x in range(s.cols):
                c = line[x]
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


run = a.run.rstrip("/")
go = float(open(run + "/rec/go-epoch.txt").read().strip())
ev = {}
for ag in AG:
    hdr, items, marks = load_events(run + "/rec/events-%s.jsonl" % ag)
    ev[ag] = dict(hdr=hdr, items=items, marks=marks)
cols, rows = ev["a"]["hdr"]["cols"], ev["a"]["hdr"]["rows"]
hud = [json.loads(l) for l in open(run + "/hud/hud.jsonl") if l.strip()]
ht = [r["t"] for r in hud]


def at(t):
    i = bisect.bisect_right(ht, t) - 1
    return hud[max(i, 0)]


def chunks_sum(r):
    return sum(r["agents"][g]["chunks"] for g in AG)


def rate(t, key, ag=None, w=6.0):
    """live tok/s from the proxy's per-agent stream counts (one SSE delta = one token on llama.cpp, checked against slots n_decoded: 11,318 vs 11,309).
    llama-server's own /metrics counters only move when a request ENDS (a 3 min request showed a flat counter), so they are a cross-check, not the HUD source."""
    r1, r0 = at(t), at(t - w)
    if ag:
        v1, v0 = r1["agents"][ag]["chunks"], r0["agents"][ag]["chunks"]
    elif key == "chunks_sum":
        v1, v0 = chunks_sum(r1), chunks_sum(r0)
    else:
        v1, v0 = r1.get(key) or 0, r0.get(key) or 0
    dt = max(r1["t"] - r0["t"], 1e-3)
    return max(0.0, (v1 - v0) / dt)


# finish time per agent: the last screen change before the end mark (done) or the wall mark
fin, state = {}, {}
for ag in AG:
    mk = ev[ag]["marks"]
    if mk.get("end_reason") == "done" and "last_change" in mk:
        fin[ag] = mk["last_change"] - go
        state[ag] = "done"
    else:
        fin[ag] = mk.get("end", go) - go
        state[ag] = mk.get("end_reason") or "unknown"
T_race = max(fin.values())
if a.max_race_s:
    T_race = min(T_race, a.max_race_s)

# leak scan pass over every screen state (the renderer masks the same cells in the clip)
scan = {}
for ag in AG:
    T = Term(cols, rows, a.font)
    n = 0
    for t, b in ev[ag]["items"]:
        T.feed(b)
        if T.leak_cells():
            n += 1
    scan[ag] = n
json.dump({"screen_states_with_a_masked_cell": scan}, open(run + "/leakscan.json", "w"))
print("leakscan (screen states with a masked cell, per agent):", scan, flush=True)
if a.leakscan_only:
    sys.exit(0)

# speed plan over race time (seconds from GO)
if a.no_timelapse or T_race <= a.open_s + a.close_s + 30:
    segs = [(-a.pre_s, T_race, 1.0)]
else:
    mid0, mid1 = a.open_s, T_race - a.close_s
    S = max(4.0, (mid1 - mid0) / a.target_main_s)
    segs = [(-a.pre_s, mid0, 1.0), (mid0, mid1, S), (mid1, T_race, 1.0)]
frames = []
for r0, r1, sp in segs:
    n = int(round((r1 - r0) / sp * FPS))
    frames += [(r0 + k * sp / FPS, sp) for k in range(n)]
hold_n = int(a.hold_s * FPS)
nvid = len(frames) + hold_n
print("race %.0f s, segments %s, %d frames (%.1f s)" % (T_race, [(round(x, 1), round(y, 1), round(s, 1)) for x, y, s in segs], nvid, nvid / FPS), flush=True)


def fmt(t):
    t = max(t, 0.0)
    m, s = int(t // 60), t - 60 * int(t // 60)
    return "%d:%04.1f" % (m, s) if m else "%4.1f s" % s


class Panel:
    def __init__(s, ag):
        s.T = Term(cols, rows, a.font)
        s.items = ev[ag]["items"]
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


P = {ag: Panel(ag) for ag in AG}
TW, TH = P["a"].T.W, P["a"].T.H
LAB = 36
GAPX = 60
X0 = (W - (2 * TW + GAPX)) // 2
XS = [X0, X0 + TW + GAPX]
Y1 = 128
Y2 = Y1 + LAB + TH + 14
POS = {"a": (XS[0], Y1), "b": (XS[1], Y1), "c": (XS[0], Y2), "d": (XS[1], Y2)}

base = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(base)
d.rectangle([0, 0, W, 114], fill=BAND)
d.rectangle([0, 114, W, 117], fill=GRID)
d.text((48, 16), "four Hermes Agent sessions, one RTX 5090, same task", font=F(36, True), fill=TEXT)
d.text((48, 72), "%s on llama.cpp, four slots on one card. every pane is a real session." % a.model_label, font=F(22), fill=MUTE)
for ag in AG:
    x, y = POS[ag]
    d.rounded_rectangle([x - 8, y + LAB - 6, x + TW + 8, y + LAB + TH + 8], radius=10, outline=COL[ag], width=3, fill=DEF_BG)
d.text((48, H - 34), "witcheer", font=F(22, True), fill=TEXT)
d.text((180, H - 31), "task: one animated isometric voxel lighthouse island as a single HTML file. same prompt in all four panes, started at the same instant.", font=F(18), fill=MUTE)

# summary numbers (also written next to the clip)
last = hud[-1]
tot_chunks = {ag: last["agents"][ag]["chunks"] for ag in AG}
exact = {ag: last["agents"][ag]["exact_pred"] for ag in AG}
agg_peak = max((rate(r["t"], "chunks_sum", w=5.0) for r in hud if r["t"] > go + 10), default=0.0)
act = [i for i, r in enumerate(hud) if go <= r["t"] <= go + T_race and sum(1 for g in AG if r["agents"][g]["inflight"] > 0) >= 3]
mean_agg = None
if len(act) > 10:
    i0, i1 = act[0], act[-1]
    dt = hud[i1]["t"] - hud[i0]["t"]
    mean_agg = (chunks_sum(hud[i1]) - chunks_sum(hud[i0])) / max(dt, 1)
vram = [r.get("vram_mib") for r in hud if r.get("vram_mib")]
metrics_total = (last.get("pred_total") or 0) - (hud[0].get("pred_total") or 0)
cross = {"sum_stream_chunks": chunks_sum(last), "sum_exact_predicted_n": sum(exact.values()), "metrics_tokens_predicted_delta": metrics_total}
SUMMARY = {"text1": "three or more agents generating at once: %s tok/s aggregate" % ("%d" % round(mean_agg) if mean_agg else "n/a"),
           "text2": ("one agent alone measured %d tok/s on the same server. " % round(a.solo_tps) if a.solo_tps else "") + "peak %d tok/s (5 s window)." % round(agg_peak)}

agg_series = {}


def agg_at(tr):
    k = int(tr)
    if k not in agg_series:
        agg_series[k] = rate(float(k), "chunks_sum", w=5.0)
    return agg_series[k]


def frame(k):
    if k < len(frames):
        tr, sp = frames[k]
        hold = False
    else:
        tr, sp, hold = T_race, 1.0, True
    tabs = go + tr
    im = base.copy()
    dr = ImageDraw.Draw(im)
    for ag in AG:
        x, y = POS[ag]
        im.paste(P[ag].at(tabs), (x, y + LAB))
        done = tr >= fin[ag]
        t_show = min(max(tr, 0.0), fin[ag])
        tps = 0.0 if done else rate(tabs, None, ag)
        dr.text((x, y), "agent %s" % ag.upper(), font=F(26, True), fill=COL[ag])
        tx = fmt(t_show)
        fnt = F(30, True, True)
        wt = dr.textlength(tx, font=fnt)
        dr.text((x + TW - wt, y - 2), tx, font=fnt, fill=(COL[ag] if done else TEXT))
        if done:
            chip = "done" if state[ag] == "done" else "stopped (%s)" % state[ag]
            cw_ = dr.textlength(chip, font=F(20, True)) + 28
            dr.rounded_rectangle([x + TW - wt - cw_ - 14, y + 2, x + TW - wt - 14, y + 32], radius=8, fill=COL[ag])
            dr.text((x + TW - wt - cw_ - 14 + 14, y + 5), chip, font=F(20, True), fill=BG)
        else:
            dr.text((x + 150, y + 2), "%3d tok/s" % round(tps), font=F(26, True, True), fill=TEXT)
    agg = agg_at(tabs) if not hold else 0.0
    big = "%d tok/s" % round(agg)
    fb = F(54, True, True)
    wb = dr.textlength(big, font=fb)
    dr.text((W - 48 - wb, 8), big, font=fb, fill="#ffffff")
    dr.text((W - 48 - dr.textlength("aggregate decode, all four agents", font=F(20)), 78), "aggregate decode, all four agents", font=F(20), fill=MUTE)
    if a.solo_tps:
        s2 = "one agent alone: %d tok/s" % round(a.solo_tps)
        dr.text((W - 48 - dr.textlength("aggregate decode, all four agents", font=F(20)) - dr.textlength(s2, font=F(20, True)) - 28, 78), s2, font=F(20, True), fill=COL["b"])
    x0, x1, y0, y1 = W - 48 - wb - 330, W - 48 - wb - 28, 14, 62
    dr.rectangle([x0, y0, x1, y1], outline=GRID, width=1)
    pts = [(x0 + (x1 - x0) * i / 120.0, tabs - 120 + i) for i in range(121) if tabs - 120 + i >= go - 5]
    mx = max(60.0, max((agg_at(tt) for _, tt in pts), default=60.0))
    line = [(px, y1 - (y1 - y0 - 4) * min(agg_at(tt) / mx, 1.0)) for px, tt in pts]
    if len(line) > 1:
        dr.line(line, fill="#ffffff", width=2)
    fast = sp != 1.0 and not hold
    pill = ("%gx speed" % round(sp, 1)) if fast else "real speed"
    pc = CORAL if fast else COL["a"]
    pw = dr.textlength(pill, font=F(22, True)) + 36
    dr.rounded_rectangle([W - 48 - pw, H - 44, W - 48, H - 12], radius=16, outline=pc, width=3, fill=(pc if fast else BG))
    dr.text((W - 48 - pw + 18, H - 41), pill, font=F(22, True), fill=(BG if fast else pc))
    if hold:
        dr.rounded_rectangle([200, 380, W - 200, 700], radius=18, fill=BAND, outline=COL["a"], width=3)
        ordered = sorted(AG, key=lambda g: fin[g])
        dr.text((250, 410), "all four finished" if all(state[g] == "done" for g in AG) else "race over", font=F(40, True), fill=TEXT)
        dr.text((250, 476), "   ".join("%s %s" % (g.upper(), fmt(fin[g]).strip()) for g in ordered), font=F(38, True, True), fill=TEXT)
        dr.text((250, 560), SUMMARY["text1"], font=F(28), fill=TEXT)
        dr.text((250, 612), SUMMARY["text2"], font=F(24), fill=MUTE)
    return im


proc = subprocess.Popen([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                         "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
still_at = set()
if a.stills:
    os.makedirs(a.stills, exist_ok=True)
    for frac in (0.02, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 0.995):
        still_at.add(min(int(nvid * frac), nvid - 1))
for k in range(nvid):
    im = frame(k)
    proc.stdin.write(im.tobytes())
    if k in still_at:
        im.save(os.path.join(a.stills, "f%05d.png" % k))
proc.stdin.close()
proc.wait()
summ = {"clip": a.out, "frames": nvid, "seconds": round(nvid / FPS, 1), "race_s": round(T_race, 1), "finish_s": {g: round(fin[g], 1) for g in AG}, "end": state,
        "chunks": tot_chunks, "exact_predicted_tokens": exact, "agg_peak_tps_5s": round(agg_peak, 1), "agg_mean_tps_3plus_active": round(mean_agg, 1) if mean_agg else None,
        "vram_peak_mib": max(vram) if vram else None, "segments": [(round(x, 1), round(y, 1), round(s, 1)) for x, y, s in segs],
        "leak_masked_states": scan, "token_count_crosscheck": cross, "ffmpeg_rc": proc.returncode}
json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
print(json.dumps(summ), flush=True)
