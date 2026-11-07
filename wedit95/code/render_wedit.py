#!/usr/bin/env python3
"""render_wedit.py <RUNDIR> <out.mp4> [--ffmpeg F] [--stills DIR] [--scale S] : the item-95 clip (CPU only, capsule only; pyte + PIL + the static ffmpeg). 1920x1080, 25 fps, ~64 s.
Reference shape (Ripple-65B item 89): result card FIRST (8 s), then the live thing, cut at the result.
  1. result card (8 s): title = the verdict in plain words; four columns (unedited, editor's edit, scripted pipeline, published edit); rows with a grey 'how measured' line; footer = claim tested + limits
  2. three panes (46 s): LEFT the unedited 0.8B model's own thinking text (saved output, first looping try in problem order, replayed), CENTRE the editor's real Hermes Agent pane (the recorded take, time-compressed, speed pill),
     RIGHT the edited 0.8B model on the SAME problem (saved output, replayed); bottom: the loop-rate counters (held-out problems that never finish thinking) and the editor's dev-set scores arriving as the take reaches them
  3. end card (10 s)
Everything on screen is read from <RUNDIR>/summary.json, arms/*.json, take/rec/events-a.jsonl, take/state/calls.jsonl. No paths, hostnames, account names: the leak regex masks cells in the Hermes pane and guards every drawn string.
Mock data (summary.mock) puts a MOCK banner on every frame."""
import argparse, base64, json, os, re, subprocess, sys, textwrap

import pyte
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("out")
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--stills", default=None)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--crf", type=int, default=21)
a = ap.parse_args()
FD = "/usr/share/fonts/truetype/dejavu/"
W, H, FPS = 1920, 1080, a.fps
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#9aa8bd", "#20293a", "#121a2a"
GREEN, GOLD, CORAL, BLUE = "#6fd6a8", "#e8b64c", "#ff6b6b", "#7aa7e8"
DEF_FG, DEF_BG = "#dde4ee", "#0b101a"
NAMES = {"black": "#1b2230", "red": "#ff6b6b", "green": "#6fd6a8", "brown": "#e8b64c", "yellow": "#e8b64c", "blue": "#7aa7e8", "magenta": "#c792ea", "cyan": "#59c3c3", "white": "#dde4ee",
         "brightblack": "#6b7a90", "brightred": "#ff8a8a", "brightgreen": "#8ff0c4", "brightbrown": "#f5cf72", "brightyellow": "#f5cf72", "brightblue": "#9bbcf2", "brightmagenta": "#dcb0f7",
         "brightcyan": "#7fdcdc", "brightwhite": "#ffffff"}
LEAK = re.compile(r"witcheer|/home/\S*|/Users/\S*|/tmp/\S*|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net|\bcapsule\b|\bmercury\b|[\w.+-]+@[\w-]+\.[\w.-]+|work-\d+|arena95\S*", re.I)
STATUS = re.compile(r"◷|⏲|~[\d.]+K/[\d.]+K")
_cm = {}


def cmap(fn):
    if fn not in _cm:
        _cm[fn] = set(TTFont(FD + fn).getBestCmap().keys())
    return _cm[fn]


def F(sz, b=False, mono=False):
    return ImageFont.truetype(FD + ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if b else "") + ".ttf", sz)


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


def safe(s):
    s = str(s)
    if LEAK.search(s):
        raise SystemExit("leak regex hit on drawn text: %r" % s[:60])
    return s


S_ = json.load(open(os.path.join(a.rundir, "summary.json")))
MOCK = bool(S_.get("mock") or json.load(open(os.path.join(a.rundir, "facts.json"))).get("mock") if os.path.exists(os.path.join(a.rundir, "facts.json")) else S_.get("mock"))
V, HLD, DEV, GATE = S_["verdict"], S_["held"], S_["dev"], S_.get("gate") or {}
ARMS = os.path.join(a.rundir, "arms")
TAKE = os.path.join(a.rundir, "take")


def rows_of(arm, st):
    p = os.path.join(ARMS, "%s-%s.json" % (arm, st))
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def Sx(sec):
    return max(1, int(round(sec * a.scale * FPS)))


def banner(d):
    if MOCK:
        d.rectangle([W - 440, 0, W, 134], fill="#5a1d1d")
        d.text((W - 424, 24), "MOCK DATA", font=F(38, True), fill="#ffd0d0")
        d.text((W - 424, 76), "NOT A RESULT, NEVER POST", font=F(21, True), fill="#ffd0d0")


def new():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


HW = W - 48 - (460 if (S_.get("mock") or os.path.exists(os.path.join(a.rundir, "facts.json")) and json.load(open(os.path.join(a.rundir, "facts.json"))).get("mock")) else 48)        # header text width: left of the MOCK box when it is drawn


def header(d, title, sub):
    d.rectangle([0, 0, W, 134], fill=BAND)
    d.rectangle([0, 134, W, 137], fill=GRID)
    sz = 42
    while sz > 26 and F(sz, True).getlength(safe(title)) > HW:
        sz -= 2
    d.text((48, 12), safe(title), font=F(sz, True), fill=TEXT)
    fs = F(21)
    lines = textwrap.wrap(safe(sub), max(20, int(HW / fs.getlength("M") * 1.12)))[:2]
    for i, l in enumerate(lines):
        d.text((48, 66 + 28 * i), l, font=fs, fill=MUTE)


def cnt(arm):
    s = HLD.get(arm)
    return "%d / %d" % (s["loops"], s["n"]) if s else "n/a"


TITLE = {"EDITOR MATCHED THE SCRIPT": "a local 27B in Hermes Agent matched the scripted pipeline at curing a 0.8B model's loops",
         "EDITOR BEAT THE SCRIPT": "a local 27B in Hermes Agent cured a 0.8B model's loops, better than the scripted pipeline",
         "EDITOR HELPED, SCRIPT BETTER": "a local 27B in Hermes Agent cut a 0.8B model's loops; the scripted pipeline cut them more",
         "EDITOR CURED, SCRIPT DID NOT": "a local 27B in Hermes Agent cured a 0.8B model's loops; the scripted pipeline did not",
         "EDITOR COULD NOT MATCH THE SCRIPT": "a local 27B in Hermes Agent could not match the scripted pipeline at curing a 0.8B model's loops",
         "NEITHER CURED IT": "neither a local 27B in Hermes Agent nor the scripted pipeline cured the 0.8B model's loops here",
         "NO CANDIDATE": "a local 27B in Hermes Agent ended its edit session without a scored candidate",
         "EDIT DID NOT REPRODUCE": "the published thinking-loop edit did not reproduce on our rig: no editor comparison",
         "INVALID": "the 0.8B model's thinking loops did not reproduce on our rig"}.get(V, "weight editor run")


def mean_tok(arm):
    s = HLD.get(arm)
    return ("%d" % s["mean_tokens"]) if s and s.get("mean_tokens") else "n/a"


def card():
    im, d = new()
    header(d, TITLE, "Qwen3.8-27B (Q6_K) as the weight editor of Qwen3.5-0.8B, no training, one RTX 5090. held-out maths problems, thinking on, 16,384-token cap")
    xs = [640, 940, 1240, 1540]
    heads = [("unedited", "0.8B as released", MUTE), ("editor's edit", "27B in Hermes Agent", GREEN), ("scripted", "no model in the loop", GOLD), ("published", "MIT CSAIL edit", BLUE)]
    for x, (h, sub, c) in zip(xs, heads):
        d.text((x, 152), h, font=F(32, True), fill=c)
        d.text((x, 192), sub, font=F(20), fill=MUTE)
    n = HLD["base"]["n"] if HLD.get("base") else 40
    rows = [("thinking never finished (loops)", [cnt(k) for k in ("base", "editor", "scripted", "published")], "of %d held-out GSM8K test problems, one try each, same seeds; loop = no answer inside the cap" % n),
            ("answered correctly", ["%s / %d" % (HLD[k]["correct"], HLD[k]["n"]) if HLD.get(k) else "n/a" for k in ("base", "editor", "scripted", "published")], "final boxed answer equals the GSM8K gold answer"),
            ("mean tokens per try", [mean_tok(k) for k in ("base", "editor", "scripted", "published")], "completion tokens from the server's own usage count"),
            ("editor's dev-set scores", ["-", "%s tried" % GATE.get("scored", 0), "-", "-"], "candidates the editor measured on 20 train problems; the best one by that score is the one tested above")]
    y = 262
    for lab, vals, note in rows:
        d.rectangle([60, y - 14, W - 60, y - 12], fill=GRID)
        d.text((80, y), safe(lab), font=F(27), fill=TEXT)
        for x, v, (h, sub, c) in zip(xs, vals, heads):
            d.text((x, y - 8), safe(v), font=F(50 if len(v) < 9 else 32, True, True), fill=c)
        d.text((80, y + 50), safe(textwrap.shorten(note, 120, placeholder="...")), font=F(22), fill=MUTE)
        y += 128
    d.rectangle([60, y - 14, W - 60, y - 12], fill=GRID)
    small = ["claim tested: a weight edit that stops a small model's thinking loops, found by a local model instead of a frontier one. the edit rule and its limits were given to the editor; it chose layers, neurons, direction, strength.",
             "limits: one 0.8B model, GSM8K word problems, %d held-out tries per arm, one editing session of one local 27B (not a benchmark of Hermes Agent), the scripted arm is a baseline we ran, not the authors' code." % n,
             "the published edit is the MIT CSAIL one (applied bit-exact); the discovery is theirs. no speed claim: the side panes are replays of saved outputs."]
    yy = y + 4
    for l in small:
        for w_ in textwrap.wrap(l, 128)[:2]:
            d.text((80, yy), safe(w_), font=F(22), fill=MUTE)
            yy += 28
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    banner(d)
    return im


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
        for y in range(s.rows - 1):
            if STATUS.search(s.line(y)):
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
                        continue
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
    items.sort(key=lambda x: x[0])
    return hdr, items, marks


def text_panel(d, x, y, w, h, title, tcol, text, shown, status, scol):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=DEF_BG, outline=GRID, width=2)
    d.text((x + 14, y + 8), safe(title), font=F(26, True), fill=tcol)
    fnt = F(17, False, True)
    cwid = max(1, int(fnt.getlength("M")))
    cols = (w - 28) // cwid
    lines = []
    for para in text[:shown].split("\n"):
        lines += textwrap.wrap(para, cols) or [""]
    maxl = (h - 100) // 22
    for i, l in enumerate(lines[-maxl:]):
        d.text((x + 14, y + 48 + 22 * i), safe(l), font=fnt, fill="#cfd8e6")
    d.text((x + 14, y + h - 40), safe(status), font=F(22, True), fill=scol)


def main():
    cols, rows_ = 62, 30
    hdr, items, marks = load_events(os.path.join(TAKE, "rec", "events-a.jsonl"))
    cols, rows_ = hdr["cols"], hdr["rows"]
    enter = marks.get("enter") or float(open(os.path.join(TAKE, "rec", "go-epoch.txt")).read().strip())
    endt = marks.get("last_change") or marks.get("end") or (items[-1][0] if items else enter)
    T_take = max(endt - enter, 1.0)
    calls = [json.loads(l) for l in open(os.path.join(TAKE, "state", "calls.jsonl")) if l.strip()]
    evs = [(c["t"] - enter, c["args"]["name"], (c["out"] or {}).get("summary") or {}) for c in calls if c["verb"] == "eval_done" and (c["out"] or {}).get("status") == "done"]
    base_rows, ed_rows = rows_of("base", "held"), rows_of("editor", "held")
    pick = next((r for r in base_rows if not r["closed"]), base_rows[0] if base_rows else None)
    other = next((r for r in ed_rows if pick and r["id"] == pick["id"]), None)
    T = Term(cols, rows_, 15)
    pane_w = max(T.W, 560)
    n2 = Sx(46)
    speed = T_take / (46.0 * a.scale)
    pill = "%dx speed" % round(speed) if speed > 1.5 else "real speed"
    c_ = card()
    sink = subprocess.Popen([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    n = [0]
    keep = {}
    want = {Sx(0.5) // 2: "s0-card", Sx(8) + n2 // 2: "s1-panes", Sx(8) + n2 - 2: "s2-late"}

    def put(im):
        if n[0] in want:
            keep[want[n[0]]] = im.copy()
        sink.stdin.write(im.tobytes())
        n[0] += 1
    for _ in range(Sx(8)):
        put(c_)
    k = 0
    best = None
    for f in range(n2):
        tr = T_take * (f + 1) / n2                       # take seconds shown
        while k < len(items) and items[k][0] - enter <= tr:
            T.feed(items[k][1])
            k += 1
        im, d = new()
        header(d, "the editor at work: three views of one 0.8B model", "left: the unedited model thinking on one problem. centre: the editor, a real Hermes Agent session. right: the edited model, same problem (saved outputs, replayed)")
        pw = 612
        gx = [24, 24 + pw + 18, 24 + 2 * (pw + 18)]
        prog = min(1.0, (f + 1) / (n2 * 0.85))
        if pick:
            ln = max(1, len(pick["think"]))
            text_panel(d, gx[0], 150, pw, 676, "unedited 0.8B", MUTE, pick["think"], int(ln * prog), ("never finished: %s tokens, no answer" % format(pick["tokens"], ",")) if not pick["closed"] and prog >= 0.99 else "thinking ... %d%%" % int(prog * 100), CORAL if prog >= 0.99 and not pick["closed"] else MUTE)
        if other:
            ln2 = max(1, len(other["think"] + "\n" + other["answer"]))
            body = other["think"] + ("\n\nANSWER: " + other["answer"] if other["closed"] else "")
            text_panel(d, gx[2], 150, pw, 676, "edited 0.8B: the editor's final pick", GREEN, body, int(len(body) * prog), ("closed after %s tokens, answer given" % format(other["tokens"], ",")) if other["closed"] and prog >= 0.99 else ("still looping at the cap" if prog >= 0.99 else "thinking ... %d%%" % int(prog * 100)), GREEN if other["closed"] and prog >= 0.99 else MUTE)
        else:
            text_panel(d, gx[2], 150, pw, 676, "edited 0.8B: the editor's final pick", GREEN, "", 0, "no candidate scored", MUTE)
        im_t = T.image()
        d.rounded_rectangle([gx[1], 150, gx[1] + pw, 826], radius=10, fill=DEF_BG, outline=GRID, width=2)
        d.text((gx[1] + 14, 158), "editor: 27B in Hermes Agent", font=F(24, True), fill=GREEN)
        sc = min(1.0, (pw - 20) / im_t.width, 600 / im_t.height)
        if sc < 1.0:
            im_t = im_t.resize((int(im_t.width * sc), int(im_t.height * sc)))
        im.paste(im_t, (gx[1] + 10, 196))
        d.text((gx[1] + pw - 150, 800), pill, font=F(22, True), fill=GOLD)
        d.text((gx[1] + 14, 792), "take clock %d:%02d" % (int(tr // 60), int(tr % 60)), font=F(22, True, True), fill=TEXT)
        done = [e for e in evs if e[0] <= tr]
        if done:
            best = min(done, key=lambda e: (e[2].get("loop_rate", 1), e[0]))
        y0 = 842
        d.rectangle([24, y0, W - 24, y0 + 176], fill=BAND)
        d.text((gx[0] + 14, y0 + 8), "unedited 0.8B: loops, held-out", font=F(24), fill=MUTE)
        d.text((gx[0] + 14, y0 + 40), cnt("base"), font=F(84, True, True), fill=CORAL)
        d.text((gx[1] + 14, y0 + 8), "editor's best dev score (20 problems)", font=F(24), fill=MUTE)
        d.text((gx[1] + 14, y0 + 40), ("%d / %d  %s" % (best[2]["loops"], best[2]["n"], best[1])) if best else "- / -", font=F(84, True, True), fill=GREEN if best else MUTE)
        d.text((gx[1] + 14, y0 + 138), "%d scored so far; base %d / %d on that set" % (len(done), (DEV.get("base") or {}).get("loops", 0), (DEV.get("base") or {}).get("n", 0)), font=F(21), fill=MUTE)
        d.text((gx[2] + 14, y0 + 8), "after the editor's edit: loops, held-out", font=F(24), fill=MUTE)
        d.text((gx[2] + 14, y0 + 40), cnt("editor") if f > n2 * 0.9 else "...", font=F(84, True, True), fill=GREEN)
        d.text((gx[2] + 14, y0 + 138), ("scripted %s  |  published %s" % (cnt("scripted"), cnt("published"))) if f > n2 * 0.9 else "", font=F(21), fill=MUTE)
        d.text((48, H - 40), "witcheer", font=F(24, True), fill=TEXT)
        banner(d)
        put(im)
    for _ in range(Sx(10)):
        put(c_)
    sink.stdin.close()
    sink.wait()
    if a.stills:
        os.makedirs(a.stills, exist_ok=True)
        for name, im in keep.items():
            im.save(os.path.join(a.stills, name + ".png"))
    print("render rc", sink.returncode, "frames", n[0], "seconds", round(n[0] / FPS, 1))
    sys.exit(sink.returncode)


main()
