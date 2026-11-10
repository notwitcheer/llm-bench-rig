#!/usr/bin/env python3
"""rc_render.py <run_dir> <out.mp4> [--ffmpeg F] [--mock] [--card-s 8] [--per-q 4.0] [--n-held 6] [--n-near 6]
The item's own postable clip (16:9 1920x1080, ~60 s): an ~8 s RESULT CARD first (title = the verdict in plain words, big numbers, a grey 'how it was measured' line
under each row, footer with the claim tested and the limits), then the live thing moving: left the 27B reasoning from scratch with a token counter running, right the
CPU cache replaying the same question (embed -> best entry -> the 1B pulls the numbers -> solve()), a running right/wrong tally per side, and a red flash every time a
near-miss question is answered confidently wrong by the cache. Everything shown is read from the run's results (never typed). PIL frames piped to the static ffmpeg.
The left side is played faster than it ran (badge says by how much); that speed-up is a playback choice, NOT a speed claim: the card carries the measured seconds.
The questions shown are the first N held-out and the first N near-miss items in the fixed run order (no picking). No paths, hostnames or tokens are drawn (leak mask)."""
import argparse, json, math, os, re, subprocess, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--mock", action="store_true")
ap.add_argument("--card-s", type=float, default=8.0)
ap.add_argument("--per-q", type=float, default=4.8)
ap.add_argument("--n-held", type=int, default=6)
ap.add_argument("--n-near", type=int, default=6)
ap.add_argument("--data", default=os.environ.get("RC_DATA", ""))
ap.add_argument("--frame", type=float, default=-1, help="write one PNG of this clip time (s) next to out and exit")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if a.data:
    os.environ["RC_DATA"] = a.data
import rc_common as C

FD = "/usr/share/fonts/truetype/dejavu/"
W, H, FPS = 1920, 1080, 25
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a"
LC, RC, RED, GREEN, AMBER = "#6fd6a8", "#e8b64c", "#ff4d4d", "#6fd6a8", "#e8b64c"
LEAK = re.compile(r"witcheer|/home/\S*|/Use" r"rs/\S*|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net|\bcapsule\b|\bmercury\b|[\w.+-]+\x40[\w-]+\.[\w.-]+", re.I)
_f = {}


def F(sz, b=False, mono=False):
    k = (sz, b, mono)
    if k not in _f:
        _f[k] = ImageFont.truetype(FD + ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if b else "") + ".ttf", sz)
    return _f[k]


def clean(s):
    return LEAK.sub("[hidden]", s)


def fit(d, text, maxw, size, bold=False, mono=False, minsz=16):
    while size > minsz and d.textlength(text, font=F(size, bold, mono)) > maxw:
        size -= 1
    return F(size, bold, mono)


run = a.run
S = json.load(open(os.path.join(run, "summary.json" if os.path.exists(os.path.join(run, "summary.json")) else "summary.pre.json")))


def ld(d):
    out = {}
    p = os.path.join(run, d)
    for f in sorted(os.listdir(p)):
        if f.endswith(".json"):
            out[f[:-5]] = json.load(open(os.path.join(p, f)))
    return out


A, B = ld("results/A"), ld("cpu/B")
tau = S["threshold"]["tau"]
selj = json.load(open(os.path.join(run, "sel.json"))) if os.path.exists(os.path.join(run, "sel.json")) else None
# the fixed run order
order = []
if selj:
    for kind, n in (("held", a.n_held), ("near", a.n_near)):
        order += C.items_of(selj, kind)[:n]
else:
    keys = sorted(B)
    order = [(k.split("-")[0], int(k.split("-")[1]), int(k.split("-")[2])) for k in [x for x in keys if x.startswith("held")][: a.n_held] + [x for x in keys if x.startswith("near")][: a.n_near]]
Q = []
for kind, tid, inst in order:
    k = C.item_key(kind, tid, inst)
    row = C.row_for(kind, tid, inst)
    hit = B[k]["served"] and B[k]["s"] >= tau - 1e-12
    Q.append({"k": k, "kind": kind, "q": clean(row["question"]), "gold": C.golden_of(row), "A": A[k], "B": B[k], "hit": hit})
NQ = len(Q)
LIVE = NQ * a.per_q
TOTAL = a.card_s + LIVE
# playback speed of the left side: the median recorded wall of the shown items mapped onto the 3.2 s the counter runs
medwall = sorted(q["A"]["wall_s"] for q in Q)[NQ // 2]
SPEED = max(1.0, medwall / 3.2)


def gnum(x):
    return ("%d" % x) if x is not None and float(x).is_integer() else ("%g" % x if x is not None else "-")


# --------------------------------------------------------------------------------------------------------------------------------------- result card
def title_of(v):
    return {"HOLDS AND TRAPS": "a CPU cache matched the 27B on new questions, and was confidently wrong on near-misses",
            "HOLDS, NO TRAP": "a CPU cache matched the 27B on new questions, and near-miss questions did not fool it",
            "DOES NOT HOLD, TRAPS": "a CPU cache fell short of the 27B, and near-miss questions fooled it",
            "DOES NOT HOLD, NO TRAP": "a CPU cache fell short of the 27B; near-miss questions did not fool it"}.get(v, "replay-cache run (no verdict)")


def card_image(p=1.0):
    h, n = S["held"], S["near"]
    up = lambda x: C.pct(x * p)             # the percentages count up from 0 over the first 2 s (motion from the first frame); p=1 is the final card
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 114], fill=BAND); d.rectangle([0, 114, W, 117], fill=GRID)
    if a.mock:
        d.text((W - 48 - d.textlength("MOCK DATA, NOT A RESULT", font=F(24, True)), 20), "MOCK DATA, NOT A RESULT", font=F(24, True), fill=RED)
    d.text((48, 14), "RESULT", font=F(22, True), fill=MUTE)
    d.text((48, 44), title_of(S["verdict"]), font=fit(d, title_of(S["verdict"]), W - 120, 40, True), fill=TEXT)
    d.text((48, 90), "Qwen3.8-27B reasoning from scratch vs its answers replayed from a cache on CPU, apple/GSM-Symbolic, 20 templates", font=F(21), fill=MUTE)
    xl, xr, xlab = 800, 1300, 80
    d.text((xl, 140), "27B from scratch", font=F(34, True), fill=LC); d.text((xl, 184), "5090, reasoning on", font=F(22), fill=MUTE)
    d.text((xr, 140), "cache replay", font=F(34, True), fill=RC); d.text((xr, 184), "CPU only, stock 1B", font=F(22), fill=MUTE)
    tm = lambda x: "%.0f s" % x if x is not None else "-"
    rows = [
        ("new questions right (n=%d)" % h["n"], "%s%%" % up(h["A_acc"]), "%s%%" % up(h["Bg_acc"]),
         "held-out instances, Apple's own answer key. cache first, the 27B's answer on a miss; cache alone %s%%, %d%% of queries hit" % (C.pct(h["Bc_acc"]), round(100 * h["hit_acc"]))),
        ("near-miss questions right (n=%d)" % n["n"], "%s%%" % up(n["A_acc"]), "%s%%" % up(n["Bg_acc"]),
         "same templates with one extra clause (p1). the cache hit %s%% of them, threshold fixed on the new questions only" % C.pct(n["hit_acc"])),
        ("near-miss: cache hit AND wrong", "-", "%s%%" % up(n["hit_wrong_acc"]), "the cache answered, confidently, with the old template's maths. rule: 20% or more is the trap"),
        ("time per question (mean)", tm(h["A_wall_s_mean"]), ("%.1f s" % (h["cpu_ms_B_mean"] / 1000.0)) if h["cpu_ms_B_mean"] else "-",
         "27B: server wall per question with 4 running at once. cache: embed + the 1B pulls the numbers + solve(), 6 CPU threads, measured per query"),
        ("the 1B on its own, no cache", "-", "%s%%" % up(max(h["C_acc"], h["Cd_acc"])), "same stock 1B, same questions, no cache: the control the paper does not run"),
    ]
    y = 250
    for lab, l, r, note in rows:
        d.rectangle([xlab - 20, y - 14, W - 60, y - 12], fill=GRID)
        d.text((xlab, y), lab, font=fit(d, lab, xl - xlab - 30, 29), fill=TEXT)
        d.text((xl, y - 8), l, font=F(56, True, True), fill=LC); d.text((xr, y - 8), r, font=F(56, True, True), fill=RC)
        d.text((xlab, y + 50), note, font=fit(d, note, W - 2 * xlab, 21), fill=MUTE)
        y += 134
    d.rectangle([xlab - 20, y - 14, W - 60, y - 12], fill=GRID)
    small = ["claim tested: NVIDIA's CPU-ARC paper (arXiv 2611.09891, 6 Nov): cached reasoning matches the source model. their cache is built from each test problem's own transcript; ours is built from one instance per template.",
             "limits: our rebuild (compiled form, no fine-tuned 1B, our embedder), 20 templates, one run, Qwen3.8-27B as source and builder. Q6_K, sampling is the server default. no speed multiplier claimed."]
    for i, line in enumerate(small):
        d.text((xlab, y + 6 + 30 * i), line, font=fit(d, line, W - 2 * xlab, 21), fill=MUTE)
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    return im


# --------------------------------------------------------------------------------------------------------------------------------------- live frame
def wrap(d, text, font, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def tally(upto, f):
    """running tallies over the finished questions before index `upto` plus the current one when its result is in (f = fraction of its slot elapsed)."""
    la = [0, 0]; rb = [0, 0, 0]
    for i in range(upto):
        q = Q[i]
        la[0 if q["A"]["correct"] else 1] += 1
        if q["hit"]:
            rb[0 if q["B"]["correct"] else 1] += 1
        else:
            rb[2] += 1
    return la, rb


def live_image(t):
    i = min(NQ - 1, int(t / a.per_q)); f = (t - i * a.per_q) / a.per_q
    q = Q[i]
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    near = q["kind"] == "near"
    d.rectangle([0, 0, W, 96], fill=BAND); d.rectangle([0, 96, W, 99], fill=GRID)
    d.text((48, 14), "same question, two ways", font=F(40, True), fill=TEXT)
    d.text((48, 62), "%s  ·  question %d of %d" % ("NEAR-MISS (one extra clause)" if near else "NEW QUESTION (same template, new numbers)", i + 1, NQ), font=F(24, True), fill=RED if near else MUTE)
    if a.mock:
        d.text((W - 48 - d.textlength("MOCK DATA, NOT A RESULT", font=F(24, True)), 30), "MOCK DATA, NOT A RESULT", font=F(24, True), fill=RED)
    # the question strip
    qf, qh = F(30), 40
    ql = wrap(d, q["q"], qf, W - 120)
    if len(ql) > 3:
        qf, qh = F(25), 30
        ql = wrap(d, q["q"], qf, W - 120)
    for j, line in enumerate(ql[:4]):
        d.text((60, 106 + qh * j), line, font=qf, fill=TEXT)
    PY0, PY1 = 236, 760
    for x0, x1, col, name in ((48, 936, LC, "Qwen3.8-27B, reasoning from scratch"), (984, 1872, RC, "cache replay")):
        d.rectangle([x0, PY0, x1, PY1], outline=GRID, width=3, fill="#0b101a")
        d.text((x0 + 20, PY0 + 12), name, font=F(30, True), fill=col)
    # ---- left: the 27B thinking, counter runs for 3.2 s of the slot then holds
    lf = min(1.0, (f * a.per_q) / 3.2)
    ntok = int(q["A"]["completion_tokens"] * lf)
    secs = q["A"]["wall_s"] * lf
    d.text((72, PY0 + 60), "%s" % format(ntok, ","), font=F(104, True, True), fill=LC)
    d.text((72, PY0 + 176), "tokens thought   %s s   (played %.0fx)" % ("%.0f" % secs, SPEED), font=F(26), fill=MUTE)
    rs = clean((q["A"].get("reasoning") or "").replace("\n", " "))
    shown = rs[: int(len(rs) * lf)]
    tl = wrap(d, shown, F(22, False, True), 800)[-5:]
    for j, line in enumerate(tl):
        d.text((72, PY0 + 232 + 30 * j), line, font=F(22, False, True), fill="#59806f")
    if lf >= 1.0:
        ok = q["A"]["correct"]
        d.text((72, PY0 + 400), "answer %s" % gnum(q["A"]["answer"]), font=F(64, True, True), fill=GREEN if ok else RED)
        d.text((72, PY0 + 480), "right" if ok else "wrong (key: %s)" % gnum(q["gold"]), font=F(26, True), fill=GREEN if ok else RED)
    # ---- right: staged reveal in milliseconds
    B_ = q["B"]
    st = [0.04, 0.12, 0.26, 0.40]            # embed, entry, 1B reads, solve (fractions of the slot)
    ex = 984 + 24
    if f >= st[0]:
        d.text((ex, PY0 + 62), "embed the question        %3.0f ms" % B_["embed_ms"], font=F(26, False, True), fill=TEXT)
    if f >= st[1]:
        d.text((ex, PY0 + 100), "best entry: template %d   s = %.2f   %s" % (B_["entry"], B_["s"], "HIT" if q["hit"] else "miss"), font=F(26, False, True), fill=TEXT if q["hit"] else AMBER)
    if f >= st[2] and q["hit"]:
        d.text((ex, PY0 + 138), "1B reads the numbers   %4.0f ms" % B_["extract_ms"], font=F(26, False, True), fill=TEXT)
        pj = clean(json.dumps({k: (int(v) if float(v).is_integer() else v) for k, v in (B_["params"] or {}).items()})) if B_["params"] else "(no params)"
        for j, line in enumerate(textwrap.wrap(pj, 50)[:3]):
            d.text((ex, PY0 + 176 + 28 * j), line, font=F(22, False, True), fill="#a98c43")
    if f >= st[3]:
        if q["hit"]:
            d.text((ex, PY0 + 270), "solve()                 %3.0f ms" % B_["solve_ms"], font=F(26, False, True), fill=TEXT)
            ok = B_["correct"]
            d.text((ex, PY0 + 316), "%s" % (gnum(B_["value"]) if B_["value"] is not None else "no number"), font=F(100, True, True), fill=GREEN if ok else RED)
            if ok:
                d.text((ex, PY0 + 450), "right", font=F(40, True), fill=GREEN)
            elif q["kind"] != "near":
                d.text((ex, PY0 + 450), "wrong (key: %s)" % gnum(q["gold"]), font=F(40, True), fill=RED)
            d.text((1872 - 20 - d.textlength("%.0f ms on CPU" % B_["cpu_ms"], font=F(34, True, True)), PY0 + 14), "%.0f ms on CPU" % B_["cpu_ms"], font=F(34, True, True), fill=RC)
        else:
            d.text((ex, PY0 + 270), "below the threshold: cache steps aside", font=F(26, False, True), fill=AMBER)
            d.text((ex, PY0 + 330), "27B answers", font=F(70, True), fill=AMBER)
    # ---- tallies
    done_left = lf >= 1.0
    la, rb = tally(i, f)
    if done_left:
        la = list(la); la[0 if q["A"]["correct"] else 1] += 1
    if f >= st[3]:
        rb = list(rb)
        if q["hit"]:
            rb[0 if B_["correct"] else 1] += 1
        else:
            rb[2] += 1
    d.text((48, 780), "27B tally", font=F(26, True), fill=MUTE)
    d.text((48, 816), "right %d" % la[0], font=F(80, True, True), fill=GREEN); d.text((470, 816), "wrong %d" % la[1], font=F(80, True, True), fill=RED if la[1] else MUTE)
    d.text((984, 780), "cache tally   (a miss goes to the 27B)", font=F(26, True), fill=MUTE)
    d.text((984, 816), "right %d" % rb[0], font=F(80, True, True), fill=GREEN); d.text((1360, 816), "wrong %d" % rb[1], font=F(80, True, True), fill=RED if rb[1] else MUTE)
    d.text((984, 912), "misses sent to the 27B: %d" % rb[2], font=F(30, True), fill=AMBER if rb[2] else MUTE)
    foot = "first %d new + first %d near-miss questions of the fixed run order, not picked. left side sped up; measured seconds are on the result card." % (a.n_held, a.n_near)
    d.text((48, 990), foot, font=fit(d, foot, W - 96, 22), fill=MUTE)
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    # ---- red flash: a near-miss answered confidently wrong by the cache
    if near and q["hit"] and not B_["correct"] and f >= st[3]:
        g = (f - st[3]) / (1 - st[3])
        alpha = max(0.0, 1.0 - g * 2.2) * 0.5
        if alpha > 0:
            ov = Image.new("RGB", (W, H), "#ff2a2a")
            im = Image.blend(im, ov, alpha)
            d = ImageDraw.Draw(im)
        d.rectangle([984 + 12, PY0 + 440, 1872 - 12, PY0 + 504], fill="#5a1212", outline=RED, width=3)
        d.text((1004, PY0 + 450), "CONFIDENTLY WRONG  (key: %s)" % gnum(q["gold"]), font=fit(d, "CONFIDENTLY WRONG  (key: %s)" % gnum(q["gold"]), 840, 40, True), fill="#ffd0d0")
    return im


def frame_at(t):
    return card_image(min(1.0, t / 2.0)) if t < a.card_s else live_image(t - a.card_s)


if a.frame >= 0:
    p = a.out + ".png" if not a.out.endswith(".png") else a.out
    frame_at(a.frame).save(p)
    print("frame", a.frame, "->", p)
    sys.exit(0)

pr = subprocess.Popen([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
n_card = int(a.card_s * FPS)
final_card = None
for n in range(n_card):
    if n < 2 * FPS:
        pr.stdin.write(card_image(min(1.0, n / (2.0 * FPS))).tobytes())
    else:
        final_card = final_card or card_image(1.0).tobytes()
        pr.stdin.write(final_card)
for n in range(int(LIVE * FPS)):
    pr.stdin.write(live_image(n / FPS).tobytes())
pr.stdin.close(); pr.wait()
print("render rc", pr.returncode, "seconds %.1f" % TOTAL, "speed", "%.1fx" % SPEED)
sys.exit(pr.returncode)
