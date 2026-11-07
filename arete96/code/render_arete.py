#!/usr/bin/env python3
"""render_arete.py: the item-96 clip. CPU only, runs on capsule (PIL + fontTools + the static ffmpeg), 16:9 1920x1080, ~45-70 s.

usage: render_arete.py RUNDIR OUT.mp4 --ffmpeg FFMPEG [--stills DIR] [--time-scale S] [--max-main-s 60] [--hold-s 6] [--card-s 8]

RUNDIR (written by arete_summary.py + the generation legs): summary.json, completions-{base,arete}.jsonl, stream-{base,arete}.jsonl, grades/grades-{base,arete}.jsonl, facts.json.
Part 1: an 8 s result card (title = the verdict in plain words; columns base Q6_K | Arete Q6_K with GB; rows any-propaganda %, fact mean, answers scoring 0 of 803, capability board;
        grey 'how measured' notes; footer with the judge, the settings, the cutter confound, 'their eval, their judge, our card').
Part 2: ONE pre-registered prompt (summary.json pick: freedom-of-expression category, base propaganda 2 and Arete 0, largest fact-score gap, tie -> lowest id), both answers replayed at the
        REAL stream speed from the saved per-chunk timestamps (both panes start together; a speed pill appears only when the longer answer needs compressing), ending on each pane's judge
        verdict, the base answer's judge-quoted `frame_adopted` phrase highlighted. Facts and counts only: the text on screen is the model's own answer and the judge's own verdict.
Mock data (facts.json {"mock": true}) is stamped MOCK on every frame. Every drawn string is scanned for account/home-path/tailnet/e-mail shapes and masked.
exit 0 = clip written, 2 = inputs missing/unusable, 4 = the verdict in summary.json is not a valid run (BROKEN/INCOMPLETE): no clip is made of an invalid run.
"""
import argparse, bisect, json, os, re, subprocess, sys

from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("out")
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--stills", default=None)
ap.add_argument("--time-scale", type=float, default=1.0, help="test knob: 0.1 plays the whole clip at 10x so a mock render takes seconds")
ap.add_argument("--max-main-s", type=float, default=60.0)
ap.add_argument("--hold-s", type=float, default=6.0)
ap.add_argument("--card-s", type=float, default=8.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--crf", type=int, default=21)
a = ap.parse_args()

FD = "/usr/share/fonts/truetype/dejavu/"
W, H, FPS = 1920, 1080, a.fps
BG, TEXT, MUTE, GRID, BAND, PANE = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a", "#0b101a"
COL = {"base": "#e8b64c", "arete": "#6fd6a8"}
CORAL, HL_BG = "#ff6b6b", "#7a2230"
LEAK = re.compile(r"witcheer|/home/\S*|/Users/\S*|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net|\bcapsule\b|\bmercury\b|[\w.+-]+@[\w-]+\.[\w.-]+", re.I)
leak_hits = [0]
_cm = {}


def cmap(fn):
    if fn not in _cm:
        _cm[fn] = set(TTFont(FD + fn).getBestCmap().keys())
    return _cm[fn]


def F(sz, b=False, mono=False):
    return ImageFont.truetype(FD + ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if b else "") + ".ttf", sz)


def clean(s, bold=False, mono=False):
    """same-length sanitiser: tabs/CR to space, glyphs the font lacks to '?', leak shapes to blocks (counted)."""
    s = (s or "").replace("\r", "").replace("\t", " ")
    fn = "DejaVuSans" + ("Mono" if mono else "") + ("-Bold" if bold else "") + ".ttf"
    cm = cmap(fn)
    s = "".join(ch if (ch == "\n" or ord(ch) in cm) else "?" for ch in s)
    def blk(m):
        leak_hits[0] += 1
        return "\u2588" * len(m.group(0)) if 0x2588 in cm else "#" * len(m.group(0))
    return LEAK.sub(blk, s)


def ffmpeg_pipe(out):
    return subprocess.Popen([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)


def rj(p):
    return json.load(open(p))


def rl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


R = a.rundir
try:
    S = rj(os.path.join(R, "summary.json"))
    FACTS = rj(os.path.join(R, "facts.json")) if os.path.exists(os.path.join(R, "facts.json")) else {}
except Exception as e:
    print("render: inputs missing: %r" % e)
    sys.exit(2)
if S.get("verdict") in ("BROKEN", "INCOMPLETE"):
    print("render: verdict is %s, no clip is made of an invalid run" % S["verdict"])
    sys.exit(4)
MOCK = bool(FACTS.get("mock"))
GB = FACTS.get("gb", {"base": 22.88, "arete": 23.62})
fb, fa = S["files"]["base"], S["files"]["arete"]
pick = S.get("pick") or {}
if not pick.get("id"):
    print("render: no pick in summary.json")
    sys.exit(2)

# ------------------------------------------------------------------------------------------------ card
def pct(x):
    return "%.1f%%" % (100 * x)


def card_title():
    v = S["verdict"]
    arrow = "%s \u2192 %s" % (pct(fb["any_prop"]), pct(fa["any_prop"]))
    return {"HOLDS": "Arete-27B's drop holds in the Q6_K file: any-propaganda %s" % arrow,
            "SHRINKS": "In the Q6_K file the drop shrinks: any-propaganda %s" % arrow,
            "PARTIAL": "Partly holds in the Q6_K file: any-propaganda %s" % arrow}.get(v, "Arete-27B vs Qwen3.8-27B, Q6_K: %s" % arrow)


def card_rows():
    rows = [("any propaganda, of %d" % fb["n"], pct(fb["any_prop"]), pct(fa["any_prop"]), "judge: share of the %d answers scored propaganda 1 or 2 (%d vs %d answers)" % (fb["n"], fb["any_prop_n"], fa["any_prop_n"])),
            ("fact mean, 0 to 5", "%.2f" % fb["fact_mean"], "%.2f" % fa["fact_mean"], "mean over all %d prompts; an unanswered prompt counts 0 (%d vs %d unanswered)" % (fb["n"], fb["unanswered"], fa["unanswered"])),
            ("answers scoring 0", str(fb["score0"]), str(fa["score0"]), "of %d (an opposed fact, no fact stated, or no answer); hit the 30,000-token cap: %d vs %d" % (fb["n"], fb["length_capped"], fa["length_capped"]))]
    bd = S.get("board") or {}
    if S.get("capability") in ("LEVEL", "DIFFERS") and all(t in bd and bd[t].get("complete") for t in ("gpqa", "he", "arc", "mmlu")):
        g, h = bd["gpqa"], bd["he"]
        rows.append(("capability board", "reference", S["capability"].lower(),
                     "same code, think off: GPQA %d/%d vs %d, HumanEval %d/%d vs %d, ARC %d vs %d, MMLU %d vs %d; paired p>=0.05 on all four = level" % (
                         g["c_arete"], g["n"], g["c_base"], h["c_arete"], h["n"], h["c_base"], bd["arc"]["c_arete"], bd["arc"]["c_base"], bd["mmlu"]["c_arete"], bd["mmlu"]["c_base"])))
    else:
        rows.append(("capability board", "reference", "n/a", "board not complete"))
    return rows


def draw_card():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 114], fill=BAND)
    d.rectangle([0, 114, W, 117], fill=GRID)
    d.text((48, 14), clean(card_title()), font=F(38, True), fill=TEXT)
    d.text((48, 72), clean("Arete-27B (Nous Research SFT) vs its base Qwen3.8-27B, both Q6_K GGUF, one RTX 5090, thinking off, temperature 0, %d prompts, one run each" % fb["n"]), font=F(21), fill=MUTE)
    x_l, x_r, x_lab = 760, 1240, 80
    d.text((x_l, 150), "Qwen3.8-27B", font=F(38, True), fill=COL["base"])
    d.text((x_l, 198), "Q6_K  \u00b7  %.2f GB" % GB["base"], font=F(24), fill=MUTE)
    d.text((x_r, 150), "Arete-27B", font=F(38, True), fill=COL["arete"])
    d.text((x_r, 198), "Q6_K  \u00b7  %.2f GB" % GB["arete"], font=F(24), fill=MUTE)
    y = 270
    for lab, l, r, note in card_rows():
        d.rectangle([x_lab - 20, y - 14, W - 60, y - 12], fill=GRID)
        d.text((x_lab, y), lab, font=F(30), fill=TEXT)
        vs = 54 if max(len(l), len(r)) < 9 else 40
        d.text((x_l, y - 8 + (54 - vs) // 2), l, font=F(vs, True, True), fill=COL["base"])
        d.text((x_r, y - 8 + (54 - vs) // 2), r, font=F(vs, True, True), fill=COL["arete"])
        d.text((x_lab, y + 62), clean(note)[:160], font=F(19), fill=MUTE)
        y += 128
    d.rectangle([x_lab - 20, y - 14, W - 60, y - 12], fill=GRID)
    small = ["their eval (Propaganda_Eval, pinned revision), their grader and judge (x-ai/grok-4.7, temperature 0), our card. one run per model; Nous report five runs.",
             "different GGUF cutters (Unsloth vs bartowski) and imatrix data: a confound we cannot remove. the base file is the Q6_K Donald serves; no speculative drafting for either.",
             "no verdict here is about any other quant, runtime or sampling setting."]
    for i, line in enumerate(small):
        d.text((x_lab, y + 4 + 30 * i), clean(line), font=F(20), fill=MUTE)
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    if MOCK:
        mock_stamp(d)
    return im


def mock_stamp(d):
    d.rectangle([W - 470, 984, W - 48, 1030], fill="#7a2230")
    d.text((W - 456, 988), "MOCK DATA (CPU proof)", font=F(30, True), fill="#ffffff")


# ------------------------------------------------------------------------------------------------ replay
def load_side(tag):
    comp = {r["id"]: r for r in rl(os.path.join(R, "completions-%s.jsonl" % tag))}[pick["id"]]
    strm = {r["id"]: r for r in rl(os.path.join(R, "stream-%s.jsonl" % tag))}[pick["id"]]
    g = {}
    for r in rl(os.path.join(R, "grades", "grades-%s.jsonl" % tag)):
        if r.get("score") is not None:
            g.setdefault(r["id"], r)
    return comp, strm, g[pick["id"]]


def build_side(tag, items):
    comp, strm, g = load_side(tag)
    chunks = strm["ch"]
    # the displayed text is the streamed content; the graded response is its final-answer part. a stub or a model that streams a think block shows it, the grade is on the response.
    full = "".join(c[1] for c in chunks)
    full_c = clean(full)
    # char offset of each chunk end (sanitiser keeps the length)
    ends, n = [], 0
    for c in chunks:
        n += len(c[1].replace("\r", ""))
        ends.append(n)
    return dict(tag=tag, comp=comp, grade=g, text=full_c, ms=[c[0] / 1000.0 for c in chunks], ends=ends, n_chunks=len(chunks), t_end=max([c[0] for c in chunks] + [comp.get("t_total_ms") or 0]) / 1000.0,
                tokens=int(comp.get("completion_tokens") or len(chunks)))


def layout(text, font, width):
    lines, pos = [], 0
    for para in text.split("\n"):
        start = pos
        if not para.strip():
            lines.append((start, start))
        else:
            ls, le = None, None
            for m in re.finditer(r"\S+", para):
                ws, we = pos + m.start(), pos + m.end()
                if ls is None:
                    ls, le = ws, we
                elif font.getlength(text[ls:we]) <= width:
                    le = we
                else:
                    lines.append((ls, le))
                    ls, le = ws, we
            if ls is not None:
                lines.append((ls, le))
        pos += len(para) + 1
    return lines


def wrap_plain(text, font, width, max_lines):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "\u2026"
    return lines


def find_spans(text, phrases):
    sp = []
    low = text.lower()
    for p in phrases or []:
        p = clean(p)
        if not p.strip():
            continue
        i = text.find(p)
        if i < 0:
            i = low.find(p.lower())
        if i < 0 and len(p) > 24:
            i = low.find(p[:24].lower())
            p = p[:24]
        if i >= 0:
            sp.append((i, i + len(p)))
    return sorted(sp)


items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "items.jsonl"))} if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "items.jsonl")) else {}
if os.path.exists(os.path.join(R, "items.jsonl")):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(R, "items.jsonl"))}
PROMPT = clean(items[pick["id"]]["prompt"]) if pick["id"] in items else "(prompt %s)" % pick["id"]

SIDE = {t: build_side(t, items) for t in ("base", "arete")}
PX, PY, PW, PH = {"base": 48, "arete": 984}, 316, 888, 528           # pane frames (x, y, width, height)
TX_W, TX_H = PW - 40, 508
fsz = 22                                                    # long answers scroll like a terminal; 22 px keeps the text readable at landscape-phone width
for sz in range(26, 22, -1):
    fnt = F(sz)
    if all(len(layout(SIDE[t]["text"], fnt, TX_W)) * int(sz * 1.32) <= TX_H for t in SIDE):
        fsz = sz
        break
FNT, LH = F(fsz), int(fsz * 1.32)
VIS = TX_H // LH
for t in SIDE:
    s = SIDE[t]
    s["lines"] = layout(s["text"], FNT, TX_W)
    s["spans"] = find_spans(s["text"], s["grade"].get("frame_adopted")) if t == "base" else []
    s["line_of"] = lambda r, L=s["lines"]: max([i for i, (x, y) in enumerate(L) if x <= max(r - 1, 0)] or [0])
    starts = [l[0] for l in s["lines"]]
    first_hl = min([s["line_of"](sp[0] + 1) for sp in s["spans"]] or [0])
    s["final_top"] = 0 if (len(s["lines"]) <= VIS or not s["spans"]) else max(0, min(first_hl - 1, len(s["lines"]) - VIS))

main_real = max(SIDE["base"]["t_end"], SIDE["arete"]["t_end"], 1.0)
speed = max(1.0, main_real / a.max_main_s)
main_video = main_real / speed
VER_AT = 0.8                                           # verdict badge appears this long after a pane's own stream ends (video seconds)
hold = a.hold_s
total_replay = main_video + VER_AT + hold
HEAD1 = clean("same prompt, same card, same settings: each answer streams at its real speed")
rate_note = "4 requests in flight, so streams run below single-stream speed"


def reveal(side, t_real):
    k = bisect.bisect_right(side["ms"], t_real)
    return (side["ends"][k - 1] if k else 0), k


def static_replay():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 100], fill=BAND)
    d.rectangle([0, 100, W, 103], fill=GRID)
    d.text((48, 12), HEAD1, font=F(32, True), fill=TEXT)
    rung = "pre-registered pick" if pick.get("rung") == "rule" else "pre-registered pick (%s)" % pick.get("rung")
    d.text((48, 58), clean("%s: freedom-of-expression prompt, largest fact-score gap of its kind (base %d, Arete %d). overall any-propaganda %s \u2192 %s on all %d" % (
        rung, pick["base_fact"], pick["arete_fact"], pct(fb["any_prop"]), pct(fa["any_prop"]), fb["n"])), font=F(21), fill=MUTE)
    d.rounded_rectangle([48, 118, W - 48, 262], radius=10, fill=BAND, outline=GRID)
    d.text((68, 128), "PROMPT (prompt %s of %d, single user turn, no system prompt)" % (pick["id"], fb["n"]), font=F(18, True), fill=MUTE)
    for i, ln in enumerate(wrap_plain(PROMPT, F(23), W - 140, 4)):
        d.text((68, 154 + 27 * i), ln, font=F(23), fill=TEXT)
    names = {"base": ("Qwen3.8-27B  Q6_K  \u00b7  %.2f GB" % GB["base"]), "arete": ("Arete-27B  Q6_K  \u00b7  %.2f GB" % GB["arete"])}
    for t in ("base", "arete"):
        x = PX[t]
        d.rounded_rectangle([x, PY - 36, x + PW, PY + PH], radius=10, fill=PANE, outline=COL[t], width=2)
        d.rectangle([x, PY - 36, x + PW, PY - 4], fill=BAND)
        d.text((x + 16, PY - 34), names[t], font=F(24, True), fill=COL[t])
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    d.text((300, H - 40), clean("their eval, their judge (grok-4.7, temp 0), our card. " + rate_note), font=F(19), fill=MUTE)
    return im


def frame_replay(base_im, tv):
    """tv = video seconds into the replay."""
    im = base_im.copy()
    d = ImageDraw.Draw(im)
    t_real = tv * speed
    if speed > 1.0:
        d.rounded_rectangle([W - 400, 20, W - 48, 62], radius=18, fill="#1d2a40")
        d.text((W - 384, 27), "%.1fx speed (time-lapse)" % speed, font=F(22, True), fill="#f5cf72")
    for t in ("base", "arete"):
        s = SIDE[t]
        x0, y0 = PX[t] + 20, PY + 6
        r, k = reveal(s, min(t_real, s["t_end"] + 1))
        done = t_real >= s["t_end"]
        cur = s["line_of"](r) if r else 0
        top = max(0, cur - VIS + 1) if len(s["lines"]) > VIS else 0
        tdone = tv - s["t_end"] / speed
        if done and tdone > VER_AT and s["final_top"] is not None and s["final_top"] != top:                      # after the stream: scroll to the line the verdict quotes
            f = min(1.0, (tdone - VER_AT) / 1.2)
            top = int(round(top + (s["final_top"] - top) * f))
        for li in range(top, min(len(s["lines"]), top + VIS)):
            a0, b0 = s["lines"][li]
            if r <= a0 and r != 0 and li > cur:
                break
            seg_end = min(b0, r)
            if seg_end <= a0 and li != 0:
                continue
            y = y0 + (li - top) * LH
            # highlight rectangles first
            for (hs, he) in s["spans"]:
                lo, hi = max(hs, a0), min(he, seg_end)
                if lo < hi:
                    xa = x0 + FNT.getlength(s["text"][a0:lo])
                    xb = x0 + FNT.getlength(s["text"][a0:hi])
                    d.rectangle([xa - 1, y - 1, xb + 1, y + LH - 2], fill=HL_BG, outline=CORAL)
            d.text((x0, y), s["text"][a0:seg_end], font=FNT, fill="#dde4ee")
        # counters
        el = min(t_real, s["t_end"])
        tok = int(round(s["tokens"] * k / max(s["n_chunks"], 1))) if not done else s["tokens"]
        d.text((PX[t] + 16, PY + PH + 10), "%5.1f s" % el, font=F(34, True, True), fill=COL[t])
        d.text((PX[t] + 220, PY + PH + 10), "%d tok" % tok, font=F(34, True, True), fill=TEXT)
        d.text((PX[t] + 470, PY + PH + 10), "%3.0f tok/s" % (tok / max(el, 0.3)), font=F(34, True, True), fill=MUTE)
        if done and tdone >= VER_AT:
            g = s["grade"]
            bad = g["propaganda"] > 0
            bx, by = PX[t], PY + PH + 62
            d.rounded_rectangle([bx, by, bx + PW, by + 66], radius=14, fill=("#7a2230" if bad else "#17402f"), outline=(CORAL if bad else COL["arete"]), width=3)
            d.text((bx + 20, by + 8), "judge: fact %d/5  \u00b7  propaganda %d/2" % (g["score"], g["propaganda"]), font=F(30, True), fill="#ffffff")
            d.text((bx + 20, by + 42), "x-ai/grok-4.7, temperature 0, Nous's grader and prompt" + ("  \u00b7  quoted phrase highlighted above" if bad and s["spans"] else ""), font=F(17), fill="#cdd6e4")
    if MOCK:
        mock_stamp(d)
    return im


# ------------------------------------------------------------------------------------------------ write
pr = ffmpeg_pipe(a.out)
nfr = 0
card = draw_card()
if a.stills:
    os.makedirs(a.stills, exist_ok=True)
    card.save(os.path.join(a.stills, "card.png"))
fr_card = card.tobytes()
for _ in range(int(a.card_s * a.time_scale * FPS)):
    pr.stdin.write(fr_card)
    nfr += 1
base_im = static_replay()
n_re = int(total_replay * a.time_scale * FPS)
still_at = {int(n_re * 0.45): "mid.png", n_re - 1: "end.png"}
for i in range(n_re):
    tv = (i / FPS) / a.time_scale
    fr = frame_replay(base_im, tv)
    if a.stills and i in still_at:
        fr.save(os.path.join(a.stills, still_at[i]))
    pr.stdin.write(fr.tobytes())
    nfr += 1
pr.stdin.close()
rc = pr.wait()
dur = nfr / FPS / (a.time_scale if a.time_scale else 1)
summ = dict(frames=nfr, video_s=round(nfr / FPS, 2), full_length_s=round(a.card_s + total_replay, 1), speed=round(speed, 2), pick=pick, font=fsz, mock=MOCK, leak_masks=leak_hits[0],
            tokens=dict(base=SIDE["base"]["tokens"], arete=SIDE["arete"]["tokens"]), stream_s=dict(base=SIDE["base"]["t_end"], arete=SIDE["arete"]["t_end"]), ffmpeg_rc=rc)
json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
print("render rc", rc, json.dumps(summ))
sys.exit(0 if rc == 0 else 1)
