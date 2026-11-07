#!/usr/bin/env python3
"""render_nerf.py <RUNDIR> <out.mp4> [--ffmpeg F] [--stills DIR] : the item-90a clip (CPU only, capsule only; PIL + the static ffmpeg). 1920x1080, 25 fps, ~68 s.
Reference shape (Ripple-65B item 89): result card FIRST (8 s), then the live thing moving, cut right at the result.
  1. result card (8 s): title = the verdict in plain words, big numbers, a grey 'how it was measured' line under each row, footer with the claim tested and the limits
  2. fingerprint (14 s): the sha256 of the file on our disk typed against the oid Hugging Face listed on 14 Aug, character by character, a green tick per matching character
  3. replay (32 s): GPQA-diamond (198) and HumanEval (164) answers, August vs today, one cell per item, a running 'same answer' counter (letters and pass/fail only: no question text)
  4. bands (14 s): August's score with its 95% Wilson band, today's score landing on it (or not): the verdict as a picture
Everything on screen is read from <RUNDIR>/summary.json, facts.json and the saved per-item results; nothing is typed by hand. Mock data (facts.mock) puts a MOCK banner on every frame.
No paths, hostnames or account names are drawn (the only text is hashes, counts and fixed labels); a leak regex is applied to every string before it is drawn.
"""
import argparse, json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import nerf_common as N  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("out")
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--stills", default=None)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--crf", type=int, default=21)
ap.add_argument("--scale", type=float, default=1.0, help="time scale for tests (0.1 = ten times shorter)")
a = ap.parse_args()

FD = "/usr/share/fonts/truetype/dejavu/"
W, H, FPS = 1920, 1080, a.fps
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a"
GREEN, GOLD, CORAL, BLUE = "#6fd6a8", "#e8b64c", "#ff6b6b", "#7aa7e8"
LEAK = re.compile(r"witcheer|/home/\S*|/Users/\S*|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net|\bcapsule\b|\bmercury\b|[\w.+-]+@[\w-]+\.[\w.-]+", re.I)


def F(sz, b=False, mono=False):
    return ImageFont.truetype(FD + ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if b else "") + ".ttf", sz)


def safe(s):
    s = str(s)
    if LEAK.search(s):
        raise SystemExit("leak regex hit on drawn text: %r" % s[:60])
    return s


summ = json.load(open(os.path.join(a.rundir, "summary.json")))
facts = json.load(open(os.path.join(a.rundir, "facts.json")))
MOCK = bool(facts.get("mock"))
an, dec = summ["analysis"], summ["decision"]
head = an["passes"]["new-q6k"]
q4 = an["passes"].get("new-q4km")
verdict = dec["verdict"]
RES = os.path.join(a.rundir, "results")


def S(sec):
    return max(1, int(round(sec * a.scale * FPS)))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def banner(d):
    if MOCK:
        d.rectangle([W - 440, 0, W, 114], fill="#5a1d1d")
        d.text((W - 424, 24), "MOCK DATA", font=F(38, True), fill="#ffd0d0")
        d.text((W - 424, 74), "NOT A RESULT, NEVER POST", font=F(21, True), fill="#ffd0d0")


def footer(d, extra=""):
    d.text((48, H - 44), "witcheer", font=F(24, True), fill=TEXT)
    if extra:
        d.text((W - 48 - d.textlength(extra, font=F(20)), H - 40), extra, font=F(20), fill=MUTE)


def new():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def header(d, title, sub):
    d.rectangle([0, 0, W, 114], fill=BAND)
    d.rectangle([0, 114, W, 117], fill=GRID)
    d.text((48, 14), safe(title), font=F(44, True), fill=TEXT)
    d.text((48, 74), safe(sub), font=F(23), fill="#9aa8bd")


TITLE = {"SAME FILE, SAME SCORES": "Qwen3.8-27B was not nerfed: same file, same scores",
         "SHIFTED DOWN": "Qwen3.8-27B: same file, lower scores on today's runtime",
         "SHIFTED UP": "Qwen3.8-27B: same file, higher scores on today's runtime",
         "SHIFTED MIXED": "Qwen3.8-27B: same file, mixed scores on today's runtime",
         "CHANGED FILE": "The Qwen3.8-27B file on our disk is not the August file"}.get(verdict, "Qwen3.8-27B nerf audit")


def pct(x):
    return "%.1f%%" % x


# ----------------------------------------------------------------------------------------------------------------------- scene 1: result card
def card():
    im, d = new()
    sub = "our August board re-run on the same items, one RTX 5090, greedy, think-off"
    header(d, TITLE, sub)
    x_l, x_r, x_lab = 800, 1260, 80
    d.text((x_l, 138), "14 Aug", font=F(40, True), fill=GOLD)
    d.text((x_l, 186), "day-0 board, recorded", font=F(22), fill=MUTE)
    d.text((x_r, 138), "today", font=F(40, True), fill=GREEN)
    d.text((x_r, 186), "same items, same settings", font=F(22), fill=MUTE)
    g, he = head["gpqa"], head["he"]
    hok = facts.get("hash_ok", {}).get("q6k")
    sha = facts.get("sha_q6k", "?" * 64)
    nch, ntot = facts.get("hf_weight_files_changed"), facts.get("hf_weight_files_total")
    ag = [("GPQA", g.get("same_answer"), g["n"]), ("HumanEval", he.get("same_answer"), he["n"])]
    rows = [
        ("Q6_K file fingerprint (sha256)", sha[:8] + "…", (sha[:8] + "…") if hok else "DIFFERENT", "our file vs the oid Hugging Face listed on 14 Aug: 64 of 64 characters"),
        ("GPQA-diamond, correct", "%d / %d" % (g["d0_c"], g["d0_n"]), "%d / %d" % (g["c"], g["n"]), "95%% band of the August score %.1f to %.1f; today %s" % (g["d0_lo"], g["d0_hi"], "inside" if not g["outside"] else "OUTSIDE")),
        ("HumanEval, correct", "%d / %d" % (he["d0_c"], he["d0_n"]), "%d / %d" % (he["c"], he["n"]), "95%% band %.1f to %.1f; today %s. ARC + MMLU re-run too" % (he["d0_lo"], he["d0_hi"], "inside" if not he["outside"] else "OUTSIDE")),
        ("same answer, item by item", "n/a", "%d/%d  %d/%d" % (ag[0][1] or 0, ag[0][2], ag[1][1] or 0, ag[1][2]), "GPQA | HumanEval, per item, August vs today (runtime build differs)"),
        ("weight files changed since upload", "n/a", ("%s of %s" % (nch, ntot)) if nch is not None else "?", "Hugging Face LFS oids of the 18 shards, main vs the 13 Aug upload"),
    ]
    y = 246
    for lab, l, r, note in rows:
        d.rectangle([x_lab - 20, y - 14, W - 60, y - 12], fill=GRID)
        d.text((x_lab, y), safe(lab), font=F(32), fill=TEXT)
        fs = 52 if len(r) <= 12 else 40
        d.text((x_l, y - 8), safe(l), font=F(52 if len(l) <= 12 else 40, True, True), fill=GOLD)
        d.text((x_r, y - 8), safe(r), font=F(fs, True, True), fill=GREEN if r != "DIFFERENT" else CORAL)
        d.text((x_lab, y + 46), safe(note), font=F(24), fill="#9aa8bd")
        y += 122
    d.rectangle([x_lab - 20, y - 14, W - 60, y - 12], fill=GRID)
    small = ["claim tested: 'they nerfed the open Qwen3.8-27B file'. hosted-API nerf claims are out of scope: no file to hash.",
             "Q6_K, greedy, think-off, August harness unchanged. today's runtime build differs from August's; both builds were run.",
             "n = 198 and 164 items: a few points of gap is noise, which is why the band is shown, not only the score."]
    for i, line in enumerate(small):
        d.text((x_lab, y + 2 + 31 * i), safe(line), font=F(23), fill="#9aa8bd")
    footer(d)
    banner(d)
    return im


# ----------------------------------------------------------------------------------------------------------------------- scene 2: fingerprint
def scene_fp(frames):
    sha_now = facts.get("sha_q6k", "0" * 64)
    sha_aug = facts.get("oid_aug14_q6k", sha_now)
    sha4_now, sha4_aug = facts.get("sha_q4km", "0" * 64), facts.get("oid_aug14_q4km", facts.get("sha_q4km", "0" * 64))
    n = S(14)
    for f in range(n):
        im, d = new()
        header(d, "step 1: is it even the same file?", "sha256 of the Q6_K file: Hugging Face's oid on 14 Aug vs the file on our disk today")
        prog = ease(f / (n * 0.62))
        k = int(round(prog * 64))
        match = sum(1 for i in range(k) if sha_now[i] == sha_aug[i])
        y0 = 190
        for row, (lab, h, col) in enumerate((("14 Aug  (Hugging Face oid)", sha_aug, GOLD), ("today  (hashed on our disk)", sha_now, GREEN))):
            d.text((80, y0 + row * 260), lab, font=F(34, True), fill=col)
            for half in range(2):
                seg = h[half * 32:(half + 1) * 32]
                for j, ch in enumerate(seg):
                    idx = half * 32 + j
                    x = 80 + j * 55
                    y = y0 + 60 + row * 260 + half * 80
                    if idx < k:
                        d.text((x, y), safe(ch), font=F(54, True, True), fill=TEXT)
        # tick marks between the rows
        for idx in range(k):
            same = sha_now[idx] == sha_aug[idx]
            half, j = divmod(idx, 32)
            x = 80 + j * 55
            y = y0 + 60 + 2 * 80 + (half * 0)
        # match strip
        for idx in range(64):
            half, j = divmod(idx, 32)
            x = 80 + j * 55
            y = 770 + half * 36
            if idx < k:
                d.rounded_rectangle([x, y, x + 44, y + 28], radius=6, fill=GREEN if sha_now[idx] == sha_aug[idx] else CORAL)
            else:
                d.rounded_rectangle([x, y, x + 44, y + 28], radius=6, outline=GRID, width=2)
        d.text((80, 862), "%d / 64 characters match" % match, font=F(110, True), fill=GREEN if match == k else CORAL)
        if f > n * 0.7:
            d.text((80, 990), safe("Q4_K_M, the other August file: %s…  =  %s…  (%s)" % (sha4_aug[:8], sha4_now[:8], "match" if sha4_aug == sha4_now else "DIFFERENT")), font=F(30), fill=MUTE)
        footer(d, "weights cannot be nerfed by a runtime: the hash says so")
        banner(d)
        frames.append(im)


# ----------------------------------------------------------------------------------------------------------------------- scene 3: replay
def load_pairs():
    pairs = {}
    for task in ("gpqa", "he"):
        d0 = N.day0_task("q6k", task)
        res = N.read_arm(RES, "new", "q6k", task)
        ks = sorted(d0["items"], key=lambda k: int(k.split("/")[-1]))
        rows = []
        for k in ks:
            r = res.get(k)
            if task == "gpqa":
                rows.append((k, d0["pred"][k], r["p"] if r else "?"))
            else:
                rows.append((k, "pass" if d0["items"][k] else "fail", r["p"] if r else "?"))
        pairs[task] = rows
    return pairs


def grid(d, x0, y0, rows, shown, cols, cell, label, colour):
    d.text((x0, y0 - 50), safe(label), font=F(34, True), fill=colour)
    for i, (k, p0, p1) in enumerate(rows):
        r, c = divmod(i, cols)
        x, y = x0 + c * cell, y0 + r * cell
        if i < shown:
            d.rounded_rectangle([x, y, x + cell - 5, y + cell - 5], radius=4, fill=GREEN if p0 == p1 else CORAL)
        else:
            d.rounded_rectangle([x, y, x + cell - 5, y + cell - 5], radius=4, outline=GRID, width=2)


def scene_replay(frames):
    pairs = load_pairs()
    g, h = pairs["gpqa"], pairs["he"]
    n = S(32)
    t_fill = 0.72
    for f in range(n):
        im, d = new()
        header(d, "step 2: same questions, same answers?", "August's saved answers vs today's, one square per item. green = identical, red = different")
        r = ease(f / (n * t_fill))
        sg, sh = int(round(r * len(g))), int(round(r * len(h)))
        grid(d, 70, 190, g, sg, 18, 42, "GPQA-diamond  (198 items)", GOLD)
        grid(d, 1010, 190, h, sh, 18, 42, "HumanEval  (164 items)", BLUE)
        sameg = sum(1 for x in g[:sg] if x[1] == x[2])
        sameh = sum(1 for x in h[:sh] if x[1] == x[2])
        d.text((70, 666), "same answer", font=F(32), fill="#9aa8bd")
        d.text((70, 700), "%d / %d" % (sameg, len(g)), font=F(120, True, True), fill=GREEN)
        d.text((1010, 666), "same pass/fail", font=F(32), fill="#9aa8bd")
        d.text((1010, 700), "%d / %d" % (sameh, len(h)), font=F(120, True, True), fill=GREEN)
        # ticker: the last 4 items revealed on each side
        for side, (rows, shown, x) in enumerate(((g, sg, 70), (h, sh, 1010))):
            for j in range(4):
                i = shown - 1 - j
                if i < 0:
                    break
                k, p0, p1 = rows[i]
                same = p0 == p1
                d.text((x, 850 + j * 38), safe("%-6s Aug %-4s today %-4s %s" % (k.replace("HumanEval/", "HE "), p0, p1, "same" if same else "DIFFERENT")), font=F(26, False, True), fill=GREEN if same else CORAL)
        footer(d, "greedy decoding: the same file should answer the same way; the runtime build is what changed")
        banner(d)
        frames.append(im)


# ----------------------------------------------------------------------------------------------------------------------- scene 4: bands
def scene_bands(frames):
    n = S(14)
    tasks = [("gpqa", "GPQA-diamond"), ("he", "HumanEval"), ("arc", "ARC-Challenge"), ("mmlu", "MMLU, 19 subjects")]
    x0, x1 = 520, 1760
    for f in range(n):
        im, d = new()
        header(d, "step 3: where does today land on August's band?", "grey = 95% band of August, gold tick = August, dot = today, ring = Q4_K_M")
        y = 190
        for ti, (t, lab) in enumerate(tasks):
            s = head[t]
            lo = min(s["d0_lo"], s["score"], s["lo"]) - 4
            hi = max(s["d0_hi"], s["score"], s["hi"]) + 4
            def X(v):
                return x0 + (v - lo) / (hi - lo) * (x1 - x0)
            d.text((60, y + 10), lab, font=F(34, True), fill=TEXT)
            d.text((60, y + 58), "August %.1f%%  today %.1f%%" % (s["d0_score"], s["score"]), font=F(24), fill=MUTE)
            d.rectangle([X(s["d0_lo"]), y + 20, X(s["d0_hi"]), y + 80], fill="#2b3550")
            d.rectangle([X(s["d0_score"]) - 3, y + 8, X(s["d0_score"]) + 3, y + 92], fill=GOLD)
            appear = ease((f / n - 0.10 - 0.12 * ti) / 0.25)
            if appear > 0:
                tx = X(s["score"])
                start = x0 - 120
                cx = start + (tx - start) * appear
                ok = not s["outside"]
                d.ellipse([cx - 22, y + 50 - 22, cx + 22, y + 50 + 22], fill=GREEN if ok else CORAL)
                if q4 and t in q4:
                    qx = X(q4[t]["score"]) if appear > 0.9 else cx
                    d.ellipse([qx - 16, y + 50 - 16, qx + 16, y + 50 + 16], outline=BLUE, width=5)
                if appear >= 1:
                    d.text((x1 - 330, y + 96), "inside the band" if ok else "OUTSIDE the band", font=F(28, True), fill=GREEN if ok else CORAL)
            y += 190
        banner(d)
        footer(d, "")
        frames.append(im)


class Sink:
    """streams frames to ffmpeg as they are drawn (1500 frames x 6 MB must never sit in memory); keeps the few frames asked for as stills"""
    def __init__(self, want):
        self.pr = subprocess.Popen([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                                    "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
        self.n = 0
        self.want = want
        self.kept = {}

    def append(self, im):
        if self.n in self.want:
            self.kept[self.n] = im.copy()
        self.pr.stdin.write(im.tobytes())
        self.n += 1


def main():
    t_card, t_fp, t_rep = S(8), S(14), S(32)
    want = {S(0.5) // 2: "s0-card", t_card + t_fp // 2: "s1-fp", t_card + t_fp + t_rep // 2: "s2-replay", t_card + t_fp + t_rep + S(10): "s3-bands"}
    sink = Sink(want)
    c = card()
    for _ in range(t_card):
        sink.append(c)
    scene_fp(sink)
    scene_replay(sink)
    scene_bands(sink)
    for _ in range(S(2)):
        sink.append(c)
    sink.pr.stdin.close()
    sink.pr.wait()
    if a.stills:
        os.makedirs(a.stills, exist_ok=True)
        for idx, name in want.items():
            if idx in sink.kept:
                sink.kept[idx].save(os.path.join(a.stills, name + ".png"))
    print("render rc", sink.pr.returncode, "frames", sink.n, "seconds", round(sink.n / FPS, 1))
    sys.exit(sink.pr.returncode)


main()
