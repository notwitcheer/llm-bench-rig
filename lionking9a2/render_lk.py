#!/usr/bin/env python3
"""render_lk.py (9a2, lion king neurons): the postable clip. CPU only; PIL frames piped to the static ffmpeg.

usage: render_lk.py --summary summary.json --out clip.mp4 [--stills 0.5,30 --stills-dir DIR]
       render_lk.py --mock --out clip.mp4 [--stills ...]   (fake numbers, MOCK banner: render check only)

Shape: result card first (8 s), then the live thing (50 s): prompt, the 12-neuron patch, the answer
share swapping, the greedy answer, the random control, subset and pair counters. ~58 s, 16:9 1920x1080.
Every number is read from summary.json (summarize.py, PREREG.md). Nothing is typed by hand.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 10
CARD_S, LIVE_S = 8.0, 50.0
FFMPEG = str(Path.home() / "voxelbench" / "ffmpeg")
FD = "/usr/share/fonts/truetype/dejavu/"
BG, FG, GREY = (14, 16, 22), (236, 236, 240), (132, 136, 148)
ACC, BLUE, GREEN, RED, DIM = (255, 184, 64), (96, 170, 255), (92, 214, 140), (240, 84, 84), (40, 44, 56)


def F(sz, bold=False):
    return ImageFont.truetype(FD + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)


def seg(t, a, b):
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    return (t - a) / (b - a)


def ease(x):
    return x * x * (3 - 2 * x)


def share(m):
    return 1.0 / (1.0 + math.exp(-m))


def wrap(d, s, font, maxw):
    lines, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def para(d, x, y, s, font, fill, maxw, lh, max_lines=None):
    lines = wrap(d, s, font, maxw)
    if max_lines:
        lines = lines[:max_lines]
    for ln in lines:
        d.text((x, y), ln, font=font, fill=fill)
        y += lh
    return y


def bar(d, x, y, w_full, h, frac, label, color):
    d.rectangle([x, y, x + w_full, y + h], fill=DIM)
    d.rectangle([x, y, x + int(w_full * max(0.0, min(1.0, frac))), y + h], fill=color)
    f = F(int(h * 0.62), True)
    tw = d.textlength(label, font=f)
    inside = w_full * max(0.0, min(1.0, frac)) > tw + 36
    d.text((x + 18, y + h // 2), label, font=f, fill=BG if inside else FG, anchor="lm")


def fake():
    return {
        "mock": True,
        "line1": "MOCK: the lion king's answer flips when we patch 12 neurons from another question, and 2,000 of 4,096 subsets flip it. 900 disjoint pairs flip it too. one chain on qwen3-1.7b-base, the paper's model",
        "prompt": "Q: What is the capital of Italy?\nA: Rome\n\nQ: Who killed the father of the main character of The Lion King?\nA:",
        "headline": {"answer": "Scar", "bridge": "Simba", "m0": 1.9, "noise_P_hop1": 0.12,
                     "full_delta": {"P_hop1": 4.0, "P_random": -0.1}, "margin_patched_full": -2.1,
                     "margin_random_full": 1.8, "gen_clean": "Scar", "gen_patched": "Simba",
                     "P_hop1_flip_subsets": 2000, "P_hop1_disjoint_pairs": 900, "uniqueness": "NOT UNIQUE",
                     "verdict": "REPRODUCED ON ONE CHAIN", "gate": "pass"},
    }


def draw_card(d, t, S):
    h = S["headline"]
    k = ease(seg(t, 0.0, 1.4))
    nflip = int(h["P_hop1_flip_subsets"] * k)
    npair = int(h["P_hop1_disjoint_pairs"] * k)
    m0, noise = h["m0"], h["noise_P_hop1"]
    d.text((80, 50), "9a2 · causal neuron circuits (arXiv 2611.13885) · qwen3-1.7b-base · one chain, one run per arm", font=F(28), fill=GREY)
    para(d, 80, 100, S["line1"], F(46, True), FG, 1740, 58, max_lines=3)
    d.text((80, 430), f"{nflip:,}", font=F(210, True), fill=ACC)
    d.text((80, 660), "of 4,096 subsets flip the answer", font=F(42), fill=FG)
    para(d, 80, 716, "how it was measured: all 4,096 subsets of the 12 neurons, one patched pass each, bf16. flip = margin below 0", F(30), GREY, 820, 38)
    d.text((1000, 430), f"{npair:,}", font=F(210, True), fill=BLUE)
    lab = "disjoint pairs of minimal flipping sets" if h["P_hop1_disjoint_pairs"] > 0 else "no disjoint pair: unique"
    para(d, 1000, 660, lab, F(42), FG, 860, 52)
    para(d, 1000, 740, "how it was measured: minimal flipping sets from the same sweep; a pair is disjoint when it shares no neuron", F(30), GREY, 860, 38)
    d.text((80, 840), f"{m0:.2f}", font=F(120, True), fill=FG)
    para(d, 400, 866, f"clean margin on the lk prompt (logit Scar minus logit Simba). bf16 noise floor {noise:.3f}; gate = 3x noise = {3 * noise:.3f}", F(34), FG, 1440, 44)
    para(d, 400, 956, "how it was measured: unpatched pass; noise = largest change over repeated bf16 runs", F(26), GREY, 1440, 34, max_lines=1)
    para(d, 80, 1012, "claim tested: a 12-neuron group from another question flips the lk answer. limits: one chain, one model, one bf16 run per arm. the example is Goodfire's, cited not reproduced.", F(24), GREY, 1760, 30, max_lines=2)


def draw_live(d, u, S):
    h = S["headline"]
    m0, mp, mr = h["m0"], h["margin_patched_full"], h["margin_random_full"]
    nflip, npair = h["P_hop1_flip_subsets"], h["P_hop1_disjoint_pairs"]
    d.text((80, 40), "the prompt", font=F(32), fill=GREY)
    y = 84
    for ln in S["prompt"].split("\n"):
        if ln.strip():
            y = para(d, 80, y, ln, F(34), FG, 1100, 42)
        else:
            y += 14
    d.text((1250, 40), "the 12-neuron group", font=F(32), fill=GREY)
    for i in range(12):
        x, yy = 1250 + (i % 6) * 100, 96 + (i // 6) * 100
        lit = u >= 4.0 + i * (5.0 / 12.0)
        d.rectangle([x, yy, x + 80, yy + 80], fill=ACC if lit else DIM)
    m = m0 if u < 9 else m0 + (mp - m0) * ease(seg(u, 9, 15))
    d.text((80, 500), "answer vs bridge, two-way share of the logit margin (derived, not full-vocab probabilities)", font=F(30), fill=GREY)
    bar(d, 80, 544, 1080, 100, share(m), f"{h['answer']}  {share(m) * 100:.0f}%", ACC)
    bar(d, 80, 664, 1080, 100, 1 - share(m), f"{h['bridge']}  {(1 - share(m)) * 100:.0f}%", GREEN)
    if u >= 20:
        mc = m0 + (mr - m0) * ease(seg(u, 20, 24))
        d.text((80, 800), "control: random 12 neurons, same layers, same prompt", font=F(30), fill=GREY)
        bar(d, 80, 840, 1080, 64, share(mc), f"{h['answer']}  {share(mc) * 100:.0f}%", ACC)
    if u >= 15:
        d.text((80, 950), "greedy answer, no patch:  " + h["gen_clean"], font=F(48), fill=FG)
    if u >= 17:
        n = int(len(h["gen_patched"]) * ease(seg(u, 17, 20)))
        d.text((80, 1010), "greedy answer, patched:  " + h["gen_patched"][:n], font=F(48), fill=GREEN)
    if u >= 28:
        n = int(nflip * ease(seg(u, 28, 38)))
        d.text((1250, 520), "subsets that flip", font=F(38), fill=GREY)
        d.text((1250, 566), f"{n:,} / 4,096", font=F(92, True), fill=ACC)
        d.rectangle([1250, 690, 1850, 712], fill=DIM)
        d.rectangle([1250, 690, 1250 + int(600 * n / 4096), 712], fill=ACC)
    if u >= 38:
        n = int(npair * ease(seg(u, 38, 46)))
        d.text((1250, 748), "disjoint pairs of minimal sets", font=F(38), fill=GREY)
        d.text((1250, 794), f"{n:,}", font=F(92, True), fill=BLUE)
    if u >= 46:
        para(d, 1250, 900, h["verdict"], F(52, True), ACC, 620, 62, max_lines=2)
        para(d, 1250, 1010, h.get("uniqueness", ""), F(38), FG, 620, 46, max_lines=1)


def frame(t, S):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if t < CARD_S:
        draw_card(d, t, S)
    else:
        draw_live(d, t - CARD_S, S)
    if S.get("mock"):
        d.text((W - 80, 8), "MOCK DATA: render check only", font=F(26, True), fill=RED, anchor="ra")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--stills", default="")
    ap.add_argument("--stills-dir")
    a = ap.parse_args()
    S = fake() if a.mock else json.loads(Path(a.summary).read_text(encoding="utf-8"))
    if not a.mock and S.get("mock"):
        sys.exit("refusing to render a mock summary as data")
    total = int(round((CARD_S + LIVE_S) * FPS))
    stills = [float(x) for x in a.stills.split(",") if x.strip()]
    sdir = Path(a.stills_dir or ".")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for i in range(total):
        img = frame(i / FPS, S)
        for st in stills:
            if i == round(st * FPS):
                sdir.mkdir(parents=True, exist_ok=True)
                img.save(sdir / f"still-{st:05.1f}s.png")
        p.stdin.write(img.tobytes())
    p.stdin.close()
    err = p.stderr.read().decode("utf-8", "replace")
    rc = p.wait()
    if rc != 0:
        sys.exit(f"ffmpeg rc={rc}: {err[:400]}")
    print(f"wrote {a.out}: {total} frames at {FPS} fps = {total / FPS:.0f} s, {W}x{H}, mock={bool(a.mock)}")


if __name__ == "__main__":
    main()
