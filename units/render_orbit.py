#!/usr/bin/env python3
"""orbit ring renderer (CPU only: numpy + PIL + ffmpeg). 16:9 1920x1080, 25 fps. Runs ON capsule (never on mercury), after the GPU leg, from the saved results.

what is on screen (all of it real data from <res>/<model>.jsonl, classified by ucommon.classify):
  * a big ring = the value the tool's schema wants (the expected argument). distance from the ring is a LOG axis: one faint circle = x10. out = too big, in = too small.
  * four sectors = the task's four forms (canonical top, sub-unit right, larger unit bottom, words left); the text at the rim is the quantity as the prompt wrote it.
  * hollow ghost = where the NUMBER IN THE PROMPT sits on that axis. each model's marker starts on the ghost and travels to the value it set (visual metaphor, labelled in the caption).
  * a correct call lands on the ring. a magnitude slip lands off it: red halo and a "x1000" / "/100" tag. no value (no call, malformed) = grey cross at the rim.
  * every landed call leaves a dot, so the ring fills with all calls of the run; the right column keeps the live slip count per model.
  * order: the task with the most slips first (frame 1 is already a slip), the other slip tasks spread evenly through the clean ones so the count keeps climbing. disclosed on screen.
  * the last seconds put the twist on screen as ONE big number (slips / calls) over the still-moving ring, with line 1 (from score.py) as the caption.
usage: render_orbit.py --res DIR --out out.mp4 [--mock] [--seconds-per-task 1.5] [--scale 1.0] [--max-tasks N] [--frame-dump DIR --dump-every S] [--sprites sprites.png]
needs: python3 with numpy + PIL, ffmpeg (-nostdin, stdin from /dev/null)."""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
import ucommon as C  # noqa: E402

W, H, FPS = 1920, 1080, 25
CX, CY = 700, 540
R0, STEP, CLAMP = 320.0, 58.0, 3.3
PANEL_X = 1470
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SECT_ANG = {"canonical": 90.0, "sub": 0.0, "larger": 270.0, "words": 180.0}
OFFS = [-26.0, -13.0, 0.0, 13.0, 26.0]
COLS = [(255, 196, 0), (255, 62, 165), (40, 224, 255), (124, 255, 77), (166, 120, 255)]
SHAPES = ["circle", "diamond", "triangle", "square", "hex"]
SHORT = {"qwen38-27b-q6k": "Qwen3.8 27B", "ornith15-35b-q4km": "Ornith 35B", "gpt-oss-20b-mxfp4": "gpt-oss 20B*", "qwen35-9b-q4km": "Qwen3.5 9B Q4", "qwen35-9b-q8_0": "Qwen3.5 9B Q8"}
RED = (255, 60, 50)
BG = (8, 9, 16)


def font(path, size):
    return ImageFont.truetype(path, size)


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def radius_of(logr):
    return R0 + STEP * max(-CLAMP, min(CLAMP, logr))


def pos(role, mi, logr, wobble=0.0):
    ang = math.radians(SECT_ANG[role] + OFFS[mi] + wobble)
    r = radius_of(logr)
    return CX + r * math.cos(ang), CY - r * math.sin(ang)


def poly(shape, x, y, r):
    if shape == "diamond":
        return [(x, y - r * 1.25), (x + r * 1.1, y), (x, y + r * 1.25), (x - r * 1.1, y)]
    if shape == "triangle":
        return [(x, y - r * 1.2), (x + r * 1.15, y + r * 0.85), (x - r * 1.15, y + r * 0.85)]
    if shape == "square":
        return [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    if shape == "hex":
        return [(x + r * 1.15 * math.cos(math.radians(60 * i)), y + r * 1.15 * math.sin(math.radians(60 * i))) for i in range(6)]
    return None


def draw_marker(d, shape, x, y, r, col, rim=(255, 255, 255), width=3):
    if shape == "circle":
        d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=rim, width=width)
    else:
        d.polygon(poly(shape, x, y, r), fill=col, outline=rim, width=width)


def logratio(c):
    if c["value"] is None:
        return None
    v = Fraction(c["value"])
    e = Fraction(c["expected"])
    if v <= 0:
        return -CLAMP
    return math.log10(float(v / e))


def load(res, mods_path, spec):
    mods = C.load_models(mods_path)
    legs = mods["legs"]
    per = {}
    for mi, m in enumerate(legs):
        p = os.path.join(res, m["key"] + ".jsonl")
        rows = {}
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    rows[(r["task"], r["role"])] = r
        per[m["key"]] = rows
    calls = {}
    for t in spec["tasks"]:
        for f in t["forms"]:
            for mi, m in enumerate(legs):
                r = per[m["key"]].get((t["id"], f["role"]))
                if r is None:
                    continue
                c = C.classify(t, f, {"n_calls": r["n_calls"], "name": r["name"], "args_raw": r["args_raw"], "finish_reason": r.get("finish_reason")})
                c["expected"] = t["expected"]
                c["lr"] = logratio(c)
                calls[(t["id"], f["role"], mi)] = c
    return legs, calls


def stated_lr(t, f):
    return math.log10(abs(float(Fraction(f["num"]) / Fraction(t["expected"]))))


def tag(c):
    k = c["k"]
    return ("x%d" % 10 ** (-k)) if k < 0 else ("/%d" % 10 ** k)


class Scene:
    def __init__(self, spec, legs, calls, args):
        self.spec, self.legs, self.calls, self.a = spec, legs, calls, args
        slips = {t["id"]: sum(1 for (tid, _r, _m), c in calls.items() if tid == t["id"] and c["cat"] == "magnitude_slip") for t in spec["tasks"]}
        tasks = [t for t in spec["tasks"] if any(k[0] == t["id"] for k in calls)]
        hot = sorted([t for t in tasks if slips[t["id"]]], key=lambda t: (-slips[t["id"]], t["id"]))
        cold = sorted([t for t in tasks if not slips[t["id"]]], key=lambda t: t["id"])
        # worst task first (frame 1 is already a slip), the other slip tasks spread evenly through the clean ones so the story keeps building
        order, n = [], len(tasks)
        if hot:
            slots = {0: hot[0]}
            rest = hot[1:]
            for i, t in enumerate(rest):
                slots[int(round((i + 1) * (n - 1) / float(len(rest) + 1)))] = t
            ci = 0
            for pos_ in range(n):
                if pos_ in slots:
                    order.append(slots[pos_])
                elif ci < len(cold):
                    order.append(cold[ci])
                    ci += 1
            used = {t["id"] for t in order}
            order += [t for t in hot + cold if t["id"] not in used]
        else:
            order = cold
        self.tasks = order
        if args.max_tasks:
            self.tasks = self.tasks[:args.max_tasks]
        self.n_task_frames = int(round(FPS * args.seconds_per_task))
        self.finale = int(round(FPS * args.finale_seconds))
        self.total = len(self.tasks) * self.n_task_frames + self.finale
        self.fb, self.fr = font(FONT_B, 40), font(FONT_R, 28)
        self.f_big, self.f_mid, self.f_sm = font(FONT_B, 54), font(FONT_B, 44), font(FONT_R, 26)
        self.f_huge = font(FONT_B, 330)
        self.persist = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.pd = ImageDraw.Draw(self.persist)
        self.landed = set()
        self.counts = [{"slip": 0, "n": 0} for _ in legs]
        self.worst = None
        self.base = self._base()
        shown = {t["id"] for t in self.tasks}
        allc = [c for k, c in calls.items() if k[0] in shown]
        self.total_calls = len(allc)
        self.total_slips = sum(1 for c in allc if c["cat"] == "magnitude_slip")
        ag = S.aggregate(args.res, args.models, spec)
        self.line1 = ag["line1"]
        self.worst_slip = ag["worst_slip"]
        self.overall = ag["overall"]

    def _base(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for i in range(60):                                           # soft vignette
            a = i / 60.0
            col = tuple(int(BG[j] + 18 * (1 - a) * (1 if j == 2 else 0.5)) for j in range(3))
            rr = 900 - i * 13
            d.ellipse([CX - rr, CY - rr, CX + rr, CY + rr], fill=col)
        for k in range(-3, 4):
            if k == 0:
                continue
            r = radius_of(k)
            d.ellipse([CX - r, CY - r, CX + r, CY + r], outline=(70, 78, 112), width=2)
            lab = ("x%d" % 10 ** k) if k > 0 else ("/%d" % 10 ** -k)
            x, y = CX + r * math.cos(math.radians(45)), CY - r * math.sin(math.radians(45))
            fl_ = font(FONT_B, 30)
            tw = d.textlength(lab, font=fl_)
            d.rectangle([x - 4, y - 20, x + 10 + tw, y + 16], fill=BG)
            d.text((x + 1, y - 21), lab, font=fl_, fill=(180, 190, 238))
        return img

    def task_state(self, f):
        ti, lf = divmod(f, self.n_task_frames)
        return ti, lf

    def land(self, t, upto_task_index, upto_local):
        pass

    def frame(self, f):
        a = self.a
        in_finale = f >= len(self.tasks) * self.n_task_frames
        ti = min(f // self.n_task_frames, len(self.tasks) - 1)
        lf = self.n_task_frames if in_finale else f - ti * self.n_task_frames
        u = lf / float(self.n_task_frames)
        t = self.tasks[ti]
        # persistent landing: every task before ti has landed; task ti lands when u>=0.72
        for k in range(ti + (0 if not in_finale else 1) + 0):
            self._land(self.tasks[k])
        if in_finale:
            self._land(self.tasks[-1])
        elif u >= 0.72:
            self._land(t)
        frame = self.base.copy()
        frame.paste(self.persist, (0, 0), self.persist)
        glow = Image.new("RGB", (W // 2, H // 2), (0, 0, 0))
        gd = ImageDraw.Draw(glow)
        pulse = 0.5 + 0.5 * math.sin(f / 7.0)
        gd.ellipse([(CX - R0) / 2, (CY - R0) / 2, (CX + R0) / 2, (CY + R0) / 2], outline=(int(70 + 60 * pulse), int(140 + 60 * pulse), 255), width=7)
        sharp = ImageDraw.Draw(frame)
        sharp.ellipse([CX - R0, CY - R0, CX + R0, CY + R0], outline=(190, 225, 255), width=6)
        self._rim_text(sharp, t, in_finale)
        # markers of the current task
        if not in_finale:
            for f_ in t["forms"]:
                role = f_["role"]
                slr = stated_lr(t, f_)
                gx, gy = pos(role, 2, slr)
                alpha = ease(u / 0.12)
                if alpha > 0:
                    r = 19
                    sharp.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(240, 240, 240), width=4)
                    gd.ellipse([(gx - 22) / 2, (gy - 22) / 2, (gx + 22) / 2, (gy + 22) / 2], outline=(60, 60, 70), width=3)
                for mi, m in enumerate(self.legs):
                    c = self.calls.get((t["id"], role, mi))
                    if c is None:
                        continue
                    col = COLS[mi % 5]
                    sx, sy = pos(role, mi, slr)
                    sx, sy = sx + 0.0, sy + 0.0
                    mu = ease((u - 0.14) / 0.50)
                    if c["lr"] is None:
                        ex, ey = pos(role, mi, CLAMP + 0.05)
                        shape_x = ex
                        shape_y = ey
                        px, py = sx + (ex - sx) * mu, sy + (ey - sy) * mu
                        if u > 0.14:
                            sharp.line([px - 11, py - 11, px + 11, py + 11], fill=(150, 150, 160), width=5)
                            sharp.line([px - 11, py + 11, px + 11, py - 11], fill=(150, 150, 160), width=5)
                        continue
                    lr = c["lr"]
                    tx, ty = pos(role, mi, lr)
                    px, py = sx + (tx - sx) * mu, sy + (ty - sy) * mu
                    if u > 0.1:
                        # trail
                        for j in range(1, 7):
                            mj = ease((u - 0.14 - 0.012 * j) / 0.50)
                            qx, qy = sx + (tx - sx) * mj, sy + (ty - sy) * mj
                            fade = max(0.0, 1 - j / 7.0)
                            gd.ellipse([(qx - 10 * fade) / 2, (qy - 10 * fade) / 2, (qx + 10 * fade) / 2, (qy + 10 * fade) / 2], fill=tuple(int(x * 0.5 * fade) for x in col))
                    slip = c["cat"] == "magnitude_slip"
                    if slip and mu > 0.05:
                        sharp.line([sx, sy, px, py], fill=(255, 90, 80), width=3)
                    rr = 21
                    gcol = RED if slip and mu > 0.95 else col
                    gd.ellipse([(px - 22) / 2, (py - 22) / 2, (px + 22) / 2, (py + 22) / 2], fill=tuple(int(x * 0.8) for x in gcol))
                    draw_marker(sharp, SHAPES[mi % 5], px, py, rr, col)
                    if slip and mu > 0.9:
                        k = (u - 0.64) / 0.3
                        rad = 30 + 110 * max(0.0, min(1.0, k))
                        al = max(0.0, 1.0 - max(0.0, k))
                        gd.ellipse([(px - rad) / 2, (py - rad) / 2, (px + rad) / 2, (py + rad) / 2], outline=tuple(int(x * al) for x in RED), width=6)
                        sharp.text((px + 26, py - 66), tag(c), font=font(FONT_B, 64), fill=RED, stroke_width=5, stroke_fill=(0, 0, 0))
                    elif c["cat"] == "other_value" and mu > 0.95:
                        sharp.text((px + 20, py - 40), "?", font=self.fb, fill=(255, 190, 80), stroke_width=4, stroke_fill=(0, 0, 0))
        gl = glow.filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)
        frame = ImageChops.add(frame, gl)
        d = ImageDraw.Draw(frame)
        self._panel(d, t, ti, in_finale, f)
        if a.mock:
            d.rectangle([0, 0, 640, 46], fill=(140, 0, 0))
            d.text((12, 6), "MOCK DATA: scripted answers, no model", font=font(FONT_B, 26), fill=(255, 255, 255))
        return frame

    def _land(self, t):
        if t["id"] in self.landed:
            return
        self.landed.add(t["id"])
        for f_ in t["forms"]:
            for mi, m in enumerate(self.legs):
                c = self.calls.get((t["id"], f_["role"], mi))
                if c is None:
                    continue
                self.counts[mi]["n"] += 1
                if c["cat"] == "magnitude_slip":
                    self.counts[mi]["slip"] += 1
                if c["lr"] is None:
                    continue
                x, y = pos(f_["role"], mi, c["lr"], wobble=((int(t["id"][1:]) * 7 + mi * 3 + len(f_["role"])) % 7 - 3) * 0.9)
                col = COLS[mi % 5]
                slip = c["cat"] == "magnitude_slip"
                r = 8 if not slip else 18
                self.pd.ellipse([x - r, y - r, x + r, y + r], fill=col + ((150 if not slip else 255),), outline=(RED + (255,)) if slip else None, width=2 if not slip else 4)

    def _rim_text(self, d, t, in_finale):
        """four numbered badges in the middle of the ring, one per form, on the same compass axes as the form's markers; the panel lists the prompt wording under the same numbers."""
        if in_finale:
            return
        fnt = font(FONT_B, 34)
        for n, role in enumerate(C.ROLES, 1):
            ang = math.radians(SECT_ANG[role])
            x, y = CX + 62 * math.cos(ang), CY - 62 * math.sin(ang)
            d.ellipse([x - 24, y - 24, x + 24, y + 24], fill=(14, 16, 30), outline=(235, 238, 250), width=3)
            d.text((x - 10, y - 21), str(n), font=fnt, fill=(255, 255, 255))

    def _wrap_px(self, d, text, fnt, maxw):
        words, lines, cur = text.split(" "), [], ""
        for w in words:
            t2 = (cur + " " + w).strip()
            if d.textlength(t2, font=fnt) <= maxw or not cur:
                cur = t2
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def _panel(self, d, t, ti, in_finale, f):
        d.rectangle([PANEL_X - 24, 0, W, H], fill=(10, 11, 22))
        d.line([PANEL_X - 24, 0, PANEL_X - 24, H], fill=(60, 66, 100), width=3)
        x, mw = PANEL_X, W - PANEL_X - 40
        if in_finale:
            big = str(self.total_slips)
            colr = RED if self.total_slips else (110, 255, 160)
            d.text((x, 36), "magnitude slips", font=font(FONT_B, 40), fill=(225, 228, 248))
            fh = font(FONT_B, 300 if len(big) < 3 else 220)
            d.text((x - 6, 84), big, font=fh, fill=colr, stroke_width=3, stroke_fill=(0, 0, 0))
            d.text((x, 410), "in %d tool calls" % self.total_calls, font=font(FONT_B, 36), fill=(225, 228, 248))
            y = 480
            for mi, m in enumerate(self.legs):
                if self.counts[mi]["n"] == 0:
                    continue
                draw_marker(d, SHAPES[mi % 5], x + 16, y + 22, 13, COLS[mi % 5])
                d.text((x + 44, y + 2), SHORT.get(m["key"], m["label"]), font=font(FONT_B, 26), fill=(235, 235, 245))
                d.text((x + 360, y - 6), str(self.counts[mi]["slip"]), font=font(FONT_B, 42), fill=RED if self.counts[mi]["slip"] else (130, 140, 180))
                y += 52
            cap = (self.line1 or ("verdict: %s" % self.overall)).split(" (")[0]
            cap = cap[:1].upper() + cap[1:]
            yy = y + 20
            fc = font(FONT_R, 24)
            for l in self._wrap_px(d, cap, fc, mw)[:7]:
                d.text((x, yy), l, font=fc, fill=(215, 218, 238))
                yy += 30
            return
        real = t["real"]
        ft = font(FONT_B, 30)
        yy = 30
        for l in self._wrap_px(d, "%s.%s" % (t["tool"], t["arg"]), ft, mw):
            d.text((x, yy), l, font=ft, fill=(255, 255, 255))
            yy += 36
        d.text((x, yy + 2), ("REAL Hermes schema" if real else "SYNTHETIC tool") + " | " + t["unit"], font=font(FONT_B, 24), fill=(120, 220, 255) if real else (255, 170, 90))
        yy += 52
        d.text((x, yy), "asked for", font=font(FONT_R, 26), fill=(150, 160, 200))
        yy += 36
        fa = font(FONT_B, 46)
        for l in self._wrap_px(d, t["ask"], fa, mw):
            d.text((x, yy), l, font=fa, fill=(255, 255, 255))
            yy += 54
        yy += 14
        d.text((x, yy), "the prompt said", font=font(FONT_R, 24), fill=(150, 160, 200))
        yy += 32
        fp = font(FONT_B, 27)
        for n, f_ in enumerate(t["forms"], 1):
            d.ellipse([x, yy + 2, x + 30, yy + 32], outline=(235, 238, 250), width=2)
            d.text((x + 8, yy + 2), str(n), font=font(FONT_B, 22), fill=(255, 255, 255))
            ls = self._wrap_px(d, f_["text"], fp, mw - 44)
            for l in ls[:2]:
                d.text((x + 42, yy), l, font=fp, fill=(240, 242, 252))
                yy += 31
            yy += 6
        yy = max(yy + 18, 470)
        tot = sum(c["slip"] for c in self.counts)
        # the running total sits on its own row under the forms (it used to reach 62 px up into the last form's text)
        d.text((x, yy + 34), "slips so far", font=font(FONT_R, 24), fill=(150, 160, 200))
        d.text((x + 200, yy), str(tot), font=font(FONT_B, 80), fill=RED if tot else (130, 140, 180), stroke_width=2, stroke_fill=(0, 0, 0))
        yy += 96
        for mi, m in enumerate(self.legs):
            if not any(k[2] == mi for k in self.calls):
                continue
            draw_marker(d, SHAPES[mi % 5], x + 16, yy + 24, 14, COLS[mi % 5])
            d.text((x + 44, yy + 6), SHORT.get(m["key"], m["label"]), font=font(FONT_B, 26), fill=(235, 235, 245))
            n = self.counts[mi]["slip"]
            d.text((x + 360, yy - 4), str(n), font=font(FONT_B, 52), fill=RED if n else (130, 140, 180))
            yy += 62
        yy = max(yy + 6, 840)
        fl = font(FONT_R, 22)
        for l in ["bright ring: the schema's value", "faint rings: x10 each", "hollow: the number as written", "task %d/%d, worst slip first" % (ti + 1, len(self.tasks)), "* gpt-oss: low effort, not off"]:
            for l2 in self._wrap_px(d, l, fl, mw):
                d.text((x, yy), l2, font=fl, fill=(170, 180, 220))
                yy += 27


def sprites(out):
    img = Image.new("RGB", (1400, 420), BG)
    d = ImageDraw.Draw(img)
    for mi, shape in enumerate(SHAPES):
        for si, r in enumerate((16, 30, 56)):
            draw_marker(d, shape, 120 + mi * 260, 60 + si * 110 if si < 2 else 250, r, COLS[mi])
        d.text((60 + mi * 260, 385), shape, font=font(FONT_B, 26), fill=(255, 255, 255))
    img.save(out)


def ffmpeg_bin():
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(p):
            return p
    g = glob.glob("/home/witcheer/Pollen/*/*/.venv/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*") + glob.glob("/home/witcheer/Pollen/*/ffvenv/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*")
    return g[0] if g else "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--out")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seconds-per-task", type=float, default=1.5)
    ap.add_argument("--finale-seconds", type=float, default=6.0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--frame-dump")
    ap.add_argument("--dump-every", type=float, default=4.0)
    ap.add_argument("--sprites")
    a = ap.parse_args()
    if a.sprites:
        sprites(a.sprites)
        print("sprites ->", a.sprites)
        return 0
    spec = C.load_tasks()
    legs, calls = load(a.res, a.models, spec)
    sc = Scene(spec, legs, calls, a)
    print("[render] %d tasks, %d frames (%.1f s), %d calls, %d slips" % (len(sc.tasks), sc.total, sc.total / FPS, sc.total_calls, sc.total_slips), flush=True)
    if a.frame_dump:
        os.makedirs(a.frame_dump, exist_ok=True)
        step = max(1, int(a.dump_every * FPS))
        for f in range(sc.total):
            fr = sc.frame(f)                                  # sequential: the persistent layer needs every earlier frame
            if f % step == 0 or f == sc.total - 1:
                fr.save(os.path.join(a.frame_dump, "f%05d.png" % f))
        print("[render] frames dumped to", a.frame_dump)
        return 0
    cmd = [ffmpeg_bin(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(sc.total):
        p.stdin.write(sc.frame(f).convert("RGB").tobytes())
        if f % 250 == 0:
            print("[render] frame %d/%d" % (f, sc.total), flush=True)
    p.stdin.close()
    rc = p.wait()
    print("[render] ffmpeg rc", rc, "->", a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
