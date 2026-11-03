#!/usr/bin/env python3
"""clover_trace_render.py <err.txt> <out.mp4> [--ffmpeg PATH] : draws a CLOVERTRACE log (llama-speculative-simple built with
the local CLOVER_TRACE patch) as a clip. each verify step: the whole drafted block appears at once (grey), then the causal pass
resolves it: accepted tokens turn white, the bonus token cyan, rejected draft tokens flash red and drop. replay steps (checkpoint
re-verify after a rejection) only commit. the picture is the model's real draft vs accept log, token for token; it is NOT a
denoising-step capture (the llama.cpp path drafts a block in ONE bidirectional pass, so there are no intermediate noise states)."""
import json, sys, subprocess, os, shutil, textwrap
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
ff = sys.argv[sys.argv.index("--ffmpeg") + 1] if "--ffmpeg" in sys.argv else "ffmpeg"
steps = []
for l in open(src, errors="replace"):
    if "CLOVERTRACE " in l:
        try: steps.append(json.loads(l.split("CLOVERTRACE ", 1)[1]))
        except Exception: pass
if not steps: print("no CLOVERTRACE lines"); sys.exit(2)
W, H = 1280, 720
fp = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf") if os.path.exists(p)), None)
F = ImageFont.truetype(fp, 24) if fp else ImageFont.load_default(); FS = ImageFont.truetype(fp, 20) if fp else F
COLS = 84; MAXL = 20
fd = os.path.splitext(out)[0] + "-frames"; shutil.rmtree(fd, ignore_errors=True); os.makedirs(fd)
n = 0
def frame(segs, hud):
    """segs = [(text, colour)], wrapped on a 84 column grid, last MAXL lines kept."""
    global n
    im = Image.new("RGB", (W, H), (14, 16, 22)); d = ImageDraw.Draw(im)
    lines = [[]]; col = 0
    for t, c in segs:
        for ch in t:
            if ch == "\n" or col >= COLS: lines.append([]); col = 0
            if ch != "\n": lines[-1].append((ch, c)); col += 1
    for i, ln in enumerate(lines[-MAXL:]):
        x = 30
        for ch, c in ln: d.text((x, 40 + i * 30), ch, font=F, fill=c); x += 14.4
    d.rectangle((0, H - 60, W, H), fill=(28, 32, 44)); d.text((30, H - 44), hud, font=FS, fill=(170, 180, 200))
    im.save(f"{fd}/f{n:05d}.png"); n += 1
WHITE, GREY, CYAN, RED = (235, 235, 240), (110, 118, 135), (90, 220, 255), (255, 90, 90)
toks = {}; total_acc = 0; total_draft = 0
def committed(upto): return "".join(toks.get(i, "") for i in range(upto))
# positions are token indices; a step's out[] starts at pos
for k, s in enumerate(steps, 1):
    pos, dr, o, acc = s["pos"], s["draft"], s["out"], s["accepted"]
    pre = committed(pos)
    hud = f"step {k}/{len(steps)}  block {len(dr)}  accepted {acc}" + ("  replay" if s["replay"] else "")
    if not s["replay"] and dr:
        for _ in range(7): frame([(pre, WHITE), ("".join(dr), GREY)], hud + "  | drafted in one pass")
        res = [("".join(o[:acc]), WHITE), ("".join(o[acc:]), CYAN)]
        rej = "".join(dr[acc:])
        for _ in range(5): frame([(pre, WHITE)] + res + ([(rej, RED)] if rej else []), hud + "  | white ok, cyan bonus, red rejected")
        total_acc += acc; total_draft += len(dr)
    for i, t in enumerate(o): toks[pos + i] = t
    for _ in range(2): frame([(committed(pos + len(o)), WHITE)], hud)
for _ in range(24): frame([(committed(max(toks) + 1), WHITE)], f"done: {len(steps)} verify steps, {total_acc} of {total_draft} drafted tokens accepted")
r = subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "12", "-i", f"{fd}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", out], capture_output=True, text=True)
print("frames", n, "steps", len(steps), "accepted", total_acc, "of", total_draft, "ffmpeg rc", r.returncode, r.stderr[:200])
shutil.rmtree(fd, ignore_errors=True)
sys.exit(r.returncode)
