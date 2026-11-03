#!/usr/bin/env python3
"""four_results.py <rundir> <out.mp4> [--seconds 12] [--ffmpeg PATH] [--concat RACE.mp4 --final FINAL.mp4]

The four finished pages side by side, same camera for all (each page's own animation, its own default view, no edits to the model's file): a 2x2 grid
1920x1080 built from <rundir>/gate/<a-d>/clip.mp4 (four_gate.py captures, 1280x720, scaled to 768x432 each) with one PIL overlay for the labels (no drawtext in
the static ffmpeg): agent letter, finish time, html KB, gate verdict. A page with no clip shows a grey 'no page' tile (an agent that never wrote its file is shown, not hidden).
--concat RACE.mp4 --final FINAL.mp4 appends this segment to the race clip (re-encoded, same size/fps).
Also writes <out>.png, a still of the mosaic (frame at 4 s) for a contact-sheet read.
"""
import argparse
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--seconds", type=int, default=12)
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--concat", default=None)
ap.add_argument("--final", default=None)
ap.add_argument("--summary", default=None, help="the race clip's .summary.json (finish times)")
a = ap.parse_args()
FD = "/usr/share/fonts/truetype/dejavu/"
W, H = 1920, 1080
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a"
COL = {"a": "#6fd6a8", "b": "#e8b64c", "c": "#7aa7e8", "d": "#c792ea"}


def F(sz, b=False):
    return ImageFont.truetype(FD + "DejaVuSans" + ("-Bold" if b else "") + ".ttf", sz)


run = a.run.rstrip("/")
fin = {}
if a.summary and os.path.exists(a.summary):
    fin = json.load(open(a.summary)).get("finish_s", {})
TW, TH = 768, 432
POS = {"a": (128, 136), "b": (1024, 136), "c": (128, 630), "d": (1024, 630)}
ov = Image.new("RGBA", (W, H), BG + "ff")
d = ImageDraw.Draw(ov)
d.rectangle([0, 0, W, 84], fill=BAND)
d.text((48, 8), "same prompt, four results", font=F(34, True), fill=TEXT)
d.text((48, 50), "each page as the agent left it, default view, real speed", font=F(20), fill=MUTE)
inputs, flt, n = [], [], 0
for ag in "abcd":
    x, y = POS[ag]
    g = {}
    gp = "%s/gate/%s/gate.json" % (run, ag)
    if os.path.exists(gp):
        g = json.load(open(gp))
    kb = "%.0f KB" % (g.get("bytes", 0) / 1024.0) if g.get("bytes") else "no file"
    verdict = "gate pass" if g.get("pass") else ("gate FAIL" if g else "no gate")
    t = ("%d:%02d" % (int(fin[ag] // 60), int(fin[ag] % 60))) if ag in fin else ""
    d.text((x, y - 46), "agent %s" % ag.upper(), font=F(30, True), fill=COL[ag])
    d.text((x + 170, y - 42), "  ".join(s for s in (t, kb, verdict) if s), font=F(24), fill=TEXT)
    d.rectangle([x - 4, y - 4, x + TW + 3, y + TH + 3], outline=COL[ag], width=3, fill="#00000000")
    cl = "%s/gate/%s/clip.mp4" % (run, ag)
    if os.path.exists(cl):
        inputs += ["-stream_loop", "-1", "-t", str(a.seconds), "-i", cl]
    else:
        inputs += ["-f", "lavfi", "-t", str(a.seconds), "-i", "color=c=0x1b2230:s=%dx%d:r=25" % (TW, TH)]
    flt.append("[%d:v]fps=25,scale=%d:%d,setsar=1[v%d]" % (n, TW, TH, n))
    n += 1
ov.save(a.out + ".overlay.png")
# frames of the four tiles sit under the overlay's transparent boxes: build the grid first (black), then overlay with the boxes punched out
graph = ";".join(flt) + ";"
# place each tile at its position using pad+overlay chain (xstack with fixed offsets wants equal tiles: use overlay on a colour canvas)
graph += "color=c=0x0e1420:s=%dx%d:r=25:d=%d[bg]" % (W, H, a.seconds)
prev = "bg"
for i, ag in enumerate("abcd"):
    x, y = POS[ag]
    graph += ";[%s][v%d]overlay=%d:%d[o%d]" % (prev, i, x, y, i)
    prev = "o%d" % i
graph += ";[%s][4:v]overlay=0:0:format=auto,format=yuv420p[out]" % prev
# the overlay png has an opaque background: punch the four tile rectangles transparent first
ovp = Image.open(a.out + ".overlay.png")
px = ovp.load()
for ag in "abcd":
    x, y = POS[ag]
    for yy in range(y, y + TH):
        for xx in range(x, x + TW):
            px[xx, yy] = (0, 0, 0, 0)
ovp.save(a.out + ".overlay.png")
cmd = [a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + inputs + ["-loop", "1", "-t", str(a.seconds), "-i", a.out + ".overlay.png",
       "-filter_complex", graph, "-map", "[out]", "-t", str(a.seconds), "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-r", "25", "-movflags", "+faststart", a.out]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print("results mosaic rc", r.returncode, r.stderr[-400:])
if r.returncode == 0:
    subprocess.run([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", "4", "-i", a.out, "-frames:v", "1", a.out + ".png"], capture_output=True)
if r.returncode == 0 and a.concat and a.final:
    r2 = subprocess.run([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", a.concat, "-i", a.out, "-filter_complex",
                         "[0:v]setsar=1,fps=25[x];[1:v]setsar=1,fps=25[y];[x][y]concat=n=2:v=1:a=0,format=yuv420p[v]", "-map", "[v]", "-c:v", "libx264", "-preset", "medium",
                         "-crf", "21", "-movflags", "+faststart", a.final], capture_output=True, text=True, timeout=900)
    print("concat rc", r2.returncode, r2.stderr[-300:])
raise SystemExit(r.returncode)
