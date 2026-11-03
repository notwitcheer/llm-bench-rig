#!/usr/bin/env python3
"""r78z_results.py <out.mp4> --left TAKEDIR --right TAKEDIR [--seconds 12] [--concat A.mp4 B.mp4 ... --final FINAL.mp4]

The two finished pages side by side, same camera for both (each page's own animation and default view, no edits to the model's file): built from <TAKEDIR>/gate/clip.mp4 (r78z_gate.py
captures, 1280x720, scaled to 880x495) with one PIL overlay for the labels (no drawtext in the static ffmpeg): file label, finish time, html KB, gate verdict. A take with no page shows a grey 'no page'
tile (shown, not hidden). --concat appends further clips (the end card) after this segment into --final (re-encoded, same size/fps). Writes <out>.png, a still of the segment.
"""
import argparse
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--left", required=True)
ap.add_argument("--right", required=True)
ap.add_argument("--seconds", type=int, default=12)
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/voxelbench/ffmpeg"))
ap.add_argument("--summary", default=None, help="the race clip's .summary.json (finish times)")
ap.add_argument("--race", default=None, help="race clip to put first in --final")
ap.add_argument("--concat", nargs="*", default=[], help="clips appended after the results segment")
ap.add_argument("--final", default=None)
a = ap.parse_args()
FD = "/usr/share/fonts/truetype/dejavu/"
W, H = 1920, 1080
BG, TEXT, MUTE, GRID, BAND = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#121a2a"
COL = {"L": "#6fd6a8", "R": "#e8b64c"}
LAB = {"L": "NVFP4  ·  vLLM", "R": "UD-Q4_K_M  ·  llama.cpp"}
D = {"L": a.left.rstrip("/"), "R": a.right.rstrip("/")}


def F(sz, b=False):
    return ImageFont.truetype(FD + "DejaVuSans" + ("-Bold" if b else "") + ".ttf", sz)


fin, st = {}, {}
if a.summary and os.path.exists(a.summary):
    j = json.load(open(a.summary))
    fin = {"L": j["left"]["finish_s"], "R": j["right"]["finish_s"]}
    st = {"L": j["left"], "R": j["right"]}
TW, TH = 880, 495
POS = {"L": (60, 230), "R": (980, 230)}
ov = Image.new("RGBA", (W, H), BG + "ff")
d = ImageDraw.Draw(ov)
d.rectangle([0, 0, W, 84], fill=BAND)
d.text((48, 8), "same prompt, two results", font=F(34, True), fill=TEXT)
d.text((48, 50), "each page as the agent left it, default view, real speed. gate = liveness check only (loads, animates, no errors), not a quality score.", font=F(20), fill=MUTE)
d.text((48, H - 40), "two takes, one per file, start synced. a picture, not a measurement.", font=F(22), fill=MUTE)
inputs, flt = [], []
for n, k in enumerate("LR"):
    x, y = POS[k]
    g = {}
    gp = "%s/gate/gate.json" % D[k]
    if os.path.exists(gp):
        g = json.load(open(gp))
    kb = "%.0f KB" % (g.get("bytes", 0) / 1024.0) if g.get("bytes") else "no file"
    verdict = "gate pass" if g.get("pass") else ("gate FAIL" if g else "no gate")
    t = ("%d:%02d" % (int(fin[k] // 60), int(fin[k] % 60))) if k in fin else ""
    d.text((x, y - 84), LAB[k], font=F(32, True), fill=COL[k])
    d.text((x, y - 42), "   ".join(s for s in (("finished " + t) if t else "", kb, verdict) if s), font=F(24), fill=TEXT)
    d.rectangle([x - 4, y - 4, x + TW + 3, y + TH + 3], outline=COL[k], width=3, fill="#00000000")
    if k in st:
        s_ = st[k]
        d.text((x, y + TH + 34), "%s tokens generated" % format(int(s_["tokens_server"] or 0), ","), font=F(30, True), fill=TEXT)
        d.text((x, y + TH + 78), "%s tok/s while generating (server count)" % (s_["tok_s_mean_while_generating"] if s_["tok_s_mean_while_generating"] is not None else "n/a"), font=F(24), fill=COL[k])
        d.text((x, y + TH + 114), "one take, ended: %s" % s_["end"], font=F(20), fill=MUTE)
    cl = "%s/gate/clip.mp4" % D[k]
    if os.path.exists(cl):
        inputs += ["-stream_loop", "-1", "-t", str(a.seconds), "-i", cl]
    else:
        inputs += ["-f", "lavfi", "-t", str(a.seconds), "-i", "color=c=0x1b2230:s=%dx%d:r=25" % (TW, TH)]
    flt.append("[%d:v]fps=25,scale=%d:%d,setsar=1[v%d]" % (n, TW, TH, n))
ovp = a.out + ".overlay.png"
px = ov.load()
for k in "LR":
    x, y = POS[k]
    for yy in range(y, y + TH):
        for xx in range(x, x + TW):
            px[xx, yy] = (0, 0, 0, 0)
ov.save(ovp)
graph = ";".join(flt) + ";color=c=0x0e1420:s=%dx%d:r=25:d=%d[bg]" % (W, H, a.seconds)
prev = "bg"
for i, k in enumerate("LR"):
    x, y = POS[k]
    graph += ";[%s][v%d]overlay=%d:%d[o%d]" % (prev, i, x, y, i)
    prev = "o%d" % i
graph += ";[%s][2:v]overlay=0:0:format=auto,format=yuv420p[out]" % prev
cmd = [a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + inputs + ["-loop", "1", "-t", str(a.seconds), "-i", ovp, "-filter_complex", graph, "-map", "[out]",
       "-t", str(a.seconds), "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-r", "25", "-movflags", "+faststart", a.out]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print("results segment rc", r.returncode, r.stderr[-400:])
if r.returncode == 0:
    subprocess.run([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", "4", "-i", a.out, "-frames:v", "1", a.out + ".png"], capture_output=True)
if r.returncode == 0 and a.final:
    parts = ([a.race] if a.race else []) + [a.out] + list(a.concat)
    ins = []
    for p_ in parts:
        ins += ["-i", p_]
    fc = "".join("[%d:v]setsar=1,fps=25[x%d];" % (i, i) for i in range(len(parts))) + "".join("[x%d]" % i for i in range(len(parts))) + "concat=n=%d:v=1:a=0,format=yuv420p[v]" % len(parts)
    r2 = subprocess.run([a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + ins + ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                         "-movflags", "+faststart", a.final], capture_output=True, text=True, timeout=1200)
    print("concat rc", r2.returncode, r2.stderr[-300:])
raise SystemExit(r.returncode)
