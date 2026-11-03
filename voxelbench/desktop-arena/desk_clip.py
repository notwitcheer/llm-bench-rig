#!/usr/bin/env python3
"""desk_clip.py: split-screen desktop-arena clip from SEPARATE captures (one frames dir per entrant per task, written by desk_runner.py).

usage: desk_clip.py --out clip.mp4 --ffmpeg <ffmpeg> [--seg 14] [--fps 25] [--title "..."] --entrant LABEL=<run dir>=<caption> [--entrant ...] [--tasks t1,t2]
For every task the three entrants' real frame sequences are played side by side at one common speed (real time, sped up so the longest one takes at most --seg
seconds; the speed is shown), an entrant that finished (or timed out) holds its last frame with a PASS / FAIL / TIMEOUT badge, and the scoreboard underneath fills as the
tasks go by. Closing card: the final table. Pure PIL + one ffmpeg raw-video pipe, CPU only. The only text drawn is ours (captions, task titles, badges): the desktop
frames carry the sandbox panel clock and window titles only (read by eye on a still before anything is posted).
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desk_tasks as T  # noqa: E402

W, H = 1920, 1080
PW, PH = 580, 362
BG = (14, 16, 20)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(sz, bold=False):
    for p in ((FONTB if bold else FONT), "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load(run, tid):
    d = os.path.join(run, "frames", tid)
    times = []
    if os.path.exists(os.path.join(d, "times.txt")):
        for line in open(os.path.join(d, "times.txt")):
            n, t = line.split()
            times.append((int(n), float(t)))
    res = None
    rp = os.path.join(run, "results.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            r = json.loads(line)
            if r["task"] == tid:
                res = r
    return d, times, res


def frame_at(d, times, t):
    if not times:
        return None
    idx = 0
    for n, ts in times:
        if ts <= t:
            idx = n
        else:
            break
    return os.path.join(d, "%05d.jpg" % idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--seg", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--title", default="which local model can drive a Linux desktop in Hermes Agent")
    ap.add_argument("--entrant", action="append", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--still", default="", help="write one PNG still of the mid-point of the last task here and still produce the clip")
    a = ap.parse_args()
    ents = []
    for e in a.entrant:
        label, run, cap = e.split("=", 2)
        ents.append({"label": label, "run": run, "cap": cap})
    want = [x for x in a.tasks.split(",") if x] or [t["id"] for t in T.TASKS]
    tasks = [t for t in T.TASKS if t["id"] in want]
    ff = subprocess.Popen([a.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(a.fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    f_t, f_s, f_c, f_b = font(34, True), font(22), font(24, True), font(26, True)
    xs = [50 + i * (PW + 40) for i in range(3)]
    state = {e["label"]: {} for e in ents}  # task -> pass/fail/timeout so far
    done_tasks = []

    def board(draw, y0, now_tid):
        draw.text((30, y0), "scoreboard (scored by a script that reads the resulting files)", font=f_s, fill=(150, 160, 170))
        cw = 150
        x0 = 520
        for i, t in enumerate(tasks):
            draw.text((x0 + i * cw + 12, y0 + 36), t["id"], font=f_c, fill=(220, 220, 220))
        for r, e in enumerate(ents):
            y = y0 + 80 + r * 74
            draw.text((30, y + 12), e["cap"].split("|")[0][:34], font=f_c, fill=(235, 235, 235))
            n_pass = sum(1 for v in state[e["label"]].values() if v == "pass")
            for i, t in enumerate(tasks):
                v = state[e["label"]].get(t["id"])
                col = {"pass": (46, 160, 90), "fail": (190, 60, 60), "timeout": (200, 140, 40)}.get(v, (50, 54, 62))
                if t["id"] == now_tid and v is None:
                    col = (70, 90, 130)
                draw.rounded_rectangle((x0 + i * cw, y, x0 + i * cw + cw - 14, y + 58), radius=8, fill=col)
                if v:
                    draw.text((x0 + i * cw + 18, y + 14), {"pass": "PASS", "fail": "FAIL", "timeout": "TIME"}[v], font=f_c, fill=(255, 255, 255))
            draw.text((x0 + len(tasks) * cw + 6, y + 12), "%d/%d" % (n_pass, len(tasks)), font=f_b, fill=(255, 255, 255))

    first_still_done = False
    for ti, t in enumerate(tasks):
        loaded = [load(e["run"], t["id"]) for e in ents]
        durs = []
        for (d, times, res) in loaded:
            durs.append(times[-1][1] if times else 0.0)
        longest = max(durs) if durs else 1.0
        speed = max(1.0, longest / a.seg)
        seg_s = min(a.seg, longest / speed) + 1.2
        nfr = int(seg_s * a.fps)
        for k in range(nfr):
            tt = k / a.fps * speed
            img = Image.new("RGB", (W, H), BG)
            dr = ImageDraw.Draw(img)
            dr.text((30, 22), a.title, font=f_t, fill=(240, 240, 240))
            dr.text((30, 68), "task %d/%d  %s: %s" % (ti + 1, len(tasks), t["title"], t["short"]), font=f_s, fill=(160, 175, 190))
            dr.text((W - 330, 70), "playback x%.1f" % speed, font=f_s, fill=(160, 175, 190))
            for i, e in enumerate(ents):
                d, times, res = loaded[i]
                fp = frame_at(d, times, tt)
                if fp and os.path.exists(fp):
                    im = Image.open(fp).convert("RGB").resize((PW, PH))
                    img.paste(im, (xs[i], 120))
                else:
                    dr.rectangle((xs[i], 120, xs[i] + PW, 120 + PH), fill=(30, 30, 34))
                dr.rectangle((xs[i] - 2, 118, xs[i] + PW + 1, 120 + PH + 1), outline=(70, 75, 85), width=2)
                dr.text((xs[i], 492), e["cap"].split("|")[0], font=f_c, fill=(235, 235, 235))
                if "|" in e["cap"]:
                    dr.text((xs[i], 524), e["cap"].split("|", 1)[1], font=f_s, fill=(140, 150, 160))
                over = times and tt >= times[-1][1]
                if res and over:
                    v = "timeout" if res.get("timeout") else ("pass" if res["pass"] else "fail")
                    col = {"pass": (46, 160, 90), "fail": (190, 60, 60), "timeout": (200, 140, 40)}[v]
                    dr.rounded_rectangle((xs[i] + PW - 190, 130, xs[i] + PW - 10, 180), radius=10, fill=col)
                    dr.text((xs[i] + PW - 175, 139), {"pass": "PASS", "fail": "FAIL", "timeout": "TIMEOUT"}[v], font=f_b, fill=(255, 255, 255))
                    state[e["label"]][t["id"]] = v
                elif times:
                    dr.text((xs[i] + 10, 130), "%.0fs" % min(tt, times[-1][1]), font=f_c, fill=(255, 255, 255))
            board(dr, 600, t["id"])
            if a.still and ti == len(tasks) - 1 and k == nfr - 1 and not first_still_done:
                img.save(a.still)
                first_still_done = True
            ff.stdin.write(img.tobytes())
        done_tasks.append(t["id"])
    # closing card: final table held 3 s
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    dr.text((30, 22), a.title, font=f_t, fill=(240, 240, 240))
    dr.text((30, 68), "final: tasks passed out of %d" % len(tasks), font=f_s, fill=(160, 175, 190))
    board(dr, 300, "")
    for _ in range(3 * a.fps):
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    if a.still:
        img.save(a.still.replace(".png", "-final.png"))
    print("clip", a.out, "rc", ff.returncode)
    return ff.returncode


if __name__ == "__main__":
    sys.exit(main())
