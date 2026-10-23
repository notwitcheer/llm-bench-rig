#!/usr/bin/env python3
"""showcase capture for a voxel run (capsule, playwright chromium on the 5090, same launch as gate.py).

usage: orbit_clip.py <html> <outdir> [--seconds 30] [--orbit-px 1257] [--keys "n@11,n@22"] [--lead 1.5]
drives the PAGE'S OWN controls only with real input events (a left-button mouse drag = its orbit, key presses = its keys),
never edits the model's file. one smooth drag of --orbit-px over the clip (1257 px = one full turn for a page
whose drag handler does theta -= dx * 0.005; velocity-style handlers need less, so read the handler and
check a contact sheet), keys fired at the given seconds after the orbit starts.
writes <outdir>/orbit.webm + orbit.mp4 (load frames trimmed), orbit_mid.png, orbit.json.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
secs = float(arg("--seconds", 30)); opx = float(arg("--orbit-px", 1257)); lead = float(arg("--lead", 1.5))
keys = [(k.split("@")[0], float(k.split("@")[1])) for k in arg("--keys", "n@11,n@22").split(",") if "@" in k]

from playwright.sync_api import sync_playwright

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = ["--use-angle=gl-egl", "--ignore-gpu-blocklist"]
res = {"html": str(html), "seconds": secs, "orbit_px": opx, "keys": keys, "errors": []}
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                          env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(out / "vid"),
                        record_video_size={"width": 1920, "height": 1080})
    tc = time.time()
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:300]))
    pg.goto(html.as_uri(), wait_until="load", timeout=60000)
    pg.wait_for_timeout(int(lead * 1000))
    t_orbit = time.time() - tc
    # REAL input (CDP mouse + keyboard), so pages that listen for pointer events (setPointerCapture) and pages that
    # listen for mouse events both orbit (2026-10-22: synthetic MouseEvents left a pointer-event page static)
    # multi-drag variant (2026-10-22, flint page): the page's handler does camYaw += dx * 0.005 with camYaw in DEGREES,
    # so one full turn needs 72,000 px of drag. we re-grab every SEG px (real input, the page's own control, no edits).
    X0, Y, SEG = 160, 540, 1600
    seg_i = 0
    pg.mouse.move(X0, Y); pg.mouse.down()
    pend = sorted(keys, key=lambda kv: kv[1]); t0 = time.time()
    while True:
        el = time.time() - t0; k = min(1.0, el / secs)
        px = opx * k; si = int(px // SEG); off = px - si * SEG
        if si != seg_i:
            pg.mouse.up(); pg.mouse.move(X0, Y); pg.mouse.down(); seg_i = si
        pg.mouse.move(X0 + off, Y)
        while pend and pend[0][1] <= el:
            pg.keyboard.press(pend.pop(0)[0])
        if k >= 1.0:
            break
        pg.wait_for_timeout(16)
    pg.mouse.up()
    pg.wait_for_timeout(800)
    vpath = pg.video.path() if pg.video else None
    ctx.close(); b.close()
res["trim_s"] = round(t_orbit, 2)
if vpath:
    shutil.move(vpath, out / "orbit.webm")
    ff = os.path.expanduser("~/voxelbench/ffmpeg")
    r = subprocess.run([ff, "-nostdin", "-y", "-ss", f"{t_orbit:.2f}", "-i", str(out / "orbit.webm"), "-t", f"{secs + 0.5:.1f}",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out / "orbit.mp4")], capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
    res["mp4_rc"] = r.returncode
    if r.returncode: res["ff_err"] = r.stderr.decode(errors="ignore")[-400:]
    m = subprocess.run([ff, "-nostdin", "-y", "-ss", f"{secs/2:.1f}", "-i", str(out / "orbit.mp4"), "-frames:v", "1", str(out / "orbit_mid.png")],
                       capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
(out / "orbit.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res))
