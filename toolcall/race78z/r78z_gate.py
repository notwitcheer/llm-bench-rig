#!/usr/bin/env python3
"""four_gate.py: liveness gate + capture for one agent's page in the item-80 race (capsule, playwright chromium, CPU rendering: no card memory, so it
can run while the model server still holds the GPU).

usage: four_gate.py <index.html> <outdir> [--video-seconds 12]
writes <outdir>/gate.json, shot_3s.png, shot_6s.png, clip.webm + clip.mp4 (1280x720)
exit 0 = pass, 1 = fail, 2 = harness error.

pass = page loads from file://, zero console errors, zero page errors, NO network request at all (the prompt says no CDN / no network), at least one canvas, frame not
blank (grey-level stddev > 8) and two frames 3 s apart differ (mean abs diff > 0.5). Same spirit as voxelbench gate.py minus its WebGL-context check (the task is a
plain 2D canvas page); a pass is necessary, not sufficient: read the shots before the page goes on screen.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
vsec = int(sys.argv[sys.argv.index("--video-seconds") + 1]) if "--video-seconds" in sys.argv else 12

res = {"html": str(html), "bytes": html.stat().st_size if html.exists() else 0, "console_errors": [], "page_errors": [], "requests": []}


def stats(png):
    from PIL import Image, ImageStat
    im = Image.open(png).convert("L")
    return round(ImageStat.Stat(im).stddev[0], 2), im


try:
    if not html.exists() or res["bytes"] < 500:
        raise FileNotFoundError("no html or under 500 bytes")
    from playwright.sync_api import sync_playwright
    chrome = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, executable_path=chrome, chromium_sandbox=False, args=["--disable-gpu"],
                              env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
        ctx = b.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(out / "vid"), record_video_size={"width": 1280, "height": 720})
        pg = ctx.new_page()
        pg.on("console", lambda m: res["console_errors"].append(m.text[:300]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
        pg.on("request", lambda r: res["requests"].append(r.url[:200]) if not r.url.startswith(("file:", "data:", "blob:")) else None)
        t0 = time.time()
        pg.goto(html.as_uri(), wait_until="load", timeout=60000)
        res["load_s"] = round(time.time() - t0, 1)
        res["canvases"] = pg.evaluate("() => document.querySelectorAll('canvas').length")
        pg.wait_for_timeout(3000)
        pg.screenshot(path=str(out / "shot_3s.png"))
        pg.wait_for_timeout(3000)
        pg.screenshot(path=str(out / "shot_6s.png"))
        pg.wait_for_timeout(max(0, vsec - 6) * 1000)
        vpath = pg.video.path() if pg.video else None
        ctx.close()
        b.close()
    sd3, a = stats(out / "shot_3s.png")
    sd6, b2 = stats(out / "shot_6s.png")
    from PIL import ImageChops, ImageStat
    diff = round(ImageStat.Stat(ImageChops.difference(a, b2)).mean[0], 2)
    res.update(stddev_3s=sd3, stddev_6s=sd6, frame_diff=diff)
    if vpath:
        shutil.move(vpath, out / "clip.webm")
        ff = os.path.expanduser("~/voxelbench/ffmpeg")
        if os.path.exists(ff):
            r = subprocess.run([ff, "-nostdin", "-y", "-i", str(out / "clip.webm"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out / "clip.mp4")],
                               capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
            res["video"] = "clip.mp4" if r.returncode == 0 else "clip.webm"
    checks = {"no_console_errors": not res["console_errors"], "no_page_errors": not res["page_errors"], "no_network": not res["requests"],
              "canvas_present": res.get("canvases", 0) >= 1, "not_blank": sd6 > 8, "animated": diff > 0.5}
    res["checks"] = checks
    res["pass"] = all(checks.values())
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("pass", "checks", "bytes", "frame_diff", "stddev_6s")}))
    sys.exit(0 if res["pass"] else 1)
except SystemExit:
    raise
except Exception as e:
    res.update(pass_=False, harness_error=str(e)[:300])
    res["pass"] = False
    res["no_html"] = (not html.exists()) or res["bytes"] < 500
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"pass": False, "error": str(e)[:200], "no_html": res["no_html"]}))
    sys.exit(2 if not res["no_html"] else 1)
