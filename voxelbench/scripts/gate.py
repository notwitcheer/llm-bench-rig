#!/usr/bin/env python3
"""voxel bench gate + capture (capsule, playwright chromium on the 5090).

usage: gate.py <html> <outdir> [--video-seconds 20]
writes <outdir>/gate.json, shot_3s.png, shot_6s.png, clip.webm (+ clip.mp4 via playwright's ffmpeg)
exit 0 = pass (page loads, zero console errors / page errors, a WebGL context exists, frame not blank,
two frames 3 s apart differ, rAF fps measured), 1 = fail, 2 = harness error.
"""
import json, shutil, subprocess, sys, time, glob, os
from pathlib import Path

html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
vsec = int(sys.argv[sys.argv.index("--video-seconds") + 1]) if "--video-seconds" in sys.argv else 20

from playwright.sync_api import sync_playwright

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = ["--use-angle=gl-egl", "--ignore-gpu-blocklist"]  # probed 2026-10-21: RTX 5090 via ANGLE GL/EGL; ANGLE/Vulkan loses the context at once
res = {"html": str(html), "bytes": html.stat().st_size if html.exists() else 0, "console_errors": [],
       "page_errors": [], "failed_requests": []}


def stats(png):
    from PIL import Image, ImageStat
    im = Image.open(png).convert("L")
    st = ImageStat.Stat(im)
    return round(st.stddev[0], 2), im


try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                              env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(out / "vid"),
                            record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        pg.on("console", lambda m: res["console_errors"].append(m.text[:300]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
        pg.on("requestfailed", lambda r: res["failed_requests"].append(r.url[:200]))
        t0 = time.time()
        pg.goto(html.as_uri(), wait_until="load", timeout=60000)
        res["load_s"] = round(time.time() - t0, 1)
        res["webgl"] = pg.evaluate("""() => { const c=[...document.querySelectorAll('canvas')];
            for (const cv of c) { const gl = cv.getContext('webgl2') || cv.getContext('webgl');
              if (gl) { const d = gl.getExtension('WEBGL_debug_renderer_info');
                return {canvases: c.length, w: cv.width, h: cv.height, lost: gl.isContextLost(),
                        renderer: d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)}; } }
            return {canvases: c.length, renderer: null}; }""")
        pg.wait_for_timeout(3000)
        pg.screenshot(path=str(out / "shot_3s.png"))
        res["fps"] = pg.evaluate("""() => new Promise(r => { let n=0; const t0=performance.now();
            function f(){ n++; if (performance.now()-t0 < 5000) requestAnimationFrame(f); else r(Math.round(n/5)); }
            requestAnimationFrame(f); })""")
        pg.screenshot(path=str(out / "shot_6s.png"))
        pg.wait_for_timeout(max(0, vsec - 11) * 1000)
        vpath = pg.video.path() if pg.video else None
        ctx.close(); b.close()
    sd3, a = stats(out / "shot_3s.png"); sd6, b2 = stats(out / "shot_6s.png")
    from PIL import ImageChops, ImageStat
    diff = round(ImageStat.Stat(ImageChops.difference(a, b2)).mean[0], 2)
    res.update(stddev_3s=sd3, stddev_6s=sd6, frame_diff=diff)
    if vpath:
        shutil.move(vpath, out / "clip.webm")
        ffbin = os.path.expanduser("~/voxelbench/ffmpeg")  # imageio-ffmpeg static build (libx264); playwright's ffmpeg is vp8-only
        ffbin = ffbin if os.path.exists(ffbin) else shutil.which("ffmpeg")
        res["video"] = "clip.webm"
        if ffbin:
            r = subprocess.run([ffbin, "-nostdin", "-y", "-i", str(out / "clip.webm"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                str(out / "clip.mp4")], capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
            if r.returncode == 0:
                res["video"] = "clip.mp4"
    checks = {"no_console_errors": not res["console_errors"], "no_page_errors": not res["page_errors"],
              "webgl_context": bool((res.get("webgl") or {}).get("renderer")) and not (res.get("webgl") or {}).get("lost"), "not_blank": sd6 > 8,
              "animated": diff > 0.5}
    res["checks"] = checks
    res["pass"] = all(checks.values())
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    summ = {k: res[k] for k in ("pass", "checks", "fps", "webgl", "load_s", "bytes")}
    summ["page_errors"] = res["page_errors"][:5]; summ["console_errors"] = res["console_errors"][:5]
    summ["failed_requests"] = res["failed_requests"][:5]  # errors in the summary: a /goal quality gate feeds this tail back to the agent
    print(json.dumps(summ))
    sys.exit(0 if res["pass"] else 1)
except Exception as e:
    res["harness_error"] = repr(e)[:500]
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"harness_error": res["harness_error"]}))
    sys.exit(2)
