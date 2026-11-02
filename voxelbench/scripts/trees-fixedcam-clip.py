#!/usr/bin/env python3
"""fixed-camera growth clip for ONE lambda (capsule, playwright chromium, launch as growth_clip.py).

usage: trees-fixedcam-clip.py <html> <outdir> --lambda 0.46 [--seed 1] [--seasons 30] [--hold 0.8] [--orbit-seconds 8]
                              [--radius 230] [--ty 66] [--elev 0.46] [--azim 0.7] [--w 960] [--h 1080] [--swiftshader]

drives the page's own hooks from outside (never edits the model's file): loads ?seed=&lambda=&seasons=&autoplay=0&orbit=0 at a
960x1080 viewport, pins the page's top-level `cam` object (target.y, radius, elev, azim) to the SAME values for every lambda,
then calls window.setSeason(n) for n = 2..seasons on an ABSOLUTE wall-clock schedule (n at lead + (n-1)*hold) so two runs line up
frame for frame, then switches the page's own auto-orbit on (cam.auto = true, 0.12 rad/s, same rate both runs).
writes <outdir>/fc.mp4 (load frames trimmed) + fc.json (stats per season, camera values, errors).
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
lam = arg("--lambda", "0.46"); seed = arg("--seed", "1"); seasons = int(arg("--seasons", 30))
hold = float(arg("--hold", 0.8)); orbit_s = float(arg("--orbit-seconds", 8)); lead = float(arg("--lead", 1.5))
radius = float(arg("--radius", 230)); ty = float(arg("--ty", 66)); elev = float(arg("--elev", 0.46)); azim = float(arg("--azim", 0.7))
W = int(arg("--w", 960)); H = int(arg("--h", 1080))
SWIFT = "--swiftshader" in sys.argv

from playwright.sync_api import sync_playwright

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = (["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] if SWIFT
        else ["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
res = {"html": str(html), "seed": seed, "lambda": lam, "seasons": seasons, "errors": [], "per_season": [],
       "camera": {"radius": radius, "target_y": ty, "elev": elev, "azim": azim, "viewport": [W, H]}, "swiftshader": SWIFT}
POLL = """() => { try { return typeof window.treeStats === 'function' ? JSON.parse(JSON.stringify(window.treeStats())) : null; } catch (e) { return null; } }"""
PIN = f"() => {{ cam.target.set(0, {ty}, 0); cam.radius = {radius}; cam.elev = {elev}; cam.azim = {azim}; cam.auto = false; }}"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                          env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(out / "vid"),
                        record_video_size={"width": W, "height": H})
    tc = time.time()
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:300]))
    pg.goto(html.as_uri() + f"?seed={seed}&lambda={lam}&seasons={seasons}&autoplay=0&orbit=0", wait_until="load", timeout=60000)
    pg.evaluate(PIN)
    pg.wait_for_timeout(300)
    res["pinned"] = pg.evaluate("() => ({r: cam.radius, ty: cam.target.y, e: cam.elev, a: cam.azim, auto: cam.auto})")
    t_start = time.time() - tc   # clip starts here (load frames trimmed)
    t0 = time.time()
    pg.wait_for_timeout(int(lead * 1000))
    for n in range(2, seasons + 1):
        due = t0 + lead + (n - 1) * hold
        w = due - time.time()
        if w > 0:
            pg.wait_for_timeout(int(w * 1000))
        ts = time.time()
        try:
            pg.evaluate(f"() => window.setSeason({n})")
        except Exception as e:
            res["errors"].append(f"step {n}: {repr(e)[:150]}")
        s = pg.evaluate(POLL)
        waited = 0
        while s and s.get("season", n) < n and waited < 60:
            pg.wait_for_timeout(250); waited += 0.25; s = pg.evaluate(POLL)
        res["per_season"].append({"n": n, "compute_s": round(time.time() - ts, 2), "late_s": round(max(0.0, time.time() - due), 2), "stats": s})
    pg.wait_for_timeout(int(1.5 * 1000))
    pg.evaluate("() => { cam.auto = true; }")
    pg.wait_for_timeout(int(orbit_s * 1000))
    res["final_stats"] = pg.evaluate(POLL)
    res["final_cam"] = pg.evaluate("() => ({r: cam.radius, ty: cam.target.y, e: cam.elev, a: cam.azim, auto: cam.auto})")
    pg.screenshot(path=str(out / "fc_last.png"))
    vpath = pg.video.path()
    ctx.close(); b.close()

res["trim_s"] = round(t_start, 2)
shutil.move(vpath, out / "fc.webm")
ff = os.path.expanduser("~/voxelbench/ffmpeg")
ff = ff if os.path.exists(ff) else shutil.which("ffmpeg")
r = subprocess.run([ff, "-nostdin", "-y", "-ss", f"{t_start:.2f}", "-i", str(out / "fc.webm"), "-r", "30", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(out / "fc.mp4")], capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
res["mp4_rc"] = r.returncode
if r.returncode:
    res["ff_err"] = r.stderr.decode(errors="ignore")[-400:]
(out / "fc.json").write_text(json.dumps(res, indent=2))
ps = res["per_season"]
print(json.dumps({"lambda": lam, "steps": len(ps), "errors": res["errors"][:3], "mp4_rc": res["mp4_rc"], "pinned": res["pinned"],
                  "final_cam": res["final_cam"], "max_late_s": max([x["late_s"] for x in ps]) if ps else None,
                  "last": {k: (res["final_stats"] or {}).get(k) for k in ("season", "metamers", "height", "crownWidth")}}))
