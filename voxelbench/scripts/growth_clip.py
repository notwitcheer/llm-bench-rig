#!/usr/bin/env python3
"""growth timelapse for the tree-growth episode (capsule, playwright chromium, same launch as gate.py / orbit_clip.py).

usage: growth_clip.py <html> <outdir> [--seed 1] [--lambda 0.46] [--seasons 30] [--from 2] [--step-hold 0.8]
                      [--orbit-seconds 8] [--orbit-px 1257] [--lead 1.2] [--swiftshader]
drives the page's OWN hooks only (never edits the model's file): opens ?seed=&lambda=&seasons=&autoplay=0&orbit=0 (paused at season 1, fixed
camera), then steps the tree from season --from to --seasons with window.setSeason(n) (fallback: the N key), holding --step-hold s per season so
every season is on screen, polls window.treeStats() for the per-season numbers; then a short camera orbit: the O key (page's own automatic orbit),
and if two frames 1 s apart do not differ, a real mouse drag of --orbit-px over --orbit-seconds instead.
writes <outdir>/growth.webm + growth.mp4 (load frames trimmed), growth_sheet.png (8 frames over the whole clip),
growth_orbit_sheet.png (4 frames of the orbit window), growth_last.png, growth.json
(method used, per-season stats, orbit method, errors).
"""
import json, os, shutil, subprocess, sys, time, io
from pathlib import Path


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
seed = arg("--seed", "1"); lam = arg("--lambda", "0.46"); seasons = int(arg("--seasons", 30)); s_from = int(arg("--from", 2))
hold = float(arg("--step-hold", 0.8)); orbit_s = float(arg("--orbit-seconds", 8)); orbit_px = float(arg("--orbit-px", 1257)); lead = float(arg("--lead", 1.2))
SWIFT = "--swiftshader" in sys.argv

from playwright.sync_api import sync_playwright

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = (["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] if SWIFT
        else ["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
res = {"html": str(html), "seed": seed, "lambda": lam, "seasons": seasons, "from": s_from, "errors": [], "per_season": [], "swiftshader": SWIFT}
POLL = """() => { try { return typeof window.treeStats === 'function' ? JSON.parse(JSON.stringify(window.treeStats())) : null; } catch (e) { return null; } }"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                          env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(out / "vid"),
                        record_video_size={"width": 1920, "height": 1080})
    tc = time.time()
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:300]))
    pg.goto(html.as_uri() + f"?seed={seed}&lambda={lam}&seasons={seasons}&autoplay=0&orbit=0", wait_until="load", timeout=60000)
    pg.wait_for_timeout(int(lead * 1000))
    t_start = time.time() - tc
    has_set = pg.evaluate("() => typeof window.setSeason === 'function'")
    res["method"] = "setSeason" if has_set else "key_n"
    res["stats_hook"] = pg.evaluate("() => typeof window.treeStats === 'function'")
    st0 = pg.evaluate(POLL)
    res["start_stats"] = st0
    pg.wait_for_timeout(int(hold * 1000))  # season 1 on screen first
    for n in range(s_from, seasons + 1):
        t0 = time.time()
        try:
            if has_set:
                pg.evaluate(f"() => window.setSeason({n})")
            else:
                pg.keyboard.press("n")
        except Exception as e:
            res["errors"].append(f"step {n}: {repr(e)[:150]}")
        s = pg.evaluate(POLL)
        waited = 0
        while s and s.get("season", n) < n and waited < 60:  # a page that grows in the background
            pg.wait_for_timeout(250); waited += 0.25; s = pg.evaluate(POLL)
        res["per_season"].append({"n": n, "compute_s": round(time.time() - t0, 2), "stats": s})
        pg.wait_for_timeout(int(hold * 1000))
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(out / "growth_last.png"))
    # short orbit: the page's own O key first; verify the camera really moved
    from PIL import Image, ImageChops, ImageStat
    pg.keyboard.press("o")
    pg.wait_for_timeout(1200)
    fa = Image.open(io.BytesIO(pg.screenshot())).convert("L")
    pg.wait_for_timeout(1000)
    fb = Image.open(io.BytesIO(pg.screenshot())).convert("L")
    d = round(ImageStat.Stat(ImageChops.difference(fa, fb)).mean[0], 3)
    res["orbit_probe_diff"] = d
    if d >= 0.3:
        res["orbit_method"] = "O key (page's own automatic orbit)"
        pg.wait_for_timeout(int(orbit_s * 1000))
    else:
        res["orbit_method"] = f"mouse drag {orbit_px:.0f} px over {orbit_s:.0f} s (O key had no visible effect)"
        pg.keyboard.press("o")  # undo a toggle that may have switched something else on
        X0, Y = 300, 540
        pg.mouse.move(X0, Y); pg.mouse.down(); t1 = time.time()
        while True:
            k = min(1.0, (time.time() - t1) / orbit_s)
            pg.mouse.move(X0 + orbit_px * k, Y)
            if k >= 1.0:
                break
            pg.wait_for_timeout(16)
        pg.mouse.up()
    pg.wait_for_timeout(600)
    res["final_stats"] = pg.evaluate(POLL)
    vpath = pg.video.path() if pg.video else None
    t_total = time.time() - tc
    ctx.close(); b.close()

res["trim_s"] = round(t_start, 2)
dur = max(1.0, t_total - t_start)
res["duration_s"] = round(dur, 1)
if vpath:
    shutil.move(vpath, out / "growth.webm")
    ff = os.path.expanduser("~/voxelbench/ffmpeg")
    ff = ff if os.path.exists(ff) else shutil.which("ffmpeg")
    r = subprocess.run([ff, "-nostdin", "-y", "-ss", f"{t_start:.2f}", "-i", str(out / "growth.webm"), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "growth.mp4")],
                       capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
    res["mp4_rc"] = r.returncode
    if r.returncode:
        res["ff_err"] = r.stderr.decode(errors="ignore")[-400:]
    else:
        m = subprocess.run([ff, "-nostdin", "-y", "-i", str(out / "growth.mp4"), "-vf", f"fps=8/{dur - 1:.2f},scale=480:-1,tile=4x2",
                            "-frames:v", "1", str(out / "growth_sheet.png")], capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
        res["sheet_rc"] = m.returncode
        # the orbit window alone (last --orbit-seconds): 4 frames, to confirm the camera really moved
        o = subprocess.run([ff, "-nostdin", "-y", "-ss", f"{max(0.0, dur - orbit_s - 0.3):.2f}", "-i", str(out / "growth.mp4"), "-t", f"{orbit_s:.1f}",
                            "-vf", f"fps=4/{orbit_s:.1f},scale=640:-1,tile=4x1", "-frames:v", "1", str(out / "growth_orbit_sheet.png")],
                           capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
        res["orbit_sheet_rc"] = o.returncode
(out / "growth.json").write_text(json.dumps(res, indent=2))
ps = res["per_season"]
print(json.dumps({"method": res["method"], "steps": len(ps), "orbit": res.get("orbit_method"), "orbit_probe_diff": res.get("orbit_probe_diff"),
                  "errors": res["errors"][:3], "mp4_rc": res.get("mp4_rc"), "duration_s": res["duration_s"],
                  "metamers_first_last": [(ps[0]["stats"] or {}).get("metamers"), (ps[-1]["stats"] or {}).get("metamers")] if ps else None}))
