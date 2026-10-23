#!/usr/bin/env python3
"""pick the showcase orbit drag for a voxel page (capsule; 2026-10-22, mercury, for unattended clips after a gate pass).

usage: orbit_pick.py <html> [--clip-seconds 30] [--swiftshader]
prints one JSON line: {"script": "orbit_clip.py"|"orbit_clip_multidrag.py", "orbit_px": N, "var": ..., "units": ...,
"method": "measured"|"static"|"default", "handler": [lines]}

reads the page's OWN drag handler (never edits the file): finds the angle variable a pointermove/mousemove handler moves,
then measures it with a REAL playwright drag (top-level let/const in a classic script is readable from page.evaluate),
at the clip's own drag speed (velocity handlers are timing-dependent). units: an angle later fed through
`* Math.PI / 180` or degToRad() counts as DEGREES (360 per turn, the first Flint page: 72,000 px per turn), else radians.
falls back to the static coefficient (turn / C) when the variable is not readable, and to 1257 px when nothing parses.
orbit px over ~1,600 (wider than one drag across the viewport) selects orbit_clip_multidrag.py.
"""
import json, os, re, sys, time
from pathlib import Path


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).resolve()
clip_s = float(arg("--clip-seconds", 30))
src = html.read_text(errors="replace")
lines = src.splitlines()
# drop minified library lines (three.js inlined): handler code is short lines
code = [(i + 1, l) for i, l in enumerate(lines) if len(l) < 400]

handler = [f"{n}: {l.strip()[:200]}" for n, l in code if re.search(r"pointermove|mousemove|movementX|clientX|\bdx\b.*\*", l)][:20]

# candidates: identifiers moved by a drag delta times a coefficient, plus common angle names
pat = re.compile(r"([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s*([+-]?=)\s*[-(]*\s*(?:[\w.]*(?:dx|movementX|deltaX|clientX)[\w.]*)[\s)]*\*\s*([0-9]*\.?[0-9]+)")
cands = []
for n, l in code:
    for m in pat.finditer(l):
        cands.append({"var": m.group(1), "op": m.group(2), "coef": float(m.group(3)), "line": n})
names = [c["var"] for c in cands]
for n, l in code:
    for m in re.finditer(r"\b((?:ctrl|orbit|controls|cam|camera|state|S|view)\.(?:theta|yaw|azimuth|angle|rotY)|camYaw|yaw|theta|azimuth|camTheta|orbitAngle)\b", l):
        if m.group(1) not in names:
            names.append(m.group(1))


def units_of(var):
    base = re.escape(var)
    if re.search(base + r"\s*\*\s*\(?\s*Math\.PI\s*/\s*180", src) or re.search(r"degToRad\(\s*" + base, src) \
            or re.search(base + r"\s*\*\s*DEG\b", src) or re.search(r"(?:radians|toRad|deg2rad)\(\s*" + base, src):
        return "deg"
    return "rad"


res = {"html": str(html), "candidates": cands[:8], "handler": handler}
turn_px = None
# 1) measured: real drag at the clip's speed (first guess 1257 px per clip -> ~42 px/s), read the variable before/after
try:
    from playwright.sync_api import sync_playwright
    CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
    ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] if "--swiftshader" in sys.argv else ["--use-angle=gl-egl", "--ignore-gpu-blocklist"]
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                              env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
        pg = b.new_context(viewport={"width": 1920, "height": 1080}).new_page()
        pg.goto(html.as_uri(), wait_until="load", timeout=60000); pg.wait_for_timeout(2000)
        readable = {}
        for v in names:
            try:
                val = pg.evaluate(f"(() => {{ try {{ const x = {v}; return (typeof x === 'number') ? x : null; }} catch (e) {{ return null; }} }})()")
            except Exception:
                val = None
            if val is not None:
                readable[v] = val
        res["readable"] = list(readable)
        if readable:
            speed = 1257.0 / clip_s
            for attempt in range(2):
                drag = min(1200.0, max(200.0, speed * 8)); secs = drag / speed
                before = {v: pg.evaluate(v) for v in readable}
                pg.mouse.move(300, 540); pg.mouse.down(); t0 = time.time()
                while True:
                    k = min(1.0, (time.time() - t0) / secs); pg.mouse.move(300 + drag * k, 540)
                    if k >= 1: break
                    pg.wait_for_timeout(16)
                pg.mouse.up(); pg.wait_for_timeout(2000)
                after = {v: pg.evaluate(v) for v in readable}
                moved = {v: abs(after[v] - before[v]) for v in readable if abs(after[v] - before[v]) > 1e-9}
                if not moved:
                    break
                v = max(moved, key=moved.get); u = units_of(v)
                per_px = moved[v] / drag
                turn_px = (360.0 if u == "deg" else 6.283185307) / per_px
                res.update({"var": v, "units": u, "method": "measured", "measured_drag_px": drag, "measured_delta": moved[v]})
                new_speed = turn_px / clip_s
                if abs(new_speed - speed) / speed < 0.25:
                    break
                speed = new_speed  # re-measure at the clip's real speed (velocity handlers)
        b.close()
except Exception as e:
    res["measure_error"] = str(e)[:200]

# 2) static: coefficient from the handler line
if turn_px is None and cands:
    c = cands[0]; u = units_of(c["var"])
    turn_px = (360.0 if u == "deg" else 6.283185307) / c["coef"]
    res.update({"var": c["var"], "units": u, "method": "static"})
if turn_px is None:
    turn_px = 1257.0; res.update({"method": "default"})

turn_px = max(300.0, min(200000.0, turn_px))
res["orbit_px"] = int(round(turn_px))
res["script"] = "orbit_clip_multidrag.py" if turn_px > 1600 else "orbit_clip.py"
print(json.dumps(res))
