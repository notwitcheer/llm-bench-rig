#!/usr/bin/env python3
"""CPU checker: loads each extracted page in headless chromium (software GL), pixel check, clicks Export GLB, validates the download.
appends one row per cell to checks.jsonl (skips cells already there). --workers N spawns N striped child processes."""
import argparse, glob, hashlib, json, os, signal, subprocess, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glbcommon as G
ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True); ap.add_argument("--models", default="all"); ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--stripe", default=""); ap.add_argument("--netcache", default=os.path.expanduser("~/glb-3d-coder/netcache"))
a = ap.parse_args()
C = G.cfg(); CK = C["checks"]
checks_path = a.res + "/checks.jsonl"
def todo():
    done = set()
    if os.path.exists(checks_path):
        for l in open(checks_path):
            if l.strip():
                r = json.loads(l); done.add((r["model"], r["prompt"], r["seed"]))
    cells = []
    for m in C["models"]:
        if a.models != "all" and m["key"] not in a.models.split(","): continue
        p = f"{a.res}/{m['key']}.jsonl"
        if not os.path.exists(p): continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if (m["key"], r["prompt"], r["seed"]) not in done: cells.append((m["key"], r["prompt"], r["seed"]))
    return sorted(cells)
if a.workers > 1 and not a.stripe:
    ps = [subprocess.Popen([sys.executable, __file__, "--res", a.res, "--models", a.models, "--stripe", f"{i}/{a.workers}", "--netcache", a.netcache]) for i in range(a.workers)]
    sys.exit(max(p.wait() for p in ps))
cells = todo()
if a.stripe: i, n = map(int, a.stripe.split("/")); cells = cells[i::n]
print(f"[check] {len(cells)} cells", flush=True)
from playwright.sync_api import sync_playwright
from PIL import Image, ImageStat, ImageChops
import numpy as np
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
os.makedirs(a.netcache, exist_ok=True)
EXP1 = r"export.*glb|glb.*export|download.*glb|save.*glb"; EXP2 = r"glb|export|download"
FIND = """([p1, p2]) => {
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const els = [...document.querySelectorAll('button, a, input, [role=button], [onclick], label, summary, div, span')].filter(vis);
  const txt = e => [e.innerText || '', e.value || '', e.title || '', e.getAttribute('aria-label') || '', e.id || ''].join(' ').slice(0, 200);
  for (const p of [p1, p2]) { const re = new RegExp(p, 'i');
    const hit = els.filter(e => re.test(txt(e)) && e.children.length <= 3).sort((x, y) => (x.innerText||'').length - (y.innerText||'').length)[0];
    if (hit) { hit.setAttribute('data-glbexp', '1'); return txt(hit).trim().slice(0, 60); } }
  return null; }"""
def fetch_cached(url):
    k = hashlib.sha1(url.encode()).hexdigest(); fp = f"{a.netcache}/{k}"
    if os.path.exists(fp + ".json"): return open(fp, "rb").read(), json.load(open(fp + ".json"))
    last = None
    for t in range(3):
        try:
            rq = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130 Safari/537.36"})
            r = urllib.request.urlopen(rq, timeout=25); body = r.read(); meta = {"url": url, "status": r.status, "ct": r.headers.get("content-type", "application/octet-stream")}
            open(fp, "wb").write(body); json.dump(meta, open(fp + ".json", "w")); return body, meta
        except urllib.error.HTTPError as e:
            if e.code in (404, 403, 410): meta = {"url": url, "status": e.code, "ct": "text/plain", "net_fail": False}; open(fp, "wb").write(b""); json.dump(meta, open(fp + ".json", "w")); return b"", meta
            last = e
        except Exception as e: last = e
        time.sleep(1.5)
    raise RuntimeError(str(last)[:100])
class Wd(Exception): pass
def alarm(*_): raise Wd("watchdog 150 s")
signal.signal(signal.SIGALRM, alarm)
def check(br, key, pid, sd):
    base = f"{a.res}/raw/{key}/{pid}-s{sd}.json"; cell = f"{key}/{pid}-s{sd}"
    row = {"model": key, "prompt": pid, "seed": sd, "html": False, "load_ok": False, "nonempty": False, "glb_valid": False, "page_valid": False, "full_pass": False,
           "net_fail": 0, "page_errors": [], "console_errors": [], "reason": ""}
    raw = json.load(open(base)); html, how = G.extract_html(raw.get("content"))
    row["extract"] = how
    if not html: row["reason"] = "no page extracted"; return row
    row["html"] = True; row["html_bytes"] = len(html)
    os.makedirs(f"{a.res}/pages/{key}", exist_ok=True); open(f"{a.res}/pages/{key}/{pid}-s{sd}.html", "w").write(html)
    ctx = br.new_context(viewport={"width": CK["viewport"][0], "height": CK["viewport"][1]}, accept_downloads=True)
    try:
        def route(r):
            u = r.request.url
            if u == "http://glb.local/page.html": return r.fulfill(status=200, content_type="text/html; charset=utf-8", body=html)
            if u.startswith("http://glb.local/") or u.startswith("data:") or u.startswith("blob:"): return r.fulfill(status=404, body="")
            try: body, meta = fetch_cached(u)
            except Exception: row["net_fail"] += 1; return r.abort()
            r.fulfill(status=meta.get("status", 200), body=body, headers={"content-type": meta["ct"], "access-control-allow-origin": "*"})
        ctx.route("**/*", route)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: row["page_errors"].append(str(e)[:200]))
        pg.on("console", lambda m: row["console_errors"].append(m.text[:200]) if m.type == "error" and "Failed to load resource" not in m.text else None)
        t0 = time.time()
        try: pg.goto("http://glb.local/page.html", wait_until="load", timeout=CK["page_timeout_s"] * 1000)
        except Exception as e: row["page_errors"].append("goto: " + str(e)[:100])
        pg.wait_for_timeout(CK["settle_ms"]); row["load_s"] = round(time.time() - t0, 1)
        row["load_ok"] = not row["page_errors"] and not row["console_errors"]
        os.makedirs(f"{a.res}/shots/{key}", exist_ok=True); shot = f"{a.res}/shots/{key}/{pid}-s{sd}.png"
        pg.screenshot(path=shot)
        im = Image.open(shot).convert("L"); arr = np.asarray(im).astype(np.int16)
        h, w = arr.shape; ps = [arr[:16, :16], arr[:16, -16:], arr[-16:, :16], arr[-16:, -16:]]
        bg = float(np.median(np.concatenate([x.ravel() for x in ps])))
        std = float(arr.std()); fg = float((np.abs(arr - bg) > CK["nonempty"]["fg_diff"]).mean())
        gl = pg.evaluate("""() => { const c=[...document.querySelectorAll('canvas')]; for (const cv of c) { const g = cv.getContext('webgl2') || cv.getContext('webgl'); if (g) return {canvases:c.length, lost:g.isContextLost()}; } return {canvases:c.length, lost:null}; }""")
        row.update(gray_std=round(std, 2), fg_frac=round(fg, 4), canvases=gl["canvases"], gl_context=gl["lost"] is not None, gl_lost=bool(gl["lost"]))
        row["nonempty"] = std >= CK["nonempty"]["gray_std_min"] and fg >= CK["nonempty"]["fg_frac_min"] and gl["lost"] is not None and not gl["lost"]
        row["page_valid"] = row["html"] and row["load_ok"] and row["nonempty"]
        label = pg.evaluate(FIND, [EXP1, EXP2]); row["export_control"] = label
        if label is None: row["reason"] = "no export control"
        else:
            try:
                with pg.expect_download(timeout=CK["export_wait_s"] * 1000) as dl: pg.evaluate("document.querySelector('[data-glbexp=\"1\"]').click()")
                os.makedirs(f"{a.res}/glb/{key}", exist_ok=True); gp = f"{a.res}/glb/{key}/{pid}-s{sd}.glb"; dl.value.save_as(gp)
                b = open(gp, "rb").read(); ok, why, info = G.glb_validate(b); row["glb_valid"] = ok; row["glb_reason"] = why; row["glb_info"] = info
                if not ok: row["reason"] = "glb invalid: " + why
            except Wd: raise
            except Exception as e: row["reason"] = "no download: " + str(e)[:80].split("\n")[0]
        row["full_pass"] = row["page_valid"] and row["glb_valid"]
        if not row["page_valid"]:
            row["export_detail"] = row["reason"]; row["reason"] = "page error" if not row["load_ok"] else "empty scene"
    finally:
        try: ctx.close()
        except Exception: pass
    return row
with sync_playwright() as p:
    br = p.chromium.launch(headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--disable-dev-shm-usage"],
                           executable_path=CHROME, chromium_sandbox=False, env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    for key, pid, sd in cells:
        signal.alarm(150)
        try: row = check(br, key, pid, sd)
        except Wd as e: row = {"model": key, "prompt": pid, "seed": sd, "html": True, "load_ok": False, "nonempty": False, "glb_valid": False, "page_valid": False, "full_pass": False, "reason": "watchdog", "harness_fault": True}
        except Exception as e: row = {"model": key, "prompt": pid, "seed": sd, "html": False, "load_ok": False, "nonempty": False, "glb_valid": False, "page_valid": False, "full_pass": False, "reason": "checker crash: " + str(e)[:100], "harness_fault": True}
        finally: signal.alarm(0)
        open(checks_path, "a").write(json.dumps(row) + "\n")
        print(f"[check] {key} {pid} s{sd}: page_valid={row['page_valid']} glb={row['glb_valid']} full={row['full_pass']} {row.get('reason','')}", flush=True)
    br.close()
