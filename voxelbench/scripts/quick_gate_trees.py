#!/usr/bin/env python3
"""in-run quality gate for the tree-growth task (capsule, playwright chromium). This is the command registered with the Hermes /goal gate;
the agent can read this file and run it itself. exit 0 = pass, 1 = fail (the JSON line says why and what to do), 2 = harness error.

usage: quick_gate_trees.py <html> <outdir> [--video-seconds N (ignored)] [--swiftshader]

what it checks, in order (the task text names every hook below):
  1. the file exists at the exact expected path (otherwise it names the path and lists any index.html it finds elsewhere under runs/)
  2. the page opens, no console errors, no page errors, a live WebGL context, a non-blank frame, a frame that changes
  3. the page's main thread keeps answering (a page that freezes the tab for > 30 s cannot autoplay)
  4. window.treeStats() answers with finite numbers and the requested season (?seed=1&lambda=0.46&seasons=5&instant=1)
  5. the tree is growing: the same load with seasons=15 holds clearly more metamers than at seasons=5, and more than 50
exit codes and messages are written for the agent: they say what is wrong and which part of the task text to re-read.
"""
import json, os, sys, time, math
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).absolute()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
SWIFT = "--swiftshader" in sys.argv
RUN_TO = float(arg("--run-timeout", 60))
BLOCK_LIMIT_S = 30
KEYS = ["season", "metamers", "buds", "height", "crownWidth", "shed"]
res = {"html": str(html), "gate": "quick_gate_trees", "swiftshader": SWIFT, "console_errors": [], "page_errors": [], "failed_requests": [],
       "main_thread": {"blocked": False, "limit_s": BLOCK_LIMIT_S}}
MT = res["main_thread"]
T_START = time.time()


def finish(code, summ):
    res["elapsed_s"] = round(time.time() - T_START, 1)
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(summ)); sys.stdout.flush()
    os._exit(code)


# 1. the path. a model that retyped the run dir (a dropped hyphen, twice now) saved its page somewhere else and then read the
# resulting ERR_FILE_NOT_FOUND as an environment bug. say the exact path and show where the strays are.
if not html.is_file():
    runs_root = html.parents[2] if len(html.parents) > 2 else html.parent
    now = time.time(); stray = []
    try:
        for p in runs_root.rglob("*.html"):
            try:
                st = p.stat()
            except OSError:
                continue
            if now - st.st_mtime < 8 * 3600 and st.st_size > 1000 and p.resolve() != html.resolve():
                stray.append((st.st_mtime, str(p)))
    except Exception:
        pass
    stray = [s for _, s in sorted(stray, reverse=True)[:5]]
    res.update(pass_=False, error="no_html_at_expected_path", expected=str(html), found_elsewhere=stray)
    msg = {"pass": False, "error": "no_html_at_expected_path", "expected_path": str(html), "found_elsewhere": stray,
           "fix": "the page must be saved at exactly expected_path. copy that path from the task text, never retype it" +
                  (": move or rewrite your page from found_elsewhere[0] to expected_path" if stray else "")}
    res["pass"] = False
    finish(1, msg)

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = (["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] if SWIFT
        else ["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
INIT_JS = """(() => { const v = window.__vx = {maxGap: 0, last: performance.now()};
 const f = () => { const n = performance.now(); const g = n - v.last; v.last = n; if (g > v.maxGap) v.maxGap = g; requestAnimationFrame(f); };
 requestAnimationFrame(f); })();"""
POLL_JS = """() => { try { if (typeof window.treeStats !== 'function') return {__nohook: true};
  const s = window.treeStats(); return s ? JSON.parse(JSON.stringify(s)) : {__null: true}; } catch (e) { return {__err: String(e).slice(0, 200)}; } }"""


def good_stats(s):
    return isinstance(s, dict) and all(isinstance(s.get(k), (int, float)) and not isinstance(s.get(k), bool) and math.isfinite(s.get(k)) for k in KEYS)


def block_probe(pg, limit_s=BLOCK_LIMIT_S):
    t = time.time()
    try:
        pg.wait_for_function("() => true", timeout=limit_s * 1000, polling=100)
        return round(time.time() - t, 2)
    except PWTimeout:
        return None


def stats_img(png):
    from PIL import Image, ImageStat
    im = Image.open(png).convert("L")
    return round(ImageStat.Stat(im).stddev[0], 2), im


def run_tree(browser, seasons):
    """fresh page, ?instant=1 load, poll treeStats() until done (or a stable final season). returns a record."""
    rec = {"seasons": seasons, "errors": [], "console": []}
    t0 = time.time()
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: rec["errors"].append(str(e)[:200]))
    pg.on("console", lambda m: rec["console"].append(m.text[:200]) if m.type == "error" else None)
    try:
        pg.goto(html.as_uri() + f"?seed=1&lambda=0.46&seasons={seasons}&instant=1&orbit=0", wait_until="load", timeout=int(RUN_TO * 1000))
        last = None; stable = 0; s = None
        while time.time() - t0 < RUN_TO:
            s = pg.evaluate(POLL_JS)
            if s.get("__nohook") or s.get("__err") or s.get("__null"):
                rec["hook_problem"] = s; break
            if s.get("done") is True:
                break
            key = (s.get("season"), s.get("metamers"))
            stable = stable + 1 if key == last else 0; last = key
            if stable >= 4 and s.get("season", 0) >= seasons:
                break
            pg.wait_for_timeout(500)
        else:
            rec["timeout"] = True
        rec["stats"] = s if (s and not (s.get("__nohook") or s.get("__err") or s.get("__null"))) else None
    except Exception as e:
        rec["exception"] = repr(e)[:240]
        if isinstance(e, PWTimeout) or "Timeout" in repr(e):
            rec["blocked"] = True
    rec["sim_s"] = round(time.time() - t0, 1)
    if not rec.get("blocked"):
        try:
            ctx.close()
        except Exception:
            pass
    rec["errors"] = rec["errors"][:3]; rec["console"] = rec["console"][:3]
    return rec


checks = {}
hints = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                              env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
        ctx = b.new_context(viewport={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        pg.add_init_script(INIT_JS)
        pg.on("console", lambda m: res["console_errors"].append(m.text[:300]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
        pg.on("requestfailed", lambda r: res["failed_requests"].append(r.url[:200]))
        t0 = time.time()
        try:
            pg.goto(html.as_uri(), wait_until="load", timeout=60000)
            res["load_s"] = round(time.time() - t0, 1)
            pr = block_probe(pg)
            if pr is None:
                MT.update(blocked=True, stage="right after load")
            else:
                res["webgl"] = pg.evaluate("""() => { const c=[...document.querySelectorAll('canvas')];
                    for (const cv of c) { const gl = cv.getContext('webgl2') || cv.getContext('webgl');
                      if (gl) { const d = gl.getExtension('WEBGL_debug_renderer_info');
                        return {canvases: c.length, w: cv.width, h: cv.height, lost: gl.isContextLost(),
                                renderer: d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)}; } }
                    return {canvases: c.length, renderer: null}; }""")
                pg.wait_for_timeout(3000)
                pr2 = block_probe(pg)
                if pr2 is None:
                    MT.update(blocked=True, stage="3 s after load")
                else:
                    pg.screenshot(path=str(out / "shot_3s.png"))
                    res["fps"] = pg.evaluate("""() => new Promise(r => { let n=0; const t0=performance.now();
                        function f(){ n++; if (performance.now()-t0 < 5000) requestAnimationFrame(f); else r(Math.round(n/5)); }
                        requestAnimationFrame(f); })""")
                    pg.screenshot(path=str(out / "shot_6s.png"))
                    vx = pg.evaluate("() => window.__vx ? Math.round(window.__vx.maxGap) : null")
                    MT["max_frame_gap_ms"] = vx
        except PWTimeout as e:
            MT.update(blocked=True, stage=MT.get("stage") or "page call timed out", error=repr(e)[:120])
        if MT["blocked"]:
            extra = block_probe(pg, 30)
            MT["blocked_s"] = round(BLOCK_LIMIT_S + extra, 1) if extra is not None else f">{BLOCK_LIMIT_S + 30}"
            checks.update(no_console_errors=not res["console_errors"], no_page_errors=not res["page_errors"], webgl_context=bool((res.get("webgl") or {}).get("renderer")),
                          not_blank=False, animated=False, main_thread_ok=False)
            hints.append("the page FREEZES the browser tab: the main thread did not answer for " + str(MT["blocked_s"]) + " s (" + MT["stage"] +
                         "). the usual cause is an infinite or runaway loop in the simulation (a loop that never reaches its exit, a recursion or "
                         "walk over a structure that contains a cycle, or work that grows with the square of the tree). find it by running the simulation "
                         "functions in node with a timeout, not by waiting on the page. a demo must stay responsive: do the heavy work in small slices per frame or per season")
        else:
            sd3, a = stats_img(out / "shot_3s.png"); sd6, b2 = stats_img(out / "shot_6s.png")
            from PIL import ImageChops, ImageStat
            diff = round(ImageStat.Stat(ImageChops.difference(a, b2)).mean[0], 2)
            res.update(stddev_3s=sd3, stddev_6s=sd6, frame_diff=diff)
            checks.update(no_console_errors=not res["console_errors"], no_page_errors=not res["page_errors"],
                          webgl_context=bool((res.get("webgl") or {}).get("renderer")) and not (res.get("webgl") or {}).get("lost"),
                          not_blank=sd6 > 8, animated=diff > 0.5, main_thread_ok=True)
            if not checks["no_console_errors"] or not checks["no_page_errors"]:
                hints.append("the page logs errors on load: fix them first (see console_errors / page_errors)")
            if not checks["not_blank"]:
                hints.append("the 6 s frame is blank")
            if not checks["animated"]:
                hints.append("the frame at 3 s and at 6 s are the same: the default page must autoplay and orbit")
        res["checks_base"] = dict(checks)
        runs = []
        if not MT["blocked"]:
            r5 = run_tree(b, 5); runs.append(r5)
            s5 = r5.get("stats")
            checks["treestats_hook"] = bool(good_stats(s5) and s5.get("season") == 5)
            if r5.get("blocked"):
                MT["blocked"] = True; MT["stage"] = "?instant=1&seasons=5 load"; checks["main_thread_ok"] = False
                hints.append("loading the page with ?instant=1&seasons=5 never finished: the page freezes while it simulates (" + r5.get("exception", "")[:80] + "). an infinite or runaway loop in the simulation is the usual cause")
            elif not checks["treestats_hook"]:
                hints.append("window.treeStats() must exist and return finite numbers for season, metamers, buds, height, crownWidth, shed, and with ?instant=1&seasons=5 it must report season 5 once finished (see hook_problem / sim_s below)")
            else:
                r15 = run_tree(b, 15); runs.append(r15)
                s15 = r15.get("stats")
                if r15.get("blocked"):
                    MT["blocked"] = True; MT["stage"] = "?instant=1&seasons=15 load"; checks["main_thread_ok"] = False
                    checks["growth"] = False
                    hints.append("loading the page with ?instant=1&seasons=15 never finished: the simulation freezes the tab or is far too slow")
                elif good_stats(s15):
                    m5, m15 = s5["metamers"], s15["metamers"]
                    res["metamers_5_15"] = [m5, m15]
                    checks["growth"] = bool(m15 > m5 and m15 > 50)
                    if not checks["growth"]:
                        hints.append(f"the tree is not growing: {m5} metamers after 5 seasons and {m15} after 15. a growing tree holds clearly more metamers after 15 "
                                     "seasons than after 5 (and well over 50). print, per season, how many buds receive enough resource to grow a shoot, and trace why the others do not")
                else:
                    checks["growth"] = False
                    hints.append("the seasons=15 load did not return usable treeStats()")
        res["runs"] = runs
        try:
            b.close()
        except Exception:
            pass
    res["checks"] = checks
    res["pass"] = bool(checks) and all(checks.values())
    summ = {"pass": res["pass"], "checks": checks, "fps": res.get("fps"), "load_s": res.get("load_s"), "metamers_5_15": res.get("metamers_5_15"),
            "main_thread": MT, "page_errors": res["page_errors"][:3], "console_errors": res["console_errors"][:3],
            "sim_seconds": {f"s{r['seasons']}": r.get("sim_s") for r in runs}}
    if hints:
        summ["fix"] = hints
    if runs and not checks.get("treestats_hook", True):
        summ["hook_help"] = [{k: r.get(k) for k in ("seasons", "hook_problem", "timeout", "exception", "errors", "sim_s")} for r in runs][:1]
    finish(0 if res["pass"] else 1, summ)
except Exception as e:
    res["harness_error"] = repr(e)[:500]
    finish(2, {"harness_error": res["harness_error"]})
