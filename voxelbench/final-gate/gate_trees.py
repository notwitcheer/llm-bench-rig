#!/usr/bin/env python3
"""voxel bench FINAL gate for the tree-growth episode (capsule, playwright chromium; sibling of gate.py, gate.py itself is untouched).

DEPLOY PATH ON CAPSULE: ~/.vxfinal/gate_trees.py (a dot directory outside ~/voxelbench and ~/benchmark-rig; HARDENING L33). It is NOT
in ~/voxelbench/scripts, it is not named in any prompt or in the /goal quality-gate command, and the queue items carry no copy of the
criteria below. The in-run gate the agent sees is scripts/quick_gate_trees.py (no paper criteria in it).

usage: gate_trees.py <html> <outdir> [--video-seconds 20] [--swiftshader] [--no-extra-seeds] [--run-timeout 120] [--budget 1200]
writes <outdir>/gate.json, shot_3s.png, shot_6s.png, clip.webm/clip.mp4 (default page), shot_lambda046.png / shot_lambda056.png
(finished trees, seed 1, 30 seasons).  exit 0 = pass, 1 = fail, 2 = harness error.

PRE-REGISTERED CRITERIA (written before any run of the prompt trees-v1; see vault voxel-trees-status.md). UNCHANGED since 2026-10-26:
  base (the series' five checks, same code path as gate.py, on the page opened with NO parameters):
    no_console_errors, no_page_errors, webgl_context, not_blank (stddev of the 6 s shot > 8), animated (3 s vs 6 s frame diff > 0.5)
  paper checks (fresh page loads with ?instant=1, seed 1, 1280x720; window.treeStats() polled until done or --run-timeout):
    treestats_hook  : window.treeStats() exists and returns finite numbers for season, metamers, buds, height, crownWidth, shed,
                      and season == the requested number of seasons (checked at ?seasons=5)
    metamers_grow   : lambda 0.46, seasons 5 / 15 / 30:  m15 >= 0.95*m5 and m30 >= 0.95*m15 and m30 > m5   ("monotonically-ish": shedding may thin a crown)
    non_trivial     : metamers at season 30 >= 100 in BOTH the lambda 0.46 and the lambda 0.56 run (not a stick)
    lambda_effect   : seed 1, 30 seasons, same page:  crownWidth/height at lambda 0.46  >  crownWidth/height at lambda 0.56
                      (Palubicki et al. 2009 Fig. 7: lambda < 0.5 = decurrent/wide, > 0.5 = excurrent/narrow). both numbers are reported.
    deterministic   : seed 1, lambda 0.46, 30 seasons loaded twice (two page loads): same metamer count
  pass = every check above. extra (reported, NOT gating): the lambda effect on seeds 2 and 3, and whether seeds differ.

h3 changes (2026-10-29, HARDENING L32-L38; none of them changes the pass rule above):
  - main_thread_ok: a page whose main thread does not answer within 30 s (infinite or runaway loop at load / per frame) used to end as
    harness_error (rc 2, \"Page.screenshot: Timeout 30000ms exceeded\"), which fails the gate anyway. It is now an explicit check
    (rc 1) with the measured block recorded in gate.json `main_thread`. For pages that do answer, `main_thread` reports the longest
    rAF gap and long-task time, NOT gating. A page that blocks the tab for 30 s cannot autoplay a demo, so it stays a FAIL.
  - reported only: `growth_flags` (runaway_growth_flag when seasons-30 metamers > 25,000; reference simulation ~4.5k). not gating.
  - --quick is gone from this file (the in-run gate is quick_gate_trees.py).
"""
import json, shutil, subprocess, sys, time, os, math
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if "--quick" in sys.argv:
    print(json.dumps({"harness_error": "--quick was moved out of the final gate; use scripts/quick_gate_trees.py"})); sys.exit(2)
html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
vsec = int(arg("--video-seconds", 20))
SWIFT = "--swiftshader" in sys.argv
EXTRA = "--no-extra-seeds" not in sys.argv
RUN_TO = float(arg("--run-timeout", 120))
BUDGET = float(arg("--budget", 1200))
BLOCK_LIMIT_S = 30
T_START = time.time()

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
if SWIFT:  # CPU-only rendering for dry-runs (no GPU use)
    ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
else:
    ARGS = ["--use-angle=gl-egl", "--ignore-gpu-blocklist"]  # same as gate.py (RTX 5090 via ANGLE GL/EGL)
res = {"html": str(html), "bytes": html.stat().st_size if html.exists() else 0, "console_errors": [],
       "page_errors": [], "failed_requests": [], "gate": "gate_trees", "swiftshader": SWIFT,
       "main_thread": {"blocked": False, "limit_s": BLOCK_LIMIT_S}}
KEYS = ["season", "metamers", "buds", "height", "crownWidth", "shed"]
MT = res["main_thread"]


def stats_img(png):
    from PIL import Image, ImageStat
    im = Image.open(png).convert("L")
    return round(ImageStat.Stat(im).stddev[0], 2), im


def good_stats(s):
    return isinstance(s, dict) and all(isinstance(s.get(k), (int, float)) and not isinstance(s.get(k), bool) and math.isfinite(s.get(k)) for k in KEYS)


POLL_JS = """() => { try { if (typeof window.treeStats !== 'function') return {__nohook: true};
  const s = window.treeStats(); return s ? JSON.parse(JSON.stringify(s)) : {__null: true}; } catch (e) { return {__err: String(e).slice(0, 200)}; } }"""

# rAF gap + long-task recorder (reported only)
INIT_JS = """(() => { const v = window.__vx = {maxGap: 0, last: performance.now(), long: 0, nlong: 0};
 const f = () => { const n = performance.now(); const g = n - v.last; v.last = n; if (g > v.maxGap) v.maxGap = g; requestAnimationFrame(f); };
 requestAnimationFrame(f);
 try { new PerformanceObserver(l => { for (const e of l.getEntries()) { v.long += e.duration; v.nlong++; } }).observe({entryTypes: ['longtask']}); } catch (e) {} })();"""


def block_probe(pg, limit_s=BLOCK_LIMIT_S):
    """seconds until the page's main thread answers a trivial wait_for_function (interval polling, so it does not rely on rAF);
    None = no answer within limit_s (a blocked main thread: playwright's own timeout fires on the driver side)."""
    t = time.time()
    try:
        pg.wait_for_function("() => true", timeout=limit_s * 1000, polling=100)
        return round(time.time() - t, 2)
    except PWTimeout:
        return None


def mark_blocked(pg, stage):
    """record a main-thread block: measured for up to 60 s more (so `blocked_s` is the lower bound or the real figure)."""
    MT["blocked"] = True; MT["stage"] = stage
    extra = block_probe(pg, 60)
    MT["blocked_s"] = round(BLOCK_LIMIT_S + extra, 1) if extra is not None else f">{BLOCK_LIMIT_S + 60}"
    MT["released"] = extra is not None


def run_tree(browser, query, tag, want, shot=None):
    """fresh context + page, ?instant=1 load, poll treeStats() until done. returns a record."""
    rec = {"tag": tag, "query": query, "errors": [], "console": [], "failed": []}
    t0 = time.time()
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: rec["errors"].append(str(e)[:200]))
    pg.on("console", lambda m: rec["console"].append(m.text[:200]) if m.type == "error" else None)
    pg.on("requestfailed", lambda r: rec["failed"].append(r.url[:160]))
    try:
        pg.goto(html.as_uri() + query, wait_until="load", timeout=int(RUN_TO * 1000))
        last = None; stable = 0; s = None
        while time.time() - t0 < RUN_TO:
            s = pg.evaluate(POLL_JS)
            if s.get("__nohook") or s.get("__err") or s.get("__null"):
                rec["hook_problem"] = s; break
            if s.get("done") is True:
                break
            key = (s.get("season"), s.get("metamers"))
            stable = stable + 1 if key == last else 0; last = key
            if stable >= 4 and s.get("season", 0) >= want:  # no `done` key: settle on a stable final season
                break
            pg.wait_for_timeout(500)
        else:
            rec["timeout"] = True
        rec["stats"] = s if (s and not (s.get("__nohook") or s.get("__err") or s.get("__null"))) else None
        rec["sim_s"] = round(time.time() - t0, 1)
        if shot and rec.get("stats"):
            pg.wait_for_timeout(1200)
            pg.screenshot(path=str(shot))
    except Exception as e:
        rec["exception"] = repr(e)[:300]
        rec["sim_s"] = round(time.time() - t0, 1)
        if isinstance(e, PWTimeout) or "Timeout" in repr(e):
            rec["blocked"] = True  # goto / evaluate / screenshot never returned: the page did not answer
    finally:
        if not rec.get("blocked"):  # closing a blocked renderer can hang; the whole browser is torn down at the end
            try:
                ctx.close()
            except Exception:
                pass
    rec["errors"] = rec["errors"][:3]; rec["console"] = rec["console"][:3]; rec["failed"] = rec["failed"][:3]
    return rec


def q(seed, lam, seasons):
    return f"?seed={seed}&lambda={lam}&seasons={seasons}&instant=1&orbit=0"


hook_ok = False
runs = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                              env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
        # ---- base section: the page opened with no parameters, exactly as gate.py ----
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(out / "vid"),
                            record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        pg.add_init_script(INIT_JS)
        pg.on("console", lambda m: res["console_errors"].append(m.text[:300]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
        pg.on("requestfailed", lambda r: res["failed_requests"].append(r.url[:200]))
        t0 = time.time()
        probes = []
        try:
            pg.goto(html.as_uri(), wait_until="load", timeout=60000)
            res["load_s"] = round(time.time() - t0, 1)
            pr = block_probe(pg); probes.append(pr)
            if pr is None:
                mark_blocked(pg, "right after load")
            else:
                res["webgl"] = pg.evaluate("""() => { const c=[...document.querySelectorAll('canvas')];
                    for (const cv of c) { const gl = cv.getContext('webgl2') || cv.getContext('webgl');
                      if (gl) { const d = gl.getExtension('WEBGL_debug_renderer_info');
                        return {canvases: c.length, w: cv.width, h: cv.height, lost: gl.isContextLost(),
                                renderer: d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)}; } }
                    return {canvases: c.length, renderer: null}; }""")
                pg.wait_for_timeout(3000)
                pr = block_probe(pg); probes.append(pr)
                if pr is None:
                    mark_blocked(pg, "3 s after load")
                else:
                    pg.screenshot(path=str(out / "shot_3s.png"))
                    res["fps"] = pg.evaluate("""() => new Promise(r => { let n=0; const t0=performance.now();
                        function f(){ n++; if (performance.now()-t0 < 5000) requestAnimationFrame(f); else r(Math.round(n/5)); }
                        requestAnimationFrame(f); })""")
                    pg.screenshot(path=str(out / "shot_6s.png"))
                    res["default_page_stats"] = pg.evaluate(POLL_JS)
                    vx = pg.evaluate("() => window.__vx ? {maxGap: Math.round(window.__vx.maxGap), long: Math.round(window.__vx.long), nlong: window.__vx.nlong} : null")
                    if vx:
                        MT.update(max_frame_gap_ms=vx["maxGap"], longtask_ms=vx["long"], longtasks=vx["nlong"])
        except PWTimeout as e:
            MT["blocked"] = True; MT["stage"] = MT.get("stage") or "goto/evaluate/screenshot timeout"; MT["error"] = repr(e)[:160]
            if "blocked_s" not in MT:
                MT["blocked_s"] = f">{BLOCK_LIMIT_S}"
        MT["probe_s"] = [x for x in probes]
        if not MT["blocked"]:
            pg.wait_for_timeout(max(0, vsec - 11) * 1000)
            vpath = pg.video.path() if pg.video else None
            ctx.close()
            sd3, a = stats_img(out / "shot_3s.png"); sd6, b2 = stats_img(out / "shot_6s.png")
            from PIL import ImageChops, ImageStat
            diff = round(ImageStat.Stat(ImageChops.difference(a, b2)).mean[0], 2)
            res.update(stddev_3s=sd3, stddev_6s=sd6, frame_diff=diff)
            if vpath:
                shutil.move(vpath, out / "clip.webm")
                ffbin = os.path.expanduser("~/voxelbench/ffmpeg")
                ffbin = ffbin if os.path.exists(ffbin) else shutil.which("ffmpeg")
                res["video"] = "clip.webm"
                if ffbin:
                    r = subprocess.run([ffbin, "-nostdin", "-y", "-i", str(out / "clip.webm"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                        str(out / "clip.mp4")], capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
                    if r.returncode == 0:
                        res["video"] = "clip.mp4"
            checks = {"no_console_errors": not res["console_errors"], "no_page_errors": not res["page_errors"],
                      "webgl_context": bool((res.get("webgl") or {}).get("renderer")) and not (res.get("webgl") or {}).get("lost"),
                      "not_blank": sd6 > 8, "animated": diff > 0.5}
        else:  # the tab froze: none of the frame checks can be measured, they fail; the cause is named in main_thread
            checks = {"no_console_errors": not res["console_errors"], "no_page_errors": not res["page_errors"],
                      "webgl_context": bool((res.get("webgl") or {}).get("renderer")) and not (res.get("webgl") or {}).get("lost"),
                      "not_blank": False, "animated": False}
        checks["main_thread_ok"] = not MT["blocked"]
        res["base_checks"] = dict(checks)
        res["base_pass"] = all(checks.values())

        # ---- paper checks ----
        r5 = run_tree(b, q(1, 0.46, 5), "s5_l046", 5); runs.append(r5)
        s5 = r5.get("stats")
        hook_ok = bool(good_stats(s5) and s5.get("season") == 5)
        checks["treestats_hook"] = hook_ok
        tr = {"runs": runs}
        res["trees"] = tr
        S = lambda r: r.get("stats") if good_stats(r.get("stats")) else None
        if hook_ok:
            r15 = run_tree(b, q(1, 0.46, 15), "s15_l046", 15); runs.append(r15)
            r30a = run_tree(b, q(1, 0.46, 30), "s30_l046_a", 30, out / "shot_lambda046.png"); runs.append(r30a)
            r30b = run_tree(b, q(1, 0.46, 30), "s30_l046_b", 30); runs.append(r30b)
            r56 = run_tree(b, q(1, 0.56, 30), "s30_l056", 30, out / "shot_lambda056.png"); runs.append(r56)
            s15, s30a, s30b, s56 = S(r15), S(r30a), S(r30b), S(r56)
            if s5 and s15 and s30a:
                m5, m15, m30 = s5["metamers"], s15["metamers"], s30a["metamers"]
                tr["metamers_5_15_30"] = [m5, m15, m30]
                checks["metamers_grow"] = bool(m15 >= 0.95 * m5 and m30 >= 0.95 * m15 and m30 > m5)
            else:
                checks["metamers_grow"] = False
            tr["non_trivial"] = {"l046": s30a["metamers"] if s30a else None, "l056": s56["metamers"] if s56 else None}
            checks["non_trivial"] = bool(s30a and s56 and s30a["metamers"] >= 100 and s56["metamers"] >= 100)
            if s30a and s56 and s30a["height"] > 0 and s56["height"] > 0:
                ra, rb = s30a["crownWidth"] / s30a["height"], s56["crownWidth"] / s56["height"]
                tr["lambda_test"] = {"seed": 1, "seasons": 30,
                                     "l046": {"crownWidth": s30a["crownWidth"], "height": s30a["height"], "ratio": round(ra, 4), "metamers": s30a["metamers"]},
                                     "l056": {"crownWidth": s56["crownWidth"], "height": s56["height"], "ratio": round(rb, 4), "metamers": s56["metamers"]}}
                checks["lambda_effect"] = bool(ra > rb)
            else:
                checks["lambda_effect"] = False
            tr["determinism"] = {"a": s30a["metamers"] if s30a else None, "b": s30b["metamers"] if s30b else None,
                                 "height_a": s30a["height"] if s30a else None, "height_b": s30b["height"] if s30b else None}
            checks["deterministic"] = bool(s30a and s30b and s30a["metamers"] == s30b["metamers"])
            # reported only, never gating: runaway growth vs the reference simulation (~4.5k metamers at season 30, lambda 0.46)
            m46 = s30a["metamers"] if s30a else None; m56 = s56["metamers"] if s56 else None
            tr["growth_flags"] = {"metamers_30_l046": m46, "metamers_30_l056": m56, "reference_sim_l046": 4515,
                                  "runaway_growth_flag": bool((m46 or 0) > 25000 or (m56 or 0) > 25000)}
            if EXTRA:  # reported only, never gating
                ex = []
                for seed in (2, 3):
                    if time.time() - T_START > BUDGET:
                        ex.append({"seed": seed, "skipped": "budget"}); continue
                    ea = run_tree(b, q(seed, 0.46, 30), f"s30_l046_seed{seed}", 30)
                    eb = run_tree(b, q(seed, 0.56, 30), f"s30_l056_seed{seed}", 30)
                    runs += [ea, eb]
                    sa, sb = S(ea), S(eb)
                    if sa and sb and sa["height"] > 0 and sb["height"] > 0:
                        x = {"seed": seed, "ratio_l046": round(sa["crownWidth"] / sa["height"], 4), "ratio_l056": round(sb["crownWidth"] / sb["height"], 4),
                             "metamers_l046": sa["metamers"], "metamers_l056": sb["metamers"]}
                        x["holds"] = x["ratio_l046"] > x["ratio_l056"]
                        ex.append(x)
                    else:
                        ex.append({"seed": seed, "failed": True})
                tr["extra_seeds"] = ex
                holds = sum(1 for x in ex if x.get("holds")) + (1 if checks.get("lambda_effect") else 0)
                tr["lambda_effect_seeds_holding"] = f"{holds}/3"
                if s30a and ex and ex[0].get("metamers_l046") is not None:
                    tr["seed_changes_tree"] = ex[0]["metamers_l046"] != s30a["metamers"]
        else:  # hook missing: the dependent checks fail without more loads
            for k in ("metamers_grow", "non_trivial", "lambda_effect", "deterministic"):
                checks[k] = False
        tr["sim_seconds"] = {r["tag"]: r.get("sim_s") for r in runs}
        tr["run_errors"] = {r["tag"]: (r["errors"] + r["console"]) for r in runs if r["errors"] or r["console"]}
        tr["blocked_loads"] = [r["tag"] for r in runs if r.get("blocked")]
        try:
            b.close()
        except Exception:
            pass
    res["checks"] = checks
    res["pass"] = all(checks.values())
    res["elapsed_s"] = round(time.time() - T_START, 1)
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    summ = {k: res.get(k) for k in ("pass", "checks", "fps", "webgl", "load_s", "bytes")}
    summ["page_errors"] = res["page_errors"][:5]; summ["console_errors"] = res["console_errors"][:5]
    summ["failed_requests"] = res["failed_requests"][:5]
    summ["main_thread"] = MT
    t = res.get("trees", {})
    summ["trees"] = {k: t[k] for k in ("metamers_5_15_30", "lambda_test", "determinism", "lambda_effect_seeds_holding", "growth_flags", "run_errors", "sim_seconds", "blocked_loads") if k in t}
    if not hook_ok:
        summ["hook_help"] = [{k: r.get(k) for k in ("tag", "hook_problem", "timeout", "exception", "errors", "sim_s")} for r in runs][:2]
    print(json.dumps(summ))
    sys.stdout.flush()
    os._exit(0 if res["pass"] else 1)  # a blocked renderer must not keep the gate alive
except Exception as e:
    res["harness_error"] = repr(e)[:500]
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"harness_error": res["harness_error"]}))
    sys.stdout.flush()
    os._exit(2)
