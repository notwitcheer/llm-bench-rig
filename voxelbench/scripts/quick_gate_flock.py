#!/usr/bin/env python3
"""in-run quality gate for the flocking task (capsule, playwright chromium, CPU only). This is the command registered with the Hermes /goal gate;
the agent can read this file and run it itself. exit 0 = pass, 1 = fail (the JSON line says why and what to do), 2 = harness error.

usage: quick_gate_flock.py <html> <outdir> [--video-seconds N (ignored)]

what it checks, in order (the task text names every hook below):
  1. the file exists at the exact expected path (otherwise it names the path and lists any html it finds elsewhere under runs/)
  2. the page opens (default URL, no parameters) with no console errors and no page errors, has a canvas, shows TWO panels (left and right half of the
     screen both have content AND both halves change between 3 s and 6 s), the picture changes, and the main thread keeps answering
  3. window.flockStats() answers with finite numbers; window.flockRun(eta, steps, avg, seed, start) returns finite svm and hovm in [0,1]
  4. the page stays alive at other noise values: flockRun at eta 0 and at eta 3.5 return in time (no freeze, no NaN)
  5. the paper's claim, as the task text states it: seed 1, ordered start, 800 steps (mean of the last 200): at eta 1.0 both models are ordered (order > 0.6);
     at eta 2.2 the triadic model has lost far more order than the pairwise model (svm - hovm > 0.30) while the pairwise model is still clearly ordered (> 0.5)
exit codes and messages are written for the agent: they say what is wrong and which part of the task text to re-read.
"""
import json, os, sys, time, math
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).absolute()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
RUN_TO = float(arg("--run-timeout", 90))
BLOCK_LIMIT_S = 30
res = {"html": str(html), "gate": "quick_gate_flock", "console_errors": [], "page_errors": [], "main_thread": {"blocked": False, "limit_s": BLOCK_LIMIT_S}}
MT = res["main_thread"]
T_START = time.time()


def finish(code, summ):
    res["elapsed_s"] = round(time.time() - T_START, 1)
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(summ)); sys.stdout.flush()
    os._exit(code)


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
    res.update(error="no_html_at_expected_path", expected=str(html), found_elsewhere=stray, **{"pass": False})
    finish(1, {"pass": False, "error": "no_html_at_expected_path", "expected_path": str(html), "found_elsewhere": stray,
               "fix": "the page must be saved at exactly expected_path. copy that path from the task text, never retype it" +
                      (": move or rewrite your page from found_elsewhere[0] to expected_path" if stray else "")})

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = ["--disable-gpu", "--disable-dev-shm-usage"]


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def block_probe(pg, limit_s=BLOCK_LIMIT_S):
    t = time.time()
    try:
        pg.wait_for_function("() => true", timeout=limit_s * 1000, polling=100)
        return round(time.time() - t, 2)
    except PWTimeout:
        return None


def sd(im):
    from PIL import ImageStat
    return round(ImageStat.Stat(im.convert("L")).stddev[0], 2)


def run_flock(browser, eta, steps, avg, seed, start="ordered", query=""):
    rec = {"eta": eta, "steps": steps, "seed": seed, "errors": []}
    t0 = time.time()
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: rec["errors"].append(str(e)[:200]))
    try:
        pg.goto(html.as_uri() + f"?autoplay=0&seed={seed}{query}", wait_until="load", timeout=60000)
        if not pg.evaluate("() => typeof window.flockRun === 'function'"):
            rec["nohook"] = True
        else:
            pg.evaluate("([a,b,c,d,e]) => { window.__fr = null; setTimeout(() => { try { window.__fr = window.flockRun(a,b,c,d,e) } catch (x) { window.__fr = {__err: String(x).slice(0,200)} } }, 0); return 0 }",
                        [eta, steps, avg, seed, start])
            pg.wait_for_function("() => window.__fr !== null", timeout=RUN_TO * 1000, polling=250)
            rec["out"] = pg.evaluate("() => JSON.parse(JSON.stringify(window.__fr))")
    except Exception as e:
        rec["exception"] = repr(e)[:240]
        if isinstance(e, PWTimeout) or "Timeout" in repr(e):
            rec["blocked"] = True
        if "crash" in repr(e).lower():
            rec["crashed"] = True
    rec["sim_s"] = round(time.time() - t0, 1)
    if not rec.get("blocked"):
        try:
            ctx.close()
        except Exception:
            pass
    rec["errors"] = rec["errors"][:2]
    return rec


def vals(rec):
    o = rec.get("out") or {}
    if isinstance(o, dict) and fin(o.get("svm")) and fin(o.get("hovm")) and 0 <= o["svm"] <= 1 and 0 <= o["hovm"] <= 1:
        return o["svm"], o["hovm"]
    return None


checks = {}; hints = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=ARGS, executable_path=CHROME, chromium_sandbox=False,
                              env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
        ctx = b.new_context(viewport={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        pg.on("console", lambda m: res["console_errors"].append(m.text[:300]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
        t0 = time.time()
        try:
            pg.goto(html.as_uri(), wait_until="load", timeout=60000)
            res["load_s"] = round(time.time() - t0, 1)
            if block_probe(pg) is None:
                MT.update(blocked=True, stage="right after load")
            else:
                res["canvases"] = pg.evaluate("() => [...document.querySelectorAll('canvas')].map(c => [c.width, c.height])")
                pg.wait_for_timeout(3000)
                if block_probe(pg) is None:
                    MT.update(blocked=True, stage="3 s after load")
                else:
                    pg.screenshot(path=str(out / "shot_3s.png")); pg.wait_for_timeout(3000); pg.screenshot(path=str(out / "shot_6s.png"))
        except PWTimeout as e:
            MT.update(blocked=True, stage=MT.get("stage") or "page call timed out", error=repr(e)[:120])
        if MT["blocked"]:
            checks.update(no_console_errors=not res["console_errors"], no_page_errors=not res["page_errors"], page_shows_two_panels=False, picture_changes=False, main_thread_ok=False)
            hints.append("the page FREEZES the browser tab: the main thread did not answer for 30 s (" + str(MT["stage"]) + "). the usual cause is an infinite or runaway loop, or "
                         "far too much work per frame (an all-pairs search for 1000 birds is fine, three nested loops over all birds are not). do heavy work in small slices per frame")
        else:
            from PIL import Image, ImageChops, ImageStat
            a = Image.open(out / "shot_3s.png").convert("L"); z = Image.open(out / "shot_6s.png").convert("L"); w, h = z.size
            sdl, sdr = sd(z.crop((0, 0, w // 2, h))), sd(z.crop((w // 2, 0, w, h)))
            dimg = ImageChops.difference(a, z)
            diff = round(ImageStat.Stat(dimg).mean[0], 2)
            dl = round(ImageStat.Stat(dimg.crop((0, 0, w // 2, h))).mean[0], 2); dr = round(ImageStat.Stat(dimg.crop((w // 2, 0, w, h))).mean[0], 2)
            res.update(stddev_left=sdl, stddev_right=sdr, frame_diff=diff, frame_diff_left=dl, frame_diff_right=dr)
            checks.update(no_console_errors=not res["console_errors"], no_page_errors=not res["page_errors"], page_shows_two_panels=bool(sdl > 6 and sdr > 6 and dl > 0.8 and dr > 0.8),
                          picture_changes=diff > 0.5, main_thread_ok=True)
            if not checks["no_console_errors"] or not checks["no_page_errors"]:
                hints.append("the page logs errors on load: fix them first (see console_errors / page_errors)")
            if not checks["page_shows_two_panels"]:
                hints.append(f"the default page must show two panels side by side, the standard Vicsek model on the left and the higher-order model on the right: the left half of the screen has contrast {sdl} and the right half {sdr} (both must be above 6), and the picture changed by {dl} on the left half and {dr} on the right half between 3 s and 6 s (both must be above 0.8: both panels must animate)")
            if not checks["picture_changes"]:
                hints.append("the picture at 3 s and at 6 s is the same: the default page must autoplay the simulation")
        try:
            ctx.close()
        except Exception:
            pass
        runs = []
        res["runs"] = runs
        if not MT["blocked"]:
            def go(eta, steps, avg, seed, tag, query=""):
                r = run_flock(b, eta, steps, avg, seed, "ordered", query); r["tag"] = tag; runs.append(r)
                if r.get("blocked"):
                    MT["blocked"] = True; MT["stage"] = f"flockRun({eta},{steps},{avg},{seed}) never returned within {RUN_TO:.0f} s"; checks["main_thread_ok"] = False
                return r
            r0 = go(1.0, 300, 100, 1, "hook")
            checks["hooks"] = bool(vals(r0) is not None)
            try:
                c2 = b.new_context(viewport={"width": 1280, "height": 720}); q2 = c2.new_page(); q2.goto(html.as_uri() + "?autoplay=0&seed=1", wait_until="load", timeout=60000)
                st = q2.evaluate("() => { try { const s = window.flockStats(); return s ? JSON.parse(JSON.stringify(s)) : null } catch (e) { return {__err: String(e)} } }")
                c2.close()
            except Exception as e:
                st = {"__err": repr(e)[:120]}
            res["flockstats"] = st
            ok_stats = bool(isinstance(st, dict) and all(fin(st.get(k)) for k in ("step", "eta", "N", "svm", "hovm")))
            checks["hooks"] = bool(checks["hooks"] and ok_stats)
            if not checks["hooks"]:
                hints.append("hooks: window.flockStats() must return finite step, eta, N, svm, hovm, and window.flockRun(1.0, 300, 100, 1, 'ordered') must return {svm, hovm} with both in [0,1]; "
                             f"flockStats gave {json.dumps(st)[:160]}, flockRun gave {json.dumps(r0.get('out') or r0.get('exception') or r0.get('nohook'))[:200]}")
            if checks["hooks"] and not MT["blocked"]:
                lz = [go(0.0, 200, 50, 1, "eta0"), go(3.5, 300, 100, 1, "eta3.5")]
                checks["alive_at_other_noise"] = bool(all(vals(r) is not None for r in lz) and not MT["blocked"])
                if not checks["alive_at_other_noise"]:
                    hints.append("flockRun at eta 0 or eta 3.5 returned no usable numbers or froze the tab (" + json.dumps([r.get("out") or r.get("exception") for r in lz])[:240] + "). the model must run at any noise from 0 to 5")
                if not MT["blocked"]:
                    lo = go(1.0, 800, 200, 1, "claim_lo"); hi = go(2.2, 800, 200, 1, "claim_hi")
                    vl, vh = vals(lo), vals(hi)
                    if vl and vh:
                        res["claim_check"] = {"eta1.0": {"svm": round(vl[0], 4), "hovm": round(vl[1], 4)}, "eta2.2": {"svm": round(vh[0], 4), "hovm": round(vh[1], 4)}}
                        checks["claim_direction"] = bool(vl[0] > 0.6 and vl[1] > 0.6 and vh[0] > 0.5 and (vh[0] - vh[1]) > 0.30)
                        if not checks["claim_direction"]:
                            hints.append(f"the paper's claim does not show in your simulation (seed 1, ordered start, 800 steps, mean of the last 200): at eta 1.0 svm {vl[0]:.3f} and hovm {vl[1]:.3f} "
                                         f"(both must be ordered, above 0.6); at eta 2.2 svm {vh[0]:.3f} and hovm {vh[1]:.3f} (the higher-order model must have lost far more order: "
                                         "svm - hovm above 0.30, with svm still above 0.5). re-read requirements 2 and 3 of the task text: which particles j and k does the triadic sum run over, what is "
                                         "normalised by what, and is noise added to the heading as eta * (u - 1/2) with u uniform in [0,1)? never special-case eta values or seeds")
                    else:
                        checks["claim_direction"] = False
                        hints.append("the claim check could not read svm and hovm from flockRun at eta 1.0 / 2.2 (" + json.dumps([lo.get("exception"), hi.get("exception")])[:200] + ")")
        try:
            b.close()
        except Exception:
            pass
    res["checks"] = checks
    res["pass"] = bool(checks) and all(checks.values())
    summ = {"pass": res["pass"], "checks": checks, "claim_check": res.get("claim_check"), "load_s": res.get("load_s"), "main_thread": MT,
            "page_errors": res["page_errors"][:3], "console_errors": res["console_errors"][:3], "sim_seconds": {r["tag"]: r.get("sim_s") for r in runs}}
    if hints:
        summ["fix"] = hints
    finish(0 if res["pass"] else 1, summ)
except Exception as e:
    res["harness_error"] = repr(e)[:500]
    finish(2, {"harness_error": res["harness_error"]})
