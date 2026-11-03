#!/usr/bin/env python3
"""voxel bench FINAL gate for paper-to-demo episode 2 (HOVM vs Vicsek flocking, arXiv 2608.10844). Capsule, playwright chromium, CPU only
(canvas 2D page: --disable-gpu, the card is never touched).

DEPLOY PATH ON CAPSULE: ~/.vxfinal/gate_flock.py (dot directory outside ~/voxelbench and ~/benchmark-rig; same rule as HARDENING L33).
NOT in ~/voxelbench/scripts, not named in the prompt or in the /goal quality-gate command, the queue item carries no copy of the numbers below.
The in-run gate the agent sees is scripts/quick_gate_flock.py (direction of the paper's claim only, no thresholds).

usage: gate_flock.py <html> <outdir> [--video-seconds N (ignored)] [--no-extra] [--budget 900]
writes <outdir>/gate.json, shot_3s.png, shot_6s.png.  exit 0 = pass, 1 = fail, 2 = harness error.

PRE-REGISTERED 2026-11-03, BEFORE ANY MODEL RUN OF THE PROMPT hovm-v1. Sources of the numbers: the authors' published curves (Zenodo
10.5281/zenodo.20793454, Main-Fig3 sigma_OP_RHO10_alpha=0/1_n=1000.txt: N=1000, rho=10, v0=0.03, R=1, random start, 10^4 steps) and our independent
reference (capsule C sim vicp.c mode 2, ordered start, 1500 steps, mean of the last 300; JS reference page reference/flock/hovm_ref.html), table in the vault
note paper-ep2-status.md.
  protocol: page opened as <html>?autoplay=0&seed=1 (N = the page default, 1000, rho 10); window.flockRun(eta, 1500, 300, seed, 'ordered') -> {svm, hovm}
            (mean order parameter of the last 300 of 1500 steps, both panels' models, all particles start with heading 0).
  base      no_console_errors, no_page_errors, canvas_present, two_panels (the default page, 1920x1080: the left half and the right half of the 6 s screenshot
            each have grey-level stddev > 6 AND each half changes between the 3 s and the 6 s screenshot, mean abs diff > 0.8; reference 3.3 and 3.0, a page that
            animates only one panel reads 0.1 on the other half), animated (3 s vs 6 s frame diff > 0.5), main_thread_ok (answers within 30 s).
  hooks     flockstats_hook: window.flockStats() returns finite step, eta, N, svm, hovm with svm and hovm in [0,1];
            flockrun_hook:   window.flockRun(1.0, 300, 100, 1, 'ordered') returns finite svm and hovm in [0,1] and finishes in time.
  paper     (every one at seed 1)                                    reference (paper)
            svm_ordered_low : svm(eta 1.0)  in [0.90, 0.99]          0.955 (0.954)
            svm_smooth      : svm(eta 2.2)  in [0.70, 0.88]          0.79 (0.788): the pairwise model loses only a little order
            hovm_ordered_low: hovm(eta 1.0) >= 0.82                   0.93 (0.909)
            hovm_collapsed  : hovm(eta 2.2) <  0.15                   0.04 (0.036): the triadic model has collapsed to a gas
            gap_at_22       : svm(2.2) - hovm(2.2) >= 0.55            0.75
            deterministic   : flockRun(1.0, 400, 100, 1, 'ordered') twice returns the same svm and hovm (|diff| < 1e-9)
  pass = every check above. extras (reported, NOT gating, the anti-tuning check): seeds 2 and 3 at eta 1.0 and 2.2 (same rules, counted as holding or not),
            eta 1.6 and 1.8 for seed 1 (the transition window, reference collapses between 1.5 and 1.7 from a random start; metastable from an ordered start),
            random-start runs (4000 steps, last 500) at eta 1.0 and 2.2 (what the viewer sees when nobody lines the birds up first).
"""
import json, os, sys, time, math
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
EXTRA = "--no-extra" not in sys.argv
BUDGET = float(arg("--budget", 900))
RUN_TO = 150  # seconds per flockRun call
BLOCK_LIMIT_S = 30
T_START = time.time()
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
ARGS = ["--disable-gpu", "--disable-dev-shm-usage"]
res = {"html": str(html), "bytes": html.stat().st_size if html.exists() else 0, "console_errors": [], "page_errors": [], "gate": "gate_flock",
       "main_thread": {"blocked": False, "limit_s": BLOCK_LIMIT_S}}
MT = res["main_thread"]


def finish(code):
    res["elapsed_s"] = round(time.time() - T_START, 1)
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res.get(k) for k in ("pass", "checks", "flock", "extras", "main_thread", "elapsed_s")}))
    sys.stdout.flush(); os._exit(code)


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def block_probe(pg, limit_s=BLOCK_LIMIT_S):
    t = time.time()
    try:
        pg.wait_for_function("() => true", timeout=limit_s * 1000, polling=100)
        return round(time.time() - t, 2)
    except PWTimeout:
        return None


def stddev(png_or_im):
    from PIL import Image, ImageStat
    im = png_or_im if hasattr(png_or_im, "convert") else Image.open(png_or_im)
    return round(ImageStat.Stat(im.convert("L")).stddev[0], 2)


def run_flock(browser, eta, steps, avg, seed, start="ordered"):
    """fresh page (?autoplay=0&seed=<seed>), one window.flockRun call, returns a record. never raises."""
    rec = {"eta": eta, "steps": steps, "seed": seed, "errors": []}
    t0 = time.time()
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: rec["errors"].append(str(e)[:200]))
    try:
        pg.goto(html.as_uri() + f"?autoplay=0&seed={seed}", wait_until="load", timeout=60000)
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
    return rec


def vals(rec):
    o = rec.get("out") or {}
    if isinstance(o, dict) and fin(o.get("svm")) and fin(o.get("hovm")) and 0 <= o["svm"] <= 1 and 0 <= o["hovm"] <= 1:
        return o["svm"], o["hovm"]
    return None


checks = {}
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
            checks.update(no_console_errors=not res["console_errors"], no_page_errors=not res["page_errors"], canvas_present=False, two_panels=False,
                          animated=False, main_thread_ok=False)
        else:
            from PIL import Image, ImageChops, ImageStat
            a = Image.open(out / "shot_3s.png").convert("L"); z = Image.open(out / "shot_6s.png").convert("L")
            w, h = z.size
            sdl = stddev(z.crop((0, 0, w // 2, h))); sdr = stddev(z.crop((w // 2, 0, w, h)))
            dimg = ImageChops.difference(a, z)
            diff = round(ImageStat.Stat(dimg).mean[0], 2)
            dl = round(ImageStat.Stat(dimg.crop((0, 0, w // 2, h))).mean[0], 2); dr = round(ImageStat.Stat(dimg.crop((w // 2, 0, w, h))).mean[0], 2)
            res.update(stddev_left=sdl, stddev_right=sdr, frame_diff=diff, frame_diff_left=dl, frame_diff_right=dr)
            checks.update(no_console_errors=not res["console_errors"], no_page_errors=not res["page_errors"], canvas_present=bool(res.get("canvases")),
                          two_panels=bool(sdl > 6 and sdr > 6 and dl > 0.8 and dr > 0.8), animated=diff > 0.5, main_thread_ok=True)
        try:
            ctx.close()
        except Exception:
            pass
        flock = {"runs": []}
        res["flock"] = flock
        if not MT["blocked"]:
            def go(eta, steps, avg, seed, tag):
                r = run_flock(b, eta, steps, avg, seed); r["tag"] = tag; flock["runs"].append(r)
                if r.get("blocked"):
                    MT["blocked"] = True; MT["stage"] = f"flockRun({eta},{steps},{avg},{seed}) never returned"; checks["main_thread_ok"] = False
                return r
            # hooks
            c = go(1.0, 300, 100, 1, "hook_probe")
            checks["flockrun_hook"] = bool(vals(c) is not None)
            try:
                c2 = b.new_context(viewport={"width": 1280, "height": 720}); q2 = c2.new_page(); q2.goto(html.as_uri() + "?autoplay=0&seed=1", wait_until="load", timeout=60000)
                st = q2.evaluate("() => { try { const s = window.flockStats(); return s ? JSON.parse(JSON.stringify(s)) : null } catch (e) { return {__err: String(e)} } }")
                c2.close()
            except Exception as e:
                st = {"__err": repr(e)[:120]}
            flock["flockstats"] = st
            checks["flockstats_hook"] = bool(isinstance(st, dict) and all(fin(st.get(k)) for k in ("step", "eta", "N", "svm", "hovm")) and 0 <= st["svm"] <= 1 and 0 <= st["hovm"] <= 1)
            if checks["flockrun_hook"] and not MT["blocked"]:
                r10 = go(1.0, 1500, 300, 1, "eta1.0_s1"); r22 = go(2.2, 1500, 300, 1, "eta2.2_s1")
                v10, v22 = vals(r10), vals(r22)
                flock["eta1.0"] = v10; flock["eta2.2"] = v22
                if v10 and v22:
                    s10, h10 = v10; s22, h22 = v22
                    checks.update(svm_ordered_low=0.90 <= s10 <= 0.99, svm_smooth=0.70 <= s22 <= 0.88, hovm_ordered_low=h10 >= 0.82,
                                  hovm_collapsed=h22 < 0.15, gap_at_22=(s22 - h22) >= 0.55)
                else:
                    checks.update(svm_ordered_low=False, svm_smooth=False, hovm_ordered_low=False, hovm_collapsed=False, gap_at_22=False)
                if not MT["blocked"]:
                    d1 = go(1.0, 400, 100, 1, "det_a"); d2 = go(1.0, 400, 100, 1, "det_b")
                    va, vb = vals(d1), vals(d2)
                    flock["determinism"] = [va, vb]
                    checks["deterministic"] = bool(va and vb and abs(va[0] - vb[0]) < 1e-9 and abs(va[1] - vb[1]) < 1e-9)
                if EXTRA and not MT["blocked"] and time.time() - T_START < BUDGET - 240:
                    ex = {}
                    for seed in (2, 3):
                        for eta in (1.0, 2.2):
                            if time.time() - T_START > BUDGET - 120:
                                break
                            r = go(eta, 1500, 300, seed, f"extra_eta{eta}_s{seed}"); ex[f"eta{eta}_s{seed}"] = vals(r)
                    for eta in (1.6, 1.8):
                        if time.time() - T_START > BUDGET - 120:
                            break
                        r = go(eta, 1500, 300, 1, f"extra_eta{eta}_s1"); ex[f"eta{eta}_s1"] = vals(r)
                    for eta in (1.0, 2.2):  # the picture: nobody is told to line up. reported only (a random start orders slowly near the transition)
                        if time.time() - T_START > BUDGET - 120:
                            break
                        r = run_flock(b, eta, 4000, 500, 1, "random"); r["tag"] = f"extra_random_eta{eta}"; flock["runs"].append(r); ex[f"random_start_eta{eta}"] = vals(r)
                    hold = 0; tot = 0
                    for seed in (2, 3):
                        a_, b_ = ex.get(f"eta1.0_s{seed}"), ex.get(f"eta2.2_s{seed}")
                        if a_ and b_:
                            tot += 1
                            hold += int(0.90 <= a_[0] <= 0.99 and 0.70 <= b_[0] <= 0.88 and a_[1] >= 0.82 and b_[1] < 0.15 and (b_[0] - b_[1]) >= 0.55)
                    res["extras"] = {"values": ex, "seeds_holding": f"{hold}/{tot}"}
        for r in flock["runs"]:
            r["errors"] = r["errors"][:2]
        try:
            b.close()
        except Exception:
            pass
    res["checks"] = checks
    res["pass"] = bool(checks) and all(checks.values())
    finish(0 if res["pass"] else 1)
except Exception as e:
    res["harness_error"] = repr(e)[:500]
    finish(2)
