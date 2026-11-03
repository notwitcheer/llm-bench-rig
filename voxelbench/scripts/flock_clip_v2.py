#!/usr/bin/env python3
"""flock clip v2 (recut, CPU only, 1920x1080 16:9) from the MODEL'S OWN page. Never edits the page.
usage: flock_clip_v2.py <html> <out.mp4> [--seed 1] [--eta-lo 1.0] [--eta-hi 2.2] [--preroll 696]
Same hooks as v1 (flockReset / setEta / flockStep / flockStats), same seed, same ordered start. The two simulation canvases are read from the
page with canvas.toDataURL (the model's own pixels, 460x460, scaled up with Lanczos; the page exposes no bird-scale hook, so birds are not redrawn).
The page's status bar and help line are never in frame (canvas-only capture), so no CSS injection was needed. Captions are drawn here from the
live flockStats values per frame. The chart is redrawn here from the same per-frame hook values (blue drawn over a thicker red) because the page
draws red over blue and hides the SVM line in the ordered phase.
Schedule (25 fps): 1.0 s teaser (the last 25 frames of this same run, tagged) -> 35 frames ordered eta_lo -> setEta(eta_hi) -> 100 frames at 3 steps/frame
(slow motion) -> rest at 6 steps/frame.
"""
import base64, io, json, os, subprocess, sys, time
from pathlib import Path


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


html = Path(sys.argv[1]).resolve(); outp = Path(sys.argv[2])
seed = int(arg("--seed", 1)); eta_lo = float(arg("--eta-lo", 1.0)); eta_hi = float(arg("--eta-hi", 2.2)); preroll = int(arg("--preroll", 696))
N_PRE, N_SLOW, N_REST, N_TEASE, FPS = 35, 100, int(arg("--rest", 490)), 25, 25
FF = os.path.expanduser("~/voxelbench/ffmpeg")
FONT = os.path.expanduser("~/voxelbench/runs/vs-3way/fonts/Manrope.ttf")
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (11, 14, 20); BLUE = (77, 166, 255); RED = (255, 90, 90); WHITE = (240, 243, 248); GREY = (150, 160, 175); DIM = (60, 70, 84)


def font(sz, bold=True):
    try:
        f = ImageFont.truetype(FONT, sz)
        try:
            f.set_variation_by_name("ExtraBold" if bold else "Medium")
        except Exception:
            try:
                f.set_variation_by_axes([800 if bold else 500])
            except Exception:
                pass
        return f
    except Exception:
        return ImageFont.load_default()


# ---------- capture ----------
cap = []   # (png_svm, png_hovm, svm, hovm, eta)
res = {"html": str(html), "seed": seed, "eta_lo": eta_lo, "eta_hi": eta_hi, "preroll_steps": preroll, "errors": []}
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--disable-gpu", "--disable-dev-shm-usage"], executable_path=CHROME, chromium_sandbox=False,
                          env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    pg = b.new_context(viewport={"width": 1920, "height": 1080}).new_page()
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:300]))
    pg.goto(html.as_uri() + f"?autoplay=0&seed={seed}&start=ordered&eta={eta_lo}", wait_until="load", timeout=60000)
    pg.wait_for_timeout(500)
    need = pg.evaluate("() => ['flockStep','setEta','flockReset','flockStats'].filter(n => typeof window[n] !== 'function')")
    if need:
        print("missing hooks", need); sys.exit(2)
    res["hooks"] = pg.evaluate("() => Object.keys(window).filter(k => /flock|setEta|scale|bird|zoom/i.test(k))")
    pg.evaluate(f"() => {{ window.flockReset({seed}, 'ordered'); window.setEta({eta_lo}); window.flockStep({preroll}); }}")
    grab = """() => ['cSvm','cHovm'].map(id => document.getElementById(id).toDataURL('image/png'))"""
    sched = [6] * N_PRE + [3] * N_SLOW + [6] * N_REST
    t0 = time.time()
    for k, spf in enumerate(sched):
        if k == N_PRE:
            pg.evaluate(f"() => window.setEta({eta_hi})")
        pg.evaluate(f"() => window.flockStep({spf})")
        st = pg.evaluate("() => window.flockStats()")
        a, c = pg.evaluate(grab)
        cap.append((a, c, st["svm"], st["hovm"], st["eta"], st["step"]))
    res["capture_s"] = round(time.time() - t0, 1)
    b.close()
M = len(cap)
print("captured", M, res["capture_s"], "s", res["hooks"], res["errors"], flush=True)

BOX = 660; GAP_C = 340
XL = (W - 2 * BOX - GAP_C) // 2; XR = XL + BOX + GAP_C; YB = 100
f_lab, f_ord, f_nz, f_nzl, f_small, f_cred, f_tag, f_ax = font(50), font(96), font(150), font(40, False), font(35, False), font(29, False), font(34), font(26, False)
CH_X0, CH_X1, CH_Y0, CH_Y1 = 130, 1790, 884, 1008


def dec(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert("RGB").resize((BOX, BOX), Image.LANCZOS)


def center(d, cx, y, txt, f, fill):
    w = d.textlength(txt, font=f); d.text((cx - w / 2, y), txt, font=f, fill=fill)


def compose(k, upto, tease):
    """k = frame index shown in the boxes, upto = chart drawn up to this frame index (inclusive)."""
    a, c, sv, hv, eta, step = cap[k]
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    im.paste(dec(a), (XL, YB)); im.paste(dec(c), (XR, YB))
    d.rectangle([XL - 2, YB - 2, XL + BOX + 1, YB + BOX + 1], outline=(43, 53, 67), width=2)
    d.rectangle([XR - 2, YB - 2, XR + BOX + 1, YB + BOX + 1], outline=(43, 53, 67), width=2)
    # labels above boxes
    center(d, XL + BOX // 2, 14, "pairwise (standard)", f_lab, BLUE)
    center(d, XR + BOX // 2, 14, "three-bird (HOVM)", f_lab, RED)
    # order numbers below boxes
    d.text((XL + BOX // 2, YB + BOX + 96), f"order {sv:.2f}", font=f_ord, fill=BLUE, anchor="ms")
    d.text((XR + BOX // 2, YB + BOX + 96), f"order {hv:.2f}", font=f_ord, fill=RED, anchor="ms")
    # centre column: noise
    cx = W // 2
    center(d, cx, 150, "noise", f_nzl, GREY)
    d.text((cx, 312), f"{eta:.1f}", font=f_nz, fill=WHITE, anchor="ms")
    lines = ([f"jumped from {eta_lo:.1f}"] if eta > eta_lo + 0.01 else ["steady"]) + ["same 1,000 birds", "same noise", "each side"]
    for i, t in enumerate(lines):
        center(d, cx, 336 + 40 * i, t, f_small, GREY)
    center(d, cx, 510, "colour = heading", f_small, (115, 125, 140))
    center(d, cx, 550, "order 1 = aligned", f_small, (115, 125, 140))
    center(d, cx, 590, "order 0 = random", f_small, (115, 125, 140))
    if tease:
        tw = d.textlength("PREVIEW", font=f_tag)
        d.rounded_rectangle([cx - 150, 648, cx + 150, 738], radius=14, fill=(190, 120, 15))
        center(d, cx, 654, "PREVIEW", f_tag, (255, 255, 255)); center(d, cx, 696, "end of this run", f_small, (255, 255, 255))
    elif k >= N_PRE and k < N_PRE + N_SLOW:
        center(d, cx, 660, "slow motion", f_tag, (255, 200, 100))
    # chart (blue over thicker red)
    d.rectangle([CH_X0, CH_Y0, CH_X1, CH_Y1], fill=(15, 19, 26), outline=(35, 42, 53))
    for g in (0.0, 0.5, 1.0):
        y = CH_Y1 - 8 - (CH_Y1 - CH_Y0 - 16) * g
        d.line([(CH_X0, y), (CH_X1, y)], fill=(32, 40, 52), width=1)
        d.text((CH_X0 - 10, y), f"{g:.1f}", font=f_ax, fill=GREY, anchor="rm")
    xj = CH_X0 + (CH_X1 - CH_X0) * N_PRE / (M - 1)
    d.line([(xj, CH_Y0), (xj, CH_Y1)], fill=(110, 120, 135), width=2)
    def pts(idx):
        return [(CH_X0 + (CH_X1 - CH_X0) * i / (M - 1), CH_Y1 - 8 - (CH_Y1 - CH_Y0 - 16) * min(1, max(0, cap[i][idx]))) for i in range(upto + 1)]
    if upto >= 1:
        d.line(pts(3), fill=RED, width=16, joint="curve")
        d.line(pts(2), fill=(15, 19, 26), width=8, joint="curve")
        d.line(pts(2), fill=BLUE, width=4, joint="curve")
    d.text((CH_X1 - 12, CH_Y0 + 80), "HOVM", font=f_ax, fill=RED, anchor="rm")
    d.text((CH_X1 - 100, CH_Y0 + 80), "pairwise", font=f_ax, fill=BLUE, anchor="rm")
    # credit
    center(d, W // 2, 1030, "Qwen3.8-27B built this page in one Hermes Agent session on one RTX 5090; paper arXiv:2608.10844", f_cred, (130, 140, 155))
    return im


proc = subprocess.Popen([FF, "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
                         "-c:v", "libx264", "-preset", "slow", "-crf", arg("--crf", "23"), "-maxrate", arg("--maxrate", "2800k"), "-bufsize", "5600k",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(outp)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
order = [("t", M - N_TEASE + i) for i in range(N_TEASE)] + [("c", 0)] * 7 + [("m", i) for i in range(M)]
sheet_frames = {}
for n, (kind, k) in enumerate(order):
    if kind == "c":
        im = Image.new("RGB", (W, H), BG); dd = ImageDraw.Draw(im)
        dd.text((W // 2, H // 2 - 20), "from the start", font=font(110), fill=WHITE, anchor="mm")
        dd.text((W // 2, H // 2 + 70), "same page, same 1,000 birds, same seed", font=f_nzl, fill=GREY, anchor="mm")
    else:
        im = compose(k, M - 1 if kind == "t" else k, kind == "t")
    proc.stdin.write(im.tobytes())
    if n in (0, 12, 25, 28, 32, 50, 70, 100, 160, 400, len(order) - 1):
        im.save(outp.parent / f"v2_frame_{n:04d}.png")
proc.stdin.close(); err = proc.stderr.read().decode()[-400:]; rc = proc.wait()
res.update({"ffmpeg_rc": rc, "frames": len(order), "seconds": len(order) / FPS, "bytes": outp.stat().st_size,
            "phi": [{"frame": i, "eta": round(cap[i][4], 2), "svm": round(cap[i][2], 3), "hovm": round(cap[i][3], 3), "step": cap[i][5]} for i in (0, N_PRE - 1, N_PRE, N_PRE + 25, N_PRE + 50, N_PRE + 100, N_PRE + 200, M - 1)]})
(outp.parent / "flock-v2.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("ffmpeg_rc", "frames", "seconds", "bytes", "phi", "errors")}), err)
