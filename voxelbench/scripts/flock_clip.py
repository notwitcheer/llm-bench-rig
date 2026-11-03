#!/usr/bin/env python3
"""fixed-camera clip for paper-to-demo episode 2 (capsule, playwright chromium, CPU only: canvas 2D page, --disable-gpu; nothing touches the card).

usage: flock_clip.py <html> <outdir> [--seed 1] [--eta-lo 1.0] [--eta-hi 2.2] [--switch 6] [--seconds 20] [--spf 6] [--fps 25] [--no-caption]
drives the page's OWN hooks only (never edits the model's file): ?autoplay=0&seed=<seed>&start=ordered, then an ABSOLUTE schedule in simulation steps,
one screenshot per frame at --spf steps per frame (so the film is identical on every machine, no dependence on the page's frame rate):
  0 .. --switch s     eta = --eta-lo (both flocks fly: SVM left, HOVM right, same birds, same box scale)
  --switch .. end     setEta(--eta-hi): the same noise hits both; SVM keeps a (looser) flock, HOVM falls apart into a gas
the page layout is the camera: both panels show the full L x L box at the same pixel scale by the task text; the script records the measured canvas size.
then ffmpeg (~/voxelbench/ffmpeg, x264, crf 18) -> flock.mp4, plus a caption band (PIL, Manrope; big live order numbers read from flockStats per frame, a neutral caption that names the noise jump only after it happens) under the page frame unless --no-caption, a 4-frame contact
sheet flock_sheet.png, the last frame flock_last.png (full-res HUD read for leaks: no paths, usernames, hostnames) and flock_clip.json (phi per second, method).
"""
import json, os, subprocess, sys, time
from pathlib import Path


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


html = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
seed = int(arg("--seed", 1)); eta_lo = float(arg("--eta-lo", 1.0)); eta_hi = float(arg("--eta-hi", 2.2)); t_switch = float(arg("--switch", 6))
seconds = float(arg("--seconds", 20)); spf = int(arg("--spf", 6)); fps = int(arg("--fps", 25)); CAPTION = "--no-caption" not in sys.argv
FF = os.path.expanduser("~/voxelbench/ffmpeg")
FONT = os.path.expanduser("~/voxelbench/runs/vs-3way/fonts/Manrope.ttf")
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
frames = out / "frames"; frames.mkdir(exist_ok=True)
for f in frames.glob("*.png"):
    f.unlink()
res = {"html": str(html), "seed": seed, "eta_lo": eta_lo, "eta_hi": eta_hi, "switch_s": t_switch, "seconds": seconds, "steps_per_frame": spf, "fps": fps, "errors": [], "phi_per_second": []}
n_frames = int(seconds * fps); switch_frame = int(t_switch * fps); per_frame = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--disable-gpu", "--disable-dev-shm-usage"], executable_path=CHROME, chromium_sandbox=False,
                          env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    ctx = b.new_context(viewport={"width": 1920, "height": 1080})
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:300]))
    pg.goto(html.as_uri() + f"?autoplay=0&seed={seed}&start=ordered&eta={eta_lo}", wait_until="load", timeout=60000)
    pg.wait_for_timeout(500)
    need = pg.evaluate("() => ['flockStep', 'setEta', 'flockReset', 'flockStats'].filter(n => typeof window[n] !== 'function')")
    if need:
        res["errors"].append("missing hooks: " + ",".join(need)); (out / "flock_clip.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res)); sys.exit(2)
    pg.evaluate(f"() => {{ window.flockReset({seed}, 'ordered'); window.setEta({eta_lo}); }}")
    res["canvases"] = pg.evaluate("() => [...document.querySelectorAll('canvas')].map(c => { const r = c.getBoundingClientRect(); return [c.width, c.height, Math.round(r.width), Math.round(r.height)]; })")
    t0 = time.time()
    for k in range(n_frames):
        if k == switch_frame:
            pg.evaluate(f"() => window.setEta({eta_hi})")
        pg.evaluate(f"() => window.flockStep({spf})")
        pg.screenshot(path=str(frames / f"f{k:05d}.png"))
        st = pg.evaluate("() => window.flockStats()"); per_frame.append((st["svm"], st["hovm"], st["eta"], st["step"]))
        if k % fps == 0:
            s = pg.evaluate("() => window.flockStats()")
            res["phi_per_second"].append({"t": k // fps, "eta": round(s["eta"], 2), "svm": round(s["svm"], 3), "hovm": round(s["hovm"], 3), "step": s["step"]})
    res["capture_s"] = round(time.time() - t0, 1)
    pg.screenshot(path=str(out / "flock_last.png"))
    b.close()
if CAPTION:
    band = 170
    big = ImageFont.truetype(FONT, 58) if os.path.exists(FONT) else ImageFont.load_default()
    small = ImageFont.truetype(FONT, 34) if os.path.exists(FONT) else ImageFont.load_default()
    for k, fp in enumerate(sorted(frames.glob("f*.png"))):
        im = Image.open(fp).convert("RGB"); W, H = im.size
        canvas = Image.new("RGB", (W, H + band), (11, 14, 20)); canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas); sv, hv, et, stp = per_frame[k]
        d.text((60, H + 14), f"order {sv:.2f}", fill=(127, 209, 255), font=big)
        d.text((W // 2 - 200, H + 14), f"noise {et:.1f}", fill=(230, 233, 239), font=big)
        d.text((W - 560, H + 14), f"order {hv:.2f}", fill=(255, 184, 107), font=big)
        txt = "same 1,000 birds, same noise, same box scale" if k < switch_frame else f"the same noise jump ({eta_lo:.1f} to {eta_hi:.1f}) hits both flocks"
        d.text((60, H + 100), txt, fill=(170, 178, 192), font=small)
        canvas.save(fp)
cmd = [FF, "-nostdin", "-y", "-framerate", str(fps), "-i", str(frames / "f%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out / "flock.mp4")]
r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
res["ffmpeg_rc"] = r.returncode
ims = [Image.open(frames / f"f{k:05d}.png") for k in (0, switch_frame - 1, switch_frame + fps * 2, n_frames - 1)]
w, h = ims[0].size; sheet = Image.new("RGB", (w, h * 2 // 1)); sheet = Image.new("RGB", (w, h))
tw, th = w // 2, h // 2
for i, im in enumerate(ims):
    sheet.paste(im.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
sheet.save(out / "flock_sheet.png")
(out / "flock_clip.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("capture_s", "ffmpeg_rc", "canvases", "errors", "phi_per_second")}))
