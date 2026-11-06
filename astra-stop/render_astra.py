#!/usr/bin/env python3
"""render_astra.py: CPU split-screen render (1920x1080, 16:9, ~65 s) of the two arms drawing pelicans, from the recorded per-try token timelines.
usage: render_astra.py <run_dir> <out.mp4> [--rounds 5] [--ffmpeg PATH]
Left = unedited, right = edited. Round r replays seed r (the first seeds, never picked) from each arm's own saved timeline at ONE shared playback
scale (the arms ran one after the other on one card; the badge says so and is not a speed claim). Then a 4x4 gallery per arm (seeds 1-16) and a result card
whose numbers come from summary(.pre).json. A MOCK file in the run dir puts a MOCK DATA banner on every frame."""
import bisect, io, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw, ImageFont
import resvg_py
from astra_common import extract_svg

W, H, FPS = 1920, 1080, 25
BG = (14, 17, 22); FG = (240, 242, 245); DIM = (120, 128, 140); RED = (235, 87, 87); GRN = (72, 214, 120); AMB = (245, 190, 70)
F = "/usr/share/fonts/truetype/dejavu/"
def font(sz, bold=True, mono=False):
    return ImageFont.truetype(F + ("DejaVuSansMono-Bold.ttf" if mono and bold else "DejaVuSansMono.ttf" if mono else "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)
FB = {k: font(k) for k in (28, 34, 44, 60, 150)}; FR = {k: font(k, False) for k in (22, 26)}; FM = font(23, False, True)


def load_try(run, arm, seed):
    base = os.path.join(run, "results", arm, f"pelican-{seed:02d}")
    if not os.path.exists(base + ".json"):
        return None
    r = json.load(open(base + ".json")); tk = json.load(open(base + ".tok.json"))
    r["think_ms"] = tk["think_ms"]; r["answer_ms"] = tk["answer_ms"]
    r["think_text"] = open(base + ".think.txt").read(); ans = open(base + ".answer.txt").read()
    svg = extract_svg(ans); r["png"] = None
    if svg and r.get("complete"):
        try:
            s = svg if "xmlns=" in svg[:400] else svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            r["png"] = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=s, width=800, background="#ffffff")))).convert("RGB")
        except Exception:
            r["png"] = None
    r["end_s"] = (r["answer_ms"][-1] if r["answer_ms"] else (r["think_ms"][-1] if r["think_ms"] else 1000)) / 1000
    return r


def tokens_at(r, t_real):
    if not r["think_ms"]:
        return 0
    n = bisect.bisect_right(r["think_ms"], t_real * 1000)
    return min(r["think_tokens"], round(n * r["think_tokens"] / len(r["think_ms"])))


def fit(img, w, h):
    img = img.copy(); img.thumbnail((w, h)); c = Image.new("RGB", (w, h), (255, 255, 255)); c.paste(img, ((w - img.width) // 2, (h - img.height) // 2)); return c


def panel(d, im, x0, title, sub, col, r, t_real, scale_note, seed):
    d.rectangle([x0, 0, x0 + 960, 130], fill=(24, 28, 36)); d.rectangle([x0, 0, x0 + 14, 130], fill=col)
    d.text((x0 + 40, 22), title, font=FB[44], fill=FG); d.text((x0 + 40, 80), sub, font=FR[26], fill=(170, 178, 190))
    done = t_real >= r["end_s"]
    n = r["think_tokens"] if done else tokens_at(r, t_real)
    d.text((x0 + 40, 140), f"{n:,}", font=FB[150], fill=(GRN if done else col))
    d.text((x0 + 40, 305), "thinking tokens" + ("  (finished)" if done else ""), font=FR[26], fill=GRN if done else DIM)
    cx, cy, cw, ch = x0 + 80, 350, 800, 600
    if done and r["png"] is not None:
        im.paste(fit(r["png"], cw, ch), (cx, cy)); d.rectangle([cx - 3, cy - 3, cx + cw + 3, cy + ch + 3], outline=GRN, width=4)
    else:
        d.rectangle([cx, cy, cx + cw, cy + ch], fill=(20, 24, 32), outline=(60, 66, 78), width=3)
        txt = r["think_text"]; k = int(len(txt) * (n / max(1, r["think_tokens"]))) if not done else len(txt)
        lines = []
        for para in txt[:k].split("\n")[-60:]:
            while len(para) > 47:
                cut = para.rfind(" ", 0, 47); cut = cut if cut > 20 else 47
                lines.append(para[:cut]); para = para[cut:].lstrip()
            lines.append(para)
        for i, ln in enumerate(lines[-19:]):
            d.text((cx + 14, cy + 12 + i * 30), ln, font=FM, fill=(185, 194, 208))
        if done:
            d.text((cx + 200, cy + 270), "no complete SVG", font=FB[44], fill=RED)


def banner(d, mock):
    if mock:
        d.rectangle([0, 1040, W, 1080], fill=(150, 20, 20)); d.text((24, 1046), "MOCK DATA: render selftest, not a result", font=FB[28], fill=(255, 255, 255))


def main():
    run, out = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 5
    ff = sys.argv[sys.argv.index("--ffmpeg") + 1] if "--ffmpeg" in sys.argv else "ffmpeg"
    mock = os.path.exists(os.path.join(run, "MOCK"))
    summ = json.load(open(os.path.join(run, "summary.json" if os.path.exists(os.path.join(run, "summary.json")) else "summary.pre.json")))
    pairs = []
    for s in range(1, 100):
        a, b = load_try(run, "base", s), load_try(run, "edited", s)
        if a and b: pairs.append((s, a, b))
        if len(pairs) >= rounds: break
    gal = {arm: [load_try(run, arm, s) for s in range(1, 17)] for arm in ("base", "edited")}
    p = subprocess.Popen([ff, "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    def emit(im):
        p.stdin.write(im.tobytes())
    PLAY, HOLD = 9.0, 1.6
    for ri, (seed, a, b) in enumerate(pairs):
        scale = PLAY / max(a["end_s"], b["end_s"])             # video seconds per real second, shared by both sides
        nf = int((PLAY + HOLD) * FPS)
        for f in range(nf):
            tv = f / FPS; tr = min(tv / scale, max(a["end_s"], b["end_s"]) + 5)
            im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
            panel(d, im, 0, "UNEDITED", "Qwen3.8-27B Q4_K_M", RED, a, tr, "", seed)
            panel(d, im, 960, "EDITED", "Astra-Stop: one matrix, +20 KB", AMB, b, tr, "", seed)
            d.rectangle([958, 0, 962, 1040], fill=(0, 0, 0))
            d.rectangle([0, 960, W, 1040], fill=(24, 28, 36))
            d.text((40, 972), f"\"pelican on a bicycle\"  ·  pelican {ri + 1} of {len(pairs)} (seed {seed}, first seeds in order)", font=FB[34], fill=FG)
            d.text((40, 1012), "same prompt, same sampler, reasoning effort xhigh · arms ran one after the other on one RTX 5090 · timelines replayed at one shared speed", font=FR[22], fill=(170, 178, 190))
            banner(d, mock); emit(im)
    for f in range(int(7.5 * FPS)):                              # gallery
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        for k, (arm, title, col) in enumerate((("base", "UNEDITED · first 16 tries", RED), ("edited", "EDITED · first 16 tries", AMB))):
            x0 = k * 960; d.rectangle([x0, 0, x0 + 14, 130], fill=col); d.text((x0 + 40, 40), title, font=FB[44], fill=FG)
            shown = int(min(16, (f / FPS) / 7.0 * 16 + 1))
            for i, r in enumerate(gal[arm][:shown]):
                gx, gy = x0 + 40 + (i % 4) * 224, 150 + (i // 4) * 200
                if r and r["png"] is not None: im.paste(fit(r["png"], 214, 190), (gx, gy))
                else: d.rectangle([gx, gy, gx + 214, gy + 190], outline=RED, width=3)
        for k, arm in enumerate(("base", "edited")):
            s = summ["arms"].get(arm, {})
            if s.get("n"): d.text((k * 960 + 40, 985), f"median {s['median']:,.0f} tokens · {s['complete']}/{s['n']} complete", font=FB[34], fill=FG)
        banner(d, mock); emit(im)
    ea, ba = summ["arms"].get("edited", {}), summ["arms"].get("base", {})
    for f in range(int(5 * FPS)):                                # result card
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        d.text((80, 90), "median thinking tokens, pelican prompt", font=FB[44], fill=DIM)
        if ba.get("n") and ea.get("n"):
            d.text((80, 190), f"{ba['median']:,.0f}", font=FB[150], fill=RED); d.text((780, 250), "→", font=FB[150], fill=DIM); d.text((1000, 190), f"{ea['median']:,.0f}", font=FB[150], fill=AMB)
            d.text((80, 400), f"{summ.get('median_change_pct', 0):+.0f}% median · {summ.get('mean_change_pct', 0):+.0f}% mean · Mann-Whitney p = {summ.get('mannwhitney_p', float('nan')):.3g} · n = {ba['n']} vs {ea['n']}", font=FB[44], fill=FG)
            d.text((80, 480), f"complete SVGs: {ba['complete']}/{ba['n']} unedited, {ea['complete']}/{ea['n']} edited", font=FB[44], fill=FG)
        d.text((80, 600), f"pre-registered verdict: {summ['verdict']}", font=FB[60], fill=GRN if summ["verdict"] == "REPRODUCED" else AMB)
        d.text((80, 760), "Q4_K_M of both arms, llama.cpp, one prompt. The author's card tested bf16: 28,474 → 17,201 mean (n = 16 per side).", font=FR[26], fill=DIM)
        d.text((80, 805), "A quantised pair is a different test from the author's. Not a claim about bf16.", font=FR[26], fill=DIM)
        banner(d, mock); emit(im)
    p.stdin.close(); rc = p.wait()
    print("render done rc", rc, out); sys.exit(rc)


if __name__ == "__main__":
    main()
