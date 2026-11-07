#!/usr/bin/env python3
"""CPU picture: per prompt a 1920x1080 contact PNG + a 4 s turntable mp4 (3 models side by side, labelled size + completion tokens), plus a 36-cell sheet and a tally card.
the turntable shows each model's OWN exported GLB in one fixed shared viewer (same camera, lights, fit-to-frame). failed cells show their page screenshot or a text card.
usage: render_glb.py --res DIR --out DIR [--mock] [--prompts id,id] [--fps 24]"""
import argparse, base64, io, json, os, subprocess, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glbcommon as G
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mock", action="store_true"); ap.add_argument("--prompts", default="all"); ap.add_argument("--fps", type=int, default=24); ap.add_argument("--secs", type=float, default=4.0)
ap.add_argument("--ffmpeg", default=""); ap.add_argument("--agg", default=""); ap.add_argument("--no-video", action="store_true")
a = ap.parse_args()
C = G.cfg(); P = G.prompts()
if a.prompts != "all": P = [p for p in P if p["id"] in a.prompts.split(",")]
VEND = os.path.expanduser("~/glb-3d-coder/vendor/package"); W, H = 1920, 1080; PW, PH = 640, 760
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f = lambda p, s: ImageFont.truetype(p, s)
def ffmpeg():
    if a.ffmpeg: return a.ffmpeg
    import glob
    for pat in ("~/voxelbench/ffmpeg/ffmpeg*", "~/Pollen/arena-talent/ffvenv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*"):
        g = [x for x in glob.glob(os.path.expanduser(pat)) if os.path.isfile(x) and os.access(x, os.X_OK)]
        if g: return g[0]
    raise SystemExit("no ffmpeg")
VIEWER = """<!doctype html><meta charset=utf-8><style>html,body{margin:0;background:#1b1f27}canvas{display:block}</style>
<script type=importmap>{"imports":{"three":"http://glb.local/three/build/three.module.js","three/addons/":"http://glb.local/three/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three'; import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const W=%d,H=%d; const r=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true}); r.setSize(W,H); r.setClearColor(0x1b1f27); document.body.appendChild(r.domElement);
const sc=new THREE.Scene(); sc.add(new THREE.HemisphereLight(0xdde8ff,0x403028,1.1)); const dl=new THREE.DirectionalLight(0xffffff,2.2); dl.position.set(3,6,4); sc.add(dl);
const cam=new THREE.PerspectiveCamera(35,W/H,0.01,1000);
const g=await new GLTFLoader().loadAsync('http://glb.local/m.glb'); const root=new THREE.Group(); root.add(g.scene); sc.add(root);
const box=new THREE.Box3().setFromObject(g.scene); const c=box.getCenter(new THREE.Vector3()); g.scene.position.sub(c);
const rad=box.getSize(new THREE.Vector3()).length()/2||1; const dist=rad/Math.sin(THREE.MathUtils.degToRad(35/2))*1.05;
cam.position.set(0,dist*0.26,dist*0.97); cam.lookAt(0,0,0);
const mx=g.animations.length?new THREE.AnimationMixer(g.scene):null; if(mx) g.animations.forEach(a=>mx.clipAction(a).play());
window.frame=(ang,t)=>{root.rotation.y=ang; if(mx) mx.setTime(t); r.render(sc,cam); return r.domElement.toDataURL('image/jpeg',0.92)};
window.info={anims:g.animations.length,rad:rad}; window.ready=true;
</script>""" % (PW, PH)
os.makedirs(a.out, exist_ok=True)
def rows(p): return {(json.loads(l)["prompt"], json.loads(l)["seed"]): json.loads(l) for l in open(p) if l.strip()} if os.path.exists(p) else {}
gen = {m["key"]: rows(f"{a.res}/{m['key']}.jsonl") for m in C["models"]}; chk = rows(a.res + "/checks.jsonl") if False else {}
chk = {}
if os.path.exists(a.res + "/checks.jsonl"):
    for l in open(a.res + "/checks.jsonl"):
        if l.strip(): r = json.loads(l); chk[(r["model"], r["prompt"], r["seed"])] = r
def pick(key, pid):
    best = None
    for sd in C["seeds"]:
        c = chk.get((key, pid, sd))
        if c and c["glb_valid"]: return sd, c, "glb"
        if c and best is None and os.path.exists(f"{a.res}/shots/{key}/{pid}-s{sd}.png"): best = (sd, c, "shot")
    if best: return best
    for sd in C["seeds"]:
        if (key, pid, sd) in chk: return sd, chk[(key, pid, sd)], "card"
    return None, None, "card"
def card(text, sub=""):
    im = Image.new("RGB", (PW, PH), (27, 31, 39)); d = ImageDraw.Draw(im)
    y = PH // 2 - 60
    for ln in textwrap.wrap(text, 24)[:4]: d.text((PW // 2, y), ln, font=f(FB, 34), fill=(230, 120, 110), anchor="mm"); y += 46
    for ln in textwrap.wrap(sub, 40)[:4]: d.text((PW // 2, y + 10), ln, font=f(FR, 22), fill=(170, 175, 185), anchor="mm"); y += 30
    return im
def fit(im):
    im = im.convert("RGB"); s = min(PW / im.width, PH / im.height); im = im.resize((int(im.width * s), int(im.height * s)))
    out = Image.new("RGB", (PW, PH), (27, 31, 39)); out.paste(im, ((PW - im.width) // 2, (PH - im.height) // 2)); return out
from playwright.sync_api import sync_playwright
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
NF = int(a.fps * a.secs); import math
def render_glb(br, glbpath):
    ctx = br.new_context(viewport={"width": PW, "height": PH}); frames = []
    def route(r):
        u = r.request.url
        if u == "http://glb.local/viewer.html": return r.fulfill(status=200, content_type="text/html", body=VIEWER)
        if u == "http://glb.local/m.glb": return r.fulfill(status=200, content_type="model/gltf-binary", body=open(glbpath, "rb").read())
        if u.startswith("http://glb.local/three/"):
            fp = os.path.join(VEND, u[len("http://glb.local/three/"):]); 
            return r.fulfill(status=200, content_type="text/javascript", body=open(fp, "rb").read()) if os.path.exists(fp) else r.fulfill(status=404, body="")
        r.fulfill(status=404, body="")
    ctx.route("**/*", route); pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:150]))
    try:
        pg.goto("http://glb.local/viewer.html", wait_until="load", timeout=60000); pg.wait_for_function("window.ready===true", timeout=60000)
        for i in range(NF):
            u = pg.evaluate("([a,t]) => window.frame(a,t)", [2 * math.pi * i / NF, i / a.fps])
            frames.append(Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB"))
    except Exception as e: errs.append("viewer: " + str(e)[:120])
    ctx.close(); return frames, errs
def label_bar(d, x, m, key, pid, sd, c, how, row):
    tok = row.get("completion_tokens") if row else None
    d.text((x + PW // 2, 905), "%s  ·  %s" % (m["size"], m["label"]), font=f(FB, 30), fill=(245, 245, 250), anchor="mm")
    d.text((x + PW // 2, 950), ("%s tokens" % format(tok, ",")) if tok else "no tokens", font=f(FB, 40), fill=(255, 205, 90), anchor="mm")
    if how == "glb": st, col = "valid GLB exported", (110, 220, 140)
    elif c is None: st, col = "no page extracted", (230, 120, 110)
    else: st, col = (c.get("reason") or "failed")[:34], (230, 120, 110)
    d.text((x + PW // 2, 1000), st, font=f(FR, 28), fill=col, anchor="mm")
    if row and row.get("finish_reason") == "length": d.text((x + PW // 2, 1040), "hit the token cap", font=f(FR, 22), fill=(230, 160, 90), anchor="mm")
def header(d, p, tag=""):
    d.rectangle([0, 0, W, 130], fill=(14, 16, 21))
    ln = textwrap.wrap(p["text"], 78)
    d.text((40, 20), "prompt: " + tag + (ln[0] if ln else ""), font=f(FB, 32), fill=(245, 245, 250))
    if len(ln) > 1: d.text((40, 66), ln[1], font=f(FB, 32), fill=(245, 245, 250))
    d.text((W - 40, 100), "same system prompt · same sampler · script-checked" + ("  ·  MOCK DATA" if a.mock else ""), font=f(FR, 22), fill=(240, 120, 120) if a.mock else (150, 156, 168), anchor="rm")
summary = {"prompts": {}}
cells_sheet = []
with sync_playwright() as p:
    br = p.chromium.launch(headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--disable-dev-shm-usage"],
                           executable_path=CHROME, chromium_sandbox=False, env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    # order: most models with a GLB first
    def nglb(pp): return -sum(1 for m in C["models"] if pick(m["key"], pp["id"])[2] == "glb")
    for pp in sorted(P, key=nglb):
        panels = []
        for m in C["models"]:
            key = m["key"]; sd, c, how = pick(key, pp["id"]); row = gen[key].get((pp["id"], sd)) if sd else None; fr = None; note = ""
            if how == "glb":
                fr, errs = render_glb(br, f"{a.res}/glb/{key}/{pp['id']}-s{sd}.glb")
                if not fr: how = "shot" if os.path.exists(f"{a.res}/shots/{key}/{pp['id']}-s{sd}.png") else "card"; note = "viewer failed: " + "; ".join(errs)[:100]
            if how == "shot": fr = [fit(Image.open(f"{a.res}/shots/{key}/{pp['id']}-s{sd}.png"))]
            if how == "card" or not fr: fr = [card("no page" if not c or not c.get("html") else "no asset", (c or {}).get("reason", "") if c else "not generated")]
            panels.append({"m": m, "key": key, "sd": sd, "c": c, "how": how, "row": row, "frames": fr, "note": note})
        summary["prompts"][pp["id"]] = [{"model": q["key"], "seed": q["sd"], "shown": q["how"], "note": q["note"]} for q in panels]
        def compose(i):
            im = Image.new("RGB", (W, H), (14, 16, 21)); d = ImageDraw.Draw(im); header(d, pp)
            for k, q in enumerate(panels):
                x = k * PW; fr = q["frames"][i % len(q["frames"])]; im.paste(fr, (x, 130 + 0)); 
                label_bar(d, x, q["m"], q["key"], pp["id"], q["sd"], q["c"], q["how"], q["row"])
                if k: d.line([x, 130, x, H], fill=(40, 44, 54), width=2)
            return im
        # contact: 4 angles per model, rows = models (cropped panels 480x270 per cell would lose the models; use a 3x4 grid)
        cs = Image.new("RGB", (W, H), (14, 16, 21)); d = ImageDraw.Draw(cs); header(d, pp, "contact · ")
        cw, ch = 380, 300; x0 = 440
        for r_, q in enumerate(panels):
            y = 150 + r_ * (ch + 14)
            d.text((20, y + ch // 2 - 26), q["m"]["size"], font=f(FB, 56), fill=(245, 245, 250), anchor="lm")
            tok = q["row"].get("completion_tokens") if q["row"] else None
            d.text((20, y + ch // 2 + 28), ("%s tok" % format(tok, ",")) if tok else "no tokens", font=f(FB, 28), fill=(255, 205, 90), anchor="lm")
            d.text((20, y + ch // 2 + 66), "valid GLB" if q["how"] == "glb" else ("failed: " + ((q["c"] or {}).get("reason") or "?")[:20]), font=f(FR, 22), fill=(110, 220, 140) if q["how"] == "glb" else (230, 120, 110), anchor="lm")
            for j in range(4):
                fr = q["frames"][(j * len(q["frames"]) // 4) % len(q["frames"])]
                t = fr.resize((cw * 1, int(cw * PH / PW))) if False else fr.crop((0, 0, PW, PH)).resize((int(ch * PW / PH), ch))
                cs.paste(t, (x0 + j * (int(ch * PW / PH) + 12) - 20, y))
        cs.save(f"{a.out}/contact-{pp['id']}.png")
        cells_sheet.append((pp, panels))
        compose(0).save(f"{a.out}/frame0-{pp['id']}.png")
        if not a.no_video:
            fd = f"{a.out}/frames-{pp['id']}"; os.makedirs(fd, exist_ok=True)
            for i in range(NF): compose(i).save(f"{fd}/{i:04d}.jpg", quality=92)
            subprocess.run([ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{fd}/%04d.jpg", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{a.out}/turntable-{pp['id']}.mp4"], stdin=subprocess.DEVNULL, check=True)
            for fn in os.listdir(fd): os.remove(f"{fd}/{fn}")
            os.rmdir(fd)
        print("[render] done", pp["id"], flush=True)
    br.close()
# one sheet of every shown cell (12 prompts x 3 models), one tally card
sheet = Image.new("RGB", (W, H), (14, 16, 21)); d = ImageDraw.Draw(sheet)
cwid, chei = W // 6, (H - 70) // 6
d.text((20, 10), "all 12 prompts x 3 models (columns 1-3 and 4-6 repeat the 4B | 27B | 35B order)" + ("   MOCK DATA" if a.mock else ""), font=f(FB, 26), fill=(245, 245, 250))
for n, (pp, panels) in enumerate(cells_sheet):
    for k, q in enumerate(panels):
        col = (n % 2) * 3 + k; row_ = n // 2; fr = q["frames"][0]
        t = fr.resize((int((chei - 30) * PW / PH), chei - 30)); x = col * cwid + (cwid - t.width) // 2; y = 60 + row_ * chei
        sheet.paste(t, (x, y))
        ok = q["how"] == "glb"
        d.text((col * cwid + 8, y + chei - 26), "%s %s" % (q["m"]["size"], pp["id"][:15]), font=f(FB, 17), fill=(110, 220, 140) if ok else (230, 120, 110))
sheet.save(f"{a.out}/sheet-all.png")
if a.agg and os.path.exists(a.agg):
    ag = json.load(open(a.agg)); NCx = ag["n_cells_planned"]
    tl = Image.new("RGB", (W, H), (14, 16, 21)); d = ImageDraw.Draw(tl)
    d.text((W // 2, 90), "valid page + valid GLB export, out of %d tries" % NCx, font=f(FB, 44), fill=(245, 245, 250), anchor="mm")
    d.text((W // 2, 150), "12 absurd prompts x 2 seeds · same system prompt · script checks only" + ("  ·  MOCK DATA" if a.mock else ""), font=f(FR, 26), fill=(240, 120, 120) if a.mock else (150, 156, 168), anchor="mm")
    for k, m in enumerate(C["models"]):
        v = ag["models"][m["key"]]; cx = 320 + k * 640
        d.text((cx, 330), m["size"], font=f(FB, 120), fill=(245, 245, 250), anchor="mm")
        d.text((cx, 470), "%d / %d" % (v["full_pass"], NCx), font=f(FB, 150), fill=(110, 220, 140) if v["full_pass"] else (230, 120, 110), anchor="mm")
        d.text((cx, 590), m["label"], font=f(FR, 28), fill=(200, 204, 214), anchor="mm")
        d.text((cx, 650), "%s tokens in all" % format(v["tokens_total"], ","), font=f(FB, 34), fill=(255, 205, 90), anchor="mm")
        d.text((cx, 710), "median %s per answer · %d hit the cap" % (format(v["tokens_median"] or 0, ","), v["truncated"]), font=f(FR, 24), fill=(170, 175, 185), anchor="mm")
    d.text((W // 2, 960), "verdict rule (pre-registered): %s" % ag["overall"], font=f(FB, 40), fill=(255, 205, 90), anchor="mm")
    tl.save(f"{a.out}/tally.png")
    if not a.no_video:
        subprocess.run([ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-loop", "1", "-t", "3", "-i", f"{a.out}/tally.png", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", f"{a.out}/tally.mp4"], stdin=subprocess.DEVNULL, check=True)
agg = os.path.join(a.res, "..", "out", "aggregate.json") if False else None
json.dump(summary, open(a.out + "/render-summary.json", "w"), indent=1)
print("[render] sheet + summary written")
