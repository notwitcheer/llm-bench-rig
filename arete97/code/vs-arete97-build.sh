#!/bin/bash
# vs-arete97-build.sh: two-pane voxel clip for item 97-arete-msm, Qwen3.8-27B Q6_K (left) vs Arete-27B Q6_K (right), same prompt, one hour of /goal in Hermes Agent each.
# runs ON capsule-w (CPU only, nice 19), never on mercury. adapted from vs-qwen-nanbeige-build.sh / vs-k31-qwen-build.sh. called by the item after all four takes; rerun by hand with PAIR_JSON=<summary dir>/pair.json.
# pre-registered pair (PREREG.md): take 1 of each model; when a take 1 failed the gate, that model's first passing take, labelled on screen. pair.json is written by arete97_summary.py.
# labels (plan): "Qwen3.8-27B Q6_K on my RTX 5090" | "Arete-27B Q6_K (Nous Research) on my RTX 5090", then code KB (page minus inlined libraries, HARDENING L26) and the take.
# the settings line (ctx, KV type, temperature, no drafting) is read from each run's run-config.json and must be IDENTICAL for both panes or the script aborts.
# layout: 1920x1080, dusk title band 0-130, label strip 130-260 (never over the image), two 960x720 panels 260-980 (centre 1440x1080 crop of each 1920x1080 orbit clip, full height, same rule for both), dusk strip 980-1080 (room for X's player controls).
# knobs for a smoke test only: PAIR_JSON= OUT= (dir name under runs/ or an absolute path) SECS=6 PRESET=ultrafast.
set -euo pipefail
R=$HOME/voxelbench/runs; F=$HOME/voxelbench/ffmpeg; PY=$HOME/benchmark-rig/.venv/bin/python
PAIR_JSON=${PAIR_JSON:-$HOME/voxel-stage/arete97/run/pair.json}
if [[ "${OUT:-}" == /* ]]; then O=$OUT; else O=$R/${OUT:-vs-qwen-arete97}; fi
SECS=${SECS:-30}; PRESET=${PRESET:-slow}; CW=${CW:-1440}; QX=$(( (1920-CW)/2 ))
[ -s "$PAIR_JSON" ] || { echo "ABORT: no pair file $PAIR_JSON"; exit 2; }
mkdir -p "$O/fonts" "$O/qa"; cd "$O"
IN=$($PY - "$PAIR_JSON" "$R" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1])); R = sys.argv[2]
if not d.get("ok"): print("ABORT"); sys.exit(0)
q, a = d["pair"]["qwen"], d["pair"]["arete"]
rc = {}
for k, t in (("q", q), ("a", a)):
    try: c = json.load(open(os.path.join(R, t["slug"], "run-config.json")))
    except Exception: c = {}
    rc[k] = c
sett = lambda c: (c.get("ctx"), c.get("kv"), c.get("sampling"), c.get("drafting"), c.get("reasoning_budget"))
print(json.dumps({"a": q["orbit_mp4"], "b": a["orbit_mp4"], "ka": q["code_kb"], "kb": a["code_kb"], "ta": q["take"], "tb": a["take"],
                  "na": q["label_note"], "nb": a["label_note"], "same": sett(rc["q"]) == sett(rc["a"]), "ctx": rc["q"].get("ctx"), "kv": rc["q"].get("kv"),
                  "sa": sett(rc["q"]), "sb": sett(rc["a"])}))
PY
)
case "$IN" in ABORT*) echo "ABORT: pair.json says no usable pair"; exit 2;; esac
echo "pair: $IN"
A=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['a'])" "$IN"); B=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['b'])" "$IN")
SAME=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['same'])" "$IN")
[ "$SAME" = True ] || { echo "ABORT: the two runs' settings differ ($IN): the settings line would not be true for both panes"; exit 2; }
for f in "$A" "$B"; do [ -s "$f" ] || { echo "ABORT: missing input $f"; exit 2; }; done
for f in Fraunces Manrope; do [ -s fonts/$f.ttf ] || cp $R/vs-k31-qwen/fonts/$f.ttf fonts/$f.ttf 2>/dev/null || true; done
[ -s fonts/Fraunces.ttf ] || curl -sSfL -o fonts/Fraunces.ttf "https://github.com/google/fonts/raw/main/ofl/fraunces/Fraunces%5BSOFT,WONK,opsz,wght%5D.ttf"
[ -s fonts/Manrope.ttf ] || curl -sSfL -o fonts/Manrope.ttf "https://github.com/google/fonts/raw/main/ofl/manrope/Manrope%5Bwght%5D.ttf"
# both inputs 1920x1080 and at least SECS long, or the crop maths and the trim differ between panels
for f in "$A" "$B"; do
  info=$($F -hide_banner -nostdin -i "$f" 2>&1 </dev/null | grep -E "Video:|Duration" || true)
  case "$info" in *1920x1080*) ;; *) echo "ABORT: $f is not 1920x1080: $info"; exit 2;; esac
  d=$(echo "$info" | sed -n 's/.*Duration: \([0-9:.]*\).*/\1/p' | head -1); ds=$(echo "$d" | awk -F: '{print $1*3600+$2*60+$3}')
  awk -v d="$ds" -v s="$SECS" 'BEGIN{exit !(d+0 >= s+0)}' || { echo "ABORT: $f is $d long, under the $SECS s the clip needs"; exit 2; }
done
export IN
$PY - <<'PY'
import json, os
from PIL import Image, ImageDraw, ImageFont
L = json.loads(os.environ["IN"])
W, H, BAND = 1920, 1080, 130; LAB = 130; PANEL_Y = BAND + LAB; PANEL_H = 720
PLUM=(42,22,34); SIENNA=(96,42,30); SAND=(246,223,184); AMBER=(240,160,75); INK=(29,18,24)
def vf(path, size, axes):
    f = ImageFont.truetype(path, size); vals = []
    for a in f.get_variation_axes():
        n = a['name'].decode() if isinstance(a['name'], bytes) else a['name']; vals.append(axes.get(n, a['default']))
    f.set_variation_by_axes(vals); return f
def fit(text, path, size, axes, maxw, d):
    while size > 18:
        f = vf(path, size, axes); bb = d.textbbox((0, 0), text, font=f)
        if bb[2] - bb[0] <= maxw: return f, size
        size -= 1
    raise SystemExit("ABORT: text cannot fit: " + text)
im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
T = vf("fonts/Fraunces.ttf", 66, {"Optical Size": 72, "Weight": 620, "Softness": 60, "Wonky": 0})
P = vf("fonts/Manrope.ttf", 28, {"Weight": 800})
for x in range(W):
    t = x / (W - 1); c = tuple(int(PLUM[i] * (1 - t) + SIENNA[i] * t) for i in range(3)); d.line([(x, 0), (x, BAND - 1)], fill=c + (255,))
d.rectangle([0, BAND - 5, W, BAND - 1], fill=AMBER + (255,))
pt = "real speed"; bb = d.textbbox((0, 0), pt, font=P); pw = bb[2] - bb[0] + 48; ph = 56; px = W - 56 - pw; py = 36
d.rounded_rectangle([px, py, px + pw, py + ph], radius=28, outline=AMBER + (255,), width=3, fill=(0, 0, 0, 60))
d.text((px + 24, py + (ph - (bb[3] - bb[1])) // 2 - bb[1]), pt, font=P, fill=AMBER + (255,))
d.text((56, -4), "same prompt, base vs fine-tune", font=T, fill=SAND + (255,))
ctx = f"{round(L['ctx'] / 1024)}k"; ksize = L["kv"]
setup = f"one hour of /goal in Hermes Agent each  ·  same settings both sides: {ctx} context, {ksize} KV, temp 1.0, no drafting"
fS, _ = fit(setup, "fonts/Manrope.ttf", 31, {"Weight": 600}, px - 56 - 60, d)
d.text((58, 80), setup, font=fS, fill=SAND + (230,))
for x in range(W):
    t = x / (W - 1); c = tuple(int(SIENNA[i] * (1 - t) + PLUM[i] * t) for i in range(3)); d.line([(x, PANEL_Y + PANEL_H), (x, H)], fill=c + (255,))
d.rectangle([0, PANEL_Y + PANEL_H, W, PANEL_Y + PANEL_H + 3], fill=AMBER + (255,))
d.rectangle([0, BAND, W, PANEL_Y - 1], fill=INK + (255,))
def take_txt(k, take, note): return f"{k} KB of code  ·  take {take}" + (f"  ({note})" if note else "")
cells = [("Qwen3.8-27B Q6_K on my RTX 5090", take_txt(L["ka"], L["ta"], L["na"])),
         ("Arete-27B Q6_K (Nous Research) on my RTX 5090", take_txt(L["kb"], L["tb"], L["nb"]))]
def maxw_of(i): return (i + 1) * 960 - 30 - (i * 960 + 30 + (4 if i else 0))
S1 = min(fit(a, "fonts/Manrope.ttf", 40, {"Weight": 800}, maxw_of(i), d)[1] for i, (a, b) in enumerate(cells))   # one size per row for both cells
S2 = min(fit(b, "fonts/Manrope.ttf", 32, {"Weight": 700}, maxw_of(i), d)[1] for i, (a, b) in enumerate(cells))
for i, (a, b) in enumerate(cells):
    x0 = i * 960; x1 = (i + 1) * 960; y0 = BAND; y1 = PANEL_Y; tx = x0 + 30 + (4 if i else 0); maxw = x1 - 30 - tx
    d.rectangle([x0 + (4 if i else 0), y0 + 8, x0 + (4 if i else 0) + 7, y1 - 8], fill=AMBER + (255,))
    f1, s1 = fit(a, "fonts/Manrope.ttf", S1, {"Weight": 800}, maxw, d); f2, s2 = fit(b, "fonts/Manrope.ttf", S2, {"Weight": 700}, maxw, d)
    d.text((tx, y0 + 12), a, font=f1, fill=SAND + (255,)); d.text((tx, y0 + 70), b, font=f2, fill=AMBER + (255,))
    print("label", i, "|", a, "|", b, "| sizes", s1, s2, "(fits its cell)")
d.rectangle([958, BAND, 961, PANEL_Y + PANEL_H], fill=INK + (255,))
im.save("overlay.png"); print("overlay ok")
PY
FC="[0:v]trim=0:$SECS,setpts=PTS-STARTPTS,crop=$CW:1080:$QX:0,scale=960:720:flags=lanczos,setsar=1[a];\
[1:v]trim=0:$SECS,setpts=PTS-STARTPTS,crop=$CW:1080:$QX:0,scale=960:720:flags=lanczos,setsar=1[b];\
[a][b]hstack=inputs=2[s];[s]pad=1920:1080:0:260:color=0x2a1622[p];[p][2:v]overlay=0:0,fps=30,format=yuv420p[v]"
nice -n 19 $F -nostdin -y -i "$A" -i "$B" -i overlay.png -filter_complex "$FC" -map "[v]" \
  -c:v libx264 -profile:v high -preset $PRESET -crf 18 -pix_fmt yuv420p -r 30 -threads 6 -movflags +faststart -an vsa97-master.mp4 </dev/null 2>qa/ff1.log
nice -n 19 $F -nostdin -y -i vsa97-master.mp4 -c:v libx264 -profile:v high -preset $PRESET -b:v 2400k -maxrate 3000k -bufsize 4800k -pass 1 -passlogfile x264vsa97 -enc_time_base 1/30 -threads 6 -an -f null /dev/null </dev/null 2>qa/ff2.log
nice -n 19 $F -nostdin -y -i vsa97-master.mp4 -c:v libx264 -profile:v high -preset $PRESET -b:v 2400k -maxrate 3000k -bufsize 4800k -pass 2 -passlogfile x264vsa97 -enc_time_base 1/30 -threads 6 -pix_fmt yuv420p -movflags +faststart -an vsa97-discord.mp4 </dev/null 2>qa/ff3.log
nice -n 19 $F -nostdin -y -i vsa97-discord.mp4 -vf "fps=8/$SECS,scale=640:-1,tile=4x2" -frames:v 1 qa/sheet.png </dev/null 2>/dev/null || true
for t in 1 $(( SECS/2 )) $(( SECS-1 )); do nice -n 19 $F -nostdin -y -ss $t -i vsa97-discord.mp4 -frames:v 1 qa/full_$t.png </dev/null 2>/dev/null || true; done
rm -rf qa/lum; mkdir qa/lum; nice -n 19 $F -nostdin -y -i vsa97-master.mp4 -vf "fps=1,scale=480:-1" qa/lum/f_%02d.png </dev/null 2>/dev/null || true
# day/night by luma per panel per second (day ~150-200, night ~55-80 on these pages): quote the night window per panel, never read it by eye
$PY - <<'PY'
from PIL import Image, ImageStat; import glob
rows = []
for p in sorted(glob.glob("qa/lum/f_*.png")):
    im = Image.open(p).convert("L"); w, h = im.size; y0 = int(h * 260 / 1080); y1 = int(h * 980 / 1080)
    v = [ImageStat.Stat(im.crop((i * w // 2 + 2, y0, (i + 1) * w // 2 - 2, y1))).mean[0] for i in range(2)]
    rows.append(f"{int(p[-6:-4]) - 1}:{v[0]:.0f}/{v[1]:.0f}")
print("luma s:qwen/arete", " ".join(rows))
PY
ls -la vsa97-master.mp4 vsa97-discord.mp4 qa/
for f in vsa97-master.mp4 vsa97-discord.mp4; do $F -hide_banner -nostdin -i $f 2>&1 </dev/null | grep -E "Duration|Video:" || true; done
md5sum vsa97-discord.mp4
echo "next: scp back ONLY the finished Discord copy + master (never composite on mercury); run the 7-frame contact sheet + luma check before sending; publication is decided at readout, not here"
