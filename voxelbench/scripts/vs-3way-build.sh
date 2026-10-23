#!/bin/bash
# three-way voxel clip (Qwen3.8 27B | Flint | Muse Glimmer 30B, msm-v2, one hour of /goal each). runs ON capsule-w, CPU only, nice 19.
# label code size = page bytes minus any inlined library (HARDENING L26): Qwen 660,756 - 603,445 (three.min.js r128 inlined) = 57,311; Flint 27,446 and Muse 31,394 load three.js from a CDN.
# layout: 1920x1080, dusk title band 0-150, label strip 150-260 (never over the image), three 640x720 panels 260-980
# (centre 960x1080 crop of each 1920x1080 orbit clip, full height, same rule for every panel), dusk strip 980-1080.
set -euo pipefail
R=$HOME/voxelbench/runs; O=$R/vs-3way; F=$HOME/voxelbench/ffmpeg; PY=$HOME/benchmark-rig/.venv/bin/python
mkdir -p $O/fonts $O/qa; cd $O
A=$R/qwen38-27b-q6k-msm2/showcase-turn/orbit.mp4
B=$R/flint-ud-q4kxl-msm-v2-rec/showcase/orbit.mp4
C=$R/muse-glimmer-30b-q4km-msm-v2-qgate/showcase-turn/orbit.mp4
CW=${CW:-960}                     # crop width from the 1920x1080 source (full height kept)
QX=${QX:-$(( (1920-CW)/2 ))}; FX=${FX:-$(( (1920-CW)/2 ))}; MX=${MX:-$(( (1920-CW)/2 ))}
[ -s fonts/Fraunces.ttf ] || curl -sSfL -o fonts/Fraunces.ttf "https://github.com/google/fonts/raw/main/ofl/fraunces/Fraunces%5BSOFT,WONK,opsz,wght%5D.ttf"
[ -s fonts/Manrope.ttf ] || curl -sSfL -o fonts/Manrope.ttf "https://github.com/google/fonts/raw/main/ofl/manrope/Manrope%5Bwght%5D.ttf"
ls -la fonts
$PY - <<'PY'
from PIL import Image, ImageDraw, ImageFont
W,H,BAND=1920,1080,150; LAB=110; PANEL_Y=BAND+LAB; PANEL_H=720
PLUM=(42,22,34); SIENNA=(96,42,30); SAND=(246,223,184); AMBER=(240,160,75); INK=(29,18,24)
def vf(path,size,axes):
    f=ImageFont.truetype(path,size)
    names=[a['name'] for a in f.get_variation_axes()]
    vals=[]
    for a in f.get_variation_axes():
        n=a['name'].decode() if isinstance(a['name'],bytes) else a['name']
        vals.append(axes.get(n, a['default']))
    f.set_variation_by_axes(vals); return f
print([ (a['name'],a['minimum'],a['default'],a['maximum']) for a in ImageFont.truetype("fonts/Fraunces.ttf",20).get_variation_axes()])
print([ (a['name'],a['minimum'],a['default'],a['maximum']) for a in ImageFont.truetype("fonts/Manrope.ttf",20).get_variation_axes()])
# axis names as FreeType reports them (Fraunces: Optical size, Weight, Softness, Wonky; Manrope: Weight)
T=vf("fonts/Fraunces.ttf",76,{"Optical Size":72,"Weight":620,"Softness":60,"Wonky":0})
S=vf("fonts/Manrope.ttf",33,{"Weight":600})
P=vf("fonts/Manrope.ttf",28,{"Weight":800})
L1=vf("fonts/Manrope.ttf",42,{"Weight":800})
L2=vf("fonts/Manrope.ttf",33,{"Weight":700})
im=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(im)
# band: horizontal plum -> sienna gradient, amber rule under it
for x in range(W):
    t=x/(W-1); c=tuple(int(PLUM[i]*(1-t)+SIENNA[i]*t) for i in range(3))
    d.line([(x,0),(x,BAND-1)],fill=c+(255,))
d.rectangle([0,BAND-5,W,BAND-1],fill=AMBER+(255,))
d.text((56,0),"same prompt, three models",font=T,fill=SAND+(255,))
d.text((58,90),"one hour of /goal in Hermes Agent, RTX 5090",font=S,fill=SAND+(230,))
# speed pill, right
pt="real speed"; bb=d.textbbox((0,0),pt,font=P); pw=bb[2]-bb[0]+48; ph=56
px=W-56-pw; py=44
d.rounded_rectangle([px,py,px+pw,py+ph],radius=28,outline=AMBER+(255,),width=3,fill=(0,0,0,60))
d.text((px+24,py+(ph-(bb[3]-bb[1]))//2-bb[1]),pt,font=P,fill=AMBER+(255,))
# bottom dusk strip (mirrored gradient) with an amber rule on top
for x in range(W):
    t=x/(W-1); c=tuple(int(SIENNA[i]*(1-t)+PLUM[i]*t) for i in range(3))
    d.line([(x,PANEL_Y+PANEL_H),(x,H)],fill=c+(255,))
d.rectangle([0,PANEL_Y+PANEL_H,W,PANEL_Y+PANEL_H+3],fill=AMBER+(255,))
# label strip: opaque, one cell per panel, never over the image
labels=[("Qwen3.8 27B","Q6_K  ·  57 KB of code"),("Flint 40B-A2.5B","UD-Q4_K_XL  ·  27 KB of code"),("Muse Glimmer 30B","Q4_K_M  ·  31 KB of code")]
d.rectangle([0,BAND,W,PANEL_Y-1],fill=INK+(255,))
for i,(a,b) in enumerate(labels):
    x0=i*640; x1=(i+1)*640; y0=BAND; y1=PANEL_Y
    d.rectangle([x0+(4 if i else 0),y0+8,x0+(4 if i else 0)+7,y1-8],fill=AMBER+(255,))
    d.text((x0+30,y0+6),a,font=L1,fill=SAND+(255,))
    d.text((x0+30,y0+60),b,font=L2,fill=AMBER+(255,))
    bb1=d.textbbox((x0+30,y0+6),a,font=L1); bb2=d.textbbox((x0+30,y0+60),b,font=L2)
    print("label",i,a,"|",b,"bbox",bb1,bb2,"cell",x0,x1,y0,y1)
print("setup bbox",d.textbbox((58,90),"one hour of /goal in Hermes Agent, RTX 5090",font=S),"title bbox",d.textbbox((56,0),"same prompt, three models",font=T))
# panel separators, full height of label strip + panels
for x in (640,1280):
    d.rectangle([x-2,BAND,x+1,PANEL_Y+PANEL_H],fill=INK+(255,))
im.save("overlay.png"); print("overlay ok")
PY
FC="[0:v]trim=0:30,setpts=PTS-STARTPTS,crop=$CW:1080:$QX:0,scale=640:720:flags=lanczos,setsar=1[a];\
[1:v]trim=0:30,setpts=PTS-STARTPTS,crop=$CW:1080:$FX:0,scale=640:720:flags=lanczos,setsar=1[b];\
[2:v]trim=0:30,setpts=PTS-STARTPTS,crop=$CW:1080:$MX:0,scale=640:720:flags=lanczos,setsar=1[c];\
[a][b][c]hstack=inputs=3[s];[s]pad=1920:1080:0:260:color=0x2a1622[p];[p][3:v]overlay=0:0,fps=30,format=yuv420p[v]"
nice -n 19 $F -nostdin -y -i $A -i $B -i $C -i overlay.png -filter_complex "$FC" -map "[v]" \
  -c:v libx264 -profile:v high -preset slow -crf 18 -pix_fmt yuv420p -r 30 -threads 6 -movflags +faststart -an vs3-master.mp4 </dev/null 2>qa/ff1.log
nice -n 19 $F -nostdin -y -i vs3-master.mp4 -c:v libx264 -profile:v high -preset slow -b:v 2400k -maxrate 3000k -bufsize 4800k -pass 1 -passlogfile x264vs3 -enc_time_base 1/30 -threads 6 -an -f null /dev/null </dev/null 2>qa/ff2.log
nice -n 19 $F -nostdin -y -i vs3-master.mp4 -c:v libx264 -profile:v high -preset slow -b:v 2400k -maxrate 3000k -bufsize 4800k -pass 2 -passlogfile x264vs3 -enc_time_base 1/30 -threads 6 -pix_fmt yuv420p -movflags +faststart -an vs3-discord.mp4 </dev/null 2>qa/ff3.log
nice -n 19 $F -nostdin -y -i vs3-discord.mp4 -vf "fps=8/30,scale=640:-1,tile=4x2" -frames:v 1 qa/sheet.png </dev/null 2>/dev/null
for t in 1 15 29; do nice -n 19 $F -nostdin -y -ss $t -i vs3-discord.mp4 -frames:v 1 qa/full_$t.png </dev/null 2>/dev/null; done
rm -rf qa/lum; mkdir qa/lum; nice -n 19 $F -nostdin -y -i vs3-master.mp4 -vf "fps=1,scale=480:-1" qa/lum/f_%02d.png </dev/null 2>/dev/null
$PY - <<'PY'
from PIL import Image, ImageStat; import glob
rows=[]
for p in sorted(glob.glob("qa/lum/f_*.png")):
    im=Image.open(p).convert("L"); w,h=im.size; y0=int(h*260/1080); y1=int(h*980/1080)
    v=[ImageStat.Stat(im.crop((i*w//3+2,y0,(i+1)*w//3-2,y1))).mean[0] for i in range(3)]
    rows.append(f"{int(p[-6:-4])-1}:{v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f}")
print("luma s:qwen/flint/muse", " ".join(rows))
PY
ls -la vs3-master.mp4 vs3-discord.mp4 qa/
for f in vs3-master.mp4 vs3-discord.mp4; do $F -hide_banner -nostdin -i $f 2>&1 | grep -E "Duration|Video:" || true; done
md5sum vs3-discord.mp4
