#!/usr/bin/env python3
"""compose the side-by-side: l0.46 | l0.56 (each 960x1080, same camera scale), small label per half, nothing else added.
usage: trees-fixedcam-compose.py <dir with l0.46/fc.mp4 and l0.56/fc.mp4> <out.mp4> [--crf 20]"""
import os, subprocess, sys, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

d = Path(sys.argv[1]); out = Path(sys.argv[2])
crf = sys.argv[sys.argv.index("--crf") + 1] if "--crf" in sys.argv else "20"
FONT = next(p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"] if os.path.exists(p))
f = ImageFont.truetype(FONT, 40)
ov = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
dr = ImageDraw.Draw(ov)
for x0, txt in ((0, "\u03bb 0.46"), (960, "\u03bb 0.56")):
    bb = dr.textbbox((0, 0), txt, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = x0 + 960 - tw - 36; y = 1080 - th - 40
    dr.rounded_rectangle((x - 16, y - 12 + bb[1], x + tw + 16, y + th + 12 + bb[1]), radius=10, fill=(10, 14, 18, 150))
    dr.text((x - bb[0], y - bb[1] + bb[1]), txt, font=f, fill=(238, 244, 238, 255))
ovp = d / "labels.png"; ov.save(ovp)
ff = os.path.expanduser("~/voxelbench/ffmpeg"); ff = ff if os.path.exists(ff) else shutil.which("ffmpeg")
r = subprocess.run([ff, "-nostdin", "-y", "-i", str(d / "l0.46/fc.mp4"), "-i", str(d / "l0.56/fc.mp4"), "-i", str(ovp),
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2:shortest=1[s];[s][2:v]overlay=0:0:format=auto[o]", "-map", "[o]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", crf, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                   capture_output=True, stdin=subprocess.DEVNULL, timeout=900)
print(r.returncode, r.stderr.decode(errors="ignore")[-300:])
