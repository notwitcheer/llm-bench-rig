#!/usr/bin/env python3
import hashlib, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
TUNER = sys.argv[1] if len(sys.argv) > 1 else "/home/witcheer/llama.cpp-moecache/scripts/moe-cache-tune.py"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
seed = subprocess.run("cd %s/seed && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64" % H, shell=True, capture_output=True, text=True).stdout.strip()
h = {"__H_SUMMARY__": sha(H + "/mc_summary.py"), "__H_GATE__": sha(H + "/rp_gate.py"), "__H_PROMPT__": sha(H + "/prompts/fix-shopcart.txt"), "__H_SEED__": seed, "__H_TUNER__": sha(TUNER), "__H_MKHOME__": sha(H + "/mc_mkhome.py")}
t = open(H + "/PREREG.md.tpl").read()
for k, v in h.items(): t = t.replace(k, v)
open(H + "/PREREG.md", "w").write(t)
open(H + "/SEED.sha256", "w").write(seed + "  seed/ (find|sort|sha256sum|sha256sum)\n")
lines = ["%s  mc_summary.py" % h["__H_SUMMARY__"], "%s  rp_gate.py" % h["__H_GATE__"], "%s  prompts/fix-shopcart.txt" % h["__H_PROMPT__"], "%s  %s" % (h["__H_TUNER__"], TUNER), "%s  mc_mkhome.py" % h["__H_MKHOME__"], "%s  PREREG.md" % sha(H + "/PREREG.md")]
open(H + "/FROZEN.sha256", "w").write("\n".join(lines) + "\n")
print(open(H + "/FROZEN.sha256").read())
print("FROZEN.sha256 sha256:", sha(H + "/FROZEN.sha256"))
