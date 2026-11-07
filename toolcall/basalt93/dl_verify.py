#!/usr/bin/env python3
"""dl_verify.py <repo> <localdir> <min_free_gb_after> <file> [<file> ...]
HF download through the xet-enabled hf CLI, size-checked against the Hub API byte count. Refuses to start when free space minus the remaining bytes would drop under <min_free_gb_after>.
exit 0 = every file present at the exact API size; 2 = guard refused; 3 = size mismatch / download failed."""
import json, os, shutil, subprocess, sys, urllib.request
repo, ldir, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
files = sys.argv[4:]
HF = "/home/witcheer/unsloth-env/bin/hf"
req = urllib.request.Request("https://huggingface.co/api/models/%s?blobs=true" % repo, headers={"User-Agent": "dl-verify"})
sib = {s["rfilename"]: s["size"] for s in json.load(urllib.request.urlopen(req, timeout=60))["siblings"]}
for f in files:
    if f not in sib:
        print("FAIL: %s not in %s" % (f, repo)); sys.exit(3)
os.makedirs(ldir, exist_ok=True)
need = sum(max(0, sib[f] - (os.path.getsize(os.path.join(ldir, f)) if os.path.exists(os.path.join(ldir, f)) else 0)) for f in files)
free = shutil.disk_usage("/").free
print("need %.1f GB more, free %.1f GB, floor after %.0f GB" % (need / 1e9, free / 1e9, floor), flush=True)
if free - need < floor * 1e9:
    print("REFUSED: free after the download would be %.1f GB < %.0f GB" % ((free - need) / 1e9, floor)); sys.exit(2)
if os.environ.get("DL_DRY") == "1":
    print("DRY: would download %s" % ", ".join(files)); sys.exit(0)
rc = subprocess.call([HF, "download", repo] + files + ["--local-dir", ldir])
bad = 0
for f in files:
    p = os.path.join(ldir, f)
    sz = os.path.getsize(p) if os.path.exists(p) else -1
    ok = sz == sib[f]
    print("%s %s %d / %d" % ("OK  " if ok else "BAD ", f, sz, sib[f]), flush=True)
    bad += (not ok)
sys.exit(0 if (rc == 0 and not bad) else 3)
