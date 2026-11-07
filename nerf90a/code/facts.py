#!/usr/bin/env python3
"""facts.py <RUNDIR> [--no-sha] : builds RUNDIR/facts.json for the nerf audit (item 90a). Everything the clip and the summary print about the FILES comes from here.
  sha_q6k / sha_q4km : sha256 of the two GGUFs on this disk (computed now, streamed, CPU)
  oid_aug14_q*       : the LFS oid Hugging Face listed for the same file name in unsloth/Qwen3.8-27B-GGUF at commit f1bfb127 (14-15 Aug, the files our day-0 board ran);
                       read live from the HF tree API when reachable, else the value frozen below (source recorded in 'oid_source')
  hf_weight_files_*  : Qwen/Qwen3.8-27B safetensors shards whose LFS oid at `main` differs from the oid at the 13 Aug upload commit 72a217af (live; else the staged read 0 of 18)
  build_old/build_new: llama-server --version first lines
The GGUF repo removed Q6_K and plain Q4_K_M on 19 Aug (commit 24fde402 / e1d8a267), so 'today' has no Hugging Face listing to compare with: the oid of record is the last listing (f1bfb127).
"""
import hashlib, json, os, subprocess, sys, urllib.request

OID_Q6 = "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727"      # unsloth/Qwen3.8-27B-GGUF @ f1bfb127c64f7072bdd2cad55f258b9c8b2910fe, read 2026-11-06
OID_Q4 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
GGUF_REV = "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"
Q6_PATH = os.path.expanduser("~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf")
Q4_PATH = os.path.expanduser("~/models/qwen38-27b-day0/Qwen3.8-27B-Q4_K_M.gguf")
SRV_OLD = os.path.expanduser("~/llama.cpp-nl/build/bin/llama-server")
SRV_NEW = os.path.expanduser("~/llama.cpp-donald/build/bin/llama-server")
UPLOAD_REV = "72a217af"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "nerf-audit"}), timeout=30).read())


def version(srv):
    try:
        out = subprocess.run([srv, "--version"], capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip().splitlines()[0]
    except Exception as e:
        return "unreadable: %s" % e


def main(rundir, no_sha=False):
    os.makedirs(rundir, exist_ok=True)
    f = dict(oid_aug14_q6k=OID_Q6, oid_aug14_q4km=OID_Q4, gguf_rev=GGUF_REV, oid_source="frozen constants (HF tree API read 2026-11-06)")
    try:
        t = get("https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/tree/%s?recursive=1" % GGUF_REV)
        live = {x["path"]: (x.get("lfs") or {}).get("oid") for x in t if x["type"] == "file"}
        if live.get("Qwen3.8-27B-Q6_K.gguf"):
            f["oid_aug14_q6k"], f["oid_aug14_q4km"] = live["Qwen3.8-27B-Q6_K.gguf"], live["Qwen3.8-27B-Q4_K_M.gguf"]
            f["oid_source"] = "HF tree API at %s, read live" % GGUF_REV[:8]
    except Exception as e:
        f["oid_source"] += " (live read failed: %s)" % str(e)[:60]
    try:
        a = {x["path"]: (x.get("lfs") or {}).get("oid") for x in get("https://huggingface.co/api/models/Qwen/Qwen3.8-27B/tree/%s?recursive=1" % UPLOAD_REV) if x["path"].endswith(".safetensors")}
        b = {x["path"]: (x.get("lfs") or {}).get("oid") for x in get("https://huggingface.co/api/models/Qwen/Qwen3.8-27B/tree/main?recursive=1") if x["path"].endswith(".safetensors")}
        f["hf_weight_files_total"] = len(b)
        f["hf_weight_files_changed"] = sum(1 for k in b if a.get(k) != b[k]) + sum(1 for k in a if k not in b)
        f["hf_weight_source"] = "HF tree API live: main vs %s" % UPLOAD_REV
    except Exception as e:
        f["hf_weight_files_total"], f["hf_weight_files_changed"] = 18, 0
        f["hf_weight_source"] = "staged read 2026-11-06 (live read failed: %s)" % str(e)[:60]
    if no_sha:
        f["sha_q6k"], f["sha_q4km"] = f["oid_aug14_q6k"], f["oid_aug14_q4km"]
        f["sha_note"] = "NOT HASHED (--no-sha), oid copied: test use only"
    else:
        f["sha_q6k"], f["sha_q4km"] = sha256(Q6_PATH), sha256(Q4_PATH)
    f["hash_ok"] = dict(q6k=(f["sha_q6k"] == f["oid_aug14_q6k"]), q4km=(f["sha_q4km"] == f["oid_aug14_q4km"]))
    f["build_old"], f["build_new"] = version(SRV_OLD), version(SRV_NEW)
    json.dump(f, open(os.path.join(rundir, "facts.json"), "w"), indent=1)
    print("facts ok: q6k hash_ok=%s q4km hash_ok=%s weight files changed %s/%s | old: %s | new: %s" % (f["hash_ok"]["q6k"], f["hash_ok"]["q4km"], f["hf_weight_files_changed"], f["hf_weight_files_total"], f["build_old"], f["build_new"]))


if __name__ == "__main__":
    main(sys.argv[1], "--no-sha" in sys.argv)
