#!/usr/bin/env python3
"""arete_facts.py RUNDIR [--no-sha] [--stub]: file facts for item 96-arete-eval, CPU only, run BEFORE the card is touched. Frozen (FROZEN.sha256).

Writes RUNDIR/facts.json: sizes, sha256 of both GGUFs (skipped with --no-sha), the Hugging Face LFS oid recorded for each file, hash_ok, the llama.cpp build line, the dataset revision and
the sha256 of the pinned eval files, and the GGUF header facts of both files that matter for the cutter confound (general.quantized_by, imatrix file/dataset/chunks, chat template hash, tensor types).
--stub marks the file mock (stub proof). Prints a one-line summary last.
"""
import collections, hashlib, json, os, subprocess, sys, threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
H = os.path.expanduser("~")
FILES = {"base": H + "/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf", "arete": H + "/models/arete-27b/NousResearch_Arete-27B-Q6_K.gguf"}
BYTES = {"base": 22884408288, "arete": 23621581824}
# oid of record: base = unsloth/Qwen3.8-27B-GGUF @ f1bfb127 (the file item 90a proved byte-identical on this box); Arete = bartowski/NousResearch_Arete-27B-GGUF LFS oid read 2026-11-07
OID = {"base": "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727", "arete": "5b4588c250b941d8dcbbfc610eb43d8303b83270229fc6da5b5ae174f52a23c3"}
SRV = H + "/llama.cpp-donald/build/bin/llama-server"
DATASET_REV = "401223a2bfcdcd99d4650a83c4d20565db0e56b8"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_small(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def gguf_meta(path):
    sys.path.insert(0, H + "/llama.cpp-donald/gguf-py")
    import gguf
    r = gguf.GGUFReader(path, "r")
    out = {}
    for k in ("general.name", "general.quantized_by", "general.file_type", "quantize.imatrix.file", "quantize.imatrix.dataset", "quantize.imatrix.entries_count", "quantize.imatrix.chunks_count",
              "qwen35.block_count", "qwen35.nextn_predict_layers"):
        f = r.fields.get(k)
        if f is None:
            out[k] = None
            continue
        try:
            v = f.contents()
        except Exception:
            v = None
        out[k] = os.path.basename(v) if k in ("quantize.imatrix.file", "quantize.imatrix.dataset") and isinstance(v, str) else v
    f = r.fields.get("tokenizer.chat_template")
    out["chat_template_sha16"] = hashlib.sha256(bytes(f.parts[f.data[0]])).hexdigest()[:16] if f is not None else None
    out["chat_template_has_enable_thinking"] = bool(f is not None and b"enable_thinking" in bytes(f.parts[f.data[0]]))
    out["tensor_types"] = dict(collections.Counter(t.tensor_type.name for t in r.tensors))
    out["n_tensors"] = len(r.tensors)
    return out


def main():
    run = sys.argv[1]
    no_sha = "--no-sha" in sys.argv or "--stub" in sys.argv
    stub = "--stub" in sys.argv
    os.makedirs(run, exist_ok=True)
    f = dict(mock=stub, dataset_revision=DATASET_REV, oid=OID, gb={k: round(BYTES[k] / 1e9, 2) for k in BYTES}, bytes=BYTES)
    f["size_ok"] = {k: (os.path.exists(FILES[k]) and os.path.getsize(FILES[k]) == BYTES[k]) for k in FILES}
    shas = {}
    if not no_sha:
        def work(k):
            shas[k] = sha256_file(FILES[k])
        ts = [threading.Thread(target=work, args=(k,)) for k in FILES]
        [t.start() for t in ts]
        [t.join() for t in ts]
        f["sha256"] = shas
        f["hash_ok"] = {k: shas.get(k) == OID[k] for k in FILES}
    else:
        f["sha256"] = None
        f["hash_ok"] = None
    f["meta"] = {}
    for k in FILES:
        try:
            f["meta"][k] = gguf_meta(FILES[k])
        except Exception as e:
            f["meta"][k] = {"error": repr(e)[:200]}
    try:
        f["build"] = subprocess.run([SRV, "--version"], capture_output=True, text=True, timeout=60).stderr.strip().splitlines()[0]
    except Exception:
        try:
            f["build"] = subprocess.run([SRV, "--version"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0]
        except Exception as e:
            f["build"] = "unknown: %r" % e
    f["pinned"] = {n: sha_small(os.path.join(ROOT, p)) for n, p in (("items.jsonl", "data/items.jsonl"), ("JUDGE_PROMPT.md", "judge/JUDGE_PROMPT.md"), ("grade.py", "judge/grade.py"), ("score.py", "judge/score.py"))}
    json.dump(f, open(os.path.join(run, "facts.json"), "w"), indent=1)
    m = f["meta"]
    print("facts ok: mock=%s hash_ok=%s sizes_ok=%s quantized_by base=%s arete=%s imatrix_chunks base=%s arete=%s" % (
        stub, f["hash_ok"], f["size_ok"], m["base"].get("general.quantized_by"), m["arete"].get("general.quantized_by"),
        m["base"].get("quantize.imatrix.chunks_count"), m["arete"].get("quantize.imatrix.chunks_count")))


if __name__ == "__main__":
    main()
