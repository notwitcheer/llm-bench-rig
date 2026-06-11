#!/usr/bin/env python3
"""ASR head-to-head — LOAD-ONCE generation on capsule (supersedes asr_bench.sh).

Each model loads its weights ONCE and transcribes the whole LibriSpeech subset, so RTFx
measures transcription, not the per-file model-LOAD that a one-shot-CLI loop accidentally
times (the bogus-RTFx trap). Emits results/asr/<model>/{transcripts.jsonl,timing.json} in
the exact contract asr_report.py (Mac, pure scoring) expects. Donald (llama-server) is
stopped for the run and ALWAYS restarted on exit.

  parakeet : native `bench --manifest --json`  -> load_ms once, per-clip proc_ms + text
  whisper  : `whisper-cli ... <all wavs>`       -> one load over the invocation, <wav>.json each

Run, Hermes-safe, in tmux:
  tmux new -d -s asrbench "~/benchmark-rig/.venv/bin/python scripts/asr_bench.py 2>&1 | tee ~/asrbench.log"
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

LIMIT = int(os.environ.get("ASR_LIMIT", "0"))  # >0 = smoke-test on first N utts/split

HOME = Path.home()
MAN = HOME / "asr-data/librispeech"   # <split>/manifest.jsonl : {id,split,audio,duration,reference}
WAV = HOME / "asr-data/wav"           # <split>/<id>.wav
OUT = HOME / "benchmark-rig/results/asr"
SPLITS = ["test-clean", "test-other"]

PARAKEET = HOME / "parakeet.cpp/build/examples/cli/parakeet-cli"
WHISPER = HOME / "whisper.cpp/build/bin/whisper-cli"
PK_MODEL = HOME / "asr-models/parakeet/tdt-0.6b-v2-f16.gguf"
WH_MODELS = {
    "whisper-large-v3-turbo": HOME / "whisper.cpp/models/ggml-large-v3-turbo.bin",
    "whisper-large-v3": HOME / "whisper.cpp/models/ggml-large-v3.bin",
}


def load_manifest():
    man = {sp: [json.loads(l) for l in (MAN / sp / "manifest.jsonl").read_text().splitlines() if l.strip()]
           for sp in SPLITS}
    if LIMIT:
        man = {sp: rows[:LIMIT] for sp, rows in man.items()}
    return man


class VramSampler:
    """Background peak-VRAM sampler (4 Hz). Donald is stopped, so this is the model's own footprint.
    Fast interval matters: parakeet transcribes the whole subset in seconds, so a 1 Hz sampler misses it."""
    def __init__(self):
        self.peak = 0
        self._stop = False

    def _run(self):
        while not self._stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            time.sleep(0.25)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=2)


def donald(action):
    subprocess.run(["sudo", "-n", "systemctl", action, "llama-server.service"], check=False)


def write_results(name, rows, audio_s, proc_s, vram):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "transcripts.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (d / "timing.json").write_text(json.dumps(
        {"audio_seconds": round(audio_s, 3), "proc_seconds": round(proc_s, 3), "peak_vram_mib": vram}))
    rtfx = audio_s / proc_s if proc_s else 0
    print(f"[asr] {name}: {len(rows)} utts | audio {audio_s/60:.1f}min | proc {proc_s:.1f}s "
          f"| RTFx {rtfx:.1f} | VRAM {vram}MiB", flush=True)


def run_parakeet(man):
    tsv = HOME / "asr-data/parakeet_manifest.tsv"
    meta = {}  # wav_path -> (id, split, duration, reference)
    with tsv.open("w") as f:
        for sp in SPLITS:
            for r in man[sp]:
                wp = str(WAV / sp / r["audio"])
                f.write(f"{wp}\t{r['reference']}\n")
                meta[wp] = (r["id"], sp, r["duration"], r["reference"])
    out_json = HOME / "asr-data/parakeet_out.json"
    with VramSampler() as vs:
        subprocess.run([str(PARAKEET), "bench", "--model", str(PK_MODEL), "--manifest", str(tsv),
                        "--decoder", "tdt", "--lang", "en", "--json", str(out_json)], check=True)
    res = json.loads(out_json.read_text())
    rows, audio_s, proc_ms = [], 0.0, 0.0
    for fr in res["files"]:
        mid, sp, dur, ref = meta[fr["path"]]
        rows.append({"id": mid, "split": sp, "reference": ref, "hypothesis": fr["text"]})
        audio_s += fr["audio_sec"]
        proc_ms += fr["proc_ms"]
    miss = sum(len(man[sp]) for sp in SPLITS) - len(rows)
    if miss:
        print(f"[asr] WARNING parakeet: {miss} utts missing from bench output", flush=True)
    write_results("parakeet-tdt-0.6b-v2", rows, audio_s, proc_ms / 1000.0, vs.peak)


def run_whisper(name, model, man):
    rows, audio_s, proc_s, vram = [], 0.0, 0.0, 0
    for sp in SPLITS:
        wavs = [str(WAV / sp / r["audio"]) for r in man[sp]]
        with VramSampler() as vs:
            t0 = time.time()
            p = subprocess.run([str(WHISPER), "-m", str(model), "-oj", "-nt", "-l", "en", *wavs],
                               capture_output=True, text=True)
            wall = time.time() - t0
        load_ms = 0.0  # one model load over the whole invocation -> subtract once for honest proc
        for ln in p.stderr.splitlines():
            if "load time" in ln:
                load_ms = float(ln.split("=")[1].strip().split()[0])
                break
        proc_s += max(wall - load_ms / 1000.0, 0.0)
        vram = max(vram, vs.peak)
        n_empty = 0
        for r in man[sp]:
            jp = WAV / sp / (r["audio"] + ".json")
            txt = ""
            try:
                jd = json.loads(jp.read_text())
                txt = "".join(s.get("text", "") for s in jd.get("transcription", [])).strip()
            except Exception:
                txt = ""
            if not txt:
                n_empty += 1
            rows.append({"id": r["id"], "split": sp, "reference": r["reference"], "hypothesis": txt})
            audio_s += r["duration"]
            jp.unlink(missing_ok=True)
        if n_empty:
            print(f"[asr] WARNING {name}/{sp}: {n_empty} empty transcripts", flush=True)
    write_results(name, rows, audio_s, proc_s, vram)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    print(f"[asr] manifest loaded: {sum(len(v) for v in man.values())} utts "
          f"({', '.join(f'{sp}={len(man[sp])}' for sp in SPLITS)})", flush=True)
    donald("stop")
    time.sleep(3)
    try:
        run_parakeet(man)
        for name, model in WH_MODELS.items():
            run_whisper(name, model, man)
    finally:
        donald("start")
    print("ASR_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
