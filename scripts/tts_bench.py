# scripts/tts_bench.py
"""Score a model's synthesized wavs (from scripts/tts_synth_*.py -> <synth_dir>/synth.json) into
per-utterance + aggregate metrics. Synthesis (GPU) is run separately per model in its own env;
this scoring phase is light (whisper-cli + SIM-o) and runs Donald-up. Resamples to 16kHz mono with
soxr (capsule has no ffmpeg). Usage:
  PYTHONPATH=. tts-metric-env/bin/python scripts/tts_bench.py <model_slug> <synth_dir> <sim_ckpt>
Writes results/tts/<slug>.json."""
import json, os, sys
import soundfile as sf
import soxr
from lib.tts.roundtrip import transcribe
from lib.tts.sim import sim_o
from lib.tts.score import aggregate
from lib.asr.wer import wer
from lib.asr.score import rtfx


def _to16k(wav):
    """Write a 16kHz mono copy next to wav (whisper + SIM-o both want 16k mono). soxr resample."""
    data, sr = sf.read(wav)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)                      # downmix to mono
    if sr != 16000:
        data = soxr.resample(data, sr, 16000)
    out = (wav[:-4] if wav.endswith(".wav") else wav) + ".16k.wav"
    sf.write(out, data, 16000)
    return out


def main():
    slug, synth_dir, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
    recs = json.load(open(os.path.join(synth_dir, "synth.json")))
    per_utt = []
    for r in recs:
        w16 = _to16k(r["wav"])
        hyp = transcribe(w16)
        ref16 = _to16k(r["ref_audio"])
        per_utt.append({
            "name": r["name"],
            "wer": wer(r["target_text"], hyp),
            "rtfx": rtfx(r["audio_seconds"], r["synth_seconds"]),
            "first_audio_s": r["first_audio_seconds"],
            "sim": sim_o(w16, ref16, ckpt),
        })
        print(f"{r['name']} wer={per_utt[-1]['wer']:.3f} rtfx={per_utt[-1]['rtfx']:.2f} "
              f"sim={per_utt[-1]['sim']:.3f}", flush=True)
    agg = aggregate(per_utt)
    os.makedirs("results/tts", exist_ok=True)
    json.dump({"model": slug, "agg": agg, "per_utt": per_utt},
              open(f"results/tts/{slug}.json", "w"), indent=2)
    print(f"TTS_SCORE_DONE {slug}", {k: round(v, 3) for k, v in agg.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
