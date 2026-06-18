# scripts/tts_synth_qwen3.py
"""Qwen3-TTS-1.7B-Base synthesis adapter. Loads once, synthesizes each utterance's target_text
voice-cloned from its ref_audio/ref_text, writes a wav + a per-utt timing record. Runs in
~/tts-qwen-env. Usage:
  PYTHONPATH=. tts-qwen-env/bin/python scripts/tts_synth_qwen3.py <manifest> <base_dir> <out_dir> [limit]
sm_120 has no stock flash-attn -> attn_implementation="sdpa". API verified against the
Qwen/Qwen3-TTS-12Hz-1.7B-Base model card: generate_voice_clone(text, language, ref_audio, ref_text)
returns (wavs, sr); ref_audio accepts a local file path."""
import json, os, sys, time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from lib.tts.dataset import load_manifest

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main():
    manifest, base_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(out_dir, exist_ok=True)
    rows = load_manifest(manifest, base_dir, limit)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    recs = []
    for u in rows:
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=u["target_text"], language="English",
            ref_audio=u["ref_audio"], ref_text=u["ref_text"],
        )
        synth_s = time.perf_counter() - t0
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else (wavs[0] if getattr(wavs, "ndim", 1) > 1 else wavs)
        wp = os.path.join(out_dir, u["name"] + ".wav")
        sf.write(wp, wav, sr)
        recs.append({"name": u["name"], "wav": wp, "ref_audio": u["ref_audio"],
                     "target_text": u["target_text"], "synth_seconds": synth_s,
                     "audio_seconds": len(wav) / sr, "first_audio_seconds": synth_s, "sr": sr})
        print(f"{u['name']} synth={synth_s:.2f}s", flush=True)
    json.dump(recs, open(os.path.join(out_dir, "synth.json"), "w"), indent=2)
    print("QWEN3_SYNTH_DONE", len(recs))


if __name__ == "__main__":
    main()
