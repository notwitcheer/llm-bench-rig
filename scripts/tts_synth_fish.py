# scripts/tts_synth_fish.py
"""Fish-S2-Pro synthesis adapter. Drives the fish-speech HTTP API server (tools/api_server.py,
started separately on :8080 with the s2-pro llama + codec checkpoints), voice-cloning each
target_text from its ref_audio. The server speaks ormsgpack-packed ServeTTSRequest (NOT JSON) —
this mirrors tools/api_client.py exactly. Writes a wav + timing per utterance. Runs in
~/tts-fish-env. Usage:
  PYTHONPATH=. tts-fish-env/bin/python scripts/tts_synth_fish.py <manifest> <base_dir> <out_dir> [limit]
Output sr is whatever the codec emits (recorded per-utt; resampled to 16k downstream for WER/SIM-o)."""
import json, os, sys, time
import requests
import ormsgpack
import soundfile as sf
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from lib.tts.dataset import load_manifest

URL = os.environ.get("FISH_URL", "http://127.0.0.1:8080/v1/tts")


def main():
    manifest, base_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(out_dir, exist_ok=True)
    rows = load_manifest(manifest, base_dir, limit)
    recs = []
    for u in rows:
        ref_bytes = open(u["ref_audio"], "rb").read()
        req = ServeTTSRequest(
            text=u["target_text"],
            references=[ServeReferenceAudio(audio=ref_bytes, text=u["ref_text"])],
            format="wav",
            normalize=True,
        )
        t0 = time.perf_counter()
        r = requests.post(URL, params={"format": "msgpack"},
                          data=ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                          headers={"content-type": "application/msgpack",
                                   "authorization": "Bearer none"},
                          timeout=300)
        synth_s = time.perf_counter() - t0
        r.raise_for_status()
        wp = os.path.join(out_dir, u["name"] + ".wav")
        with open(wp, "wb") as f:
            f.write(r.content)
        info = sf.info(wp)
        recs.append({"name": u["name"], "wav": wp, "ref_audio": u["ref_audio"],
                     "target_text": u["target_text"], "synth_seconds": synth_s,
                     "audio_seconds": info.frames / info.samplerate,
                     "first_audio_seconds": synth_s, "sr": info.samplerate})
        print(f"{u['name']} synth={synth_s:.2f}s sr={info.samplerate}", flush=True)
    json.dump(recs, open(os.path.join(out_dir, "synth.json"), "w"), indent=2)
    print("FISH_SYNTH_DONE", len(recs))


if __name__ == "__main__":
    main()
