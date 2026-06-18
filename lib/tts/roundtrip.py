"""Round-trip transcription for TTS WER: shell whisper-cli (large-v3) over a synthesized wav,
read the <wav>.json it writes, join the transcription segments. Round-trip WER itself is just
lib.asr.wer(target_text, transcription) (it normalizes internally). Matches the ASR bench's
whisper usage exactly (`-oj -nt -l en`)."""
import json
import os
import subprocess

WHISPER = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-large-v3.bin")


def parse_whisper_json(jd):
    """Join whisper-cli -oj transcription segments into one normalized-ready string."""
    return "".join(s.get("text", "") for s in jd.get("transcription", [])).strip()


def transcribe(wav_path, model=WHISPER_MODEL):
    """Transcribe one wav with whisper-cli; returns the text. Writes <wav>.json next to it."""
    subprocess.run([WHISPER, "-m", model, "-oj", "-nt", "-l", "en", wav_path],
                   check=True, capture_output=True)
    with open(wav_path + ".json") as f:
        return parse_whisper_json(json.load(f))


def transcribe_batch(wav_paths, model=WHISPER_MODEL):
    """Transcribe many wavs with ONE whisper-cli invocation: the model loads once (not per clip),
    which is far faster AND avoids the per-clip GPU reload that contends with a co-resident server
    (e.g. Donald) -> intermittent CUDA failures. Writes <wav>.json next to each; returns
    {wav_path: text}. Matches the ASR bench's load-once multi-file usage."""
    if not wav_paths:
        return {}
    subprocess.run([WHISPER, "-m", model, "-oj", "-nt", "-l", "en", *wav_paths],
                   check=True, capture_output=True)
    out = {}
    for w in wav_paths:
        with open(w + ".json") as f:
            out[w] = parse_whisper_json(json.load(f))
    return out
