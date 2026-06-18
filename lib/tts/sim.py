# lib/tts/sim.py
"""SIM-o speaker similarity: cosine between WavLM-large+ECAPA embeddings of two wavs (16kHz
mono), using F5-TTS's ECAPA_TDNN_SMALL + the wavlm_large_finetune.pth checkpoint — the metric
the Seed-TTS / F5-TTS / CosyVoice papers report, so numbers are comparable. Torch; capsule.

The ckpt lives on capsule at ~/tts-data/wavlm_large_finetune.pth (gdown'd in Task 0); pass its
path to embed/sim_o. The f5_tts import is lazy so the pure cosine() unit runs without the model."""
import torch
import torch.nn.functional as F
import soundfile as sf


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())


_MODEL = None


def _load(ckpt_path, device="cuda"):
    global _MODEL
    if _MODEL is None:
        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL  # vendored WavLM+ECAPA (wavlm_large via s3prl)
        m = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state["model"] if "model" in state else state, strict=False)
        _MODEL = m.to(device).eval()
    return _MODEL


def embed(wav_path, ckpt_path, device="cuda"):
    wav, sr = sf.read(wav_path)
    x = torch.tensor(wav, dtype=torch.float32)
    if x.ndim > 1:
        x = x.mean(-1)            # mono
    assert sr == 16000, f"SIM-o expects 16kHz, got {sr} for {wav_path} (resample upstream)"
    with torch.no_grad():
        return _load(ckpt_path, device)(x.unsqueeze(0).to(device))


def sim_o(synth_wav, ref_wav, ckpt_path, device="cuda"):
    return cosine(embed(synth_wav, ckpt_path, device), embed(ref_wav, ckpt_path, device))
