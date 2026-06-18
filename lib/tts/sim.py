# lib/tts/sim.py
"""SIM-o speaker similarity: cosine between WavLM-large+ECAPA embeddings of two wavs (16kHz
mono), using F5-TTS's ECAPA_TDNN_SMALL + the wavlm_large_finetune.pth checkpoint — the metric
the Seed-TTS / F5-TTS / CosyVoice papers report, so numbers are comparable. Torch; capsule.

The ckpt lives on capsule at ~/tts-data/wavlm_large_finetune.pth (gdown'd in Task 0); pass its
path to embed/sim_o. The f5_tts import is lazy so the pure cosine() unit runs without the model."""
import os
import shutil
import subprocess
import torch
import torch.nn.functional as F
import soundfile as sf

_S3PRL_HUB = os.path.expanduser("~/.cache/torch/hub/s3prl_s3prl_main")


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())


def _torchaudio_compat():
    """s3prl's wavlm upstream imports torchaudio APIs removed in 2.x (set/get/list_audio_backend).
    Shim them as no-ops so the import passes (backend is auto-dispatched in torchaudio 2.x)."""
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda *a, **k: None
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda *a, **k: []


def _ensure_s3prl_repo():
    """ECAPA_TDNN_SMALL loads wavlm_large via torch.hub. The GitHub path blocks on an interactive
    trust prompt (EOFError in batch runs), so it prefers a local cache at
    ~/.cache/torch/hub/s3prl_s3prl_main. Clone it there and overwrite the default hubconf — which
    eagerly imports the whole upstream zoo (several upstreams break on torchaudio 2.x: byol_s's
    set_audio_backend, mos_prediction's sox_effects) — with a wavlm-only hubconf."""
    wavlm_hubconf = os.path.join(_S3PRL_HUB, "s3prl", "upstream", "wavlm", "hubconf.py")
    if not os.path.exists(wavlm_hubconf):
        shutil.rmtree(_S3PRL_HUB, ignore_errors=True)
        os.makedirs(os.path.dirname(_S3PRL_HUB), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/s3prl/s3prl", _S3PRL_HUB],
                       check=True, capture_output=True)
    with open(os.path.join(_S3PRL_HUB, "hubconf.py"), "w") as f:
        f.write("from s3prl.upstream.wavlm.hubconf import wavlm_large\n")


_MODEL = None


def _load(ckpt_path, device="cuda"):
    global _MODEL
    if _MODEL is None:
        _torchaudio_compat()
        _ensure_s3prl_repo()
        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL  # WavLM+ECAPA (wavlm_large via local s3prl cache)
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
