"""TTS head-to-head — metric layer (stdlib units on Mac; sim test is torch-gated -> capsule)."""
from lib.tts.dataset import parse_manifest_line, load_manifest


def test_parse_manifest_line():
    line = "utt1|hello there|prompts/utt1.wav|the quick brown fox|gt/utt1.wav"
    u = parse_manifest_line(line, base_dir="/data")
    assert u["name"] == "utt1"
    assert u["ref_text"] == "hello there"
    assert u["ref_audio"] == "/data/prompts/utt1.wav"
    assert u["target_text"] == "the quick brown fox"


def test_parse_manifest_line_no_gt():
    # Seed-TTS-eval EN lines have only 4 fields (no ground-truth audio)
    line = "u|prompt text|prompt-wavs/u.wav|synthesize this"
    u = parse_manifest_line(line, base_dir="/d")
    assert u["target_text"] == "synthesize this" and u["ref_audio"] == "/d/prompt-wavs/u.wav"


def test_load_manifest_limit(tmp_path):
    m = tmp_path / "meta.lst"
    m.write_text("\n".join(f"u{i}|r{i}|p{i}.wav|t{i}|g{i}.wav" for i in range(5)) + "\n")
    rows = load_manifest(str(m), base_dir=str(tmp_path), limit=3)
    assert len(rows) == 3 and rows[0]["name"] == "u0"


from lib.tts.roundtrip import parse_whisper_json
from lib.asr.wer import wer


def test_parse_whisper_json_joins_segments():
    jd = {"transcription": [{"text": " the quick"}, {"text": " brown fox "}]}
    assert parse_whisper_json(jd) == "the quick brown fox"


def test_roundtrip_wer_via_lib_asr():
    assert wer("the cat sat", parse_whisper_json({"transcription": [{"text": "the cat sat"}]})) == 0.0
    assert wer("the quick brown fox", "the quick red fox") == 0.25


from lib.tts.score import aggregate


def test_aggregate_means_and_handles_missing_sim():
    per_utt = [
        {"wer": 0.0, "rtfx": 2.0, "first_audio_s": 0.1, "sim": 0.8},
        {"wer": 0.2, "rtfx": 4.0, "first_audio_s": 0.3, "sim": None},
    ]
    a = aggregate(per_utt)
    assert abs(a["wer_mean"] - 0.1) < 1e-9
    assert abs(a["rtfx_mean"] - 3.0) < 1e-9
    assert abs(a["sim_mean"] - 0.8) < 1e-9   # None ignored
    assert a["n"] == 2


def test_sim_cosine_identity_and_orthogonal():
    # torch-gated -> skips on the Mac (no torch), runs on capsule
    import pytest
    torch = pytest.importorskip("torch")
    from lib.tts.sim import cosine
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    assert abs(cosine(a, a) - 1.0) < 1e-6
    assert abs(cosine(a, b) - 0.0) < 1e-6
