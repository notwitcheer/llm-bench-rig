"""card_fit: fit classes from measured VRAM peaks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import card_fit as cf  # noqa: E402


def test_fit_classes():
    # 22.3 GiB peak, 21.3 GiB file: resident on 24 and 32, file too big for 16
    assert cf.fit(22835, 21.3, 32) == "yes"
    assert cf.fit(22835, 21.3, 24) == "yes"
    assert cf.fit(22835, 21.3, 16) == "no"
    # 23.9 GiB peak on a 24 GB card: file fits, 16k peak + headroom does not
    assert cf.fit(24470, 21.3, 24) == "tight"
    # unknown peak
    assert cf.fit(None, 10.0, 16) == "?"
    assert cf.fit(None, 20.0, 16) == "no"
    # offloaded rows: the measured point either fits or the card needs more offload
    assert cf.fit(28484, 73.5, 32, offloaded=True) == "offload"
    assert cf.fit(28484, 73.5, 24, offloaded=True) == "no"


def _mk(root, slug, quality, speed, meta):
    d = root / slug
    d.mkdir()
    (d / "quality.json").write_text(json.dumps(quality))
    (d / "speed.json").write_text(json.dumps(speed))
    (d / "meta.json").write_text(json.dumps(meta))
    return d


Q = {"mmlu": {"score": 90}, "arc_challenge": {"score": 90}, "hellaswag": {"score": 90},
     "gsm8k": {"score": 90}, "humaneval": {"score": 90}}


def test_collect_detects_offload_and_skips_vllm(tmp_path):
    _mk(tmp_path, "dense-q6-k", Q, {"engine": "llama.cpp", "vram_peak_mib": 22835, "model_size_gib": 21.3},
        {"name": "Dense", "quant": "Q6_K", "think": False})
    _mk(tmp_path, "moe-offload", Q, {"engine": "llama.cpp", "vram_peak_mib": 3762, "model_size_gib": 58.3},
        {"name": "MoE", "quant": "Q3_K_M", "think": False})
    _mk(tmp_path, "nvfp4", Q, {"engine": "vllm", "vram_peak_mib": 30000, "model_size_gib": 21.0},
        {"name": "NV", "quant": "NVFP4", "think": False})
    (tmp_path / "dense-q6-k" / "gpqa.json").write_text(json.dumps({"score": 49.0}))
    rows = cf.collect(tmp_path)
    assert sorted(r["slug"] for r in rows) == ["dense-q6-k", "moe-offload"]  # vllm row skipped
    dense = next(r for r in rows if r["slug"] == "dense-q6-k")
    moe = next(r for r in rows if r["slug"] == "moe-offload")
    assert dense["fits_24gb"] == "yes" and dense["fits_16gb"] == "no" and dense["gpqa"] == 49.0
    assert moe["offload_n_cpu_moe"] == "all"
    assert all(moe[f"fits_{c}gb"] == "offload" for c in cf.CARDS_GB)
    md = cf.markdown(rows)
    assert "| Dense | Q6_K | 21.3 | 22.3 | 90.0 | 49.0 | ⬜ | ⬜ | ⬜ | ✅ | ✅ |" in md
    assert "`--n-cpu-moe all`" in md and "🟠" in md
