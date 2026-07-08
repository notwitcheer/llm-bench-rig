"""Tests for lib.ltx — prompt-set validation, video-config invariants, summariser.

The config invariants (H,W divisible by 64 for the two-stage pipeline; frame
count of the form 8k+1) live in tested code per ADR-0003: a bad matrix entry
must fail at load time on the Mac, not 30 GPU-minutes into a capsule run.
"""
import json

import pytest

from lib.ltx.config import validate_video_config
from lib.ltx.dataset import load_prompts
from lib.ltx.score import summarize


def _write_prompts(tmp_path, rows):
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_load_prompts_valid(tmp_path):
    rows = [{"name": "a", "prompt": "x", "category": "motion"}]
    assert load_prompts(_write_prompts(tmp_path, rows)) == rows


def test_load_prompts_missing_key_raises(tmp_path):
    rows = [{"name": "a", "prompt": "x"}]
    with pytest.raises(ValueError, match="category"):
        load_prompts(_write_prompts(tmp_path, rows))


def test_load_prompts_limit(tmp_path):
    rows = [{"name": str(i), "prompt": "x", "category": "c"} for i in range(4)]
    assert len(load_prompts(_write_prompts(tmp_path, rows), limit=2)) == 2


@pytest.mark.parametrize("w,h,f", [(768, 512, 97), (1280, 704, 97), (512, 320, 33)])
def test_validate_video_config_accepts(w, h, f):
    validate_video_config(width=w, height=h, num_frames=f)


@pytest.mark.parametrize(
    "w,h,f,msg",
    [
        (770, 512, 97, "divisible by 64"),
        (768, 500, 97, "divisible by 64"),
        (768, 512, 96, "8k\\+1"),
        (768, 512, 98, "8k\\+1"),
        (768, 512, 0, "8k\\+1"),
    ],
)
def test_validate_video_config_rejects(w, h, f, msg):
    with pytest.raises(ValueError, match=msg):
        validate_video_config(width=w, height=h, num_frames=f)


def _rec(name, w, h, gen_s, peak, ok=True):
    return {
        "name": name, "category": "c", "width": w, "height": h,
        "num_frames": 97, "fps": 24.0, "seed": 42,
        "gen_seconds": gen_s, "peak_vram_mib": peak, "nvidia_global_mib": peak,
        "ok": ok,
    }


def test_summarize_groups_by_config_and_computes_rtf():
    records = [
        _rec("a", 768, 512, 40.0, 25000),
        _rec("b", 768, 512, 44.0, 26000),
        _rec("a", 1280, 704, 120.0, 31000),
    ]
    s = summarize(records)
    c1 = s["configs"]["768x512"]
    # 97 frames @ 24fps = 4.0417s of video
    assert c1["n_ok"] == 2
    assert c1["mean_gen_seconds"] == pytest.approx(42.0)
    assert c1["video_seconds"] == pytest.approx(97 / 24.0, abs=1e-4)
    assert c1["seconds_per_video_second"] == pytest.approx(42.0 / (97 / 24.0), rel=1e-3)
    assert c1["max_peak_vram_mib"] == 26000
    assert s["configs"]["1280x704"]["n_ok"] == 1


def test_summarize_counts_failures_separately():
    records = [_rec("a", 1280, 704, None, None, ok=False)]
    s = summarize(records)
    assert s["configs"]["1280x704"]["n_ok"] == 0
    assert s["configs"]["1280x704"]["n_failed"] == 1
    assert s["configs"]["1280x704"]["mean_gen_seconds"] is None
