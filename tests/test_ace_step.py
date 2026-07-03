"""Unit tests for the ACE-Step music-gen metric layer (pure, deterministic — ADR-0003)."""
import json

from lib.ace_step.score import rtf, aggregate
from lib.ace_step.dataset import load_prompts


def test_rtf_faster_than_realtime():
    # a 240s song generated in 4s => 60x faster than it plays
    assert rtf(240.0, 4.0) == 60.0


def test_rtf_realtime_is_one():
    assert rtf(30.0, 30.0) == 1.0


def test_rtf_zero_gen_is_safe():
    assert rtf(30.0, 0.0) == 0.0


def test_aggregate_groups_and_maxes_vram():
    recs = [
        {"model_tier": "2b", "duration_s": 30, "gen_seconds": 2.0, "song_seconds": 30.0, "peak_vram_mib": 3000},
        {"model_tier": "2b", "duration_s": 30, "gen_seconds": 4.0, "song_seconds": 30.0, "peak_vram_mib": 3500},
    ]
    agg = aggregate(recs)
    cell = agg[("2b", 30)]
    assert cell["n"] == 2
    assert cell["gen_seconds_mean"] == 3.0
    assert cell["peak_vram_mib_max"] == 3500
    # rtf(30,2)=15.0, rtf(30,4)=7.5 -> mean 11.25
    assert round(cell["rtf_mean"], 3) == 11.25


def test_aggregate_single_record_zero_std():
    recs = [{"model_tier": "xl", "duration_s": 240, "gen_seconds": 10.0, "song_seconds": 240.0, "peak_vram_mib": 9000}]
    cell = aggregate(recs)[("xl", 240)]
    assert cell["n"] == 1
    assert cell["gen_seconds_std"] == 0.0
    assert cell["rtf_mean"] == 24.0


def test_load_prompts_fields_and_limit(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps([
        {"name": "edm", "caption": "upbeat EDM heavy bass", "lyrics": "", "duration_s": 30},
        {"name": "lofi", "caption": "chill lofi", "lyrics": "", "duration_s": 30},
    ]))
    rows = load_prompts(str(p), limit=1)
    assert len(rows) == 1
    assert rows[0]["name"] == "edm"
    assert rows[0]["caption"].startswith("upbeat")


def test_load_prompts_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"name": "x", "caption": "y"}]))  # missing lyrics, duration_s
    try:
        load_prompts(str(p))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "duration_s" in str(e)


def test_real_prompt_set_is_valid():
    rows = load_prompts("dataset/ace_step/prompts.json")
    assert len(rows) == 6
    assert {r["name"] for r in rows} == {"pop", "lofi", "orchestral", "edm", "acoustic", "hiphop"}
