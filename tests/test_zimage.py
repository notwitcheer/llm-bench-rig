"""Unit tests for the Z-Image image-gen metric layer (pure, deterministic — ADR-0003)."""
import json

from lib.zimage.score import images_per_min, megapixels, aggregate
from lib.zimage.dataset import load_prompts


def test_images_per_min_basic():
    # a 2s image => 30 images/min
    assert images_per_min(2.0) == 30.0


def test_images_per_min_subsecond():
    assert images_per_min(0.5) == 120.0


def test_images_per_min_zero_is_safe():
    assert images_per_min(0.0) == 0.0
    assert images_per_min(-1.0) == 0.0


def test_megapixels():
    assert megapixels(1024, 1024) == round(1024 * 1024 / 1_000_000, 3)
    assert megapixels(2048, 2048) == round(2048 * 2048 / 1_000_000, 3)


def test_aggregate_groups_by_res_and_steps_and_maxes_vram():
    recs = [
        {"resolution": 1024, "steps": 8, "gen_seconds": 1.0, "wall_seconds": 1.1, "peak_vram_mib": 20000},
        {"resolution": 1024, "steps": 8, "gen_seconds": 3.0, "wall_seconds": 3.2, "peak_vram_mib": 21000},
    ]
    cell = aggregate(recs)[(1024, 8)]
    assert cell["n"] == 2
    assert cell["gen_seconds_mean"] == 2.0
    assert cell["peak_vram_mib_max"] == 21000
    # images/min off the mean compute (2.0s) => 30/min
    assert cell["images_per_min"] == 30.0


def test_aggregate_single_record_zero_std():
    recs = [{"resolution": 512, "steps": 8, "gen_seconds": 0.4, "peak_vram_mib": 18000}]
    cell = aggregate(recs)[(512, 8)]
    assert cell["n"] == 1
    assert cell["gen_seconds_std"] == 0.0
    # wall_seconds defaults to gen_seconds when absent
    assert cell["wall_seconds_mean"] == 0.4


def test_aggregate_separates_step_cells():
    recs = [
        {"resolution": 1024, "steps": 4, "gen_seconds": 0.5},
        {"resolution": 1024, "steps": 8, "gen_seconds": 1.0},
        {"resolution": 1024, "steps": 16, "gen_seconds": 2.0},
    ]
    agg = aggregate(recs)
    assert set(agg) == {(1024, 4), (1024, 8), (1024, 16)}
    assert agg[(1024, 16)]["gen_seconds_mean"] == 2.0


def test_load_prompts_fields_and_limit(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps([
        {"name": "a", "category": "photoreal", "prompt": "x"},
        {"name": "b", "category": "spatial", "prompt": "y"},
    ]))
    rows = load_prompts(str(p), limit=1)
    assert len(rows) == 1
    assert rows[0]["name"] == "a"


def test_load_prompts_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"name": "x", "prompt": "y"}]))  # missing category
    try:
        load_prompts(str(p))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "category" in str(e)


def test_real_prompt_set_is_valid():
    rows = load_prompts("dataset/zimage/prompts.json")
    assert len(rows) == 8
    assert {r["name"] for r in rows} == {
        "portrait", "two-object", "counting", "text",
        "spatial", "landscape", "complex", "illustration",
    }
