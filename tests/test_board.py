import json

from lib.board import quality_average, think_mode, build_quality_board, THINK_ON, THINK_OFF


def test_quality_average_dict_scores():
    q = {
        "mmlu": {"score": 80.0, "metric": "acc"},
        "arc_challenge": {"score": 90.0},
        "hellaswag": {"score": 70.0},
        "humaneval": {"score": 100.0, "metric": "pass@1"},
        "gsm8k": {"score": 60.0},
    }
    assert quality_average(q) == 80.0


def test_quality_average_bare_numbers_and_missing():
    assert quality_average({"mmlu": 50.0, "humaneval": 70.0}) == 60.0
    assert quality_average({}) is None
    assert quality_average({"mmlu": None}) is None


def test_think_mode_prefers_meta_then_sets():
    assert think_mode({"think": True}, "anything") is True
    assert think_mode({"think": False}, "gpt-oss-120b-mxfp4") is False  # meta wins over set
    assert think_mode({}, "gpt-oss-120b-mxfp4") is True   # set fallback (ON)
    assert think_mode({}, "gemma-4-31b-it-q6-k") is False  # set fallback (OFF)
    assert think_mode({}, "brand-new-model") is None       # unknown surfaced


def test_think_sets_disjoint():
    assert THINK_ON.isdisjoint(THINK_OFF)


def _write(d, slug, think, humaneval, mmlu=80.0):
    p = d / slug
    p.mkdir()
    meta = {"slug": slug, "name": slug}
    if think is not None:
        meta["think"] = think
    (p / "meta.json").write_text(json.dumps(meta))
    (p / "quality.json").write_text(json.dumps({
        "mmlu": {"score": mmlu}, "humaneval": {"score": humaneval},
    }))


def test_build_quality_board_groups_and_sorts(tmp_path):
    _write(tmp_path, "off-hi", False, 95.0)   # q_avg 87.5
    _write(tmp_path, "off-lo", False, 50.0)   # q_avg 65.0
    _write(tmp_path, "on-one", True, 90.0)    # q_avg 85.0
    _write(tmp_path, "mystery", None, 40.0)   # unknown slug -> unknown group
    board = build_quality_board(tmp_path)

    assert [e["slug"] for e in board["off"]] == ["off-hi", "off-lo"]  # sorted desc
    assert [e["slug"] for e in board["on"]] == ["on-one"]
    assert [e["slug"] for e in board["unknown"]] == ["mystery"]
    assert board["off"][0]["q_avg"] == 87.5
