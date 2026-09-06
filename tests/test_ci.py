"""Wilson intervals and q_avg error propagation, on synthetic result dirs only."""
import csv
import json
import math

import pytest

from lib.ci import (BOARD_TASKS, dominant_variance_task, qavg_halfwidth, slug_intervals,
                    task_row_from_detail, wilson_interval)


def test_wilson_reference_values():
    # textbook check: 153/164 (a real humaneval row) -> roughly 0.88 to 0.96
    lo, hi = wilson_interval(153, 164)
    assert math.isclose(lo, 0.88390, abs_tol=1e-4)
    assert math.isclose(hi, 0.96214, abs_tol=1e-4)
    # symmetric case, large n: half-width close to 1.96 * sqrt(0.25/10000) = 0.0098
    lo, hi = wilson_interval(5000, 10000)
    assert math.isclose((hi - lo) / 2, 0.0098, abs_tol=0.0002)


def test_wilson_edges_stay_in_unit_interval():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = wilson_interval(50, 50)
    assert 0.9 < lo < 1.0 and hi == 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_task_row_from_detail_handles_correct_and_passed():
    row = task_row_from_detail({"score": 97.27, "correct": 1283, "total": 1319, "parse_failures": 2})
    assert row["correct"] == 1283 and row["parse_failures"] == 2
    assert row["ci_low"] < 97.27 < row["ci_high"]
    he = task_row_from_detail({"score": 93.29, "passed": 153, "total": 164})
    assert he["correct"] == 153 and he["parse_failures"] is None
    assert task_row_from_detail({"score": 50.0}) is None
    assert task_row_from_detail(None) is None


def test_task_row_sums_mmlu_per_subject_parse_failures():
    detail = {"score": 80.0, "correct": 80, "total": 100,
              "per_subject": {"a": {"parse_failures": 2}, "b": {"parse_failures": 3}}}
    assert task_row_from_detail(detail)["parse_failures"] == 5


def test_qavg_halfwidth_propagates_independent_variances():
    rows = {t: {"correct": 90, "total": 100} for t in BOARD_TASKS}
    # each task var = 0.9*0.1/100 * 1e4 = 9 pts^2; mean of five -> 9/5; sd = 1.3416; *1.96
    assert math.isclose(qavg_halfwidth(rows), 1.96 * math.sqrt(9 / 5), rel_tol=1e-3)
    # a missing board task means no q_avg interval
    partial = dict(rows)
    del partial["humaneval"]
    assert qavg_halfwidth(partial) is None
    # gpqa in the dict does not change the result
    with_gpqa = dict(rows, gpqa={"correct": 1, "total": 198})
    assert qavg_halfwidth(with_gpqa) == qavg_halfwidth(rows)


def test_dominant_variance_task_is_the_small_n_one():
    rows = {t: {"correct": 900, "total": 1000} for t in BOARD_TASKS}
    rows["humaneval"] = {"correct": 150, "total": 164}
    t, share = dominant_variance_task(rows)
    assert t == "humaneval" and share > 0.5


def _write_slug(tmp_path, slug, counts, quality=None, gpqa=None):
    d = tmp_path / slug
    d.mkdir()
    for t, (c, n) in counts.items():
        body = {"score": round(c / n * 100, 2), "total": n, "parse_failures": 0}
        body["passed" if t == "humaneval" else "correct"] = c
        (d / f"{t}_detail.json").write_text(json.dumps(body))
    if quality is not None:
        (d / "quality.json").write_text(json.dumps(quality))
    if gpqa is not None:
        (d / "gpqa.json").write_text(json.dumps(gpqa))
    return d


def test_slug_intervals_reads_details_gpqa_and_quality(tmp_path):
    counts = {"mmlu": (6163, 7010), "arc_challenge": (1136, 1172), "hellaswag": (4792, 5021),
              "gsm8k": (1283, 1319), "humaneval": (153, 164)}
    quality = {t: {"score": round(c / n * 100, 2)} for t, (c, n) in counts.items()}
    d = _write_slug(tmp_path, "m", counts, quality=quality,
                    gpqa={"score": 54.55, "correct": 108, "total": 198, "parse_failures": 0})
    res = slug_intervals(d)
    assert set(res["tasks"]) == set(BOARD_TASKS) | {"gpqa"}
    assert res["tasks"]["gpqa"]["correct"] == 108
    assert "published_score" not in res["tasks"]["mmlu"]
    q = res["q_avg"]
    assert q["score"] == pytest.approx(94.17, abs=0.01)
    # about 0.8 pts for this row, humaneval dominated
    assert 0.7 < q["halfwidth"] < 0.9
    assert q["ci_low"] == pytest.approx(q["score"] - q["halfwidth"], abs=0.02)


def test_slug_intervals_flags_published_score_mismatch(tmp_path):
    counts = {"mmlu": (80, 100)}
    d = _write_slug(tmp_path, "m", counts, quality={"mmlu": {"score": 70.0}})
    res = slug_intervals(d)
    assert res["tasks"]["mmlu"]["published_score"] == 70.0
    assert res["q_avg"] is None


def test_board_ci_script_writes_csv_with_qavg_rows(tmp_path):
    from scripts.board_ci import COLUMNS, collect_rows, summarise, write_csv

    counts = {"mmlu": (6163, 7010), "arc_challenge": (1136, 1172), "hellaswag": (4792, 5021),
              "gsm8k": (1283, 1319), "humaneval": (153, 164)}
    quality = {t: {"score": round(c / n * 100, 2)} for t, (c, n) in counts.items()}
    _write_slug(tmp_path, "full", counts, quality=quality)
    _write_slug(tmp_path, "no-quality", counts)  # excluded by default
    rows, summaries = collect_rows(tmp_path)
    assert {r["slug"] for r in rows} == {"full"}
    assert [r["task"] for r in rows] == list(BOARD_TASKS) + ["q_avg"]
    out = tmp_path / "board_ci.csv"
    write_csv(rows, out)
    with open(out, newline="") as f:
        read = list(csv.DictReader(f))
    assert list(read[0].keys()) == COLUMNS
    assert read[-1]["task"] == "q_avg"
    assert "humaneval" in summarise(summaries)
    rows_all, _ = collect_rows(tmp_path, include_all=True)
    assert {r["slug"] for r in rows_all} == {"full", "no-quality"}
