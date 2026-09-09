"""Tokens-per-correct as a standing gpqa metric: eval fields, sidecar, board_ci row.

Fake client and fake `datasets` module throughout; nothing here touches a server.
"""
import csv
import json
import sys
import types

import pytest

from lib.evals.gpqa import CAP_SLACK, GPQAEval, token_summary

LETTER_OF = "ABCD"


class _TokenClient:
    """Answers with a scripted letter per call and reports scripted token counts."""

    def __init__(self, answers, tokens):
        self.answers = list(answers)
        self.tokens = list(tokens)
        self.calls = []
        self.last_completion_tokens = None
        self.reasoning_fallback_count = 0

    def chat(self, messages, **kw):
        self.calls.append(kw)
        i = len(self.calls) - 1
        self.last_completion_tokens = self.tokens[i]
        return self.answers[i]


def _patch_datasets(monkeypatch, n=4):
    raw = {"Question": "q?", "Correct Answer": "right", "Incorrect Answer 1": "w1",
           "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}

    def _fake_load_dataset(*args, **kwargs):
        return [raw] * n

    fake = types.ModuleType("datasets")
    fake.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _expected_letters(n):
    from lib.evals.gpqa import _build_item
    raw = {"Question": "q?", "Correct Answer": "right", "Incorrect Answer 1": "w1",
           "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
    return [LETTER_OF[_build_item(raw, i)["answer"]] for i in range(n)]


# --- token_summary (pure) ---

def test_token_summary_over_recorded_entries_only():
    entries = [
        {"correct": True, "completion_tokens": 100, "capped": False},
        {"correct": True, "completion_tokens": 300, "capped": False},
        {"correct": False, "completion_tokens": 2048, "capped": True},
        {"correct": True, "predicted": "A", "expected": "A"},  # pre-token entry
        {"correct": False, "completion_tokens": None, "capped": False},  # usage missing
    ]
    s = token_summary(entries, max_tokens=2048)
    assert s == {
        "max_tokens": 2048,
        "tokens_recorded": 3,
        "completion_tokens_total": 2448,
        "completion_tokens_median": 300,
        "tokens_per_correct": 1224.0,   # 2448 / 2 correct among the recorded three
        "capped_count": 1,
        "capped_rate": round(1 / 3, 4),
    }


def test_token_summary_zero_correct_and_empty():
    s = token_summary([{"correct": False, "completion_tokens": 50}], max_tokens=64)
    assert s["tokens_per_correct"] is None
    assert s["capped_count"] == 0 and s["capped_rate"] == 0.0
    empty = token_summary([], max_tokens=2048)
    assert empty["tokens_recorded"] == 0
    assert empty["completion_tokens_total"] == 0
    assert empty["completion_tokens_median"] is None
    assert empty["tokens_per_correct"] is None
    assert empty["capped_rate"] is None


def test_token_summary_derives_capped_when_flag_missing():
    # an entry with tokens but no `capped` key (partial upgrade) uses the budget rule
    entries = [{"correct": True, "completion_tokens": 2048 - CAP_SLACK},
               {"correct": True, "completion_tokens": 2048 - CAP_SLACK - 1}]
    assert token_summary(entries, 2048)["capped_count"] == 1


# --- GPQAEval wiring ---

def test_default_max_tokens_is_2048_and_passed_to_chat(monkeypatch, tmp_path):
    _patch_datasets(monkeypatch, n=2)
    client = _TokenClient(answers=["A", "B"], tokens=[5, 6])
    ev = GPQAEval(client=client)
    result = ev.evaluate()
    assert all(c["max_tokens"] == 2048 for c in client.calls)
    assert result["max_tokens"] == 2048
    assert result["total"] == 2
    # no results_dir: nothing written anywhere
    assert not list(tmp_path.iterdir())


def test_max_tokens_flows_through_and_records_per_item(monkeypatch, tmp_path):
    n = 4
    _patch_datasets(monkeypatch, n=n)
    exp = _expected_letters(n)
    # item 0 right, item 1 wrong, item 2 right but capped, item 3 no usage reported
    answers = [exp[0], "Z" if exp[1] == "A" else "A", exp[2], exp[3]]
    tokens = [100, 200, 16384 - 3, None]
    client = _TokenClient(answers=answers, tokens=tokens)
    ev = GPQAEval(client=client, results_dir=tmp_path, max_tokens=16384)
    result = ev.evaluate()

    assert all(c["max_tokens"] == 16384 for c in client.calls)
    assert result["correct"] == 3
    assert result["max_tokens"] == 16384
    assert result["tokens_recorded"] == 3            # the None item is not counted
    assert result["completion_tokens_total"] == 100 + 200 + 16381
    assert result["completion_tokens_median"] == 200
    assert result["tokens_per_correct"] == round(16681 / 2, 1)  # 2 correct among recorded
    assert result["capped_count"] == 1
    assert result["capped_rate"] == round(1 / 3, 4)
    # existing keys untouched
    for k in ("score", "metric", "correct", "total", "parse_failures",
              "reasoning_fallback_count", "n_shot", "shuffle_seed",
              "completion_tokens_mean", "caveat"):
        assert k in result

    progress = json.loads((tmp_path / "gpqa_progress.json").read_text())["completed"]
    assert progress["0"] == {"correct": True, "predicted": exp[0], "expected": exp[0],
                             "completion_tokens": 100, "capped": False}
    assert progress["2"]["capped"] is True
    assert progress["3"]["completion_tokens"] is None and progress["3"]["capped"] is False

    side = [json.loads(l) for l in (tmp_path / "gpqa_tokens.jsonl").read_text().splitlines()]
    assert side == [
        {"idx": 0, "completion_tokens": 100, "correct": True, "capped": False},
        {"idx": 1, "completion_tokens": 200, "correct": False, "capped": False},
        {"idx": 2, "completion_tokens": 16381, "correct": True, "capped": True},
        {"idx": 3, "completion_tokens": None, "correct": True, "capped": False},
    ]


def test_resume_from_old_progress_without_token_fields(monkeypatch, tmp_path):
    """Progress written before this change has no completion_tokens; the resumed
    run must still load it, count its correctness, and report token coverage as
    partial rather than inventing numbers."""
    n = 4
    _patch_datasets(monkeypatch, n=n)
    exp = _expected_letters(n)
    old = {"completed": {
        "0": {"correct": True, "predicted": exp[0], "expected": exp[0]},
        "1": {"correct": False, "predicted": None, "expected": exp[1]},
    }, "timestamp": "2026-08-20T00:00:00"}
    (tmp_path / "gpqa_progress.json").write_text(json.dumps(old))
    # a hand-rolled sidecar from the old run must be appended to, not truncated
    (tmp_path / "gpqa_tokens.jsonl").write_text(
        json.dumps({"completion_tokens": 9000, "wall_s": 40.0}) + "\n")

    client = _TokenClient(answers=[exp[2], exp[3]], tokens=[400, 600])
    result = GPQAEval(client=client, results_dir=tmp_path).evaluate()

    assert len(client.calls) == 2                    # items 0 and 1 were skipped
    assert result["total"] == 4 and result["correct"] == 3
    assert result["parse_failures"] == 1
    assert result["tokens_recorded"] == 2
    assert result["completion_tokens_total"] == 1000
    assert result["completion_tokens_median"] == 500
    assert result["tokens_per_correct"] == 500.0
    assert result["capped_count"] == 0 and result["capped_rate"] == 0.0

    progress = json.loads((tmp_path / "gpqa_progress.json").read_text())["completed"]
    assert "completion_tokens" not in progress["0"]  # old entries left as they were
    assert progress["2"]["completion_tokens"] == 400

    side_lines = (tmp_path / "gpqa_tokens.jsonl").read_text().splitlines()
    assert len(side_lines) == 3
    assert json.loads(side_lines[0])["completion_tokens"] == 9000
    assert json.loads(side_lines[1]) == {"idx": 2, "completion_tokens": 400,
                                         "correct": True, "capped": False}


def test_zero_correct_gives_none_tokens_per_correct(monkeypatch):
    _patch_datasets(monkeypatch, n=2)
    exp = _expected_letters(2)
    wrong = ["B" if e == "A" else "A" for e in exp]
    result = GPQAEval(client=_TokenClient(wrong, [10, 20])).evaluate()
    assert result["correct"] == 0
    assert result["tokens_per_correct"] is None
    assert result["completion_tokens_total"] == 30


# --- board_ci: gpqa_thinkon row ---

def _write_slug(root, slug, counts=None, quality=None, gpqa=None, sidecar=None):
    from lib.ci import BOARD_TASKS
    d = root / slug
    d.mkdir()
    for t, (c, n) in (counts or {}).items():
        body = {"score": round(c / n * 100, 2), "total": n, "parse_failures": 0}
        body["passed" if t == "humaneval" else "correct"] = c
        (d / f"{t}_detail.json").write_text(json.dumps(body))
    if quality is not None:
        (d / "quality.json").write_text(json.dumps(quality))
    if gpqa is not None:
        (d / "gpqa.json").write_text(json.dumps(gpqa))
    if sidecar is not None:
        (d / "gpqa_tokens.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sidecar))
    return d


FULL_COUNTS = {"mmlu": (6163, 7010), "arc_challenge": (1136, 1172), "hellaswag": (4792, 5021),
               "gsm8k": (1283, 1319), "humaneval": (153, 164)}


def test_board_ci_emits_gpqa_thinkon_row_with_token_columns(tmp_path):
    from lib.ci import BOARD_TASKS
    from scripts.board_ci import COLUMNS, TOKEN_COLUMNS, collect_rows, write_csv

    assert COLUMNS[:8] == ["slug", "task", "score", "correct", "total", "parse_failures",
                           "ci_low", "ci_high"]
    assert COLUMNS[8:] == ["tokens_per_correct", "completion_tokens_median", "capped_rate"]
    assert tuple(COLUMNS[8:]) == TOKEN_COLUMNS

    quality = {t: {"score": round(c / n * 100, 2)} for t, (c, n) in FULL_COUNTS.items()}
    _write_slug(tmp_path, "m", FULL_COUNTS, quality=quality,
                gpqa={"score": 54.55, "correct": 108, "total": 198, "parse_failures": 0})
    # new-style think-on dir: no quality.json, gpqa.json carries the metric keys
    _write_slug(tmp_path, "m-thinkon", gpqa={
        "score": 79.8, "correct": 158, "total": 198, "parse_failures": 3,
        "max_tokens": 16384, "tokens_recorded": 198, "completion_tokens_total": 1_600_000,
        "completion_tokens_median": 7400, "tokens_per_correct": 10126.6,
        "capped_count": 14, "capped_rate": 0.0707,
    })

    rows, summaries = collect_rows(tmp_path)
    by = {(r["slug"], r["task"]): r for r in rows}
    assert ("m-thinkon", "gpqa_thinkon") in by
    assert ("m-thinkon", "gpqa") not in by
    assert [r["task"] for r in rows if r["slug"] == "m"] == list(BOARD_TASKS) + ["gpqa", "q_avg"]
    # only the summaries of complete five-task slugs; the think-on dir adds none
    assert [s["slug"] for s in summaries] == ["m"]

    t = by[("m-thinkon", "gpqa_thinkon")]
    assert t["correct"] == 158 and t["total"] == 198 and t["parse_failures"] == 3
    assert t["ci_low"] < 79.8 < t["ci_high"]
    assert t["tokens_per_correct"] == 10126.6
    assert t["completion_tokens_median"] == 7400
    assert t["capped_rate"] == 0.0707
    # every other row leaves the three new columns empty
    for r in rows:
        if r["task"] != "gpqa_thinkon":
            assert all(r[c] == "" for c in TOKEN_COLUMNS), r

    out = tmp_path / "board_ci.csv"
    write_csv(rows, out)
    with open(out, newline="") as f:
        read = list(csv.DictReader(f))
    assert list(read[0].keys()) == COLUMNS
    csv_t = next(r for r in read if r["task"] == "gpqa_thinkon")
    assert csv_t["tokens_per_correct"] == "10126.6" and csv_t["capped_rate"] == "0.0707"
    csv_m = next(r for r in read if r["task"] == "gpqa")
    assert csv_m["tokens_per_correct"] == "" and csv_m["capped_rate"] == ""


def test_board_ci_thinkon_detected_by_regime_and_legacy_sidecar(tmp_path):
    from scripts.board_ci import collect_rows, is_thinkon_dir, token_extras

    # regime field, no -thinkon suffix, and the legacy sidecar shape (no correctness,
    # no capped flag) with a max_tokens budget in gpqa.json
    legacy = [{"completion_tokens": t, "wall_s": 1.0, "reasoning_fallback": False,
               "answer_head": "B"} for t in (1000, 3000, 16384, 16380)]
    d = _write_slug(tmp_path, "r", gpqa={"score": 50.0, "correct": 2, "total": 4,
                                        "parse_failures": 0, "regime": "think-on",
                                        "max_tokens": 16384}, sidecar=legacy)
    assert is_thinkon_dir(d)
    ex = token_extras(d, json.loads((d / "gpqa.json").read_text()))
    assert ex["completion_tokens_median"] == 3000 + (16380 - 3000) / 2
    assert ex["tokens_per_correct"] == round((1000 + 3000 + 16384 + 16380) / 2, 1)
    assert ex["capped_rate"] == 0.5

    # `think: true` also counts; without sidecar or keys the columns stay empty
    d2 = _write_slug(tmp_path, "t", gpqa={"score": 25.0, "correct": 1, "total": 4,
                                         "parse_failures": 0, "think": True})
    assert is_thinkon_dir(d2)
    assert token_extras(d2, json.loads((d2 / "gpqa.json").read_text())) == {
        "tokens_per_correct": None, "completion_tokens_median": None, "capped_rate": None}

    # legacy sidecar with no budget anywhere: median + tokens_per_correct only
    d3 = _write_slug(tmp_path, "u-thinkon", gpqa={"score": 50.0, "correct": 1, "total": 2,
                                                  "parse_failures": 0},
                     sidecar=[{"completion_tokens": 10}, {"completion_tokens": 30}])
    ex3 = token_extras(d3, json.loads((d3 / "gpqa.json").read_text()))
    assert ex3 == {"tokens_per_correct": 40.0, "completion_tokens_median": 20,
                   "capped_rate": None}

    # a think-off dir without quality.json is still excluded by default
    _write_slug(tmp_path, "plain", gpqa={"score": 50.0, "correct": 1, "total": 2,
                                        "parse_failures": 0})
    rows, _ = collect_rows(tmp_path)
    assert {(r["slug"], r["task"]) for r in rows} == {
        ("r", "gpqa_thinkon"), ("t", "gpqa_thinkon"), ("u-thinkon", "gpqa_thinkon")}
    u = next(r for r in rows if r["slug"] == "u-thinkon")
    assert u["capped_rate"] == "" and u["tokens_per_correct"] == 40.0
    with pytest.raises(StopIteration):
        next(r for r in rows if r["slug"] == "plain")
