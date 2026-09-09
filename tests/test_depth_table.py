"""depth_table: merge llama-bench depth sweeps into speed.json and build the speed-at-depth table."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import depth_table as dt  # noqa: E402


def _raw(pp0, tg0, pp8, tg8, pp32, tg32, build="abc1234"):
    def row(n_prompt, n_gen, depth, ts):
        return {"build_commit": build, "n_prompt": n_prompt, "n_gen": n_gen, "n_depth": depth,
                "avg_ts": ts, "stddev_ts": 0.1}
    return [row(512, 0, 0, pp0), row(0, 128, 0, tg0), row(512, 0, 8192, pp8), row(0, 128, 8192, tg8),
            row(512, 0, 32768, pp32), row(0, 128, 32768, tg32)]


def _mk(root, slug, speed, raw=None, meta=None):
    d = root / slug
    d.mkdir()
    (d / "speed.json").write_text(json.dumps(speed))
    if raw is not None:
        (d / "bench.depth.json").write_text(json.dumps(raw))
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_merge_adds_depth_keys_and_keeps_d0_rows(tmp_path):
    d = _mk(tmp_path, "m-q6-k", {"pp512": {"tokens_per_sec": 3000.0, "stddev": 1.0},
                                 "tg128": {"tokens_per_sec": 60.0, "stddev": 0.1}, "engine": "llama.cpp"},
            raw=_raw(3100, 61, 2900, 58, 2400, 55))
    assert dt.merge(tmp_path) == ["m-q6-k"]
    sp = json.loads((d / "speed.json").read_text())
    assert sp["tg128"]["tokens_per_sec"] == 60.0          # original d=0 row untouched
    assert sp["tg128@d32768"]["tokens_per_sec"] == 55.0
    assert sp["pp512@d8192"]["tokens_per_sec"] == 2900.0
    assert sp["depth_source"]["build_commit"] == "abc1234"
    # idempotent
    assert dt.merge(tmp_path) == []


def test_merge_skips_non_dict_speed_and_existing_depth(tmp_path):
    _mk(tmp_path, "quant_tax", [{"quant": "x"}], raw=_raw(1, 1, 1, 1, 1, 1))
    _mk(tmp_path, "has-depth", {"tg128": {"tokens_per_sec": 1}, "tg128@d32768": {"tokens_per_sec": 1}},
        raw=_raw(1, 1, 1, 1, 1, 1))
    assert dt.merge(tmp_path) == []


def test_table_derives_ttft_and_retention(tmp_path):
    _mk(tmp_path, "a-q4", {"tg128": {"tokens_per_sec": 80.0}, "tg128@d32768": {"tokens_per_sec": 72.0},
                           "pp512@d8192": {"tokens_per_sec": 4096.0}}, meta={"name": "A", "quant": "Q4_K_M"})
    _mk(tmp_path, "b-off", {"tg128": {"tokens_per_sec": 60.0}, "tg128@d32768": {"tokens_per_sec": 51.0},
                            "pp512@d8192": {"tokens_per_sec": 819.2}, "n_cpu_moe": 22,
                            "ttft_16k_s": 21.3}, meta={"name": "B", "quant": "UD-Q2_K_XL"})
    _mk(tmp_path, "c-nodepth", {"tg128": {"tokens_per_sec": 100.0}})
    rows = dt.collect(tmp_path)
    assert [r["slug"] for r in rows] == ["a-q4", "b-off"]       # sorted by tg128@32k desc, no-depth row dropped
    a, b = rows
    assert a["decode_retention_32k"] == 90.0
    assert a["ttft_16k_est_s"] == 4.0                          # 16384 / 4096
    assert a["ttft_16k_measured_s"] is None
    assert b["ttft_16k_est_s"] == 20.0 and b["ttft_16k_measured_s"] == 21.3
    out = tmp_path / "speed_at_depth.csv"
    dt.write_csv(rows, out)
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["slug"] for r in got] == ["a-q4", "b-off"]
    assert got[0]["ttft_16k_measured_s"] == ""
    md = dt.markdown(rows)
    assert "| A | Q4_K_M | 80.0 | 72.0 | 90% | 4,096 | ~4.0 s |" in md
    assert "`--n-cpu-moe 22`" in md and "| 21.3 s |" in md   # measured TTFT preferred over the estimate
