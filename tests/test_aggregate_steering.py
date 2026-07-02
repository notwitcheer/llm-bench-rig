from scripts.aggregate_steering import build_table


def test_build_table_orders_and_deltas():
    rows = [
        {"label": "base", "benchmark": "math500", "acc": 53.4, "rep_rate": .01, "mean_len": 600},
        {"label": "steer-b20", "benchmark": "math500", "acc": 61.0, "rep_rate": .01, "mean_len": 650},
    ]
    t = build_table(rows)
    assert t["math500"]["steer-b20"]["delta_vs_base"] == 7.6
    assert t["math500"]["base"]["delta_vs_base"] == 0.0


def test_build_table_multi_benchmark():
    rows = [
        {"label": "base", "benchmark": "math500", "acc": 50.0, "rep_rate": 0, "mean_len": 1},
        {"label": "base", "benchmark": "amc23", "acc": 40.0, "rep_rate": 0, "mean_len": 1},
        {"label": "lora-b20", "benchmark": "amc23", "acc": 45.0, "rep_rate": 0, "mean_len": 1},
    ]
    t = build_table(rows)
    assert t["amc23"]["lora-b20"]["delta_vs_base"] == 5.0
    assert "lora-b20" not in t["math500"]
