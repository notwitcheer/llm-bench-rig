import os
import pytest
from lib.cacheback import load_workload


@pytest.mark.parametrize("wl", ["code", "chat", "copyctx"])
def test_workload_shape(wl):
    path = f"dataset/cacheback/{wl}.jsonl"
    if not os.path.exists(path):
        pytest.skip("workloads not built yet (built on capsule in T5)")
    rows = load_workload(path)
    assert len(rows) == 30
    assert all(r["workload"] == wl and r["prompt"].strip() for r in rows)
