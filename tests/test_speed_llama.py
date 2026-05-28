import pytest
from pathlib import Path
from lib.speed_llama import parse_llama_bench_output

FIXTURE = Path(__file__).parent / "fixtures" / "llama-bench-output.txt"

def test_parse_llama_bench_output():
    raw = FIXTURE.read_text()
    results = parse_llama_bench_output(raw)
    assert results["pp128"]["tokens_per_sec"] == pytest.approx(10234.56, abs=0.01)
    assert results["pp512"]["tokens_per_sec"] == pytest.approx(9302.33, abs=0.01)
    assert results["pp2048"]["tokens_per_sec"] == pytest.approx(8150.21, abs=0.01)
    assert results["pp4096"]["tokens_per_sec"] == pytest.approx(7420.88, abs=0.01)
    assert results["pp8192"]["tokens_per_sec"] == pytest.approx(6135.44, abs=0.01)
    assert results["pp16384"]["tokens_per_sec"] == pytest.approx(4802.17, abs=0.01)
    assert results["tg128"]["tokens_per_sec"] == pytest.approx(273.05, abs=0.01)
    assert results["pp512"]["stddev"] == pytest.approx(52.36, abs=0.01)

def test_parse_extracts_model_info():
    raw = FIXTURE.read_text()
    results = parse_llama_bench_output(raw)
    assert results["model_size_gib"] == pytest.approx(20.60, abs=0.01)
    assert results["params"] == "34.66 B"
    assert results["backend"] == "CUDA"
