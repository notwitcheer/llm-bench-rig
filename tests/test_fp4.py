import pytest
from lib.fp4 import median_tps, speedup, parse_quant_kernel


def test_median_tps():
    assert median_tps([100.0, 102.0, 98.0]) == 100.0
    assert median_tps([50.0, 60.0]) == 55.0


def test_speedup():
    assert speedup(200.0, 100.0) == 2.0


def test_parse_marlin_for_nvfp4():
    # NVFP4 dense on sm_120 falls back to Marlin -> the whole point of the bench
    log = "INFO ... Using MarlinLinearKernel for ModelOptNvFp4Config ... gpu sm_120"
    assert parse_quant_kernel(log) == "marlin"


def test_parse_cutlass_fp4():
    log = "INFO ... cutlass_scaled_mm fp4 kernel selected ..."
    assert parse_quant_kernel(log) == "cutlass-fp4"


def test_parse_flashinfer():
    assert parse_quant_kernel("... FlashInfer ... MoE ...") == "flashinfer"


def test_parse_unknown():
    assert parse_quant_kernel("nothing relevant here") == "unknown"
