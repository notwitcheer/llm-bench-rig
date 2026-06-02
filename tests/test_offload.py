from lib.offload import resolve_offload

CFG = {"offload": {"gpt-oss-120b-mxfp4": {"n_cpu_moe": 28, "n_gpu_layers": 99}}}

def test_resolve_known_slug():
    off = resolve_offload("gpt-oss-120b-mxfp4", CFG)
    assert off == {"n_cpu_moe": 28, "n_gpu_layers": 99}

def test_resolve_unknown_slug_returns_empty():
    assert resolve_offload("mellum2-12b", CFG) == {}

def test_resolve_no_offload_section():
    assert resolve_offload("anything", {}) == {}
