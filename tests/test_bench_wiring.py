import bench


def test_run_speed_forwards_offload(monkeypatch, tmp_path):
    captured = {}

    def fake_run_llama_bench(model_path, **kwargs):
        captured.update(kwargs)
        return {"tg128": {"tokens_per_sec": 1.0, "stddev": 0.0}}

    monkeypatch.setattr(bench, "run_llama_bench", fake_run_llama_bench)
    monkeypatch.setattr(bench, "get_vram_used_mib", lambda: 0)

    class P:  # minimal progress stub
        def update(self, *a, **k): pass

    bench._run_speed("/m/x.gguf", "llama.cpp", {"slug": "x"}, tmp_path, P(), {"n_cpu_moe": 28})
    assert captured == {"n_cpu_moe": 28}
