"""Provenance block: pure helpers and the never-crash wiring, no network, no gpu."""
import hashlib
import json

import pytest

import lib.provenance as prov
from lib.provenance import (attach_to_meta, chat_template_sha256, gguf_sha256,
                            parse_git_state, record_provenance, resolve_quality_config,
                            summarise_props)


def test_chat_template_sha256_matches_sha256_of_utf8():
    tpl = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    assert chat_template_sha256(tpl) == hashlib.sha256(tpl.encode()).hexdigest()
    assert chat_template_sha256("") is None
    assert chat_template_sha256(None) is None


def test_summarise_props_keeps_only_score_shaping_fields():
    props = {
        "build_info": "b10371-abc1234",
        "chat_template": "{{ x }}",
        "default_generation_settings": {"n_ctx": 8192, "temperature": 0.8},
        "total_slots": 4,
        "model_path": "/m/x.gguf",
        "something_else": [1, 2, 3],
    }
    out = summarise_props(props)
    assert out["build_info"] == "b10371-abc1234"
    assert out["version"] is None
    assert out["n_ctx"] == 8192
    assert out["total_slots"] == 4
    assert out["chat_template_sha256"] == hashlib.sha256(b"{{ x }}").hexdigest()
    assert "something_else" not in out
    assert summarise_props(None)["error"]


def test_parse_git_state():
    sha = "8163907" + "0" * 33
    assert parse_git_state(sha + "\n", "") == {"sha": sha, "dirty": False}
    assert parse_git_state(sha + "\n", " M lib/quality.py\n") == {"sha": sha, "dirty": True}
    assert parse_git_state("fatal: not a git repository", "") == {"sha": "unknown", "dirty": None}
    assert parse_git_state("", "") == {"sha": "unknown", "dirty": None}


def test_resolve_quality_config_reads_five_keys_with_run_defaults():
    cfg = {"quality.sample": 0.5, "quality.think": False, "quality.mc_gate_tokens": None}

    def get(key, default=None):
        return cfg[key] if key in cfg else default

    out = resolve_quality_config(get)
    assert out == {"sample": 0.5, "think": False, "mmlu_limit": None,
                   "mc_gate_tokens": None, "limit": None}
    # defaults mirror run_quality_bench when the key is absent
    assert resolve_quality_config(lambda k, d=None: d)["mc_gate_tokens"] == 50
    assert resolve_quality_config(lambda k, d=None: d)["think"] is True


def test_gguf_sha256_computes_and_caches_sidecar(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"gguf" * 1000)
    expected = hashlib.sha256(b"gguf" * 1000).hexdigest()
    first = gguf_sha256(f)
    assert first["sha256"] == expected and first["cached"] is False
    sidecar = tmp_path / "m.gguf.sha256"
    assert sidecar.read_text().split()[0] == expected
    # second call reads the sidecar instead of hashing
    f.write_bytes(b"changed")
    second = gguf_sha256(f)
    assert second["sha256"] == expected and second["cached"] is True


def test_gguf_sha256_skips_oversized_with_reason(tmp_path):
    f = tmp_path / "big.gguf"
    f.write_bytes(b"x" * 100)
    out = gguf_sha256(f, max_bytes=10)
    assert out["sha256"] is None and out["skipped"] is True
    assert "larger than" in out["reason"]
    assert not (tmp_path / "big.gguf.sha256").exists()


def test_gguf_sha256_readonly_dir_hashes_but_does_not_cache(tmp_path, monkeypatch):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"abc")
    monkeypatch.setattr(prov.os, "access", lambda p, mode: False)
    out = gguf_sha256(f)
    assert out["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert not (tmp_path / "m.gguf.sha256").exists()


def test_gguf_sha256_missing_file():
    out = gguf_sha256("/nonexistent/x.gguf")
    assert out["skipped"] is True and out["sha256"] is None


def test_attach_to_meta_keeps_existing_keys(tmp_path):
    mp = tmp_path / "meta.json"
    mp.write_text(json.dumps({"slug": "x", "think": False, "size_gib": 1.5}))
    attach_to_meta(mp, {"source": "run"})
    meta = json.loads(mp.read_text())
    assert meta["slug"] == "x" and meta["think"] is False and meta["size_gib"] == 1.5
    assert meta["provenance"] == {"source": "run"}


def test_record_provenance_never_raises_and_writes_error(tmp_path, monkeypatch):
    mp = tmp_path / "meta.json"
    mp.write_text(json.dumps({"slug": "x"}))

    def boom(*a, **k):
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(prov, "collect_provenance", boom)
    out = record_provenance(tmp_path, "http://127.0.0.1:8090/v1", "/m/x.gguf", ["srv"], {})
    assert out["error"].startswith("RuntimeError")
    meta = json.loads(mp.read_text())
    assert meta["slug"] == "x"
    assert meta["provenance"]["error"].startswith("RuntimeError")


def test_collect_provenance_assembles_block_with_mocked_io(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"weights")
    monkeypatch.setattr(prov, "fetch_props", lambda api_base, timeout=10: {
        "build_info": "b1-deadbeef", "chat_template": "t",
        "default_generation_settings": {"n_ctx": 4096}, "total_slots": 1})
    monkeypatch.setattr(prov, "harness_git_state",
                        lambda repo_root=None: {"sha": "a" * 40, "dirty": False})
    monkeypatch.setattr(prov, "package_versions",
                        lambda names=("datasets", "httpx"): {"datasets": "4.8.5", "httpx": "0.28.1"})
    cmd = ["llama-server", "-m", str(model), "--jinja"]
    cfg = {"sample": 0.5, "think": False, "mmlu_limit": None, "mc_gate_tokens": 50, "limit": None}
    block = prov.collect_provenance("http://127.0.0.1:8090/v1", str(model), cmd, cfg)
    assert block["source"] == "run"
    assert block["server_props"]["build_info"] == "b1-deadbeef"
    assert block["server_props"]["chat_template_sha256"] == hashlib.sha256(b"t").hexdigest()
    assert block["server_command"] == cmd
    assert block["quality_config"] == cfg
    assert block["gguf"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert block["harness_git"]["sha"] == "a" * 40
    assert block["packages"]["datasets"] == "4.8.5"
    assert block["timestamp_utc"].endswith("+00:00")


def test_collect_provenance_records_props_error_without_raising(tmp_path, monkeypatch):
    def fail(api_base, timeout=10):
        raise ConnectionError("down")

    monkeypatch.setattr(prov, "fetch_props", fail)
    monkeypatch.setattr(prov, "harness_git_state", lambda repo_root=None: {"sha": "unknown", "dirty": None})
    block = prov.collect_provenance("http://x/v1", str(tmp_path / "missing.gguf"), None, None)
    assert block["server_props"]["error"].startswith("ConnectionError")
    assert block["gguf"]["skipped"] is True


def test_run_quality_bench_records_provenance_before_evals(tmp_path, monkeypatch):
    """The wiring: run_quality_bench writes meta.json['provenance'] using the live
    server command, api_base and resolved config, and does so even if evals fail."""
    import lib.quality as q

    (tmp_path / "meta.json").write_text(json.dumps({"slug": "x", "think": False}))
    cfg = {"quality.tasks": ["mmlu"], "quality.think": False, "quality.sample": 0.5,
           "llama_cpp.server_port": 8090}
    monkeypatch.setattr(q, "get", lambda k, d=None: cfg.get(k, d))

    class FakeProc:
        args = ["llama-server", "-m", "/m/x.gguf", "--jinja"]

    monkeypatch.setattr(q, "start_llama_server", lambda *a, **k: FakeProc())
    monkeypatch.setattr(q, "stop_llama_server", lambda p: None)
    seen = {}

    def fake_record(results_dir, api_base, model_path, server_command, quality_config):
        seen.update(results_dir=results_dir, api_base=api_base, model_path=model_path,
                    server_command=server_command, quality_config=quality_config)
        return {}

    monkeypatch.setattr(q, "record_provenance", fake_record)

    def fail_evals(*a, **k):
        raise RuntimeError("no gpu here")

    monkeypatch.setattr(q, "_run_evals", fail_evals)
    with pytest.raises(RuntimeError):
        q.run_quality_bench("/m/x.gguf", "llama.cpp", results_dir=tmp_path)
    assert seen["results_dir"] == tmp_path
    assert seen["api_base"] == "http://127.0.0.1:8090/v1"
    assert seen["server_command"] == FakeProc.args
    assert seen["quality_config"]["sample"] == 0.5 and seen["quality_config"]["think"] is False


def test_backfill_block_marks_source_and_unknown_sha(tmp_path):
    from scripts.backfill_provenance import backfill_block, backfill_dir

    model = tmp_path / "m.gguf"
    model.write_bytes(b"old weights")
    block = backfill_block({"path": str(model)})
    assert block["source"] == "backfill"
    assert block["harness_git"]["sha"] == "unknown"
    assert block["gguf"]["sha256"] == hashlib.sha256(b"old weights").hexdigest()
    assert backfill_block({})["gguf"]["skipped"] is True

    slug = tmp_path / "slug"
    slug.mkdir()
    (slug / "meta.json").write_text(json.dumps({"slug": "slug", "think": True, "path": str(model)}))
    assert backfill_dir(slug) == "backfilled"
    meta = json.loads((slug / "meta.json").read_text())
    assert meta["think"] is True and meta["provenance"]["source"] == "backfill"
    # a run-time block is not overwritten unless forced
    meta["provenance"] = {"source": "run"}
    (slug / "meta.json").write_text(json.dumps(meta))
    assert backfill_dir(slug) == "kept run provenance"
    assert backfill_dir(slug, force=True) == "backfilled"
