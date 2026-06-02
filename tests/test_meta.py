import pytest
from pathlib import Path
from lib.meta import detect_engine, parse_model_name, extract_metadata

def test_detect_engine_gguf():
    assert detect_engine(Path("/models/Qwen3-8B-Q4_K_M.gguf")) == "llama.cpp"

def test_detect_engine_safetensors():
    assert detect_engine(Path("/models/Qwen3-8B/")) == "vllm"

def test_parse_model_name_gguf():
    info = parse_model_name("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    assert info["base_name"] == "Qwen3.6-35B-A3B"
    assert info["quant"] == "UD-Q4_K_M"

def test_parse_model_name_simple():
    info = parse_model_name("gpt-oss-20b-Q4_K_M.gguf")
    assert info["base_name"] == "gpt-oss-20b"
    assert info["quant"] == "Q4_K_M"

def test_parse_sharded_mxfp4():
    info = parse_model_name("gpt-oss-120b-mxfp4-00001-of-00003.gguf")
    assert info["base_name"] == "gpt-oss-120b"
    assert info["quant"] == "mxfp4"

def test_parse_dot_separated_quant():
    info = parse_model_name("Mellum2-12B-A2.5B-Thinking.Q6_K.gguf")
    assert info["base_name"] == "Mellum2-12B-A2.5B-Thinking"
    assert info["quant"] == "Q6_K"

def test_parse_sharded_keeps_qquant():
    info = parse_model_name("Qwen3-Foo-Q4_K_M-00001-of-00002.gguf")
    assert info["base_name"] == "Qwen3-Foo"
    assert info["quant"] == "Q4_K_M"

def test_extract_metadata_sharded_mxfp4_slug(tmp_path):
    f = tmp_path / "gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    f.write_bytes(b"x")
    meta = extract_metadata(f)
    assert meta["slug"] == "gpt-oss-120b-mxfp4"
    assert meta["quant"] == "mxfp4"
