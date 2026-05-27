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
