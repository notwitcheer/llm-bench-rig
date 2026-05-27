import pytest
from pathlib import Path
from lib.config import load_config, get
import lib.config as config_module

@pytest.fixture(autouse=True)
def reset_config():
    config_module._config = None
    yield
    config_module._config = None

def test_load_config():
    cfg = load_config()
    assert "llama_cpp" in cfg
    assert "speed" in cfg
    assert cfg["speed"]["context_lengths"] == [128, 512, 2048]

def test_get_nested():
    assert get("speed.context_lengths") == [128, 512, 2048]
    assert get("llama_cpp.server_port") == 8090

def test_get_missing():
    assert get("nonexistent.key", "fallback") == "fallback"
