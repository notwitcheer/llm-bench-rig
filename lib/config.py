import yaml
from pathlib import Path

_config = None

def load_config(path: Path = None) -> dict:
    global _config
    if _config is not None:
        return _config
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config

def get(key: str, default=None):
    cfg = load_config()
    keys = key.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val
