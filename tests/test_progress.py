import pytest
import json
from pathlib import Path
from lib.progress import Progress

def test_progress_init(tmp_path):
    p = Progress(tmp_path / "progress.json", model="test-model")
    assert p.data["model"] == "test-model"
    assert p.data["step"] == "init"
    assert p.data["pct"] == 0

def test_progress_update(tmp_path):
    p = Progress(tmp_path / "progress.json", model="test-model")
    p.update(step="speed_pp128", pct=25, partial={"pp128": 9000})
    data = json.loads((tmp_path / "progress.json").read_text())
    assert data["step"] == "speed_pp128"
    assert data["pct"] == 25
    assert data["partial"]["pp128"] == 9000

def test_progress_done(tmp_path):
    p = Progress(tmp_path / "progress.json", model="test-model")
    p.done()
    data = json.loads((tmp_path / "progress.json").read_text())
    assert data["step"] == "done"
    assert data["pct"] == 100
