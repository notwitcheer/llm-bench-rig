import json
import time
from pathlib import Path

class Progress:
    def __init__(self, path: Path, model: str):
        self.path = Path(path)
        self.data = {
            "model": model,
            "step": "init",
            "pct": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
            "partial": {},
        }
        self._write()

    def update(self, step: str, pct: int, partial: dict = None):
        self.data["step"] = step
        self.data["pct"] = pct
        self.data["updated_at"] = time.time()
        if partial:
            self.data["partial"].update(partial)
        self._write()

    def done(self):
        self.data["step"] = "done"
        self.data["pct"] = 100
        self.data["updated_at"] = time.time()
        self._write()

    def fail(self, error: str):
        self.data["step"] = "error"
        self.data["error"] = error
        self.data["updated_at"] = time.time()
        self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        tmp.rename(self.path)
