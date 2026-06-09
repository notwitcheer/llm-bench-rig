"""run_native orchestration: a per-task exception must be recorded as a failed task,
not crash the whole run (so one 128K context-overflow can't void a tier)."""
import json
import lib.agentic.native.run_native as rn


def test_task_exception_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def boom(client, goal, tools, max_steps=8):
        raise RuntimeError("simulated server context overflow")

    monkeypatch.setattr(rn, "run_agent", boom)
    # long32k mode has a small fixed task set; the run must complete and write a file
    rn.run("dummy-model", "long32k")
    out = json.loads((tmp_path / "results/dummy-model/agentic_longctx_32k.json").read_text())
    assert out["success_pct"] == 0.0
    assert all(d["success"] is False and "error" in d for d in out["details"])
