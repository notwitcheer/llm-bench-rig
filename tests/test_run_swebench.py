import subprocess
from lib.agentic.native.tools_repo import TmpRepoBackend
from lib.agentic.native.run_swebench import solve_instance


class ScriptedClient:
    def __init__(self, turns): self.turns = list(turns)
    def chat(self, messages, tools): return self.turns.pop(0)


def test_solve_instance_extracts_patch(tmp_path):
    # a tiny git repo with a bug
    (tmp_path / "bug.py").write_text("def add(a, b):\n    return a - b\n")
    for cmd in ["git init -q", "git add -A", "git -c user.email=t@t -c user.name=t commit -qm init"]:
        subprocess.run(cmd, shell=True, cwd=tmp_path, check=True)
    turns = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
         "function": {"name": "edit_file",
                      "arguments": '{"path": "bug.py", "old": "return a - b", "new": "return a + b"}'}}]},
        {"role": "assistant", "content": "fixed the sign bug", "tool_calls": None},
    ]
    out = solve_instance(ScriptedClient(turns), "add() subtracts instead of adds",
                         TmpRepoBackend(str(tmp_path)), max_steps=5)
    assert "return a + b" in out["patch"] and out["patch"].startswith("diff --git")
    assert out["n_tool_calls"] == 1 and out["stalled"] is False
    assert out["patch_empty"] is False
