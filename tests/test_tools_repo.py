import subprocess
from pathlib import Path
from lib.agentic.native.tools_repo import TmpRepoBackend, _cap


def test_cap_truncates_long_output():
    assert _cap("x" * 10, 4) == "xxxx\n...[truncated 6 chars]"
    assert _cap("short", 100) == "short"


def test_tmp_backend_read_write_run(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n")
    b = TmpRepoBackend(str(tmp_path))
    content, ok = b.read("a.py")
    assert ok and "print(1)" in content
    assert b.write("a.py", "print(2)\n") is True
    assert b.read("a.py")[0] == "print(2)\n"
    out, rc = b.run("echo hi")
    assert rc == 0 and "hi" in out


def test_tmp_backend_read_missing_file(tmp_path):
    b = TmpRepoBackend(str(tmp_path))
    content, ok = b.read("nope.py")
    assert ok is False


from lib.agentic.native.tools_repo import make_repo_tools, make_dispatch, REPO_TOOLS


def test_tools_read_edit_run(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    schemas, ns = make_repo_tools(TmpRepoBackend(str(tmp_path)))
    assert {t["function"]["name"] for t in schemas} == {
        "read_file", "list_dir", "search", "edit_file", "run_bash"}
    assert "return 1" in ns["read_file"]("m.py")
    assert "edited" in ns["edit_file"]("m.py", "return 1", "return 2")
    assert "return 2" in ns["read_file"]("m.py")
    assert "[exit 0]" in ns["run_bash"]("echo ok")


def test_edit_file_missing_old_text_is_error(tmp_path):
    (tmp_path / "m.py").write_text("a = 1\n")
    _, ns = make_repo_tools(TmpRepoBackend(str(tmp_path)))
    assert ns["edit_file"]("m.py", "NOT THERE", "x").startswith("ERROR")


def test_make_dispatch_shape(tmp_path):
    _, ns = make_repo_tools(TmpRepoBackend(str(tmp_path)))
    d = make_dispatch(ns)
    out = d("run_bash", {"cmd": "echo hi"})
    assert out["ok"] is True and "hi" in str(out["result"])
    assert d("nope", {})["ok"] is False


from lib.agentic.native.tools_repo import DockerRepoBackend


def test_docker_backend_builds_exec_argv():
    calls = []
    def fake_run(argv, **kw):
        calls.append(argv)
        class R: stdout = "file contents"; stderr = ""; returncode = 0
        return R()
    b = DockerRepoBackend("cont123", root="/testbed", _runner=fake_run)
    content, ok = b.read("pkg/m.py")
    assert ok and content == "file contents"
    assert calls[0][:5] == ["docker", "exec", "-w", "/testbed", "cont123"]
    assert "cat" in calls[0] and "pkg/m.py" in calls[0]
