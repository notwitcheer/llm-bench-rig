"""Real repo tools for the SWE-bench anchor, backed by a RepoBackend so the tool logic is
testable without Docker. TmpRepoBackend runs against a local dir (unit tests); the Docker
backend shells into the instance container. Outputs are size-capped to protect the model's
context window."""
import base64
import shlex
import subprocess

from lib.agentic.native.schemas import to_openai_tools

FILE_CAP = 4000
BASH_CAP = 3000


def _cap(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


class TmpRepoBackend:
    """Backend over a local directory — used by unit tests and local dry-runs."""
    def __init__(self, root: str):
        self.root = root

    def read(self, path: str):
        try:
            p = subprocess.run(["cat", path], cwd=self.root, capture_output=True, text=True, timeout=30)
            return (p.stdout, True) if p.returncode == 0 else (p.stderr or "not found", False)
        except Exception as e:
            return (f"{type(e).__name__}: {e}", False)

    def write(self, path: str, content: str) -> bool:
        try:
            import os
            full = os.path.join(self.root, path)
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
            return True
        except Exception:
            return False

    def run(self, cmd: str):
        try:
            p = subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=180)
            return (p.stdout + p.stderr, p.returncode)
        except subprocess.TimeoutExpired:
            return ("[timeout]", 124)

    def search(self, pattern: str):
        p = subprocess.run(["grep", "-rn", "--include=*.py", pattern, "."],
                           cwd=self.root, capture_output=True, text=True)
        return p.stdout


REPO_TOOLS = [
    {"name": "read_file", "signature": "read_file(path: str) -> str",
     "description": "Read a file from the repo (path relative to the repo root)."},
    {"name": "list_dir", "signature": "list_dir(path: str) -> str",
     "description": "List a directory in the repo (use '.' for the root)."},
    {"name": "search", "signature": "search(pattern: str) -> str",
     "description": "grep -rn for a regex/string across the repo's .py files."},
    {"name": "edit_file", "signature": "edit_file(path: str, old: str, new: str) -> str",
     "description": "Replace the exact text `old` with `new` (first occurrence) in a file. `old` must match exactly."},
    {"name": "run_bash", "signature": "run_bash(cmd: str) -> str",
     "description": "Run a shell command in the repo root (e.g. run the project's tests). Returns output + exit code."},
]


def make_repo_tools(backend):
    """Returns (openai_tool_schemas, name->callable). Tools call the backend + cap output."""
    def read_file(path):
        content, ok = backend.read(path)
        return _cap(content, FILE_CAP) if ok else f"ERROR: {content}"

    def list_dir(path="."):
        out, rc = backend.run(f"ls -la {path}")
        return _cap(out, BASH_CAP) if rc == 0 else f"ERROR: {out.strip()}"

    def search(pattern):
        return _cap(backend.search(pattern), BASH_CAP) or "no matches"

    def edit_file(path, old, new):
        content, ok = backend.read(path)
        if not ok:
            return f"ERROR: cannot read {path}"
        if old not in content:
            return "ERROR: `old` text not found in file (it must match the file exactly)"
        return f"edited {path}" if backend.write(path, content.replace(old, new, 1)) else f"ERROR: write failed {path}"

    def run_bash(cmd):
        out, rc = backend.run(cmd)
        return _cap(out, BASH_CAP) + f"\n[exit {rc}]"

    ns = {"read_file": read_file, "list_dir": list_dir, "search": search,
          "edit_file": edit_file, "run_bash": run_bash}
    return to_openai_tools(REPO_TOOLS), ns


def make_dispatch(ns):
    """A dispatch(name, args)->{ok,result,error} bound to a repo-tools namespace
    (same contract as native.dispatch.dispatch_tool, for run_agent's injectable dispatch)."""
    def dispatch(name, args):
        fn = ns.get(name)
        if fn is None:
            return {"ok": False, "result": None, "error": f"unknown tool '{name}'"}
        try:
            return {"ok": True, "result": fn(**args), "error": ""}
        except Exception as e:
            return {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}
    return dispatch


class DockerRepoBackend:
    """Backend that shells into a running instance container via `docker exec`.
    `_runner` is injectable for tests (defaults to subprocess.run)."""
    def __init__(self, container: str, root: str = "/testbed", _runner=None):
        self.c = container
        self.root = root
        self._run = _runner or (lambda argv, **kw: subprocess.run(argv, capture_output=True, text=True, **kw))

    def _exec(self, argv):
        return self._run(["docker", "exec", "-w", self.root, self.c] + argv, timeout=200)

    def read(self, path: str):
        r = self._exec(["cat", path])
        return (r.stdout, True) if r.returncode == 0 else (r.stderr or "not found", False)

    def write(self, path: str, content: str) -> bool:
        b64 = base64.b64encode(content.encode()).decode()
        r = self._exec(["bash", "-c", f"echo {b64} | base64 -d > {shlex.quote(path)}"])
        return r.returncode == 0

    def run(self, cmd: str):
        r = self._exec(["bash", "-c", cmd])
        return (r.stdout + r.stderr, r.returncode)

    def search(self, pattern: str):
        r = self._exec(["grep", "-rn", "--include=*.py", pattern, "."])
        return r.stdout
