"""agenttest v2 shared helpers: task registry, file helpers, deterministic git fixtures.

Every task is a class registered with @register(...). A task has
  seed(T)   write the starting state into the task dir T (deterministic, path independent)
  check(T)  -> (passed: bool, reason: str, frac: float)  frac = partial credit in [0, 1], pass means frac == 1.0 AND the hard conditions
  solve(T)  a hand solution (proof only: what a correct agent leaves on disk)
  near(T)   a plausible wrong answer applied to a FRESH seed (proof only; must FAIL check)
The prompt uses {T} for the absolute task dir. Seeds never contain an absolute path, so the seed hash is the same on
mercury (proof) and capsule (runs). Expected values are derived from the seed code, never from files the agent can edit.
"""
import hashlib
import os
import random
import shutil
import subprocess
import sys

REGISTRY = []


def register(tid, name, cat, diff, budget, prompt):
    def deco(cls):
        cls.id, cls.name, cls.cat, cls.diff, cls.budget, cls.prompt = tid, name, cat, diff, budget, prompt
        REGISTRY.append(cls)
        return cls
    return deco


def rng(tid):
    return random.Random("agenttest-v2-" + tid)


def w(T, rel, s, mode=None):
    p = os.path.join(T, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)
    if mode is not None:
        os.chmod(p, mode)


def r(T, rel):
    p = os.path.join(T, rel)
    if not os.path.isfile(p):
        return None
    return open(p, encoding="utf-8", errors="replace").read()


def sha(s):
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode()).hexdigest()


def file_sha(p):
    if not os.path.isfile(p):
        return ""
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sh(cmd, cwd, timeout=60, env=None, inp=None):
    e = dict(os.environ)
    e.update({"LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"})
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=e, input=inp,
                          shell=isinstance(cmd, str))


def py(T, code, timeout=60):
    return sh(["python3", "-c", code], T, timeout)


def one_line(s):
    """a file that must hold exactly one line: one trailing newline allowed, nothing else multi-line."""
    if s is None:
        return None
    if s.endswith("\r\n"):
        s = s[:-2]
    elif s.endswith("\n"):
        s = s[:-1]
    return None if "\n" in s else s


def lines(s):
    return [] if s is None else [x for x in s.replace("\r\n", "\n").split("\n")]


def rm_tree(p):
    if os.path.isdir(p):
        subprocess.run(["chmod", "-R", "u+rwX", p], capture_output=True)
        shutil.rmtree(p)


def tree_hash(T):
    """hash of every file (relative path + bytes), skipping .git internals and caches; git repos contribute HEAD."""
    h = hashlib.sha256()
    for dp, dn, fns in os.walk(T):
        dn.sort()
        if ".git" in dn:
            dn.remove(".git")
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dp, capture_output=True, text=True).stdout.strip()
            h.update(("git:" + os.path.relpath(dp, T) + ":" + head + "\n").encode())
        dn[:] = [d for d in dn if d != "__pycache__"]
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            h.update(os.path.relpath(p, T).encode() + b"\0")
            if fn.endswith(".db"):  # sqlite files differ in header bytes across versions: hash the logical dump
                import sqlite3
                con = sqlite3.connect(p)
                h.update("\n".join(con.iterdump()).encode() + b"\0")
                con.close()
                continue
            try:
                h.update(open(p, "rb").read() + b"\0")
            except PermissionError:
                h.update(b"<unreadable>\0")
            m = os.stat(p).st_mode  # umask differs between boxes: only the owner read and exec bits are part of the seed
            h.update(("r" if m & 0o400 else "-").encode() + ("x" if m & 0o100 else "-").encode())
    return h.hexdigest()


GIT_ENV = {
    "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
}


def git(cwd, *args, date=None, check=True):
    env = dict(GIT_ENV)
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, **env})
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr}")
    return p.stdout.strip()


def git_init(d):
    os.makedirs(d, exist_ok=True)
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "core.autocrlf", "false")
    git(d, "config", "commit.gpgsign", "false")


def git_commit(d, msg, n):
    """deterministic commit number n (fixed date, author)."""
    git(d, "add", "-A")
    git(d, "commit", "-q", "-m", msg, date=f"2026-01-{n:02d}T10:00:00+00:00")
    return git(d, "rev-parse", "HEAD")


def loaded_tasks():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import importlib
    for m in ("t_files", "t_code", "t_refactor", "t_recovery", "t_shell_data", "t_long"):
        try:
            importlib.import_module("tasks." + m)
        except ModuleNotFoundError as e:
            if e.name != "tasks." + m or os.environ.get("AT2_STRICT"):
                raise
    return sorted(REGISTRY, key=lambda c: c.id)
