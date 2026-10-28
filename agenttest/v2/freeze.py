#!/usr/bin/env python3
"""freeze the v2 task set: writes tasks.json (prompts, budgets, seed hashes) and FROZEN.sha256 (hashes of every file that defines a task or a checker).
run ONCE, before any model output; the queue item's preflight re-checks FROZEN.sha256 (sha256sum -c) and the seed hashes, so any later edit aborts the run.

usage: freeze.py [--check]"""
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

FROZEN_FILES = ["common.py", "tasks/__init__.py", "tasks/t_files.py", "tasks/t_code.py", "tasks/t_refactor.py", "tasks/t_recovery.py",
                "tasks/t_shell_data.py", "tasks/t_long.py", "tasks.json"]


def build():
    out = []
    for cls in common.loaded_tasks():
        d = tempfile.mkdtemp(prefix="at2-freeze-")
        try:
            cls.seed(d)
            sh = common.tree_hash(d)
        finally:
            common.rm_tree(d)
        out.append({"id": cls.id, "name": cls.name, "cat": cls.cat, "diff": cls.diff, "budget_s": cls.budget, "prompt": cls.prompt,
                    "prompt_sha256": hashlib.sha256(cls.prompt.encode()).hexdigest(), "seed_sha256": sh})
    return {"version": "agenttest-v2", "n": len(out), "work_placeholder": "{T}", "tasks": out}


if __name__ == "__main__":
    os.environ["AT2_STRICT"] = "1"
    spec = build()
    path = os.path.join(HERE, "tasks.json")
    if "--check" in sys.argv:
        cur = json.load(open(path))
        same = json.dumps(cur, sort_keys=True) == json.dumps(spec, sort_keys=True)
        print("tasks.json matches the registry + seeds" if same else "MISMATCH: tasks.json differs from the registry or the seed hashes")
        sys.exit(0 if same else 1)
    json.dump(spec, open(path, "w"), indent=1)
    with open(os.path.join(HERE, "FROZEN.sha256"), "w") as f:
        for rel in FROZEN_FILES:
            f.write(f"{hashlib.sha256(open(os.path.join(HERE, rel), 'rb').read()).hexdigest()}  {rel}\n")
    print(f"froze {spec['n']} tasks; wrote tasks.json and FROZEN.sha256")
