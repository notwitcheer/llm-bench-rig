"""t04 fix a bug: test.py untouched, `python3 test.py` passes, and a hidden test of stats.py passes."""
import hashlib
import os
import subprocess

from _common import SEED, main


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


HIDDEN = (
    "import sys; sys.path.insert(0, '.'); from stats import mean, total;"
    "assert mean([1, 2, 3, 4]) == 2.5; assert mean([5]) == 5; assert mean([-2, 2]) == 0;"
    "assert total([1.5, 2.5]) == 4.0; assert total([]) == 0; print('hidden ok')"
)


def check(work):
    d = os.path.join(work, "t04")
    t = os.path.join(d, "test.py")
    if not os.path.isfile(t) or sha(t) != sha(os.path.join(SEED, "t04/test.py")):
        return False, "test.py missing or modified"
    r = subprocess.run(["python3", "test.py"], cwd=d, capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "all tests passed" not in r.stdout:
        return False, f"test.py rc={r.returncode} {r.stderr.strip()[-200:]}"
    h = subprocess.run(["python3", "-c", HIDDEN], cwd=d, capture_output=True, text=True, timeout=30)
    if h.returncode != 0:
        return False, f"hidden test failed: {h.stderr.strip()[-200:]}"
    return True, "test.py + hidden test pass, test.py unchanged"


if __name__ == "__main__":
    main(check)
