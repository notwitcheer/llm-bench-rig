"""t09 recover from failure: out/report.txt == `report ok 42` and run_report.sh is unchanged."""
import hashlib
import os

from _common import gen_seeds, main, one_line, read


def check(work):
    p = os.path.join(work, "t09/run_report.sh")
    if not os.path.isfile(p):
        return False, "run_report.sh missing"
    if open(p, "rb").read() != gen_seeds.expected_bytes("t09/run_report.sh", os.path.abspath(work)):
        return False, "run_report.sh was modified"
    s = read(work, "t09/out/report.txt")
    if s is None:
        return False, "out/report.txt missing"
    if one_line(s) != "report ok 42":
        return False, f"report.txt = {s!r}"
    return True, "report produced, script unchanged"


if __name__ == "__main__":
    main(check)
