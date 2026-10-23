"""t07 log count: count.txt holds only the number of lines containing the uppercase word ERROR."""
import re

from _common import main, read, seed_text


def expected():
    return sum(1 for l in seed_text("t07/app.log").splitlines() if re.search(r"\bERROR\b", l))


def check(work):
    s = read(work, "t07/count.txt")
    if s is None:
        return False, "count.txt missing"
    v = s.strip()
    if v != str(expected()):
        return False, f"{s!r} != {expected()}"
    return True, f"{v} == {expected()}"


if __name__ == "__main__":
    main(check)
