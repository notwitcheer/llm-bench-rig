"""t01 create a file: hello.txt holds exactly the single line `muse glimmer was here`."""
from _common import main, one_line, read

WANT = "muse glimmer was here"


def check(work):
    s = read(work, "t01/hello.txt")
    if s is None:
        return False, "hello.txt missing"
    line = one_line(s)
    if line is None:
        return False, f"not a single line: {s!r}"
    if line != WANT:
        return False, f"content {line!r} != {WANT!r}"
    return True, "exact single line"


if __name__ == "__main__":
    main(check)
