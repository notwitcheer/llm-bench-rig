"""t05 multi-file: project/{a,b,c}.txt hold alpha/beta/gamma, index.txt lists a.txt b.txt c.txt sorted."""
from _common import main, read

WANT = {"a.txt": "alpha", "b.txt": "beta", "c.txt": "gamma"}


def check(work):
    for fn, word in WANT.items():
        s = read(work, f"t05/project/{fn}")
        if s is None:
            return False, f"{fn} missing"
        if s.strip() != word:
            return False, f"{fn} = {s!r}"
    s = read(work, "t05/project/index.txt")
    if s is None:
        return False, "index.txt missing"
    lines = s.rstrip("\n").split("\n")
    if [l.strip() for l in lines] != ["a.txt", "b.txt", "c.txt"]:
        return False, f"index.txt = {s!r}"
    return True, "3 files + sorted index"


if __name__ == "__main__":
    main(check)
