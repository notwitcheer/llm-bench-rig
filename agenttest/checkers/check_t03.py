"""t03 find a phrase: answer.txt holds only the file name of the one doc containing the phrase."""
import os

from _common import SEED, main, one_line, read

PHRASE = "blue heron protocol"


def expected():
    d = os.path.join(SEED, "t03/docs")
    hits = [f for f in sorted(os.listdir(d)) if PHRASE in open(os.path.join(d, f)).read()]
    assert len(hits) == 1, hits
    return hits[0]


def check(work):
    s = read(work, "t03/answer.txt")
    if s is None:
        return False, "answer.txt missing"
    line = one_line(s)
    if line is None or line.strip() != expected():
        return False, f"{s!r} != {expected()!r}"
    return True, f"{expected()}"


if __name__ == "__main__":
    main(check)
