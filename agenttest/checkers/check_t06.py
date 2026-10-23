"""t06 web fact: fact.txt is one line naming the product the docs page documents (Hermes Agent)."""
from _common import main, read


def check(work):
    s = read(work, "t06/fact.txt")
    if s is None:
        return False, "fact.txt missing"
    lines = [l for l in s.split("\n") if l.strip()]
    if len(lines) != 1:
        return False, f"{len(lines)} non-empty lines: {s!r}"
    line = lines[0].strip()
    if "hermes agent" not in line.lower():
        return False, f"{line!r} does not name Hermes Agent"
    if len(line) > 80:
        return False, f"line too long ({len(line)} chars)"
    return True, repr(line)


if __name__ == "__main__":
    main(check)
