"""t02 CSV column total: total.txt holds only the sum of the amount column (to the cent)."""
import csv
import io
import re

from _common import main, read, seed_text


def expected():
    rows = csv.DictReader(io.StringIO(seed_text("t02/sales.csv")))
    return round(sum(float(r["amount"]) for r in rows), 2)


def check(work):
    s = read(work, "t02/total.txt")
    if s is None:
        return False, "total.txt missing"
    v = s.strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", v):
        return False, f"not a bare number: {s!r}"
    if abs(float(v) - expected()) > 0.005:
        return False, f"{v} != {expected():.2f}"
    return True, f"{v} == {expected():.2f}"


if __name__ == "__main__":
    main(check)
