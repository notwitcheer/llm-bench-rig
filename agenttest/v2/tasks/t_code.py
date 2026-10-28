"""agenttest v2, category code (t07-t11): single-module bug fixing, implementing from a spec, py2 to py3."""
import hashlib
import os
import subprocess
import sys
import unicodedata
from decimal import Decimal, ROUND_HALF_UP

from common import file_sha, py, r, register, rng, sh, w

# ---------------------------------------------------------------- t07
T07_WINDOW = '''"""small helpers for time series."""


def moving_average(xs, n):
    """average of every window of n consecutive items, as a list of floats (len(xs) - n + 1 values)."""
    out = []
    for i in range(len(xs) - n):
        out.append(sum(xs[i:i + n]) / n)
    return out


def chunk(xs, size):
    """split xs into consecutive lists of at most `size` items; the last one may be shorter."""
    out = []
    for i in range(0, len(xs), size):
        out.append(xs[i:i + size - 1])
    return out
'''
T07_TEST = '''import window

assert window.moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5], window.moving_average([1, 2, 3, 4], 2)
assert window.moving_average([5, 5, 5], 3) == [5.0], window.moving_average([5, 5, 5], 3)
assert window.chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]], window.chunk([1, 2, 3, 4, 5], 2)
print("all tests passed")
'''
T07_HIDDEN = ("import sys; sys.path.insert(0, '.'); from window import moving_average as m, chunk as c;"
              "assert m([2], 1) == [2.0]; assert m([1, 2, 3], 3) == [2.0]; assert m([1, 2], 3) == []; assert m([], 1) == [];"
              "assert m([1, 3, 5, 7, 9], 2) == [2.0, 4.0, 6.0, 8.0]; assert c([], 3) == []; assert c([1, 2, 3], 3) == [[1, 2, 3]];"
              "assert c([1, 2, 3, 4, 5, 6, 7], 3) == [[1, 2, 3], [4, 5, 6], [7]]; assert c([1], 5) == [[1]]; print('hidden ok')")


@register("t07", "fix an off-by-one module", "code", "A", 420,
          "In {T} there is a small Python module window.py and a test script test.py. Running python3 test.py in that folder currently fails. "
          "Fix the bugs in window.py (do not edit test.py) so that python3 test.py passes, and run it to confirm.")
class T07:
    @staticmethod
    def seed(T):
        w(T, "window.py", T07_WINDOW)
        w(T, "test.py", T07_TEST)

    @staticmethod
    def check(T):
        t = os.path.join(T, "test.py")
        if not os.path.isfile(t) or file_sha(t) != hashlib.sha256(T07_TEST.encode()).hexdigest():
            return False, "test.py missing or modified", 0.0
        p = sh(["python3", "test.py"], T)
        if p.returncode != 0 or "all tests passed" not in p.stdout:
            return False, f"test.py rc={p.returncode} {p.stderr.strip()[-160:]}", 0.0
        h = sh(["python3", "-c", T07_HIDDEN], T)
        if h.returncode != 0:
            return False, f"hidden cases failed: {h.stderr.strip()[-160:]}", 0.0
        return True, "test.py and hidden cases pass", 1.0

    @staticmethod
    def solve(T):
        s = r(T, "window.py").replace("range(len(xs) - n)", "range(len(xs) - n + 1)").replace("xs[i:i + size - 1]", "xs[i:i + size]")
        w(T, "window.py", s)

    @staticmethod
    def near(T):
        w(T, "test.py", "print('all tests passed')\n")


# ---------------------------------------------------------------- t08
T08_INVOICE = '''"""invoice totals.

rules: subtotal = sum(qty * unit_price). orders of 100.00 or more get a 5% discount, orders of 500.00 or more get 10%.
tax (20%) is charged on the amount AFTER the discount. the total is rounded half up to whole cents (Decimal, never float).
"""
from decimal import Decimal, ROUND_HALF_EVEN

TAX = Decimal("0.20")


def cents(x):
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def discount_rate(subtotal):
    if subtotal > Decimal("500"):
        return Decimal("0.10")
    if subtotal > Decimal("100"):
        return Decimal("0.05")
    return Decimal("0")


def invoice_total(items):
    """items: list of (qty, unit_price_str). returns the total as a Decimal with 2 places."""
    subtotal = sum((Decimal(p) * q for q, p in items), Decimal("0"))
    disc = subtotal * discount_rate(subtotal)
    tax = subtotal * TAX
    return cents(subtotal - disc + tax)
'''
T08_TEST = '''import unittest
from decimal import Decimal
import invoice


class T(unittest.TestCase):
    def test_small(self):
        self.assertEqual(invoice.invoice_total([(2, "10.00")]), Decimal("24.00"))

    def test_exact_100(self):
        self.assertEqual(invoice.invoice_total([(1, "100.00")]), Decimal("114.00"))

    def test_exact_500(self):
        self.assertEqual(invoice.invoice_total([(5, "100.00")]), Decimal("540.00"))

    def test_discount_then_tax(self):
        self.assertEqual(invoice.invoice_total([(1, "200.00")]), Decimal("228.00"))

    def test_half_cent_up(self):
        self.assertEqual(invoice.invoice_total([(1, "0.25")]), Decimal("0.30"))

    def test_rate(self):
        self.assertEqual(invoice.discount_rate(Decimal("99.99")), Decimal("0"))
        self.assertEqual(invoice.discount_rate(Decimal("100")), Decimal("0.05"))

    def test_empty(self):
        self.assertEqual(invoice.invoice_total([]), Decimal("0.00"))


if __name__ == "__main__":
    unittest.main()
'''


def t08_ref(items):
    sub = sum((Decimal(p) * q for q, p in items), Decimal("0"))
    rate = Decimal("0.10") if sub >= 500 else Decimal("0.05") if sub >= 100 else Decimal("0")
    return ((sub - sub * rate) * Decimal("1.20")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def t08_cases():
    rg = rng("t08")
    out = [[(1, "99.99")], [(1, "100.00")], [(1, "499.99")], [(1, "500.00")], [(3, "0.05")], [(7, "0.15")], [(1, "0.25")], [(1, "1.04")], [(1, "100.25")], [(1, "150.25")], [(1, "300.25")]]
    for _ in range(22):
        out.append([(rg.randint(1, 9), f"{rg.uniform(0.01, 140):.2f}") for _ in range(rg.randint(1, 4))])
    return out


@register("t08", "fix three bugs in a money module", "code", "B", 600,
          "In {T} there is invoice.py (the rules are in its docstring) and test_invoice.py. Run python3 -m unittest in that folder: several tests fail. "
          "Fix invoice.py so that it follows the docstring rules exactly and all tests pass; do not edit test_invoice.py. Hidden extra cases will be checked too.")
class T08:
    @staticmethod
    def seed(T):
        w(T, "invoice.py", T08_INVOICE)
        w(T, "test_invoice.py", T08_TEST)

    @staticmethod
    def check(T):
        t = os.path.join(T, "test_invoice.py")
        if not os.path.isfile(t) or file_sha(t) != hashlib.sha256(T08_TEST.encode()).hexdigest():
            return False, "test_invoice.py missing or modified", 0.0
        p = sh(["python3", "-m", "unittest", "-q"], T)
        vis_ok = p.returncode == 0
        code = ("import sys, json; sys.path.insert(0, '.'); from decimal import Decimal; import invoice; cases = json.loads(sys.stdin.read()); res = [];\n"
                "for c in cases:\n    try: res.append(str(invoice.invoice_total([(q, pr) for q, pr in c])))\n    except Exception as e: res.append('ERR ' + str(e))\nprint(json.dumps(res))")
        import json
        cs = t08_cases()
        h = sh(["python3", "-c", code], T, inp=json.dumps(cs))
        try:
            got = json.loads(h.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return False, f"hidden run failed: {h.stderr.strip()[-160:]}", 0.0
        good = sum(1 for c, g in zip(cs, got) if g == str(t08_ref(c)))
        frac = good / len(cs)
        if vis_ok and good == len(cs):
            return True, f"unittest ok, {good}/{len(cs)} hidden cases", 1.0
        return False, f"unittest {'ok' if vis_ok else 'fails'}, hidden {good}/{len(cs)}", round(min(frac, 0.99) * (1 if vis_ok else 0.5), 3)

    @staticmethod
    def solve(T):
        s = r(T, "invoice.py").replace("ROUND_HALF_EVEN", "ROUND_HALF_UP").replace('subtotal > Decimal("500")', 'subtotal >= Decimal("500")') \
            .replace('subtotal > Decimal("100")', 'subtotal >= Decimal("100")').replace("tax = subtotal * TAX", "tax = (subtotal - disc) * TAX")
        w(T, "invoice.py", s)

    @staticmethod
    def near(T):  # fixes thresholds and tax base but keeps the banker's rounding
        s = r(T, "invoice.py").replace('subtotal > Decimal("500")', 'subtotal >= Decimal("500")') \
            .replace('subtotal > Decimal("100")', 'subtotal >= Decimal("100")').replace("tax = subtotal * TAX", "tax = (subtotal - disc) * TAX")
        w(T, "invoice.py", s)


# ---------------------------------------------------------------- t09
T09_STUB = '''"""text helpers."""


def slugify(s, max_len=None):
    """turn a title into a URL slug. rules, applied in this order:
    1. lowercase everything.
    2. fold accented letters to plain ASCII (e with acute -> e); the German sharp s becomes ss.
    3. delete apostrophes (straight ' and curly U+2019) without leaving a separator: "don't" -> "dont".
    4. every run of characters that are not a-z or 0-9 becomes ONE hyphen.
    5. no leading or trailing hyphen.
    6. if max_len is given, cut the slug to at most max_len characters, then strip a trailing hyphen the cut may have left.
    7. if nothing is left, return "untitled".
    """
    raise NotImplementedError
'''
T09_TEST = '''import text

assert text.slugify("Hello, World!") == "hello-world"
assert text.slugify("  Already--slugged  ") == "already-slugged"
assert text.slugify("Caf\\u00e9 d\\u00e9j\\u00e0 vu") == "cafe-deja-vu"
assert text.slugify("don't stop") == "dont-stop"
assert text.slugify("***") == "untitled"
assert text.slugify("a very long title indeed", 10) == "a-very-lon"
print("visible tests passed")
'''


def t09_ref(s, max_len=None):
    s = s.lower().replace("\u00df", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("\u2019", "")
    out, prev = [], True
    for c in s:
        if c.isascii() and c.isalnum():
            out.append(c)
            prev = False
        elif not prev:
            out.append("-")
            prev = True
    s = "".join(out).strip("-")
    if max_len is not None:
        s = s[:max_len].rstrip("-")
    return s or "untitled"


T09_CASES = [("Hello, World!", None), ("  Already--slugged  ", None), ("Caf\u00e9 d\u00e9j\u00e0 vu", None), ("don't stop", None), ("***", None),
             ("a very long title indeed", 10), ("Stra\u00dfe nach \u00dcberlingen", None), ("rock\u2019n\u2019roll 101", None), ("MiXeD CaSe_and_underscores", None),
             ("trailing cut here ok", 12), ("trailing cut here ok", 13), ("", None), ("   ", None), ("2026-10-28 release notes", None), ("a", 1), ("-a-b-", None),
             ("Na\u00efve fa\u00e7ade", None), ("one  two\tthree\nfour", None), ("x" * 40, 5), ("Ol\u00e9! \u00c7a va?", 6), ("it's 100% done", None),
             ("\u00fcber \u00e9t\u00e9", 4), ("a-b-c-d", 4), ("a b c d", 3), ("\u2019\u2019\u2019", None), ("CamelCaseTitle", None), ("tab\ttab", 4), ("e\u0301cole", None),
             ("ends with number 42", 18), ("sp ace-s", 2)]


@register("t09", "implement slugify from a spec", "code", "B", 600,
          "In {T}/text.py the function slugify is a stub whose docstring lists seven rules. Implement it to follow every rule (python3 standard library only). "
          "{T}/test.py has a few visible cases; run it with python3 test.py. A larger hidden case list will be checked.")
class T09:
    @staticmethod
    def seed(T):
        w(T, "text.py", T09_STUB)
        w(T, "test.py", T09_TEST)

    @staticmethod
    def check(T):
        import json
        code = ("import sys, json; sys.path.insert(0, '.'); import text; cs = json.loads(sys.stdin.read()); res = [];\n"
                "for s, m in cs:\n    try: res.append(text.slugify(s) if m is None else text.slugify(s, m))\n    except Exception as e: res.append('ERR ' + type(e).__name__)\nprint(json.dumps(res))")
        h = sh(["python3", "-c", code], T, inp=json.dumps(T09_CASES))
        try:
            got = json.loads(h.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return False, f"run failed: {h.stderr.strip()[-160:]}", 0.0
        good = [c for c, g in zip(T09_CASES, got) if g == t09_ref(*c)]
        frac = len(good) / len(T09_CASES)
        if len(good) == len(T09_CASES):
            return True, f"{len(good)}/{len(T09_CASES)} cases", 1.0
        bad = [c for c, g in zip(T09_CASES, got) if g != t09_ref(*c)]
        return False, f"{len(good)}/{len(T09_CASES)} cases; first miss {bad[0]!r}", round(frac, 3)

    @staticmethod
    def solve(T):
        w(T, "text.py", '''import unicodedata


def slugify(s, max_len=None):
    s = s.lower().replace("\\u00df", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("\\u2019", "")
    out, prev = [], True
    for c in s:
        if c.isascii() and c.isalnum():
            out.append(c)
            prev = False
        elif not prev:
            out.append("-")
            prev = True
    s = "".join(out).strip("-")
    if max_len is not None:
        s = s[:max_len].rstrip("-")
    return s or "untitled"
''')

    @staticmethod
    def near(T):  # plausible: no accent folding of sharp s, regex without apostrophe rule
        w(T, "text.py", '''import re


def slugify(s, max_len=None):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    if max_len:
        s = s[:max_len]
    return s or "untitled"
''')


# ---------------------------------------------------------------- t10
T10_FILES = {
    "a.txt": "one two\nthree four five\n\nsix\n",
    "b.txt": "caf\u00e9 cr\u00e8me br\u00fbl\u00e9e\nx\n",
    "c.txt": "no newline at end",
}


def t10_ref(args, files, stdin=""):
    flags = [a for a in args if a in ("-l", "-w", "-c", "--longest")]
    names = [a for a in args if a not in ("-l", "-w", "-c", "--longest")]
    show = {"-l": "-l" in flags, "-w": "-w" in flags, "-c": "-c" in flags, "--longest": "--longest" in flags}
    if not flags:
        show.update({"-l": True, "-w": True, "-c": True})
    out, err, rc = [], [], 0
    tot = [0, 0, 0, 0]
    seen = 0
    for n in (names or ["-"]):
        if n == "-" or not names:
            text = stdin
            shown = "-"
        else:
            if n not in files:
                err.append(f"minicount: {n}: No such file")
                rc = 1
                continue
            text = files[n]
            shown = n
        ls = text.split("\n")
        if ls and ls[-1] == "":
            ls = ls[:-1]
        nl = text.count("\n")
        nw = len(text.split())
        nc = len(text)
        lg = max([len(x) for x in ls] + [0])
        vals = [nl, nw, nc, lg]
        tot = [tot[0] + nl, tot[1] + nw, tot[2] + nc, max(tot[3], lg)]
        seen += 1
        out.append(" ".join(str(v) for v, k in zip(vals, ("-l", "-w", "-c", "--longest")) if show[k]) + " " + shown)
    if seen >= 2:
        out.append(" ".join(str(v) for v, k in zip(tot, ("-l", "-w", "-c", "--longest")) if show[k]) + " total")
    return "\n".join(out) + ("\n" if out else ""), "\n".join(err) + ("\n" if err else ""), rc


T10_SCEN = [(["a.txt"], ""), (["-l", "a.txt"], ""), (["-w", "-c", "b.txt"], ""), (["--longest", "c.txt"], ""), (["a.txt", "b.txt", "c.txt"], ""),
            (["-c", "-l", "a.txt", "b.txt"], ""), (["a.txt", "nope.txt", "c.txt"], ""), ([], "alpha beta\ngamma\n"), (["-w", "-"], "x y z\n"),
            (["nope.txt"], ""), (["--longest", "-l", "a.txt", "b.txt"], "")]


@register("t10", "write a small CLI from a spec", "code", "B", 720,
          "Write {T}/minicount.py, a command line tool (python3 standard library only). Usage: python3 minicount.py [-l] [-w] [-c] [--longest] [FILE ...]. "
          "It counts lines (number of newline characters), words (whitespace separated), characters (decoded as UTF-8, not bytes) and the length in characters of the longest line (without its newline). "
          "Flags choose which counts are printed; with no flag it prints lines, words and characters. The order is ALWAYS lines, words, characters, longest, whatever the flag order. "
          "One output line per file: the selected counts separated by single spaces, then a space and the file name, e.g. `4 6 29 11 a.txt`. With two or more readable files print a final line with the sums (longest = the maximum) and the name `total`. "
          "With no FILE (or the name -) read standard input and show the name - . A file that cannot be opened prints `minicount: NAME: No such file` to standard error, is skipped (not in the total), and makes the exit code 1 at the end. "
          "Sample input files are in {T}/samples.")
class T10:
    @staticmethod
    def seed(T):
        for n, s in T10_FILES.items():
            w(T, "samples/" + n, s)

    @staticmethod
    def check(T):
        if not os.path.isfile(os.path.join(T, "minicount.py")):
            return False, "minicount.py missing", 0.0
        good = 0
        first = ""
        sd = os.path.join(T, "samples")
        for args, stdin in T10_SCEN:
            exp_out, exp_err, exp_rc = t10_ref(args, T10_FILES, stdin)
            try:
                p = sh(["python3", os.path.join(T, "minicount.py"), *args], sd, inp=stdin, timeout=20)
            except Exception as e:  # noqa: BLE001
                first = first or f"{args}: {type(e).__name__}"
                continue
            if p.stdout == exp_out and p.stderr.strip() == exp_err.strip() and p.returncode == exp_rc:
                good += 1
            elif not first:
                first = f"{args}: out {p.stdout!r} want {exp_out!r} rc {p.returncode} want {exp_rc}"
        frac = good / len(T10_SCEN)
        if good == len(T10_SCEN):
            return True, f"{good}/{len(T10_SCEN)} scenarios", 1.0
        return False, f"{good}/{len(T10_SCEN)} scenarios; {first[:200]}", round(frac, 3)

    @staticmethod
    def solve(T):
        w(T, "minicount.py", '''import sys


def main(argv):
    flags = [a for a in argv if a in ("-l", "-w", "-c", "--longest")]
    names = [a for a in argv if a not in ("-l", "-w", "-c", "--longest")]
    keys = ("-l", "-w", "-c", "--longest")
    show = {k: (k in flags) for k in keys}
    if not flags:
        show.update({"-l": True, "-w": True, "-c": True})
    rc = 0
    tot = [0, 0, 0, 0]
    seen = 0
    for n in (names or ["-"]):
        if n == "-":
            text = sys.stdin.read()
        else:
            try:
                text = open(n, encoding="utf-8").read()
            except OSError:
                sys.stderr.write("minicount: %s: No such file\\n" % n)
                rc = 1
                continue
        ls = text.split("\\n")
        if ls and ls[-1] == "":
            ls = ls[:-1]
        vals = [text.count("\\n"), len(text.split()), len(text), max([len(x) for x in ls] + [0])]
        tot = [tot[0] + vals[0], tot[1] + vals[1], tot[2] + vals[2], max(tot[3], vals[3])]
        seen += 1
        print(" ".join(str(v) for v, k in zip(vals, keys) if show[k]) + " " + n)
    if seen >= 2:
        print(" ".join(str(v) for v, k in zip(tot, keys) if show[k]) + " total")
    return rc


sys.exit(main(sys.argv[1:]))
''')

    @staticmethod
    def near(T):  # prints bytes instead of characters and ignores the exit code
        w(T, "minicount.py", '''import sys
for n in sys.argv[1:]:
    if n.startswith("-"):
        continue
    try:
        b = open(n, "rb").read()
    except OSError:
        continue
    print(b.count(b"\\n"), len(b.split()), len(b), n)
''')


# ---------------------------------------------------------------- t11
T11_REPORT = '''import sys


def main(path):
    counts = {}
    f = open(path)
    for line in f.readlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        name, qty = parts[0], int(parts[1])
        if counts.has_key(name):
            counts[name] += qty
        else:
            counts[name] = qty
    f.close()
    for name in sorted(counts.keys(), cmp=lambda a, b: cmp(counts[b], counts[a]) or cmp(a, b)):
        print "%-10s %5d" % (name, counts[name])
    print "total:", reduce(lambda a, b: a + b, counts.values())
    print "average:", sum(counts.values()) / len(counts)
    for i in xrange(2):
        print "pass", i
    try:
        open("/nonexistent/x")
    except IOError, e:
        print "missing file handled"


if __name__ == "__main__":
    main(sys.argv[1])
'''
T11_DATA = "apple,10\nbanana,7\napple,5\ncherry,7\n\nbad line\ndate,3\nbanana,2\n"


def t11_ref(data):
    counts = {}
    for line in data.split("\n"):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
    out = []
    for n in sorted(counts, key=lambda k: (-counts[k], k)):
        out.append("%-10s %5d" % (n, counts[n]))
    out.append("total: %d" % sum(counts.values()))
    out.append("average: %d" % (sum(counts.values()) // len(counts)))
    out += ["pass 0", "pass 1", "missing file handled"]
    return "\n".join(out) + "\n"


@register("t11", "port a python2 script to python3", "code", "B", 600,
          "{T}/report.py is a Python 2 script. {T}/data.csv is its input and {T}/expected_output.txt is exactly what it printed under Python 2. "
          "Port report.py in place so that python3 report.py data.csv prints exactly the same text (the average must keep Python 2's integer division result). "
          "It will also be run on other csv files, so do not hard code the output.")
class T11:
    @staticmethod
    def seed(T):
        w(T, "report.py", T11_REPORT)
        w(T, "data.csv", T11_DATA)
        w(T, "expected_output.txt", t11_ref(T11_DATA))

    @staticmethod
    def check(T):
        other = "kiwi,4\nfig,4\nkiwi,1\nlime,9\n"
        good = 0
        for data in (T11_DATA, other):
            w(T, "_hidden.csv", data)
            p = sh(["python3", "report.py", "_hidden.csv"], T, timeout=20)
            if p.returncode == 0 and p.stdout == t11_ref(data):
                good += 1
            os.remove(os.path.join(T, "_hidden.csv"))
        if good == 2:
            return True, "same output on the seed and a hidden csv", 1.0
        return False, f"{good}/2 datasets match", 0.0

    @staticmethod
    def solve(T):
        w(T, "report.py", '''import sys
from functools import reduce


def main(path):
    counts = {}
    f = open(path)
    for line in f.readlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        name, qty = parts[0], int(parts[1])
        if name in counts:
            counts[name] += qty
        else:
            counts[name] = qty
    f.close()
    for name in sorted(counts.keys(), key=lambda k: (-counts[k], k)):
        print("%-10s %5d" % (name, counts[name]))
    print("total:", reduce(lambda a, b: a + b, counts.values()))
    print("average:", sum(counts.values()) // len(counts))
    for i in range(2):
        print("pass", i)
    try:
        open("/nonexistent/x")
    except IOError as e:
        print("missing file handled")


if __name__ == "__main__":
    main(sys.argv[1])
''')

    @staticmethod
    def near(T):  # prints the seed's expected text without computing anything
        w(T, "report.py", "print(open(__file__.replace('report.py', 'expected_output.txt')).read(), end='')\n")
