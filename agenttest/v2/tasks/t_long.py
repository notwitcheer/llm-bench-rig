"""agenttest v2, category long (t27-t30): longer multi-step tasks (build a package from a spec, merge twelve input files, answer questions about a codebase, repair a failing suite)."""
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from common import file_sha, lines, py, r, register, rm_tree, rng, sh, w

# ---------------------------------------------------------------- t27 build a package from a README spec
T27_SPEC = """# inv: a tiny inventory tool

Build a python3 package `inv` (standard library only) in this folder, run as `python3 -m inv --store FILE COMMAND ...`.
The store is a JSON file mapping sku -> {"name": str, "qty": int}. If FILE does not exist it is an empty store; it is only created when a
command changes something. Layout: inv/__init__.py, inv/store.py (load/save logic), inv/cli.py (argument handling), inv/__main__.py.

Commands (all output on standard output unless it says error):
- add SKU NAME QTY       add a new item (QTY integer >= 0). If SKU exists: print `error: SKU exists` to standard error, exit code 1, store unchanged.
- remove SKU             delete an item. Unknown SKU: `error: unknown SKU` on standard error, exit 1.
- adjust SKU DELTA       add DELTA (integer, may be negative) to the quantity. Unknown SKU: `error: unknown SKU`, exit 1.
                         If the new quantity would be below 0: `error: insufficient stock` on standard error, exit 1, store unchanged.
- list                   one line per item sorted by sku: "%-8s %-12s %5d" % (sku, name, qty).
- low THRESHOLD          the skus whose quantity is strictly below THRESHOLD (integer), one per line, sorted. Nothing printed when none.
- report                 exactly three lines: `items: N`, `units: M` (sum of quantities), `lowest: SKU (QTY)` (the item with the smallest quantity, ties -> smaller sku),
                         or `lowest: none` when the store is empty.
Successful add/remove/adjust print nothing and exit 0. Any other bad usage (unknown command, missing arguments, non-integer number): print `error: usage` to standard error and exit 2.

Also write tests in tests/ (unittest, at least 3 test methods) and run them: python3 -m unittest discover -s tests -t .
"""


def t27_scenarios():
    """(argv list, expected stdout, expected rc, optional check of store after)  sequence run against ONE store file."""
    return [
        (["list"], "", 0),
        (["report"], "items: 0\nunits: 0\nlowest: none\n", 0),
        (["add", "B2", "bolt", "40"], "", 0),
        (["add", "A1", "anchor", "5"], "", 0),
        (["add", "C3", "clamp", "12"], "", 0),
        (["add", "A1", "dup", "1"], "", 1),
        (["list"], "A1       anchor           5\nB2       bolt            40\nC3       clamp           12\n", 0),
        (["low", "13"], "A1\nC3\n", 0),
        (["low", "5"], "", 0),
        (["adjust", "A1", "-5"], "", 0),
        (["adjust", "A1", "-1"], "", 1),
        (["adjust", "ZZ", "1"], "", 1),
        (["adjust", "C3", "x"], "", 2),
        (["bogus"], "", 2),
        (["add", "D4", "dowel"], "", 2),
        (["report"], "items: 3\nunits: 52\nlowest: A1 (0)\n", 0),
        (["remove", "A1"], "", 0),
        (["remove", "A1"], "", 1),
        (["add", "E5", "eye", "12"], "", 0),
        (["report"], "items: 3\nunits: 64\nlowest: C3 (12)\n", 0),
        (["list"], "B2       bolt            40\nC3       clamp           12\nE5       eye             12\n", 0),
    ]


@register("t27", "build a small CLI package from a README spec", "long", "C", 1200,
          "Read {T}/README.md: it specifies a small inventory tool, the python package inv. Build it completely in {T} as specified (package, command line behaviour, error handling, and your own unit tests), "
          "run your tests, and also try the commands by hand against a scratch store file. A hidden scenario will be run against your implementation.")
class T27:
    @staticmethod
    def seed(T):
        w(T, "README.md", T27_SPEC)

    @staticmethod
    def check(T):
        if not os.path.isfile(os.path.join(T, "inv", "__main__.py")):
            return False, "inv/__main__.py missing", 0.0
        d = tempfile.mkdtemp(prefix="at2s-")
        store = os.path.join(d, "store.json")
        good, total, first = 0, 0, ""
        try:
            for argv, out, rc in t27_scenarios():
                total += 2
                try:
                    p = sh(["python3", "-m", "inv", "--store", store, *argv], T, timeout=20)
                except Exception as e:  # noqa: BLE001
                    first = first or f"{argv}: {type(e).__name__}"
                    continue
                good += p.stdout == out
                good += p.returncode == rc
                if (p.stdout != out or p.returncode != rc) and not first:
                    first = f"{argv}: out {p.stdout!r} rc {p.returncode}, want {out!r} rc {rc}"
            # error text goes to stderr
            q = sh(["python3", "-m", "inv", "--store", store, "remove", "NOPE"], T, timeout=20)
            total += 1
            good += "error: unknown SKU" in q.stderr and q.stdout == ""
            s = None
            try:
                s = json.load(open(store))
            except Exception:  # noqa: BLE001
                pass
            total += 1
            good += isinstance(s, dict) and sorted(s) == ["B2", "C3", "E5"] and s["B2"] == {"name": "bolt", "qty": 40}
            # store must not be created by read-only commands
            ghost = os.path.join(d, "ghost.json")
            sh(["python3", "-m", "inv", "--store", ghost, "list"], T, timeout=20)
            total += 1
            good += not os.path.exists(ghost)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        tests = ""
        for dp, _dn, fns in os.walk(os.path.join(T, "tests")):
            for fn in fns:
                if fn.endswith(".py"):
                    tests += open(os.path.join(dp, fn)).read()
        t = sh(["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."], T, timeout=60)
        tests_ok = tests.count("def test_") >= 3 and t.returncode == 0
        frac = good / total
        if good == total and tests_ok:
            return True, f"{good}/{total} hidden checks, own tests pass", 1.0
        return False, f"{good}/{total} hidden checks; own tests {'ok' if tests_ok else 'missing/failing'}; {first[:160]}", round(frac * (1 if tests_ok else 0.9) * 0.99, 3)

    @staticmethod
    def solve(T):
        w(T, "inv/__init__.py", "")
        w(T, "inv/store.py", '''import json
import os


def load(path):
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
''')
        w(T, "inv/cli.py", '''import sys

from . import store


class Usage(Exception):
    pass


def num(x):
    try:
        return int(x)
    except ValueError:
        raise Usage()


def run(argv):
    if len(argv) < 3 or argv[0] != "--store":
        raise Usage()
    path, cmd, a = argv[1], argv[2], argv[3:]
    data = store.load(path)
    if cmd == "add":
        if len(a) != 3:
            raise Usage()
        q = num(a[2])
        if q < 0:
            raise Usage()
        if a[0] in data:
            return err("SKU exists")
        data[a[0]] = {"name": a[1], "qty": q}
        store.save(path, data)
    elif cmd == "remove":
        if len(a) != 1:
            raise Usage()
        if a[0] not in data:
            return err("unknown SKU")
        del data[a[0]]
        store.save(path, data)
    elif cmd == "adjust":
        if len(a) != 2:
            raise Usage()
        d = num(a[1])
        if a[0] not in data:
            return err("unknown SKU")
        if data[a[0]]["qty"] + d < 0:
            return err("insufficient stock")
        data[a[0]]["qty"] += d
        store.save(path, data)
    elif cmd == "list":
        for s in sorted(data):
            print("%-8s %-12s %5d" % (s, data[s]["name"], data[s]["qty"]))
    elif cmd == "low":
        if len(a) != 1:
            raise Usage()
        t = num(a[0])
        for s in sorted(data):
            if data[s]["qty"] < t:
                print(s)
    elif cmd == "report":
        print("items: %d" % len(data))
        print("units: %d" % sum(v["qty"] for v in data.values()))
        if data:
            s = min(sorted(data), key=lambda k: data[k]["qty"])
            print("lowest: %s (%d)" % (s, data[s]["qty"]))
        else:
            print("lowest: none")
    else:
        raise Usage()
    return 0


def err(msg):
    sys.stderr.write("error: %s\\n" % msg)
    return 1


def main(argv):
    try:
        return run(argv)
    except Usage:
        return err("usage") and 2
''')
        w(T, "inv/__main__.py", "import sys\nfrom .cli import main\nsys.exit(main(sys.argv[1:]))\n")
        w(T, "tests/__init__.py", "")
        w(T, "tests/test_inv.py", '''import os, subprocess, sys, tempfile, unittest


def run(*a):
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run([sys.executable, "-m", "inv", *a], cwd=d, capture_output=True, text=True)


class T(unittest.TestCase):
    def test_add_list(self):
        s = os.path.join(tempfile.mkdtemp(), "s.json")
        run("--store", s, "add", "A1", "a", "2")
        self.assertIn("A1", run("--store", s, "list").stdout)

    def test_dup(self):
        s = os.path.join(tempfile.mkdtemp(), "s.json")
        run("--store", s, "add", "A1", "a", "2")
        self.assertEqual(run("--store", s, "add", "A1", "b", "3").returncode, 1)

    def test_usage(self):
        self.assertEqual(run("--store", "x", "nope").returncode, 2)
''')

    @staticmethod
    def near(T):  # list/add only
        T27.solve(T)
        s = r(T, "inv/cli.py").replace('    elif cmd == "low":', '    elif cmd == "low_disabled":')
        w(T, "inv/cli.py", s)


# ---------------------------------------------------------------- t28 merge twelve input files
T28_SPEC = """# merge the inbox

`inbox/` holds 12 order files in three formats (orders_01..04.csv, orders_05..08.jsonl, orders_09..12.tsv). Write merged.csv in this folder.

Output columns, with this exact header line: order_id,date,customer,amount_eur
- csv files: columns order_id,customer,date (YYYY-MM-DD),amount,currency
- jsonl files: one json object per line with keys id, cust, ts (ISO datetime like 2026-03-04T10:00:00Z, the date part is the date), total (number), cur
- tsv files: tab separated with header Date, Order, Customer, Amount EUR; Date is DD/MM/YYYY, the amount uses a decimal comma (12,50) and is always in EUR
Rules:
1. order_id is text exactly as given (strip spaces). date is written YYYY-MM-DD. customer is stripped and lowercased.
2. amount_eur = amount converted with this fixed table: EUR 1.00, USD 0.90, GBP 1.15; rounded half up to 2 decimals (use decimal arithmetic).
3. Drop rows with a missing date, or an amount that is not greater than 0.
4. The same order_id can appear in several files: keep only the FIRST occurrence, reading the files in alphabetical order of file name and rows in file order.
5. Sort the output by date, then by order_id (plain text order). Amounts always have 2 decimals.
"""


def t28_files():
    rg = rng("t28")
    custs = ["Acme Co", "  bolt ltd", "Cargo&Sons ", "Delta GmbH", "echo SA"]
    files = {}
    oid = 1000
    pool = []
    for i in range(1, 13):
        rows = []
        for _ in range(12):
            if pool and rg.random() < 0.18:
                o = rg.choice(pool)
            else:
                oid += 1
                o = str(oid)
                pool.append(o)
            cust = rg.choice(custs)
            day = f"2026-{rg.randint(1, 6):02d}-{rg.randint(1, 28):02d}"
            amt = round(rg.uniform(-3, 250), 2) if rg.random() < 0.08 else round(rg.uniform(2, 250), 2)
            cur = rg.choice(["EUR", "USD", "GBP", "EUR"])
            missing = rg.random() < 0.06
            rows.append((o, cust, "" if missing else day, amt, cur))
        files[i] = rows
    return files


def t28_render(files):
    out = {}
    for i, rows in files.items():
        if i <= 4:
            b = io.StringIO()
            cw = csv.writer(b, lineterminator="\n")
            cw.writerow(["order_id", "customer", "date", "amount", "currency"])
            for o, c, d, a, cur in rows:
                cw.writerow([o, c, d, f"{a:.2f}", cur])
            out[f"orders_{i:02d}.csv"] = b.getvalue()
        elif i <= 8:
            ls = []
            for o, c, d, a, cur in rows:
                ls.append(json.dumps({"id": o, "cust": c, "ts": (d + "T10:00:00Z") if d else "", "total": a, "cur": cur}))
            out[f"orders_{i:02d}.jsonl"] = "\n".join(ls) + "\n"
        else:
            ls = ["Date\tOrder\tCustomer\tAmount EUR"]
            for o, c, d, a, cur in rows:
                dd = f"{d[8:10]}/{d[5:7]}/{d[0:4]}" if d else ""
                eur = Decimal(str(a)) * {"EUR": Decimal(1), "USD": Decimal("0.9"), "GBP": Decimal("1.15")}[cur]
                ls.append(f"{dd}\t{o}\t{c}\t{str(eur.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)).replace('.', ',')}")
            out[f"orders_{i:02d}.tsv"] = "\n".join(ls) + "\n"
    return out


def t28_expected():
    files = t28_files()
    rate = {"EUR": Decimal(1), "USD": Decimal("0.9"), "GBP": Decimal("1.15")}
    seen, rows = set(), []
    for i in sorted(files):
        for o, c, d, a, cur in files[i]:
            cur2 = cur if i <= 8 else "EUR"
            amt = Decimal(str(a)) * (rate[cur] if i <= 8 else rate[cur])
            if i > 8:
                amt = (Decimal(str(a)) * rate[cur]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # tsv amounts are already EUR
                cur2 = "EUR"
            if not d or amt <= 0:
                continue
            if o in seen:
                continue
            seen.add(o)
            rows.append((o, d, c.strip().lower(), amt.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)))
    rows.sort(key=lambda x: (x[1], x[0]))
    b = io.StringIO()
    cw = csv.writer(b, lineterminator="\n")
    cw.writerow(["order_id", "date", "customer", "amount_eur"])
    for o, d, c, a in rows:
        cw.writerow([o, d, c, str(a)])
    return b.getvalue()


@register("t28", "merge twelve inbox files in three formats", "long", "C", 1200,
          "Follow the spec in {T}/SPEC.md: merge the twelve order files in {T}/inbox (csv, jsonl and tsv) into {T}/merged.csv. Check your result against the rules (duplicates, currencies, dropped rows, sort order) before you finish.")
class T28:
    @staticmethod
    def seed(T):
        w(T, "SPEC.md", T28_SPEC)
        for fn, s in t28_render(t28_files()).items():
            w(T, "inbox/" + fn, s)

    @staticmethod
    def check(T):
        s = r(T, "merged.csv")
        if s is None:
            return False, "merged.csv missing", 0.0
        got, want = s.replace("\r\n", "\n").split("\n"), t28_expected().split("\n")
        if got == want:
            return True, f"{len(want) - 2} rows exact", 1.0
        sg, sw = set(got), set(want)
        frac = len(sg & sw) / max(1, len(sw))
        return False, f"{len(sg & sw)}/{len(sw)} expected lines present, {len(got)} lines vs {len(want)}", round(frac * 0.99, 3)

    @staticmethod
    def solve(T):
        w(T, "merged.csv", t28_expected())

    @staticmethod
    def near(T):  # naive concatenation sorted by date, no dedupe and no currency conversion
        rows = []
        for i, rs in t28_files().items():
            for o, c, d, a, cur in rs:
                if d and a > 0:
                    rows.append((o, d, c.strip().lower(), Decimal(str(a)).quantize(Decimal("0.01"))))
        rows.sort(key=lambda x: (x[1], x[0]))
        w(T, "merged.csv", "order_id,date,customer,amount_eur\n" + "".join(f"{o},{d},{c},{a}\n" for o, d, c, a in rows))


# ---------------------------------------------------------------- t29 questions about a codebase
T29_MODULES = ["config", "client", "auth", "billing", "cache", "export", "sync", "quota", "ids", "jobs", "logs", "metrics", "notify", "parser", "queue", "render",
               "retry", "schema", "search", "store", "utils", "webhook"]


def t29_files():
    rg = rng("t29")
    files = {}
    for m in T29_MODULES:
        body = [f'"""svc.{m}: generated filler module."""', ""]
        for k in range(rg.randint(3, 5)):
            a, b = rg.randint(2, 9), rg.randint(10, 99)
            body += [f"def {m}_op{k}(x, y={b}):", f"    \"\"\"combine x and y using factor {a}.\"\"\"", f"    z = (x * {a} + y) % {b + 7}", "    return z if z > 0 else -z", ""]
        files[f"svc/{m}.py"] = "\n".join(body)
    files["svc/__init__.py"] = ""

    def add(m, text):
        files[f"svc/{m}.py"] += "\n" + text

    # Q1: env var read in exactly one file
    add("config", 'import os\n\n\ndef retry_limit():\n    return int(os.environ.get("SVC_RETRY_LIMIT", "4"))\n')
    add("retry", 'import os\n\n\ndef retry_limits_report():\n    # not the same variable: SVC_RETRY_LIMITS is a legacy alias that nothing reads\n    return os.environ.get("SVC_RETRY_LIMITS", "")\n')
    add("jobs", "# note: SVC_RETRY_LIMIT is read in the config module, do not read it here\n")
    # Q2: default timeout of client.fetch
    add("client", 'def fetch(url, timeout=17, headers=None):\n    """fetch a url."""\n    return (url, timeout, headers)\n\n\ndef fetch_all(urls, timeout=30):\n    return [fetch(u, timeout=timeout) for u in urls]\n')
    # Q3: nine call sites of normalize_id (plus def and a comment)
    add("ids", 'def normalize_id(x):\n    return str(x).strip().lower()\n')
    sites = {"auth": 2, "billing": 2, "cache": 1, "export": 2, "sync": 1, "store": 1}
    for m, n in sites.items():
        add(m, "from svc.ids import normalize_id\n\n\ndef " + m + "_ids(rows):\n" + "".join(f"    a{i} = normalize_id(rows[{i}])\n" for i in range(n)) + f"    return [{', '.join(f'a{i}' for i in range(n))}]\n")
    add("utils", "# TODO: use normalize_id(x) here once the schema settles\n")
    # Q4: the function that raises QuotaExceeded
    add("quota", 'class QuotaExceeded(Exception):\n    pass\n\n\ndef consume_quota(used, limit, n):\n    if used + n > limit:\n        raise QuotaExceeded(f"{used + n} > {limit}")\n    return used + n\n')
    add("jobs", "from svc.quota import QuotaExceeded\n\n\ndef run_job(job):\n    try:\n        return job()\n    except QuotaExceeded:\n        return None\n")
    add("notify", "from svc.quota import QuotaExceeded\n\n\ndef describe(exc):\n    return isinstance(exc, QuotaExceeded)\n")
    # Q5: modules importing legacy_codec
    for m in ("billing", "export", "sync"):
        add(m, "import legacy_codec  # noqa: F401\n")
    add("parser", "# legacy_codec was removed from the parser in 2025\n")
    return files


T29_ANS = {"q1": "svc/config.py", "q2": 17, "q3": 9, "q4": "consume_quota", "q5": ["billing", "export", "sync"]}


@register("t29", "answer five questions about a 23-file codebase", "long", "B", 720,
          "{T}/svc is a small python package (23 files). Without changing any code, answer these questions and write them as one JSON object with exactly the keys q1..q5 to {T}/answers.json: "
          "q1 = the path (relative to {T}) of the one file that READS the environment variable SVC_RETRY_LIMIT (exactly that name); "
          "q2 = the default value of the timeout parameter of the function client.fetch (integer); "
          "q3 = the number of CALL sites of normalize_id( in the package, not counting its definition and not counting comments; "
          "q4 = the name of the function that raises QuotaExceeded; "
          "q5 = the sorted list of module names (file name without .py) that import legacy_codec (a real import statement, not a comment).")
class T29:
    @staticmethod
    def seed(T):
        for rel, s in t29_files().items():
            w(T, rel, s)

    @staticmethod
    def check(T):
        try:
            got = json.loads(r(T, "answers.json"))
        except Exception:  # noqa: BLE001
            return False, "answers.json missing or invalid", 0.0
        if not isinstance(got, dict):
            return False, "answers.json must be an object", 0.0
        ok = [k for k in T29_ANS if got.get(k) == T29_ANS[k] and type(got.get(k)) is type(T29_ANS[k])]
        if len(ok) == 5 and set(got) == set(T29_ANS):
            return True, "5/5 answers", 1.0
        return False, f"{len(ok)}/5 correct (wrong: {[k for k in T29_ANS if k not in ok]})", round(len(ok) / 5 * 0.99, 3)

    @staticmethod
    def solve(T):
        w(T, "answers.json", json.dumps(T29_ANS))

    @staticmethod
    def near(T):  # counts the comment and the def, takes the alias variable
        a = dict(T29_ANS)
        a.update({"q3": 11, "q1": "svc/retry.py"})
        w(T, "answers.json", json.dumps(a))


# ---------------------------------------------------------------- t30 five bugs behind a 40-test suite
T30_GOOD = {
    "lib/__init__.py": "",
    "lib/dates.py": '''import datetime


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def days_between(a, b):
    """absolute number of days between two (y, m, d) tuples."""
    return abs((datetime.date(*b) - datetime.date(*a)).days)


def days_in_month(y, m):
    if m == 2:
        return 29 if is_leap(y) else 28
    return 30 if m in (4, 6, 9, 11) else 31
''',
    "lib/wrapx.py": '''def wrap(text, width):
    """greedy word wrap: lines are at most `width` characters (a longer single word stays on its own line)."""
    lines, cur = [], ""
    for word in text.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
''',
    "lib/stats2.py": '''def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def mode(xs):
    """most common value; ties -> smallest."""
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(k for k, v in counts.items() if v == best)
''',
    "lib/urls.py": '''def join_url(base, path):
    """join with exactly one slash between base and path."""
    return base.rstrip("/") + "/" + path.lstrip("/")


def query(params):
    """build a query string with keys sorted, e.g. a=1&b=2 ('' for no params)."""
    return "&".join(f"{k}={params[k]}" for k in sorted(params))
''',
    "lib/roman.py": '''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(n):
    if not 0 < n < 4000:
        raise ValueError(n)
    out = ""
    for v, s in PAIRS:
        while n >= v:
            out += s
            n -= v
    return out


def from_roman(s):
    total, i = 0, 0
    for v, sym in PAIRS:
        while s.startswith(sym, i):
            total += v
            i += len(sym)
    return total
''',
}
T30_BUG = {
    "lib/dates.py": [("return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)", "return y % 4 == 0 and y % 100 != 0"),
                     ("return abs((datetime.date(*b) - datetime.date(*a)).days)", "return (datetime.date(*b) - datetime.date(*a)).days")],
    "lib/wrapx.py": [("<= width", "< width")],
    "lib/stats2.py": [("(s[mid - 1] + s[mid]) / 2", "s[mid - 1]")],
    "lib/urls.py": [('base.rstrip("/") + "/" + path.lstrip("/")', 'base + "/" + path')],
    "lib/roman.py": [('(900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV")',
                      '(500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V")')],
}
T30_TESTS = {
    "tests/__init__.py": "",
    "tests/test_dates.py": '''import unittest
from lib.dates import is_leap, days_between, days_in_month


class T(unittest.TestCase):
    def test_leap_plain(self):
        self.assertTrue(is_leap(2024))
        self.assertFalse(is_leap(2023))

    def test_leap_century(self):
        self.assertFalse(is_leap(1900))
        self.assertTrue(is_leap(2000))

    def test_between_forward(self):
        self.assertEqual(days_between((2026, 1, 1), (2026, 1, 31)), 30)

    def test_between_backward(self):
        self.assertEqual(days_between((2026, 3, 1), (2026, 2, 1)), 28)

    def test_between_same(self):
        self.assertEqual(days_between((2026, 5, 5), (2026, 5, 5)), 0)

    def test_feb(self):
        self.assertEqual(days_in_month(2024, 2), 29)
        self.assertEqual(days_in_month(1900, 2), 28)

    def test_thirty(self):
        self.assertEqual(days_in_month(2026, 4), 30)

    def test_thirty_one(self):
        self.assertEqual(days_in_month(2026, 12), 31)
''',
    "tests/test_wrapx.py": '''import unittest
from lib.wrapx import wrap


class T(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(wrap("the quick brown fox", 9), ["the quick", "brown fox"])

    def test_exact_fit(self):
        self.assertEqual(wrap("abc def", 7), ["abc def"])

    def test_one_over(self):
        self.assertEqual(wrap("abc def", 6), ["abc", "def"])

    def test_long_word(self):
        self.assertEqual(wrap("a incomprehensible b", 5), ["a", "incomprehensible", "b"])

    def test_empty(self):
        self.assertEqual(wrap("", 10), [])

    def test_spaces(self):
        self.assertEqual(wrap("  a   b  ", 10), ["a b"])

    def test_width_one(self):
        self.assertEqual(wrap("a b c", 1), ["a", "b", "c"])

    def test_many(self):
        self.assertEqual(wrap("aa bb cc dd ee", 8), ["aa bb cc", "dd ee"])
''',
    "tests/test_stats2.py": '''import unittest
from lib.stats2 import median, mode


class T(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_even_equal(self):
        self.assertEqual(median([5, 5]), 5)

    def test_single(self):
        self.assertEqual(median([9]), 9)

    def test_empty(self):
        with self.assertRaises(ValueError):
            median([])

    def test_mode(self):
        self.assertEqual(mode([1, 2, 2, 3]), 2)

    def test_mode_tie(self):
        self.assertEqual(mode([3, 1, 3, 1, 2]), 1)

    def test_mode_single(self):
        self.assertEqual(mode([7]), 7)
''',
    "tests/test_urls.py": '''import unittest
from lib.urls import join_url, query


class T(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(join_url("http://x", "a"), "http://x/a")

    def test_base_slash(self):
        self.assertEqual(join_url("http://x/", "a"), "http://x/a")

    def test_path_slash(self):
        self.assertEqual(join_url("http://x", "/a"), "http://x/a")

    def test_both(self):
        self.assertEqual(join_url("http://x/", "/a/b"), "http://x/a/b")

    def test_query(self):
        self.assertEqual(query({"b": 2, "a": 1}), "a=1&b=2")

    def test_query_empty(self):
        self.assertEqual(query({}), "")

    def test_query_one(self):
        self.assertEqual(query({"k": "v"}), "k=v")

    def test_many_slashes(self):
        self.assertEqual(join_url("h///", "///p"), "h/p")
''',
    "tests/test_roman.py": '''import unittest
from lib.roman import to_roman, from_roman


class T(unittest.TestCase):
    def test_small(self):
        self.assertEqual(to_roman(3), "III")

    def test_four(self):
        self.assertEqual(to_roman(4), "IV")

    def test_nine(self):
        self.assertEqual(to_roman(9), "IX")

    def test_forty(self):
        self.assertEqual(to_roman(40), "XL")

    def test_big(self):
        self.assertEqual(to_roman(1994), "MCMXCIV")

    def test_range(self):
        with self.assertRaises(ValueError):
            to_roman(0)

    def test_from(self):
        self.assertEqual(from_roman("MCMXCIV"), 1994)

    def test_roundtrip(self):
        self.assertTrue(all(from_roman(to_roman(n)) == n for n in range(1, 400)))
''',
}


def t30_buggy(rel):
    s = T30_GOOD[rel]
    for old, new in T30_BUG.get(rel, []):
        assert old in s, (rel, old)
        s = s.replace(old, new)
    return s


@register("t30", "repair five bugs behind a 40-test suite", "long", "C", 1200,
          "{T} holds a small library (lib/) with a test suite (tests/, 5 files, 40 tests). Run python3 -m unittest discover -s tests -t . from {T}: many tests fail because of bugs in lib/. "
          "Find and fix the bugs in lib/ so that all 40 tests pass. Do not edit anything under tests/. Hidden extra checks will also be run.")
class T30:
    @staticmethod
    def seed(T):
        for rel in T30_GOOD:
            w(T, rel, t30_buggy(rel))
        for rel, s in T30_TESTS.items():
            w(T, rel, s)

    @staticmethod
    def check(T):
        for rel, s in T30_TESTS.items():
            if file_sha(os.path.join(T, rel)) != hashlib.sha256(s.encode()).hexdigest():
                return False, f"{rel} was modified or removed", 0.0
        p = sh(["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."], T, timeout=120)
        import re
        m = re.search(r"Ran (\d+) tests", p.stderr)
        n = int(m.group(1)) if m else 0
        f = re.search(r"failures=(\d+)", p.stderr)
        e = re.search(r"errors=(\d+)", p.stderr)
        bad = (int(f.group(1)) if f else 0) + (int(e.group(1)) if e else 0)
        passing = max(0, n - bad) if n else 0
        d = tempfile.mkdtemp(prefix="at2l-")
        try:
            for rel, s in T30_GOOD.items():
                w(d, "good/" + rel, s)
            hid = ("import sys, importlib, json\nsys.path.insert(0, '.')\nsys.path.insert(0, %r)\n"
                   "import lib.dates as a, lib.wrapx as b, lib.stats2 as c, lib.urls as u, lib.roman as ro\n"
                   "ok = 0\n"
                   "ok += a.is_leap(2100) is False and a.is_leap(2400) is True and a.days_between((2024,2,28),(2024,3,1)) == 2 and a.days_between((2026,12,31),(2026,1,1)) == 364\n"
                   "ok += b.wrap('x yy zzz wwww', 6) == ['x yy', 'zzz', 'wwww'] and b.wrap('ab cd', 5) == ['ab cd']\n"
                   "ok += c.median([1, 2, 3, 4, 5, 6]) == 3.5 and c.median([10, 0]) == 5.0 and c.mode([2, 2, 1, 1]) == 1\n"
                   "ok += u.join_url('a', 'b') == 'a/b' and u.join_url('a//', '/b/') == 'a/b/'\n"
                   "ok += ro.to_roman(444) == 'CDXLIV' and ro.to_roman(3999) == 'MMMCMXCIX' and ro.from_roman('XLIX') == 49\n"
                   "print(ok)\n") % os.path.join(d, "good")
            h = sh(["python3", "-c", hid], T, timeout=60)
            hid_ok = h.stdout.strip().endswith("5")
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if n == 40 and bad == 0 and hid_ok:
            return True, "40/40 tests, hidden extras pass", 1.0
        return False, f"{passing}/{n or 40} tests pass, hidden {'ok' if hid_ok else 'fail'}", round(passing / 40 * 0.99, 3)

    @staticmethod
    def solve(T):
        for rel, s in T30_GOOD.items():
            w(T, rel, s)

    @staticmethod
    def near(T):  # fixes four of five bugs (the roman table stays broken)
        for rel, s in T30_GOOD.items():
            if rel != "lib/roman.py":
                w(T, rel, s)
