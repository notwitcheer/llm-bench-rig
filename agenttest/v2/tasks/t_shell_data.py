"""agenttest v2, categories shell and data (t21-t26): git from scratch, bisecting a history, a shell filter, sqlite, a csv join, ini to json."""
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from decimal import Decimal

from common import git, git_commit, git_init, lines, one_line, r, register, rm_tree, rng, sh, w

IDENT = "agenttest <agenttest@example.invalid>"


# ---------------------------------------------------------------- t21 build a small git history by hand
@register("t21", "build a git history with a branch and an annotated tag", "shell", "B", 600,
          "Create a new git repository in {T}/proj (default branch main). Make exactly three commits, in this order, each adding one file with exactly this message: "
          "(1) message `add a`, file a.txt containing the text one; (2) message `add b`, file b.txt containing two; (3) message `add c`, file c.txt containing three. "
          "Then create a branch named hotfix that points at the FIRST commit (stay on main), and an annotated tag v0.1 on the last commit with the tag message `first release`. "
          "No git identity is configured on this machine: use agenttest <agenttest@example.invalid> (set it in the repo or pass it via git -c / environment variables).")
class T21:
    @staticmethod
    def seed(T):
        os.makedirs(T, exist_ok=True)

    @staticmethod
    def check(T):
        repo = os.path.join(T, "proj")
        if not os.path.isdir(os.path.join(repo, ".git")):
            return False, "proj is not a git repository", 0.0
        g = lambda *a: git(repo, *a, check=False)  # noqa: E731
        msgs = g("log", "--reverse", "--format=%s", "main").split("\n")
        if msgs != ["add a", "add b", "add c"]:
            return False, f"commit messages {msgs}", 0.0
        revs = g("rev-list", "--reverse", "main").split()
        want = [("a.txt", "one"), ("b.txt", "two"), ("c.txt", "three")]
        for rev, (fn, content) in zip(revs, want):
            files = g("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", rev).split()
            if files != [fn] or g("show", f"{rev}:{fn}").strip() != content:
                return False, f"commit {rev[:7]} should add only {fn}={content!r}, has {files}", 0.0
        if g("rev-parse", "hotfix") != revs[0]:
            return False, "branch hotfix does not point at the first commit", 0.0
        if g("cat-file", "-t", "v0.1") != "tag":
            return False, "v0.1 is not an annotated tag", 0.0
        if g("rev-parse", "v0.1^{commit}") != revs[-1]:
            return False, "v0.1 does not point at the last commit", 0.0
        if g("tag", "-l", "--format=%(contents:subject)", "v0.1") != "first release":
            return False, "tag message is not `first release`", 0.0
        if g("rev-parse", "--abbrev-ref", "HEAD") != "main":
            return False, "not on main", 0.0
        if g("status", "--porcelain"):
            return False, "working tree not clean", 0.0
        return True, "history, branch and annotated tag as specified", 1.0

    @staticmethod
    def solve(T):
        repo = os.path.join(T, "proj")
        git_init(repo)
        for i, (fn, c, m) in enumerate((("a.txt", "one", "add a"), ("b.txt", "two", "add b"), ("c.txt", "three", "add c")), 1):
            w(repo, fn, c + "\n")
            git_commit(repo, m, i)
        first = git(repo, "rev-list", "--max-parents=0", "main")
        git(repo, "branch", "hotfix", first)
        git(repo, "tag", "-a", "v0.1", "-m", "first release")

    @staticmethod
    def near(T):  # lightweight tag, branch on the last commit
        repo = os.path.join(T, "proj")
        git_init(repo)
        for i, (fn, c, m) in enumerate((("a.txt", "one", "add a"), ("b.txt", "two", "add b"), ("c.txt", "three", "add c")), 1):
            w(repo, fn, c + "\n")
            git_commit(repo, m, i)
        git(repo, "branch", "hotfix")
        git(repo, "tag", "v0.1")


# ---------------------------------------------------------------- t22 find the commit that broke the check
def t22_build(repo):
    git_init(repo)
    w(repo, "check.sh", "#!/bin/bash\ncd \"$(dirname \"$0\")\"\npython3 -c \"import score; assert score.total(['a','b','c']) == 6, score.total(['a','b','c'])\"\n", 0o755)
    steps = [
        ("initial scoring", {"score.py": 'WEIGHTS = {"a": 1, "b": 2, "c": 3}\n\n\ndef total(items):\n    return sum(WEIGHTS[x] for x in items)\n', "README.md": "# score\n"}),
        ("add readme usage", {"README.md": "# score\n\nusage: score.total(['a', 'b'])\n"}),
        ("add changelog", {"CHANGELOG.md": "0.1 first cut\n"}),
        ("refactor score names", {"score.py": 'W = {"a": 1, "b": 2, "c": 3}\n\n\ndef total(items):\n    return sum(W[x] for x in items)\n'}),
        ("tune display weights", {"README.md": "# score\n\nusage: score.total(['a', 'b'])\ndisplay weights are cosmetic\n"}),
        ("add docstring", {"score.py": 'W = {"a": 1, "b": 2, "c": 3}\n\n\ndef total(items):\n    """sum of the weights of items."""\n    return sum(W[x] for x in items)\n'}),
        ("tune scoring weights", {"score.py": 'W = {"a": 1, "b": 3, "c": 3}\n\n\ndef total(items):\n    """sum of the weights of items."""\n    return sum(W[x] for x in items)\n'}),
        ("add license", {"LICENSE": "MIT\n"}),
        ("touch changelog", {"CHANGELOG.md": "0.1 first cut\n0.2 weights\n"}),
        ("add type comment", {"score.py": 'W = {"a": 1, "b": 3, "c": 3}  # weights per item\n\n\ndef total(items):\n    """sum of the weights of items."""\n    return sum(W[x] for x in items)\n'}),
        ("readme badge", {"README.md": "# score [ok]\n\nusage: score.total(['a', 'b'])\ndisplay weights are cosmetic\n"}),
        ("add gitignore", {".gitignore": "__pycache__/\n"}),
        ("bump version note", {"CHANGELOG.md": "0.1 first cut\n0.2 weights\n0.3 polish\n"}),
        ("final tidy", {"README.md": "# score [ok]\n\nusage: score.total(['a', 'b'])\n"}),
    ]
    culprit = None
    for n, (msg, files) in enumerate(steps, 1):
        for fn, s in files.items():
            w(repo, fn, s)
        h = git_commit(repo, msg, n)
        if msg == "tune scoring weights":
            culprit = h
    return culprit


def t22_expected():
    d = tempfile.mkdtemp(prefix="at2g-")
    try:
        return t22_build(os.path.join(d, "r"))
    finally:
        rm_tree(d)


@register("t22", "find the commit that broke a check", "shell", "C", 720,
          "{T}/repo is a git repository with 14 commits. The script check.sh (bash check.sh in the repo) passed in the early history and fails at HEAD. "
          "Find the FIRST commit at which it fails, and write its full 40-character commit hash, alone on one line, to {T}/answer.txt. Do not rewrite history.")
class T22:
    @staticmethod
    def seed(T):
        t22_build(os.path.join(T, "repo"))

    @staticmethod
    def check(T):
        s = one_line(r(T, "answer.txt"))
        if s is None:
            return False, "answer.txt missing or not one line", 0.0
        if s.strip() != t22_expected():
            return False, f"answer {s.strip()[:12]} is not the first failing commit", 0.0
        return True, "first bad commit found", 1.0

    @staticmethod
    def solve(T):
        w(T, "answer.txt", t22_expected() + "\n")

    @staticmethod
    def near(T):  # the commit that last touched score.py before HEAD
        repo = os.path.join(T, "repo")
        w(T, "answer.txt", git(repo, "log", "-1", "--format=%H", "--", "score.py") + "\n")


# ---------------------------------------------------------------- t23 shell filter
T23_SAMPLE = "category,amount\nfood,12.50\ntools,99.99\nfood,7.50\ntoys,20.00\ntools,0.01\n\ntoys,5.00\nbooks,30.00\n"
T23_IN2 = "category,amount\nz,5\na,5\nb,5\nc,1\n"
T23_IN3 = "category,amount\nsolo,3.333\nsolo,1.002\n"


def t23_ref(text):
    tot = {}
    for ln in text.split("\n")[1:]:
        if not ln.strip():
            continue
        c, a = ln.split(",")
        tot[c] = tot.get(c, Decimal(0)) + Decimal(a)
    top = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return "".join(f"{c} {v.quantize(Decimal('0.01'))}\n" for c, v in top)


@register("t23", "write a shell filter from a spec", "shell", "B", 600,
          "Write {T}/topcat.sh, a bash script that reads CSV from standard input (header line category,amount; blank lines may appear and must be ignored) and prints the three categories with the largest total amount, "
          "one per line as `<category> <total with exactly two decimals>`, largest first, ties broken alphabetically by category; if there are fewer than three categories print what exists. "
          "Example: bash topcat.sh < {T}/sample.csv must print exactly the lines of {T}/sample_expected.txt. It will be run on other inputs too. Standard tools (awk, sort, python3) are fine.")
class T23:
    @staticmethod
    def seed(T):
        w(T, "sample.csv", T23_SAMPLE)
        w(T, "sample_expected.txt", t23_ref(T23_SAMPLE))

    @staticmethod
    def check(T):
        if not os.path.isfile(os.path.join(T, "topcat.sh")):
            return False, "topcat.sh missing", 0.0
        for data in (T23_SAMPLE, T23_IN2, T23_IN3):
            try:
                p = sh(["bash", os.path.join(T, "topcat.sh")], T, inp=data, timeout=20)
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}", 0.0
            if p.stdout != t23_ref(data):
                return False, f"output {p.stdout!r} != {t23_ref(data)!r}", 0.0
        return True, "3 inputs exact", 1.0

    @staticmethod
    def solve(T):
        w(T, "topcat.sh", '#!/bin/bash\npython3 -c "\nimport sys\nfrom decimal import Decimal\nt={}\nfor ln in sys.stdin.read().split(chr(10))[1:]:\n    if ln.strip():\n        c,a=ln.split(\',\'); t[c]=t.get(c,Decimal(0))+Decimal(a)\nfor c,v in sorted(t.items(), key=lambda kv:(-kv[1],kv[0]))[:3]:\n    print(c, v.quantize(Decimal(\'0.01\')))\n"\n')

    @staticmethod
    def near(T):  # awk with float formatting but no tie rule and wrong ordering on ties
        w(T, "topcat.sh", "#!/bin/bash\nawk -F, 'NR>1 && NF==2 {t[$1]+=$2} END {for (c in t) printf \"%s %.2f\\n\", c, t[c]}' | sort -k2 -nr | head -3\n")


# ---------------------------------------------------------------- t24 sqlite questions
def t24_build_db(path):
    rg = rng("t24")
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("create table customers(id integer primary key, name text, country text);"
                      "create table products(sku text primary key, name text);"
                      "create table orders(id integer primary key, customer_id integer, status text, placed_on text);"
                      "create table items(order_id integer, sku text, qty integer, unit_cents integer);")
    names = ["Ada", "Bruno", "Chloe", "Dmitri", "Elif", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kira", "Lars"]
    for i, n in enumerate(names, 1):
        cur.execute("insert into customers values (?,?,?)", (i, n, rg.choice(["FR", "DE", "FR", "ES"])))
    skus = [f"SKU{i:02d}" for i in range(1, 13)]
    for s in skus:
        cur.execute("insert into products values (?,?)", (s, "product " + s))
    oid = 0
    for _ in range(90):
        oid += 1
        st = rg.choice(["shipped", "shipped", "shipped", "cancelled", "pending"])
        cur.execute("insert into orders values (?,?,?,?)", (oid, rg.randint(1, 12), st, f"2026-0{rg.randint(1, 4)}-{rg.randint(10, 28)}"))
        for _ in range(rg.randint(1, 3)):
            cur.execute("insert into items values (?,?,?,?)", (oid, rg.choice(skus[:9]), rg.randint(1, 4), rg.choice([250, 990, 1500, 4000])))
    cur.execute("update items set sku='SKU07' where sku='SKU08' and order_id in (select id from orders where status<>'cancelled')")
    cur.execute("insert into orders values (91, 3, 'cancelled', '2026-02-11')")
    cur.execute("insert into items values (91, 'SKU08', 2, 990)")
    con.commit()
    con.close()


def t24_expected():
    d = tempfile.mkdtemp(prefix="at2d-")
    try:
        p = os.path.join(d, "x.db")
        t24_build_db(p)
        c = sqlite3.connect(p)
        q1 = c.execute("select count(distinct c.id) from customers c join orders o on o.customer_id=c.id where c.country='FR' and o.status='shipped'").fetchone()[0]
        q2 = c.execute("select coalesce(sum(i.qty*i.unit_cents),0) from items i join orders o on o.id=i.order_id where o.status='shipped' and o.placed_on like '2026-03%'").fetchone()[0]
        rows = c.execute("select c.name, sum(i.qty*i.unit_cents) t from customers c join orders o on o.customer_id=c.id join items i on i.order_id=o.id where o.status='shipped' group by c.id order by t desc").fetchall()
        assert rows[0][1] != rows[1][1]
        q3 = rows[0][0]
        q4 = [x[0] for x in c.execute("select p.sku from products p where p.sku not in (select i.sku from items i join orders o on o.id=i.order_id where o.status<>'cancelled') order by p.sku")]
        c.close()
        return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
    finally:
        rm_tree(d)


@register("t24", "answer four questions from a sqlite database", "data", "B", 600,
          "{T}/shop.db is a sqlite3 database (tables customers, products, orders, items; amounts in cents, an item's revenue is qty * unit_cents). Only the python3 sqlite3 module may be available (no sqlite3 command). "
          "Answer, and write them as one JSON object with exactly the keys q1..q4 to {T}/answers.json: "
          "q1 = number of distinct customers from country FR that have at least one order with status shipped (integer); "
          "q2 = total revenue in cents of shipped orders placed in March 2026, placed_on starts with 2026-03 (integer); "
          "q3 = name of the customer with the highest total revenue over shipped orders (string); "
          "q4 = list of product skus that appear in no non-cancelled order (orders with status cancelled do not count, a product never ordered at all counts), sorted ascending.")
class T24:
    @staticmethod
    def seed(T):
        os.makedirs(T, exist_ok=True)
        t24_build_db(os.path.join(T, "shop.db"))

    @staticmethod
    def check(T):
        s = r(T, "answers.json")
        try:
            got = json.loads(s)
        except Exception:  # noqa: BLE001
            return False, "answers.json missing or invalid", 0.0
        exp = t24_expected()
        if not isinstance(got, dict):
            return False, "answers.json must be an object", 0.0
        ok = [k for k in exp if got.get(k) == exp[k] and type(got.get(k)) is type(exp[k])]
        if len(ok) == 4 and set(got) == set(exp):
            return True, "q1-q4 correct", 1.0
        return False, f"{len(ok)}/4 correct ({[k for k in exp if k not in ok]} wrong)", round(len(ok) / 4 * 0.99, 3)

    @staticmethod
    def solve(T):
        w(T, "answers.json", json.dumps(t24_expected()))

    @staticmethod
    def near(T):  # q4 counts only products never ordered at all (ignores the cancelled rule)
        e = t24_expected()
        c = sqlite3.connect(os.path.join(T, "shop.db"))
        e["q4"] = [x[0] for x in c.execute("select sku from products where sku not in (select sku from items) order by sku")]
        c.close()
        w(T, "answers.json", json.dumps(e))


# ---------------------------------------------------------------- t25 csv join with rules
def t25_data():
    rg = rng("t25")
    cust = [(f"C{i:02d}", n) for i, n in enumerate(["Ada Lin", "Bo Chen", "Cy Park", "Di Rao", "Eli Roy", "Fay Qin", "Gus Ode"], 1)]
    cust.append(("C03", "Cyrus Park"))  # duplicate id: the LAST row wins
    orders = []
    for i in range(1, 61):
        cid = rg.choice([c for c, _ in cust] + ["C99", " C02", "C05 "])
        orders.append((i, cid, f"{rg.uniform(3, 300):.2f}", rg.choice(["paid", "paid", "cancelled", "shipped"])))
    return cust, orders


def t25_expected():
    cust, orders = t25_data()
    names = {}
    for c, n in cust:
        names[c] = n
    tot = {}
    for _i, cid, amt, st in orders:
        if st == "cancelled":
            continue
        nm = names.get(cid.strip(), "UNKNOWN")
        tot[nm] = tot.get(nm, Decimal(0)) + Decimal(amt)
    rows = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))
    b = io.StringIO()
    cw = csv.writer(b, lineterminator="\n")
    cw.writerow(["name", "total"])
    for n, v in rows:
        cw.writerow([n, str(v.quantize(Decimal("0.01")))])
    return b.getvalue()


@register("t25", "join two csv files with rules", "data", "B", 720,
          "{T}/orders.csv (order_id,customer_id,amount,status) and {T}/customers.csv (customer_id,name) are given. Write {T}/totals.csv with header name,total: one row per customer name with the sum of amount over their orders, "
          "skipping orders whose status is cancelled. Rules: ignore spaces around customer ids when matching; if customers.csv lists an id twice, the LAST row wins; orders whose customer is not in customers.csv are grouped under the name UNKNOWN; "
          "totals have exactly two decimals; sort by total descending, then name ascending.")
class T25:
    @staticmethod
    def seed(T):
        cust, orders = t25_data()
        w(T, "customers.csv", "customer_id,name\n" + "".join(f"{c},{n}\n" for c, n in cust))
        w(T, "orders.csv", "order_id,customer_id,amount,status\n" + "".join(f"{i},{c},{a},{s}\n" for i, c, a, s in orders))

    @staticmethod
    def check(T):
        s = r(T, "totals.csv")
        if s is None:
            return False, "totals.csv missing", 0.0
        exp = t25_expected()
        if s.replace("\r\n", "\n") != exp:
            got, want = s.replace("\r\n", "\n").split("\n"), exp.split("\n")
            n = sum(1 for a, b in zip(got, want) if a == b)
            return False, f"{n}/{len(want) - 1} lines match; first diff {[x for x in zip(got, want) if x[0] != x[1]][:1]}", 0.0
        return True, "totals.csv exact", 1.0

    @staticmethod
    def solve(T):
        w(T, "totals.csv", t25_expected())

    @staticmethod
    def near(T):  # forgets the trim rule and the last-wins rule
        cust, orders = t25_data()
        names = {}
        for c, n in cust:
            names.setdefault(c, n)
        tot = {}
        for _i, cid, amt, st in orders:
            if st == "cancelled":
                continue
            nm = names.get(cid, "UNKNOWN")
            tot[nm] = tot.get(nm, Decimal(0)) + Decimal(amt)
        rows = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))
        w(T, "totals.csv", "name,total\n" + "".join(f"{n},{v.quantize(Decimal('0.01'))}\n" for n, v in rows))


# ---------------------------------------------------------------- t26 ini to json
T26_INI = """; sample config
[server]
port = 8080
host = localhost
debug = yes
zip = 02134
offset = -5
version = 1.10

[paths]
root = /srv/app
logs = ${paths.root}/logs
cache = ${paths.logs}/cache

[limits]
max_conn = 250
ratio = 0.75
allowed_list = a, b , c
empty_list =
enabled = Off
# trailing comment
"""
T26_OTHER = """[a]
n = 3
flag = TRUE
mode = fast
dirs_list = x,y
neg = -0.5
v = 007
base = /opt
full = ${a.base}/bin
[b]
n = -12
off = no
name = hello world
ref = ${a.full}/tool
"""


def t26_ref(text):
    secs, cur = {}, None
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or s[0] in ";#":
            continue
        if s.startswith("["):
            cur = s[1:-1]
            secs[cur] = {}
            continue
        k, _, v = s.partition("=")
        secs[cur][k.strip()] = v.strip()

    def interp(v):
        for _ in range(5):
            v2 = re.sub(r"\$\{(\w+)\.(\w+)\}", lambda m: secs[m.group(1)][m.group(2)], v)
            if v2 == v:
                break
            v = v2
        return v

    out = {}
    for sec, kv in secs.items():
        out[sec] = {}
        for k, v in kv.items():
            v = interp(v)
            if k.endswith("_list"):
                out[sec][k] = [x.strip() for x in v.split(",") if x.strip()]
            elif re.fullmatch(r"-?(0|[1-9]\d*)", v):
                out[sec][k] = int(v)
            elif re.fullmatch(r"-?(0|[1-9]\d*)\.\d+", v):
                out[sec][k] = float(v)
            elif v.lower() in ("yes", "true", "on"):
                out[sec][k] = True
            elif v.lower() in ("no", "false", "off"):
                out[sec][k] = False
            else:
                out[sec][k] = v
    return out


@register("t26", "convert ini to typed json with interpolation", "data", "B", 720,
          "Write {T}/ini2json.py (python3 standard library; usage: python3 ini2json.py IN.ini OUT.json) converting an ini file to JSON, and run it on {T}/app.ini to produce {T}/app.json. Rules: each section becomes an object; "
          "blank lines and lines starting with ; or # are ignored; a value that is an integer (optional minus sign, digits, NO leading zero unless it is exactly 0) becomes a JSON integer; "
          "a decimal like 0.75 or -0.5 becomes a float; yes/true/on and no/false/off in any letter case become true/false; keys whose name ends in _list become a list of trimmed strings (empty items dropped); "
          "everything else stays a string (so 02134 stays a string); ${section.key} inside a value is replaced by that other value (which may itself contain references). It will be run on other ini files too.")
class T26:
    @staticmethod
    def seed(T):
        w(T, "app.ini", T26_INI)

    @staticmethod
    def check(T):
        if not os.path.isfile(os.path.join(T, "ini2json.py")):
            return False, "ini2json.py missing", 0.0
        try:
            if json.loads(r(T, "app.json")) != t26_ref(T26_INI):
                return False, "app.json differs from the expected typed conversion", 0.0
        except Exception:  # noqa: BLE001
            return False, "app.json missing or invalid", 0.0
        w(T, "_o.ini", T26_OTHER)
        p = sh(["python3", "ini2json.py", "_o.ini", "_o.json"], T, timeout=20)
        ok = p.returncode == 0
        try:
            ok = ok and json.loads(r(T, "_o.json")) == t26_ref(T26_OTHER)
        except Exception:  # noqa: BLE001
            ok = False
        for f in ("_o.ini", "_o.json"):
            if os.path.exists(os.path.join(T, f)):
                os.remove(os.path.join(T, f))
        if not ok:
            return False, "the script fails on the hidden ini", 0.0
        return True, "seed and hidden ini convert correctly", 1.0

    @staticmethod
    def solve(T):
        w(T, "ini2json.py", '''import json
import re
import sys

secs, cur = {}, None
for ln in open(sys.argv[1]).read().split("\\n"):
    s = ln.strip()
    if not s or s[0] in ";#":
        continue
    if s.startswith("["):
        cur = s[1:-1]
        secs[cur] = {}
        continue
    k, _, v = s.partition("=")
    secs[cur][k.strip()] = v.strip()


def interp(v):
    for _ in range(5):
        v2 = re.sub(r"\\$\\{(\\w+)\\.(\\w+)\\}", lambda m: secs[m.group(1)][m.group(2)], v)
        if v2 == v:
            break
        v = v2
    return v


out = {}
for sec, kv in secs.items():
    out[sec] = {}
    for k, v in kv.items():
        v = interp(v)
        if k.endswith("_list"):
            out[sec][k] = [x.strip() for x in v.split(",") if x.strip()]
        elif re.fullmatch(r"-?(0|[1-9]\\d*)", v):
            out[sec][k] = int(v)
        elif re.fullmatch(r"-?(0|[1-9]\\d*)\\.\\d+", v):
            out[sec][k] = float(v)
        elif v.lower() in ("yes", "true", "on"):
            out[sec][k] = True
        elif v.lower() in ("no", "false", "off"):
            out[sec][k] = False
        else:
            out[sec][k] = v
json.dump(out, open(sys.argv[2], "w"), indent=2)
''')
        subprocess.run(["python3", "ini2json.py", "app.ini", "app.json"], cwd=T, check=True, capture_output=True)

    @staticmethod
    def near(T):  # the typed output for the seed, written by hand, plus a script that only copies it
        w(T, "app.json", json.dumps(t26_ref(T26_INI)))
        w(T, "ini2json.py", "import shutil, sys\nshutil.copy('app.json', sys.argv[2])\n")
