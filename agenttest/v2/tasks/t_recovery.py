"""agenttest v2, category recovery (t17-t20): scripts that fail on purpose, stale state, a git merge conflict, bad input rows."""
import csv
import hashlib
import io
import os
import subprocess

from common import GIT_ENV, file_sha, git, git_commit, git_init, lines, one_line, r, register, rm_tree, sh, w

# ---------------------------------------------------------------- t17 two environment faults, script must not be edited
T17_SCRIPT = '''#!/bin/bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
IN="$HERE/data/input.txt"
OUT="$HERE/out"
[ -d "$OUT" ] || { echo "error: output directory $OUT does not exist" >&2; exit 1; }
[ -r "$IN" ] || { echo "error: cannot read $IN (permission denied)" >&2; exit 1; }
n=$(wc -l < "$IN")
echo "report ok $n" > "$OUT/report.txt"
'''
T17_INPUT = "".join(f"record {i}\n" for i in range(1, 43))


@register("t17", "recover from two environment faults", "recovery", "B", 480,
          "Run the script {T}/run_report.sh with bash. If it fails, work out why, fix the environment (do NOT edit the script itself) and run it again until it succeeds. "
          "It should produce {T}/out/report.txt.")
class T17:
    @staticmethod
    def seed(T):
        w(T, "run_report.sh", T17_SCRIPT, 0o755)
        w(T, "data/input.txt", T17_INPUT, 0o644)
        os.chmod(os.path.join(T, "data/input.txt"), 0o000)

    @staticmethod
    def check(T):
        if file_sha(os.path.join(T, "run_report.sh")) != hashlib.sha256(T17_SCRIPT.encode()).hexdigest():
            return False, "run_report.sh was modified", 0.0
        s = r(T, "out/report.txt")
        if one_line(s) != "report ok 42":
            return False, f"out/report.txt = {s!r}", 0.0
        return True, "report produced, script unchanged", 1.0

    @staticmethod
    def solve(T):
        os.makedirs(os.path.join(T, "out"), exist_ok=True)
        os.chmod(os.path.join(T, "data/input.txt"), 0o644)
        subprocess.run(["bash", os.path.join(T, "run_report.sh")], check=True, capture_output=True)

    @staticmethod
    def near(T):  # fixes the directory, edits the script to skip the read check instead of fixing the permission
        os.makedirs(os.path.join(T, "out"), exist_ok=True)
        s = r(T, "run_report.sh").replace('[ -r "$IN" ] ||', 'true ||')
        w(T, "run_report.sh", s)
        os.chmod(os.path.join(T, "data/input.txt"), 0o644)
        subprocess.run(["bash", os.path.join(T, "run_report.sh")], capture_output=True)


# ---------------------------------------------------------------- t18 stale lock + missing generated config
T18_BUILD = '''#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f .build.lock ]; then
  pid=$(cat .build.lock)
  if kill -0 "$pid" 2>/dev/null; then echo "build: another build (pid $pid) is running" >&2; exit 3
  else echo "build: stale lock from pid $pid (not running), remove .build.lock to continue" >&2; exit 4; fi
fi
[ -f config.local ] || { echo "build: config.local missing, run ./gen_config.sh first" >&2; exit 2; }
. ./config.local
mkdir -p dist
sum=$(cat src/*.txt | sha256sum | cut -c1-12)
echo "app $VERSION built from $sum" > dist/app.txt
echo "build ok"
'''
T18_GEN = '''#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "VERSION=$(cat "$HERE/VERSION")" > "$HERE/config.local"
echo "wrote config.local"
'''
T18_SRC = {"src/a.txt": "alpha module\n", "src/b.txt": "beta module\nsecond line\n"}


def t18_expected():
    s = hashlib.sha256(("alpha module\n" + "beta module\nsecond line\n").encode()).hexdigest()[:12]
    return f"app 1.4.2 built from {s}"


@register("t18", "build blocked by a stale lock and a missing config", "recovery", "C", 720,
          "In {T}, running bash build.sh fails. Get the build to succeed so that dist/app.txt is produced, by fixing the situation the error messages describe: "
          "do NOT edit build.sh or gen_config.sh, and do not fake the output file. (Generated files and stale lock files are fair game.)")
class T18:
    @staticmethod
    def seed(T):
        w(T, "build.sh", T18_BUILD, 0o755)
        w(T, "gen_config.sh", T18_GEN, 0o755)
        w(T, "VERSION", "1.4.2\n")
        for rel, s in T18_SRC.items():
            w(T, rel, s)
        w(T, ".build.lock", "999999\n")

    @staticmethod
    def check(T):
        for fn, src in (("build.sh", T18_BUILD), ("gen_config.sh", T18_GEN)):
            if file_sha(os.path.join(T, fn)) != hashlib.sha256(src.encode()).hexdigest():
                return False, f"{fn} was modified", 0.0
        if os.path.exists(os.path.join(T, ".build.lock")):
            return False, "stale lock still present", 0.0
        if not os.path.isfile(os.path.join(T, "config.local")):
            return False, "config.local not generated", 0.0
        s = one_line(r(T, "dist/app.txt"))
        if s != t18_expected():
            return False, f"dist/app.txt = {s!r}", 0.0
        return True, "lock cleared, config generated, build output exact", 1.0

    @staticmethod
    def solve(T):
        os.remove(os.path.join(T, ".build.lock"))
        subprocess.run(["bash", os.path.join(T, "gen_config.sh")], check=True, capture_output=True)
        subprocess.run(["bash", os.path.join(T, "build.sh")], check=True, capture_output=True)

    @staticmethod
    def near(T):  # hand-writes the output file
        w(T, "dist/app.txt", "app 1.4.2 built from 000000000000\n")


# ---------------------------------------------------------------- t19 a real git merge conflict
T19 = {
    "base": {"config.ini": "[net]\ntimeout = 30\nretries = 3\n[log]\nlevel = info\n",
             "calc.py": "def price(x):\n    return x * 2\n",
             "README.md": "# calc\n",
             "test_calc.py": "import calc\nassert calc.price(3) == 7\nassert calc.price_with_tax(3) == 8.4\nassert calc.price_rounded(3.14159) == 3.14\nprint('calc ok')\n"},
    "main": {"config.ini": "[net]\ntimeout = 60\nretries = 3\n[log]\nlevel = warn\n",
             "calc.py": "def price(x):\n    return x * 2 + 1\n\n\ndef price_rounded(x):\n    return round(x, 2)\n"},
    "feature": {"config.ini": "[net]\ntimeout = 45\nretries = 5\n[log]\nlevel = info\n",
                "calc.py": "def price(x):\n    return x * 2\n\n\ndef price_with_tax(x):\n    return round(price(x) * 1.2, 2)\n",
                "README.md": "# calc\n\nfeature: tax helper added.\n"},
}


def t19_build(repo):
    git_init(repo)
    for fn, s in T19["base"].items():
        w(repo, fn, s)
    git_commit(repo, "base", 1)
    git(repo, "checkout", "-q", "-b", "feature")
    for fn, s in T19["feature"].items():
        w(repo, fn, s)
    git_commit(repo, "feature: tax helper, more retries", 2)
    git(repo, "checkout", "-q", "main")
    for fn, s in T19["main"].items():
        w(repo, fn, s)
    git_commit(repo, "main: price offset, rounding helper, warn logs", 3)
    git(repo, "config", "user.name", "agenttest")
    git(repo, "config", "user.email", "agenttest@example.invalid")


@register("t19", "resolve a real git merge conflict", "recovery", "C", 900,
          "{T}/repo is a git repository on branch main with a second branch called feature. Merge feature into main. The merge conflicts: resolve it so that "
          "config.ini keeps main's timeout value and feature's retries value (and main's log level), and calc.py keeps main's version of price plus BOTH new functions "
          "(price_rounded from main and price_with_tax from feature); README.md keeps feature's addition. Finish with a normal merge commit (a git identity is already configured inside the repo), "
          "leave the working tree clean, and make sure python3 test_calc.py passes in the repo.")
class T19C:
    @staticmethod
    def seed(T):
        t19_build(os.path.join(T, "repo"))

    @staticmethod
    def check(T):
        repo = os.path.join(T, "repo")
        if not os.path.isdir(os.path.join(repo, ".git")):
            return False, "repo missing", 0.0
        parents = git(repo, "rev-list", "--parents", "-n", "1", "HEAD", check=False).split()
        if len(parents) != 3:
            return False, f"HEAD is not a merge commit ({len(parents) - 1} parent(s))", 0.0
        if git(repo, "status", "--porcelain", check=False):
            return False, "working tree not clean", 0.0
        if git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False) != "main":
            return False, "not on main", 0.0
        for fn in ("config.ini", "calc.py", "README.md"):
            t = r(repo, fn) or ""
            if "<<<<<<<" in t or ">>>>>>>" in t or "=======" in t:
                return False, f"conflict markers left in {fn}", 0.0
        cfg = r(repo, "config.ini") or ""
        if "timeout = 60" not in cfg or "retries = 5" not in cfg or "level = warn" not in cfg or "timeout = 45" in cfg:
            return False, f"config.ini wrong: {cfg!r}", 0.0
        if "feature: tax helper added." not in (r(repo, "README.md") or ""):
            return False, "README lost the feature line", 0.0
        p = sh(["python3", "test_calc.py"], repo)
        if p.returncode != 0 or "calc ok" not in p.stdout:
            return False, f"test_calc fails: {p.stderr.strip()[-160:]}", 0.0
        names = git(repo, "branch", "--format=%(refname:short)", "--merged", "HEAD").split()
        if "feature" not in names:
            return False, "feature is not merged into HEAD", 0.0
        return True, "merge commit with 2 parents, resolved as specified, tests pass", 1.0

    @staticmethod
    def solve(T):
        repo = os.path.join(T, "repo")
        git(repo, "merge", "--no-commit", "feature", check=False)
        w(repo, "config.ini", "[net]\ntimeout = 60\nretries = 5\n[log]\nlevel = warn\n")
        w(repo, "calc.py", "def price(x):\n    return x * 2 + 1\n\n\ndef price_rounded(x):\n    return round(x, 2)\n\n\ndef price_with_tax(x):\n    return round(price(x) * 1.2, 2)\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "merge feature")

    @staticmethod
    def near(T):  # commits the conflict markers as they are
        repo = os.path.join(T, "repo")
        git(repo, "merge", "--no-commit", "feature", check=False)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "merge feature")


# ---------------------------------------------------------------- t20 crash on bad rows: split good and rejected
T20_SCRIPT = '''import csv
import sys


def main(inp, out, rejects):
    with open(inp) as f, open(out, "w", newline="") as o:
        wr = csv.writer(o)
        wr.writerow(["id", "name", "qty", "price_cents"])
        for row in csv.DictReader(f):
            wr.writerow([int(row["id"]), row["name"].strip().title(), int(row["qty"]), int(round(float(row["price"]) * 100))])


if __name__ == "__main__":
    main(*sys.argv[1:4])
'''
T20_IN = """id,name,qty,price
1, bolt ,40,0.29
2,anchor,5,12.5
x3,clamp,12,3.10
4,,7,1.00
5,washer,seven,0.05
6,nut,100,
7,rivet,8,1.15
8,pin,3,abc
9,hinge,2,19.99
"""


def t20_ref(text):
    good, bad = [], []
    for row in csv.DictReader(io.StringIO(text)):
        reason = None
        try:
            int(row["id"])
        except ValueError:
            reason = "bad_id"
        if reason is None and not row["name"].strip():
            reason = "empty_name"
        if reason is None:
            try:
                int(row["qty"])
            except ValueError:
                reason = "bad_qty"
        if reason is None:
            try:
                float(row["price"])
            except ValueError:
                reason = "bad_price"
        if reason:
            bad.append([row["id"], row["name"], row["qty"], row["price"], reason])
        else:
            from decimal import Decimal, ROUND_HALF_UP
            good.append([str(int(row["id"])), row["name"].strip().title(), str(int(row["qty"])),
                         str(int((Decimal(row["price"]) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))])

    def dump(rows, head):
        b = io.StringIO()
        cw = csv.writer(b, lineterminator="\n")
        cw.writerow(head)
        cw.writerows(rows)
        return b.getvalue()
    return dump(good, ["id", "name", "qty", "price_cents"]), dump(bad, ["id", "name", "qty", "price", "reason"])


T20_OTHER = "id,name,qty,price\n10, gear ,4,2.30\n11,cog,1,0.29\nq,belt,2,5.00\n12,chain,,4.40\n13,sprocket,6,1.99\n14,\" \",1,1.00\n15,link,3,0.58\n"


@register("t20", "make a migration survive bad rows", "recovery", "B", 600,
          "{T}/migrate.py converts a csv: python3 migrate.py IN OUT REJECTS. On {T}/orders_in.csv it crashes. Change migrate.py so that it never crashes on a bad row: good rows go to OUT exactly as now "
          "(price_cents = price in euros times 100, rounded to a whole number), and each bad row goes to REJECTS, a csv with header id,name,qty,price,reason holding the ORIGINAL four values and a reason. "
          "A row is bad when: id is not an integer (reason bad_id), else name is empty after trimming (empty_name), else qty is not an integer (bad_qty), else price is not a number (bad_price). "
          "Row order is kept in both files. Exit code 0. Run it on orders_in.csv to produce {T}/out.csv and {T}/rejects.csv. It will also be run on another input file.")
class T20:
    @staticmethod
    def seed(T):
        w(T, "migrate.py", T20_SCRIPT)
        w(T, "orders_in.csv", T20_IN)

    @staticmethod
    def check(T):
        ok = 0
        for name, data in (("orders_in.csv", T20_IN), ("_other.csv", T20_OTHER)):
            if name == "_other.csv":
                w(T, name, data)
            outp, rej = os.path.join(T, "_o.csv"), os.path.join(T, "_r.csv")
            for p_ in (outp, rej):
                if os.path.exists(p_):
                    os.remove(p_)
            try:
                p = sh(["python3", "migrate.py", name, outp, rej], T, timeout=30)
            except Exception:  # noqa: BLE001
                continue
            exp_out, exp_rej = t20_ref(data)
            if p.returncode == 0 and r(T, "_o.csv") == exp_out and r(T, "_r.csv") == exp_rej:
                ok += 1
            for p_ in (outp, rej, os.path.join(T, "_other.csv")):
                if os.path.exists(p_):
                    os.remove(p_)
        if ok != 2:
            return False, f"{ok}/2 inputs produce the expected out/rejects", 0.0
        eo, er = t20_ref(T20_IN)
        if r(T, "out.csv") != eo or r(T, "rejects.csv") != er:
            return False, "out.csv / rejects.csv in the task folder are missing or differ", 0.0
        return True, "script handles seed and hidden input; deliverable files exact", 1.0

    @staticmethod
    def solve(T):
        w(T, "migrate.py", '''import csv
import sys
from decimal import Decimal, ROUND_HALF_UP


def main(inp, out, rejects):
    with open(inp) as f, open(out, "w", newline="") as o, open(rejects, "w", newline="") as rj:
        wr, rw = csv.writer(o, lineterminator="\\n"), csv.writer(rj, lineterminator="\\n")
        wr.writerow(["id", "name", "qty", "price_cents"])
        rw.writerow(["id", "name", "qty", "price", "reason"])
        for row in csv.DictReader(f):
            reason = None
            try:
                int(row["id"])
            except ValueError:
                reason = "bad_id"
            if reason is None and not row["name"].strip():
                reason = "empty_name"
            if reason is None:
                try:
                    int(row["qty"])
                except ValueError:
                    reason = "bad_qty"
            if reason is None:
                try:
                    float(row["price"])
                except ValueError:
                    reason = "bad_price"
            if reason:
                rw.writerow([row["id"], row["name"], row["qty"], row["price"], reason])
            else:
                wr.writerow([int(row["id"]), row["name"].strip().title(), int(row["qty"]),
                             int((Decimal(row["price"]) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))])


if __name__ == "__main__":
    main(*sys.argv[1:4])
''')
        subprocess.run(["python3", "migrate.py", "orders_in.csv", "out.csv", "rejects.csv"], cwd=T, check=True, capture_output=True)

    @staticmethod
    def near(T):  # try/except that drops bad rows silently, no rejects file
        s = T20_SCRIPT.replace("            wr.writerow([int", "            try:\n                wr.writerow([int").replace("* 100))])", "* 100))])\n            except Exception:\n                pass")
        w(T, "migrate.py", s)
        subprocess.run(["python3", "migrate.py", "orders_in.csv", "out.csv", "rejects.csv"], cwd=T, capture_output=True)
