"""agenttest v2, category refactor (t12-t16): multi-file edits, signature changes, writing tests, import errors."""
import ast
import hashlib
import os
import shutil
import tempfile

from common import file_sha, py, r, register, rm_tree, sh, w

# ---------------------------------------------------------------- t12 rename a function across a package
T12_FILES = {
    "shop/__init__.py": "",
    "shop/pricing.py": '''"""pricing rules."""


def calc_total(order):
    """total of an order dict: sum(price * qty) with tax added, rounded to cents. calc_total is the public entry point."""
    sub = sum(i["price"] * i["qty"] for i in order["items"])
    return round(sub * (1 + order.get("tax", 0.0)), 2)
''',
    "shop/cart.py": '''from .pricing import calc_total


class Cart:
    def __init__(self, tax=0.0):
        self.items = []
        self.tax = tax

    def add(self, price, qty=1):
        self.items.append({"price": price, "qty": qty})

    def total(self):
        return calc_total({"items": self.items, "tax": self.tax})
''',
    "shop/invoice.py": '''from .pricing import calc_total as ct


def invoice_line(order):
    return "TOTAL %.2f" % ct(order)
''',
    "shop/report.py": '''from . import pricing


def totals(orders):
    return [pricing.calc_total(o) for o in orders]


def grand_total(orders):
    # sums the values returned by calc_total for every order
    return round(sum(totals(orders)), 2)
''',
    "shop/registry.py": '''from . import pricing

HANDLERS = {"calc_total": "pricing"}


def dispatch(name, *args):
    """call a pricing function by its registered name."""
    if name not in HANDLERS:
        raise KeyError(name)
    return getattr(pricing, name)(*args)
''',
    "shop/cli.py": '''from .registry import dispatch


def main(argv):
    order = {"items": [{"price": float(p), "qty": 1} for p in argv], "tax": 0.0}
    return dispatch("calc_total", order)
''',
    "tests/__init__.py": "",
    "tests/test_shop.py": '''import unittest

from shop.pricing import calc_total
from shop.cart import Cart
from shop import report, cli
from shop.registry import dispatch
from shop.invoice import invoice_line

O = {"items": [{"price": 10.0, "qty": 2}, {"price": 5.0, "qty": 1}], "tax": 0.2}


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(calc_total(O), 30.0)

    def test_cart(self):
        c = Cart(0.1)
        c.add(10.0, 3)
        self.assertEqual(c.total(), 33.0)

    def test_report(self):
        self.assertEqual(report.grand_total([O, O]), 60.0)

    def test_dispatch_and_cli(self):
        self.assertEqual(dispatch("calc_total", O), 30.0)
        self.assertEqual(cli.main(["1", "2.5"]), 3.5)

    def test_invoice(self):
        self.assertEqual(invoice_line(O), "TOTAL 30.00")


if __name__ == "__main__":
    unittest.main()
''',
    "README.md": "# shop\n\nUse `calc_total(order)` from `shop.pricing` to price an order. The CLI dispatches to `calc_total` by name.\n",
}


@register("t12", "rename a function across a package", "refactor", "B", 720,
          "The project in {T} is a small package (shop/, tests/, README.md). Rename the function calc_total to compute_invoice_total everywhere: its definition, every import and call (including aliased imports), "
          "the name string used for dispatch by name, comments and docs, and the tests. When you are done no occurrence of calc_total may remain in any .py or .md file, and "
          "python3 -m unittest discover -s tests -t . (run from {T}) must still pass. Do not delete or weaken any test.")
class T12:
    @staticmethod
    def seed(T):
        for rel, s in T12_FILES.items():
            w(T, rel, s)

    @staticmethod
    def check(T):
        left = []
        for dp, _dn, fns in os.walk(T):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                if fn.endswith((".py", ".md")):
                    t = open(os.path.join(dp, fn), encoding="utf-8").read()
                    if "calc_total" in t:
                        left.append(os.path.relpath(os.path.join(dp, fn), T))
        if left:
            return False, f"calc_total still in {left}", 0.0
        tests = r(T, "tests/test_shop.py") or ""
        if tests.count("def test_") < 5:
            return False, "tests deleted or weakened (fewer than 5 test methods)", 0.0
        p = sh(["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."], T)
        if p.returncode != 0:
            return False, f"unittest failed: {p.stderr.strip()[-200:]}", 0.0
        hid = ("from shop.pricing import compute_invoice_total as c; from shop.registry import dispatch; from shop import cli; import shop.invoice, shop.report, shop.cart\n"
               "o={'items':[{'price':4.0,'qty':3}],'tax':0.5}\nassert c(o)==18.0\nassert dispatch('compute_invoice_total', o)==18.0\nassert cli.main(['2','3'])==5.0\nprint('ok')")
        h = sh(["python3", "-c", hid], T)
        if h.returncode != 0:
            return False, f"hidden check failed: {h.stderr.strip()[-200:]}", 0.0
        return True, "no old name left, tests pass, new name dispatches", 1.0

    @staticmethod
    def solve(T):
        for dp, _dn, fns in os.walk(T):
            for fn in fns:
                if fn.endswith((".py", ".md")):
                    p = os.path.join(dp, fn)
                    s = open(p, encoding="utf-8").read()
                    open(p, "w", encoding="utf-8").write(s.replace("calc_total", "compute_invoice_total"))

    @staticmethod
    def near(T):  # renames definition, imports and calls but misses the dispatch string and the docs
        for rel in ("shop/pricing.py", "shop/cart.py", "shop/invoice.py", "shop/report.py", "tests/test_shop.py"):
            s = r(T, rel).replace("calc_total", "compute_invoice_total")
            w(T, rel, s)


# ---------------------------------------------------------------- t13 split a module
T13_UTILS = '''"""misc helpers (one big module that should be two)."""


def title_case_words(s):
    return " ".join(wd[:1].upper() + wd[1:].lower() for wd in s.split())


def truncate(s, n):
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."


def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")


def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def percent(part, whole):
    return 0.0 if whole == 0 else round(100.0 * part / whole, 1)


def gcd_many(*nums):
    from math import gcd
    g = 0
    for n in nums:
        g = gcd(g, n)
    return g
'''
T13_TEST = '''import unittest
import utils


class T(unittest.TestCase):
    def test_strings(self):
        self.assertEqual(utils.title_case_words("hello wORLD"), "Hello World")
        self.assertEqual(utils.truncate("abcdefghij", 8), "abcde...")
        self.assertEqual(utils.count_vowels("Education"), 5)
        self.assertTrue(utils.is_palindrome("A man, a plan, a canal: Panama"))

    def test_numbers(self):
        self.assertEqual(utils.clamp(15, 0, 10), 10)
        self.assertEqual(utils.mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(utils.percent(1, 3), 33.3)
        self.assertEqual(utils.gcd_many(12, 18, 24), 6)


if __name__ == "__main__":
    unittest.main()
'''
STR_FNS = {"title_case_words", "truncate", "count_vowels", "is_palindrome"}
NUM_FNS = {"clamp", "mean", "percent", "gcd_many"}


def defs(path):
    try:
        t = ast.parse(open(path, encoding="utf-8").read())
    except Exception:  # noqa: BLE001
        return None
    return {n.name for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}


@register("t13", "split a module in two, keep the old imports working", "refactor", "C", 900,
          "{T}/utils.py mixes string helpers and number helpers. Split it: move title_case_words, truncate, count_vowels and is_palindrome into a new strutil.py, "
          "and clamp, mean, percent and gcd_many into a new numutil.py (the functions must be DEFINED in the new modules). Keep utils.py working for old code: "
          "it must still expose all eight names (import them from the new modules, do not copy the code). {T}/test_utils.py must keep passing unchanged (python3 -m unittest in {T}).")
class T13:
    @staticmethod
    def seed(T):
        w(T, "utils.py", T13_UTILS)
        w(T, "test_utils.py", T13_TEST)

    @staticmethod
    def check(T):
        if file_sha(os.path.join(T, "test_utils.py")) != hashlib.sha256(T13_TEST.encode()).hexdigest():
            return False, "test_utils.py modified", 0.0
        ds, dn, du = defs(os.path.join(T, "strutil.py")), defs(os.path.join(T, "numutil.py")), defs(os.path.join(T, "utils.py"))
        if ds is None or dn is None or du is None:
            return False, "strutil.py / numutil.py missing or not parseable", 0.0
        if not STR_FNS <= ds:
            return False, f"strutil lacks {sorted(STR_FNS - ds)}", 0.0
        if not NUM_FNS <= dn:
            return False, f"numutil lacks {sorted(NUM_FNS - dn)}", 0.0
        if (STR_FNS | NUM_FNS) & du:
            return False, f"utils.py still defines {sorted((STR_FNS | NUM_FNS) & du)}", 0.0
        p = sh(["python3", "-m", "unittest", "-q"], T)
        if p.returncode != 0:
            return False, f"tests fail: {p.stderr.strip()[-200:]}", 0.0
        hid = ("import utils, strutil, numutil\nfor n in %r: assert getattr(utils, n) is getattr(strutil, n), n\n"
               "for n in %r: assert getattr(utils, n) is getattr(numutil, n), n\n"
               "assert numutil.percent(0, 0) == 0.0 and numutil.clamp(-1, 0, 5) == 0 and strutil.truncate('ab', 5) == 'ab'\nprint('ok')") % (sorted(STR_FNS), sorted(NUM_FNS))
        h = sh(["python3", "-c", hid], T)
        if h.returncode != 0:
            return False, f"re-export identity failed: {h.stderr.strip()[-200:]}", 0.0
        return True, "split done, old imports work, same function objects", 1.0

    @staticmethod
    def solve(T):
        src = T13_UTILS.split("\n\n\n")
        head, fns = src[0], src[1:]
        by = {}
        for f in fns:
            name = f.split("(")[0].replace("def ", "").strip()
            by[name] = f.rstrip("\n")
        w(T, "strutil.py", '"""string helpers."""\n\n\n' + "\n\n\n".join(by[n] for n in ("title_case_words", "truncate", "count_vowels", "is_palindrome")) + "\n")
        w(T, "numutil.py", '"""number helpers."""\n\n\n' + "\n\n\n".join(by[n] for n in ("clamp", "mean", "percent", "gcd_many")) + "\n")
        w(T, "utils.py", '"""compat: everything lives in strutil and numutil now."""\nfrom strutil import title_case_words, truncate, count_vowels, is_palindrome  # noqa: F401\n'
          'from numutil import clamp, mean, percent, gcd_many  # noqa: F401\n')

    @staticmethod
    def near(T):  # copies the code into the new modules but leaves utils.py as it was
        T13.solve(T)
        w(T, "utils.py", T13_UTILS)


# ---------------------------------------------------------------- t14 a signature change reaches nine call sites
T14_FILES = {
    "money.py": '''def fmt_money(amount):
    """format an amount for display."""
    return "%.2f EUR" % amount
''',
    "orders.py": '''from money import fmt_money


def order_lines(order):
    cur = order["currency"]
    lines = []
    for it in order["items"]:
        lines.append("%s x%d = %s" % (it["name"], it["qty"], fmt_money(it["price"] * it["qty"])))
    lines.append("shipping: " + fmt_money(order["shipping"]))
    lines.append("order total (%s): " % cur + fmt_money(sum(i["price"] * i["qty"] for i in order["items"]) + order["shipping"]))
    return lines
''',
    "reports.py": '''from money import fmt_money

REPORT_CURRENCY = "GBP"


def daily_report(rows):
    out = ["best day: " + fmt_money(max(rows))]
    out.append("worst day: " + fmt_money(min(rows)))
    out.append("sum: " + fmt_money(sum(rows)))
    return out
''',
    "invoice.py": '''from money import fmt_money


class Invoice:
    def __init__(self, net, cur):
        self.net = net
        self.cur = cur

    def lines(self):
        return ["net " + fmt_money(self.net), "vat " + fmt_money(self.net * 0.2)]
''',
    "cli.py": '''import argparse

from money import fmt_money


def run(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("amount", type=float)
    ap.add_argument("--currency", default="CHF")
    a = ap.parse_args(argv)
    return "pay " + fmt_money(a.amount)
''',
    "demo.py": '''import cli
from invoice import Invoice
from orders import order_lines
from reports import daily_report

order = {"currency": "USD", "shipping": 4.5, "items": [{"name": "pen", "qty": 3, "price": 1.25}, {"name": "ink", "qty": 1, "price": 8.0}]}
for line in order_lines(order):
    print(line)
for line in daily_report([10.0, 25.5, 7.25]):
    print(line)
for line in Invoice(200.0, "SEK").lines():
    print(line)
print(cli.run(["12.5", "--currency", "JPY"]))
''',
}
T14_EXPECTED = """pen x3 = 3.75 USD
ink x1 = 8.00 USD
shipping: 4.50 USD
order total (USD): 16.25 USD
best day: 25.50 GBP
worst day: 7.25 GBP
sum: 42.75 GBP
net 200.00 SEK
vat 40.00 SEK
pay 12.50 JPY
"""


@register("t14", "add a required parameter to nine call sites", "refactor", "C", 900,
          "In {T}, money.fmt_money(amount) always prints EUR. Change it to fmt_money(amount, currency): both parameters required (no default), output \"<amount with 2 decimals> <currency>\". "
          "Then update every caller so each passes the currency that belongs to its context (the order's currency, the report's REPORT_CURRENCY, the invoice's cur, the CLI's --currency). "
          "python3 demo.py (run in {T}) must then print exactly the lines in {T}/expected_demo.txt.")
class T14:
    @staticmethod
    def seed(T):
        for rel, s in T14_FILES.items():
            w(T, rel, s)
        w(T, "expected_demo.txt", T14_EXPECTED)

    @staticmethod
    def check(T):
        p = sh(["python3", "demo.py"], T)
        if p.returncode != 0:
            return False, f"demo.py failed: {p.stderr.strip()[-200:]}", 0.0
        if p.stdout != T14_EXPECTED:
            return False, "demo output differs: " + repr(p.stdout[:160]), 0.0
        sig = py(T, "import inspect, money; s = inspect.signature(money.fmt_money); print([(k, v.default is inspect._empty) for k, v in s.parameters.items()])")
        if sig.stdout.strip() != "[('amount', True), ('currency', True)]":
            return False, f"signature must be (amount, currency) both required: {sig.stdout.strip()}", 0.0
        bad = 0
        for fn in ("orders.py", "reports.py", "invoice.py", "cli.py"):
            tree = ast.parse(r(T, fn) or "")
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "fmt_money" and len(n.args) + len(n.keywords) != 2:
                    bad += 1
        if bad:
            return False, f"{bad} call(s) without two arguments", 0.0
        return True, "all nine sites updated, output exact", 1.0

    @staticmethod
    def solve(T):
        w(T, "money.py", 'def fmt_money(amount, currency):\n    """format an amount for display."""\n    return "%.2f %s" % (amount, currency)\n')
        s = r(T, "orders.py").replace("fmt_money(it[\"price\"] * it[\"qty\"])", "fmt_money(it[\"price\"] * it[\"qty\"], cur)") \
            .replace("fmt_money(order[\"shipping\"])", "fmt_money(order[\"shipping\"], cur)") \
            .replace("+ order[\"shipping\"])", "+ order[\"shipping\"], cur)")
        w(T, "orders.py", s)
        s = r(T, "reports.py").replace("fmt_money(max(rows))", "fmt_money(max(rows), REPORT_CURRENCY)").replace("fmt_money(min(rows))", "fmt_money(min(rows), REPORT_CURRENCY)") \
            .replace("fmt_money(sum(rows))", "fmt_money(sum(rows), REPORT_CURRENCY)")
        w(T, "reports.py", s)
        s = r(T, "invoice.py").replace("fmt_money(self.net)", "fmt_money(self.net, self.cur)").replace("fmt_money(self.net * 0.2)", "fmt_money(self.net * 0.2, self.cur)")
        w(T, "invoice.py", s)
        w(T, "cli.py", r(T, "cli.py").replace("fmt_money(a.amount)", "fmt_money(a.amount, a.currency)"))

    @staticmethod
    def near(T):  # the lazy fix: a default currency keeps every caller running
        w(T, "money.py", 'def fmt_money(amount, currency="EUR"):\n    return "%.2f %s" % (amount, currency)\n')


# ---------------------------------------------------------------- t15 write tests that catch mutants
T15_CART = '''"""a tiny shopping cart. prices are integer cents."""


class Cart:
    def __init__(self):
        self.items = {}

    def add(self, sku, price_cents, qty=1):
        if qty <= 0:
            raise ValueError("qty must be positive")
        if sku in self.items:
            p, q = self.items[sku]
            if p != price_cents:
                raise ValueError("price mismatch")
            self.items[sku] = (p, q + qty)
        else:
            self.items[sku] = (price_cents, qty)

    def remove(self, sku, qty=None):
        if sku not in self.items:
            raise KeyError(sku)
        p, q = self.items[sku]
        if qty is None or qty >= q:
            del self.items[sku]
        else:
            self.items[sku] = (p, q - qty)

    def subtotal(self):
        return sum(p * q for p, q in self.items.values())

    def total(self, coupon=None):
        s = self.subtotal()
        if coupon == "TEN":
            s = s - s // 10
        elif coupon == "FIVEOFF":
            s = max(0, s - 500)
        elif coupon is not None:
            raise ValueError("unknown coupon")
        return s
'''
T15_MUTANTS = [("qty <= 0", "qty < 0"), ("qty >= q", "qty > q"), ("s - s // 10", "s - s // 100"), ("max(0, s - 500)", "s - 500"), ("if p != price_cents:", "if False:")]


@register("t15", "write tests that catch five hidden bugs", "refactor", "C", 900,
          "{T}/cart.py is a small shopping cart (read its code; prices are integer cents). Write {T}/test_cart.py using the standard unittest module with at least 6 test methods that "
          "pass on the current cart.py and are strong enough to catch subtle bugs. Hidden mutated copies of cart.py (an off-by-one in a comparison, a wrong discount, a missing validation, ...) will be run against your tests: "
          "a mutant counts as caught only if your tests fail on it. Do not edit cart.py. Run your tests (python3 -m unittest in {T}) before you finish.")
class T15:
    @staticmethod
    def seed(T):
        w(T, "cart.py", T15_CART)

    @staticmethod
    def check(T):
        if file_sha(os.path.join(T, "cart.py")) != hashlib.sha256(T15_CART.encode()).hexdigest():
            return False, "cart.py modified", 0.0
        t = r(T, "test_cart.py")
        if t is None:
            return False, "test_cart.py missing", 0.0
        p = sh(["python3", "-m", "unittest", "-v", "test_cart"], T, timeout=60)
        n = p.stderr.count(" ... ok")
        if p.returncode != 0 or n < 6:
            return False, f"tests must pass on the original with >= 6 methods (ok={n}, rc={p.returncode}) {p.stderr.strip()[-150:]}", 0.0
        killed = 0
        for old, new in T15_MUTANTS:
            d = tempfile.mkdtemp(prefix="at2m-")
            try:
                assert old in T15_CART
                open(os.path.join(d, "cart.py"), "w").write(T15_CART.replace(old, new, 1))
                shutil.copy(os.path.join(T, "test_cart.py"), d)
                try:
                    q = sh(["python3", "-m", "unittest", "test_cart"], d, timeout=30)
                    killed += q.returncode != 0
                except Exception:  # noqa: BLE001
                    killed += 1
            finally:
                shutil.rmtree(d, ignore_errors=True)
        frac = killed / len(T15_MUTANTS)
        if killed == len(T15_MUTANTS):
            return True, f"{n} tests pass; caught {killed}/5 mutants", 1.0
        return False, f"{n} tests pass; caught {killed}/5 mutants", round(frac * 0.99, 3)

    @staticmethod
    def solve(T):
        w(T, "test_cart.py", '''import unittest
from cart import Cart


class T(unittest.TestCase):
    def test_add_merges(self):
        c = Cart(); c.add("a", 100, 2); c.add("a", 100, 3)
        self.assertEqual(c.subtotal(), 500)

    def test_add_zero_qty_rejected(self):
        with self.assertRaises(ValueError):
            Cart().add("a", 100, 0)

    def test_add_price_mismatch(self):
        c = Cart(); c.add("a", 100)
        with self.assertRaises(ValueError):
            c.add("a", 120)

    def test_remove_exact_qty_deletes(self):
        c = Cart(); c.add("a", 100, 2); c.remove("a", 2)
        self.assertEqual(c.items, {})

    def test_remove_partial(self):
        c = Cart(); c.add("a", 100, 3); c.remove("a", 1)
        self.assertEqual(c.subtotal(), 200)

    def test_ten_percent(self):
        c = Cart(); c.add("a", 1000)
        self.assertEqual(c.total("TEN"), 900)

    def test_fiveoff_floor(self):
        c = Cart(); c.add("a", 300)
        self.assertEqual(c.total("FIVEOFF"), 0)

    def test_unknown_coupon(self):
        with self.assertRaises(ValueError):
            Cart().total("NOPE")


if __name__ == "__main__":
    unittest.main()
''')

    @staticmethod
    def near(T):  # six passing tests that never probe a boundary
        w(T, "test_cart.py", '''import unittest
from cart import Cart


class T(unittest.TestCase):
    def test_a(self):
        c = Cart(); c.add("a", 100); self.assertEqual(c.subtotal(), 100)

    def test_b(self):
        c = Cart(); c.add("a", 100, 2); self.assertEqual(c.subtotal(), 200)

    def test_c(self):
        c = Cart(); c.add("a", 100, 3); c.remove("a", 1); self.assertEqual(c.subtotal(), 200)

    def test_d(self):
        c = Cart(); c.add("a", 100); c.add("b", 50); self.assertEqual(c.subtotal(), 150)

    def test_e(self):
        c = Cart(); c.add("a", 1000); self.assertEqual(c.total(), 1000)

    def test_f(self):
        with self.assertRaises(KeyError):
            Cart().remove("zzz")
''')


# ---------------------------------------------------------------- t16 import errors in a package
T16_FILES = {
    "inventory/__init__.py": "",
    "inventory/models.py": '''from .store import DEFAULT_PATH


class Item:
    def __init__(self, sku, name, qty):
        self.sku, self.name, self.qty = sku, name, qty

    def source(self):
        return DEFAULT_PATH
''',
    "inventory/store.py": '''import json

from .models import Item

DEFAULT_PATH = "data.json"


def load(path=DEFAULT_PATH):
    return [Item(r["sku"], r["name"], r["qty"]) for r in json.load(open(path))]
''',
    "inventory/fmt.py": '''def format_row(item):
    return "%-6s %-10s %3d" % (item.sku, item.name, item.qty)
''',
    "inventory/main.py": '''import sys

from models import Item
from . import store
from .fmt import fmt_row


def main(argv):
    path = argv[0] if argv else store.DEFAULT_PATH
    items = store.load(path)
    for it in sorted(items, key=lambda i: i.sku):
        print(fmt_row(it))
    print("items: %d, units: %d" % (len(items), sum(i.qty for i in items)))


if __name__ == "__main__":
    main(sys.argv[1:])
''',
    "data.json": '[{"sku": "B2", "name": "bolt", "qty": 40}, {"sku": "A1", "name": "anchor", "qty": 5}, {"sku": "C3", "name": "clamp", "qty": 12}]\n',
}


def t16_expected(rows):
    out = ["%-6s %-10s %3d" % (r_["sku"], r_["name"], r_["qty"]) for r_ in sorted(rows, key=lambda x: x["sku"])]
    out.append("items: %d, units: %d" % (len(rows), sum(r_["qty"] for r_ in rows)))
    return "\n".join(out) + "\n"


@register("t16", "fix broken imports in a package", "refactor", "B", 600,
          "In {T} the package inventory does not run: python3 -m inventory.main (from {T}) crashes with import errors. Fix the package (the code in inventory/) so that the command prints "
          "the three item rows sorted by sku followed by the totals line, using data.json. It will also be run with another data file as argument. Do not change data.json.")
class T16:
    @staticmethod
    def seed(T):
        for rel, s in T16_FILES.items():
            w(T, rel, s)

    @staticmethod
    def check(T):
        import json
        base = json.loads(T16_FILES["data.json"])
        p = sh(["python3", "-m", "inventory.main"], T)
        if p.returncode != 0:
            return False, f"crash: {p.stderr.strip()[-200:]}", 0.0
        if p.stdout != t16_expected(base):
            return False, "output differs: " + repr(p.stdout[:120]), 0.0
        other = [{"sku": "Z9", "name": "zip", "qty": 3}, {"sku": "D4", "name": "dowel", "qty": 100}]
        w(T, "_other.json", json.dumps(other))
        q = sh(["python3", "-m", "inventory.main", "_other.json"], T)
        os.remove(os.path.join(T, "_other.json"))
        if q.returncode != 0 or q.stdout != t16_expected(other):
            return False, "hidden data file run differs", 0.0
        if file_sha(os.path.join(T, "data.json")) != hashlib.sha256(T16_FILES["data.json"].encode()).hexdigest():
            return False, "data.json modified", 0.0
        return True, "runs on the seed and on another data file", 1.0

    @staticmethod
    def solve(T):
        w(T, "inventory/models.py", '''class Item:
    def __init__(self, sku, name, qty):
        self.sku, self.name, self.qty = sku, name, qty

    def source(self):
        return "data.json"
''')
        s = r(T, "inventory/main.py").replace("from models import Item\n", "").replace("fmt_row", "format_row")
        w(T, "inventory/main.py", s)

    @staticmethod
    def near(T):  # hard-codes the seed output
        w(T, "inventory/main.py", "import sys\nprint(open('data.json').read() and %r, end='')\n" % t16_expected(__import__("json").loads(T16_FILES["data.json"])))
