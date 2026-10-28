"""agenttest v2, category files (t01-t06): plain file and terminal work with tight, checkable outputs."""
import json
import os
import re

from common import lines, one_line, r, register, rng, w

WORDS = ("alder birch cedar delta ember falcon granite harbor indigo juniper kestrel lantern meadow nectar orchid pebble quartz "
         "ripple saffron timber umber velvet willow xenon yarrow zephyr anchor bridge copper dagger engine forest glacier").split()


def filler(rg, n):
    return "\n".join(" ".join(rg.choice(WORDS) for _ in range(rg.randint(5, 9))) for _ in range(n)) + "\n"


# ---------------------------------------------------------------- t01
T01_LINES = ["release: v2.4.1", "  indented by two spaces", "quotes 'single' and \"double\" stay as typed"]


@register("t01", "exact three-line file", "files", "A", 240,
          "Create the folder {T}/out and, inside it, the file notes.txt containing exactly these three lines (the second line starts with two spaces, keep them):\n"
          + "\n".join(T01_LINES) + "\nThen read the file back to confirm and tell me it is done.")
class T01:
    @staticmethod
    def seed(T):
        os.makedirs(T, exist_ok=True)

    @staticmethod
    def check(T):
        s = r(T, "out/notes.txt")
        if s is None:
            return False, "out/notes.txt missing", 0.0
        got = lines(s[:-1] if s.endswith("\n") else s)
        if got != T01_LINES:
            return False, f"content {got!r}", 0.0
        return True, "exact three lines", 1.0

    @staticmethod
    def solve(T):
        w(T, "out/notes.txt", "\n".join(T01_LINES) + "\n")

    @staticmethod
    def near(T):
        w(T, "out/notes.txt", "\n".join(x.strip() for x in T01_LINES) + "\n")


# ---------------------------------------------------------------- t02
def t02_rows():
    rg = rng("t02")
    out = []
    for i in range(1, 241):
        reg = rg.choice(["north", "south", "east", "west"])
        st = rg.choice(["paid", "paid", "pending", "refunded", "Paid", "paid "])
        amt = round(rg.uniform(5, 900), 2)
        out.append((i, reg, st, amt, f"2026-0{rg.randint(1, 9)}-{rg.randint(10, 28)}"))
    return out


def t02_expected():
    return round(sum(a for _i, reg, st, a, _d in t02_rows() if reg == "north" and st == "paid"), 2)


@register("t02", "csv filtered total", "files", "A", 300,
          "The file {T}/sales.csv has a header row (id,region,status,amount,date). Compute the total of the amount column over the rows whose region is exactly north AND whose status is exactly paid "
          "(lowercase, no surrounding spaces; rows like Paid or 'paid ' do not count). Write just that number with two decimals (no other text) to {T}/total.txt.")
class T02:
    @staticmethod
    def seed(T):
        w(T, "sales.csv", "id,region,status,amount,date\n" + "\n".join(f"{i},{reg},{st},{a:.2f},{d}" for i, reg, st, a, d in t02_rows()) + "\n")

    @staticmethod
    def check(T):
        s = one_line(r(T, "total.txt"))
        if s is None:
            return False, "total.txt missing or not one line", 0.0
        if not re.fullmatch(r"\d+\.\d{2}", s.strip()):
            return False, f"not a two-decimal number: {s!r}", 0.0
        if abs(float(s) - t02_expected()) > 0.005:
            return False, f"total {s} != {t02_expected():.2f}", 0.0
        return True, "total correct", 1.0

    @staticmethod
    def solve(T):
        w(T, "total.txt", f"{t02_expected():.2f}\n")

    @staticmethod
    def near(T):
        v = round(sum(a for _i, reg, st, a, _d in t02_rows() if reg == "north" and st.strip().lower() == "paid"), 2)
        w(T, "total.txt", f"{v:.2f}\n")


# ---------------------------------------------------------------- t03
def t03_files():
    rg = rng("t03")
    dirs = ["docs/ops", "docs/dev", "docs/legal", "archive/2024", "archive/2025", "notes"]
    out = {}
    k = 0
    for d in dirs:
        for j in range(7):
            k += 1
            out[f"{d}/file_{k:02d}.txt"] = filler(rg, rg.randint(40, 70))
    keys = sorted(out)
    target = keys[23]
    ls = out[target].split("\n")
    ls[36] = "TODO(rotate-key) replace the signing key before the audit"
    out[target] = "\n".join(ls)
    d1, d2 = keys[5], keys[31]
    out[d1] = out[d1].replace("\n", "\nTODO(rotate-keys) plural, not the one\n", 1)
    out[d2] = out[d2].replace("\n", "\nplease rotate-key soon\n", 1)
    return out, target, 37


@register("t03", "find a marker line in a tree", "files", "A", 300,
          "Under {T}/tree there are 42 text files in nested folders. Exactly one of them contains a line with the exact marker TODO(rotate-key) (singular; TODO(rotate-keys) and plain 'rotate-key' are decoys). "
          "Write to {T}/answer.txt one line of the form <path relative to {T}/tree>:<line number, counting from 1>, for example docs/x/file_01.txt:12.")
class T03:
    @staticmethod
    def seed(T):
        for p, s in t03_files()[0].items():
            w(T, "tree/" + p, s)

    @staticmethod
    def check(T):
        _f, target, ln = t03_files()
        s = one_line(r(T, "answer.txt"))
        if s is None:
            return False, "answer.txt missing or not one line", 0.0
        if s.strip() != f"{target}:{ln}":
            return False, f"answer {s!r} != {target}:{ln}", 0.0
        return True, "path:line correct", 1.0

    @staticmethod
    def solve(T):
        _f, target, ln = t03_files()
        w(T, "answer.txt", f"{target}:{ln}\n")

    @staticmethod
    def near(T):
        _f, target, ln = t03_files()
        w(T, "answer.txt", f"{T}/tree/{target}:{ln}\n")


# ---------------------------------------------------------------- t04
def t04_log():
    rg = rng("t04")
    ips = sorted({f"10.0.{rg.randint(0, 3)}.{rg.randint(1, 250)}" for _ in range(26)})
    heavy = set(rg.sample(ips, 6))
    light = {ip: 0 for ip in rg.sample([i for i in ips if i not in heavy], 6)}
    rows = []
    for i in range(800):
        ip = rg.choice(ips)
        if ip in heavy and rg.random() < 0.18:
            st = rg.choice([500, 502, 503])
        elif ip in light and light[ip] < 2 and rg.random() < 0.2:
            st = rg.choice([500, 504])
            light[ip] += 1
        else:
            st = rg.choice([200, 200, 200, 204, 301, 404, 404, 403])
        path = rg.choice(["/", "/api/items", "/api/items/500", "/login", "/static/app.js", "/health"])
        size = rg.choice([512, 500, 1500, 233, 5000])
        rows.append((ip, st, f'{ip} - - [12/Oct/2026:10:{i // 60 % 60:02d}:{i % 60:02d} +0000] "GET {path} HTTP/1.1" {st} {size}'))
    return rows


def t04_expected(minimum=3):
    cnt = {}
    for ip, st, _l in t04_log():
        if 500 <= st <= 599:
            cnt[ip] = cnt.get(ip, 0) + 1
    return sorted(ip for ip, c in cnt.items() if c >= minimum), sorted(ip for ip, c in cnt.items() if c >= 1)


@register("t04", "log analysis: noisy clients", "files", "B", 420,
          "{T}/access.log is a web server access log (client ip first, HTTP status code right after the quoted request). List every client IP address that received at least 3 responses with a 5xx status "
          "(500-599). Write the IPs to {T}/ips.txt, one per line, sorted in plain text order (as `sort` with LC_ALL=C would), nothing else in the file. Careful: other numbers like 500 can appear in paths and sizes.")
class T04:
    @staticmethod
    def seed(T):
        w(T, "access.log", "\n".join(x[2] for x in t04_log()) + "\n")

    @staticmethod
    def check(T):
        s = r(T, "ips.txt")
        if s is None:
            return False, "ips.txt missing", 0.0
        got = [x.strip() for x in s.strip().split("\n") if x.strip()]
        exp = t04_expected()[0]
        if got != exp:
            return False, f"got {len(got)} ips, expected {len(exp)}: diff {sorted(set(got) ^ set(exp))[:4]}", 0.0
        return True, f"{len(exp)} ips correct", 1.0

    @staticmethod
    def solve(T):
        w(T, "ips.txt", "\n".join(t04_expected()[0]) + "\n")

    @staticmethod
    def near(T):
        w(T, "ips.txt", "\n".join(t04_expected(1)[1]) + "\n")


# ---------------------------------------------------------------- t05
def t05_data():
    rg = rng("t05")
    svc = []
    for i in range(12):
        s = {"name": f"svc-{i:02d}", "tier": rg.choice(["gold", "silver", "bronze", "gold"]), "replicas": rg.randint(1, 5),
             "env": {"LOG": rg.choice(["info", "warn"]), "REGION": rg.choice(["eu", "us"])}}
        if rg.random() < 0.5:
            s["deprecated"] = rg.choice([True, "yes"])
        svc.append(s)
    return {"version": 3, "services": svc}


def t05_expected():
    d = t05_data()
    for s in d["services"]:
        s.pop("deprecated", None)
        if s["tier"] == "gold":
            s["replicas"] = max(s["replicas"], 3)
            s["monitored"] = True
    return d


@register("t05", "bulk JSON edit with rules", "files", "B", 420,
          "Edit {T}/services.json in place, keeping it valid JSON and keeping every other key and value unchanged: (1) for every service whose tier is gold, raise replicas to at least 3 (never lower it) and add the key \"monitored\" with the value true; "
          "(2) remove the key \"deprecated\" from every service, whatever its tier.")
class T05:
    @staticmethod
    def seed(T):
        w(T, "services.json", json.dumps(t05_data(), indent=2) + "\n")

    @staticmethod
    def check(T):
        s = r(T, "services.json")
        try:
            got = json.loads(s)
        except Exception as e:  # noqa: BLE001
            return False, f"invalid json: {e}", 0.0
        exp = t05_expected()
        if got != exp:
            bad = [a["name"] for a, b in zip(got.get("services", []), exp["services"]) if a != b]
            return False, f"differs from expected for {bad[:5]}", 0.0
        return True, "all rules applied", 1.0

    @staticmethod
    def solve(T):
        w(T, "services.json", json.dumps(t05_expected(), indent=2) + "\n")

    @staticmethod
    def near(T):
        d = t05_expected()
        for s in t05_data()["services"]:
            pass
        d2 = t05_data()
        for s in d2["services"]:
            if s["tier"] == "gold":
                s["replicas"] = max(s["replicas"], 3)
                s["monitored"] = True  # deprecated keys left in place
        w(T, "services.json", json.dumps(d2, indent=2) + "\n")


# ---------------------------------------------------------------- t06
def t06_notes():
    rg = rng("t06")
    facts = ["Launch date is fixed for 14 november, screenshots are due 3 november.",
             "The approved budget for this quarter is 48,000 euros (last quarter's 41,500 euros is closed).",
             "Priya owns the database migration to the new cluster, Marco owns the vendor contract."]
    out = []
    for i in range(40):
        out.append(filler(rg, 1).strip() + f" (item {i + 1})")
    for k, f in zip((6, 19, 31), facts):
        out[k] = f
    return "\n".join(out) + "\n"


@register("t06", "40 notes to 3 bullets", "files", "A", 300,
          "Read {T}/notes.txt (40 lines of meeting notes). Write a summary to {T}/summary.txt that is exactly 3 lines, each line a bullet starting with \"- \", no blank lines and nothing else in the file. "
          "The summary must mention the launch date, the approved budget figure for this quarter, and the owner of the database migration.")
class T06:
    @staticmethod
    def seed(T):
        w(T, "notes.txt", t06_notes())

    @staticmethod
    def check(T):
        s = r(T, "summary.txt")
        if s is None:
            return False, "summary.txt missing", 0.0
        ls = s.replace("\r\n", "\n").rstrip("\n").split("\n")
        if len(ls) != 3 or any(not x.startswith("- ") or not x[2:].strip() for x in ls):
            return False, f"not exactly 3 non-empty '- ' lines ({len(ls)} lines)", 0.0
        t = s.lower()
        miss = []
        if not re.search(r"14 nov|nov(ember)? 14", t):
            miss.append("launch date")
        if not re.search(r"48,?000", t):
            miss.append("budget 48,000")
        if "priya" not in t:
            miss.append("owner priya")
        if re.search(r"41,?500", t):
            miss.append("used the closed budget")
        if miss:
            return False, "problems: " + ", ".join(miss), 0.0
        return True, "3 bullets with all three facts", 1.0

    @staticmethod
    def solve(T):
        w(T, "summary.txt", "- launch date is 14 november.\n- the approved budget this quarter is 48,000 euros.\n- priya owns the database migration.\n")

    @staticmethod
    def near(T):
        w(T, "summary.txt", "- launch date is 14 november.\n- the budget is 41,500 euros.\n- priya owns the database migration.\n")
