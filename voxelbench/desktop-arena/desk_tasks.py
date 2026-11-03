#!/usr/bin/env python3
"""desk_tasks.py: the 8 desktop-arena tasks. Each task = prompt + seed(c) + score(c) + hand(c).
Everything here runs from the HARNESS through docker exec (desk_lib.dx); the agent only ever sees the desktop through computer_use. A task is scored from the
resulting FILE STATE (or xfconf value) read in the container after the agent ended: the gate sits outside the agent. hand(c) is the reference solution done
with shell commands (used only by `desk_tasks.py --prove` to show that a scorer passes a correct end state and fails the fresh seed).
Container user pn, HOME /home/pn. Apps are PRE-OPENED by the seed (the agent is not asked to find a launcher); the desktop is Xfce (xfwm4, no compositor).
"""
import base64
import json
import sys
import time

import desk_lib as L

CHROME = "/opt/playwright/chromium-1248/chrome-linux64/chrome"
H = "/home/pn"

FORM1 = """<!doctype html><html><head><meta charset="utf-8"><title>Signup form</title></head><body style="font:20px sans-serif;margin:40px">
<h2>Signup form</h2><form method="post" action="/submit/form1">
<p><label>Name <input type="text" name="name" size="30"></label></p>
<p><label>Colour <select name="colour"><option value="">choose</option><option>red</option><option>green</option><option>blue</option></select></label></p>
<p><label><input type="checkbox" name="subscribe"> subscribe</label></p>
<p><button type="submit">Submit</button></p></form></body></html>"""
FORM2 = """<!doctype html><html><head><meta charset="utf-8"><title>Ticket form</title></head><body style="font:20px sans-serif;margin:40px">
<h2>Ticket form</h2><form method="post" action="/submit/form2">
<p><label>Code <input type="text" name="code" size="30"></label></p>
<p><button type="submit">Submit</button></p></form></body></html>"""
SERVER = """import http.server, json, os, urllib.parse
ROOT = "/home/pn/site"; OUT = "/home/pn/out"; os.makedirs(OUT, exist_ok=True)
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n).decode()
        d = {k: v[0] for k, v in urllib.parse.parse_qs(body, keep_blank_values=True).items()}
        page = self.path.rsplit("/", 1)[-1]
        json.dump(d, open(os.path.join(OUT, page + ".json"), "w"))
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
        self.wfile.write(b"<html><body style='font:24px sans-serif;margin:40px'><h2>Thanks, received.</h2></body></html>")
http.server.ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
"""
# comm names of every process a task may leave behind (matched by awk on `ps`, never by a name-kill tool)
LEFTOVERS = "mousepad|thunar|gnumeric|xfce4-appearanc|xfce4-terminal|chrome|python3"


def sh(c, cmd, user="pn", env=True):
    return L.dx(c, cmd, user=user, env=env)


def put(c, path, text, user="pn"):
    b = base64.b64encode(text.encode()).decode()
    return sh(c, "mkdir -p $(dirname %s) && echo %s | base64 -d > %s" % (path, b, path), user=user)


def reset(c):
    """wipe every task artefact and end every app a task opened (container-local; the desktop itself is restarted by the runner)."""
    sh(c, "ps -u pn -o pid=,comm= | awk '$2 ~ /^(%s)$/ {print $1}' | xargs -r kill -9; true" % LEFTOVERS, env=False)
    sh(c, "rm -rf %s/Documents %s/inbox %s/site %s/out %s/count.txt /tmp/cprof; mkdir -p %s/Documents" % ((H,) * 6), env=False)


def app(c, cmd):
    return sh(c, "cd %s && (%s >/dev/null 2>&1 &)" % (H, cmd))


def web_stack(c):
    put(c, H + "/site/form1.html", FORM1)
    put(c, H + "/site/form2.html", FORM2)
    put(c, H + "/site/server.py", SERVER)
    sh(c, "cd %s/site && (setsid python3 server.py >/dev/null 2>&1 &)" % H)
    time.sleep(1.0)


def chrome(c, url):
    app(c, "%s --no-sandbox --user-data-dir=/tmp/cprof --no-first-run --no-default-browser-check --force-renderer-accessibility --window-position=60,60 --window-size=1280,800 %s" % (CHROME, url))


# ---- seeds -------------------------------------------------------------------------------------------------------------------------------------------
def seed_t1(c):
    app(c, "mousepad")


def seed_t2(c):
    put(c, H + "/Documents/draft.txt", "quarterly numbers\n")
    put(c, H + "/Documents/other.txt", "keep me\n")
    put(c, H + "/Documents/data.csv", "a,b\n1,2\n")
    app(c, "thunar %s/Documents" % H)


def seed_t3(c):
    put(c, H + "/Documents/old.log", "line one\nline two\n")
    put(c, H + "/Documents/keep.txt", "stay here\n")
    app(c, "thunar %s/Documents" % H)


def seed_t4(c):
    sh(c, "xfconf-query -c xsettings -p /Xft/Antialias -n -t int -s 1 || xfconf-query -c xsettings -p /Xft/Antialias -s 1")
    app(c, "xfce4-appearance-settings")


def seed_t5(c):
    web_stack(c)
    chrome(c, "http://127.0.0.1:8000/form1.html")


def seed_t6(c):
    put(c, "/tmp/sales.csv", "12,30,8\n5,7,9\n")
    sh(c, "ssconvert /tmp/sales.csv %s/Documents/sales.gnumeric >/dev/null 2>&1" % H)
    app(c, "gnumeric %s/Documents/sales.gnumeric" % H)


def seed_t7(c):
    sh(c, "mkdir -p %s/inbox/sub1 %s/inbox/sub2; for n in a b c d e f g; do echo $n > %s/inbox/file-$n.txt; done" % (H, H, H))
    app(c, "xfce4-terminal --geometry=100x30 --working-directory=%s" % H)


def seed_t8(c):
    put(c, H + "/Documents/ticket.txt", "Support ticket\nTicket code: K7-ALPHA-4821\nPriority: low\n")
    web_stack(c)
    app(c, "mousepad %s/Documents/ticket.txt" % H)
    time.sleep(1.5)
    chrome(c, "http://127.0.0.1:8000/form2.html")


# ---- scorers: (passed, detail) --------------------------------------------------------------------------------------------------------------------------
def cat(c, path):
    rc, out = L.dx(c, "cat %s" % path, user="root", env=False)
    return out if rc == 0 else None


def score_t1(c):
    t = cat(c, H + "/Documents/note.txt")
    if t is None:
        return False, "no note.txt"
    ok = t.strip().splitlines() == ["deploy ok", "build 4182", "owner: ops"]
    return ok, repr(t[:120])


def score_t2(c):
    rc, ls = L.dx(c, "ls %s/Documents" % H, user="root", env=False)
    names = sorted(ls.split())
    ok = names == ["data.csv", "other.txt", "report-final.txt"] and cat(c, H + "/Documents/report-final.txt") == "quarterly numbers\n" and cat(c, H + "/Documents/other.txt") == "keep me\n"
    return ok, names


def score_t3(c):
    rc, ls = L.dx(c, "cd %s/Documents && find . -type f | sort" % H, user="root", env=False)
    files = ls.split()
    ok = files == ["./archive/old.log", "./keep.txt"] and cat(c, H + "/Documents/archive/old.log") == "line one\nline two\n"
    return ok, files


def score_t4(c):
    rc, out = L.dx(c, "xfconf-query -c xsettings -p /Xft/Antialias")
    return out.strip() == "0", out.strip()[:60]


def score_t5(c):
    t = cat(c, H + "/out/form1.json")
    if t is None:
        return False, "no submission"
    d = json.loads(t)
    ok = d.get("name") == "Ada Lovelace" and d.get("colour") == "green" and d.get("subscribe") == "on"
    return ok, d


def score_t6(c):
    cmd = ("ssconvert -T Gnumeric_stf:stf_csv %s/Documents/sales.gnumeric fd://1 2>/dev/null; echo ---; "
           "gzip -dc %s/Documents/sales.gnumeric | tr -d '\\n' | grep -o '<gnm:Cell Row=\"0\" Col=\"3\"[^>]*>[^<]*'") % (H, H)
    rc, out = L.dx(c, cmd, user="root", env=False)
    csv, _, cell = out.partition("---")
    row0 = csv.strip().splitlines()[0].split(",") if csv.strip() else []
    formula = "=" in cell
    ok = len(row0) >= 4 and row0[3].strip() == "50" and formula
    return ok, {"row0": row0, "d1_cell": cell.strip()[:100]}


def score_t7(c):
    t = cat(c, H + "/count.txt")
    return (t is not None and t.strip() == "7"), repr(t)


def score_t8(c):
    t = cat(c, H + "/out/form2.json")
    if t is None:
        return False, "no submission"
    d = json.loads(t)
    return d.get("code") == "K7-ALPHA-4821", d


# ---- hand solutions (reference end states, shell only) ---------------------------------------------------------------------------------------------------
def hand_t1(c):
    put(c, H + "/Documents/note.txt", "deploy ok\nbuild 4182\nowner: ops\n")


def hand_t2(c):
    sh(c, "mv %s/Documents/draft.txt %s/Documents/report-final.txt" % (H, H))


def hand_t3(c):
    sh(c, "cd %s/Documents && mkdir archive && mv old.log archive/" % H)


def hand_t4(c):
    sh(c, "xfconf-query -c xsettings -p /Xft/Antialias -s 0")


def hand_t5(c):
    sh(c, "curl -s -d 'name=Ada+Lovelace&colour=green&subscribe=on' http://127.0.0.1:8000/submit/form1")


def hand_t6(c):
    put(c, "/tmp/sales2.csv", "12,30,8,=SUM(A1:C1)\n5,7,9\n")
    sh(c, "ssconvert /tmp/sales2.csv %s/Documents/sales.gnumeric >/dev/null 2>&1" % H)


def hand_t7(c):
    sh(c, "ls -p %s/inbox | grep -v / | wc -l > %s/count.txt" % (H, H))


def hand_t8(c):
    sh(c, "curl -s -d 'code=K7-ALPHA-4821' http://127.0.0.1:8000/submit/form2")


def mk(i, title, short, prompt, wins):
    g = globals()
    return {"id": "t%d" % i, "title": title, "short": short, "prompt": prompt, "windows": wins,
            "seed": g["seed_t%d" % i], "score": g["score_t%d" % i], "hand": g["hand_t%d" % i]}


TASKS = [
    mk(1, "editor-save", "type + save in a text editor",
       "Mousepad (a text editor) is open and empty. Type exactly these three lines, with a line break between them: deploy ok / build 4182 / owner: ops. "
       "Then save the file as /home/pn/Documents/note.txt (use the full path).", ["Mousepad"]),
    mk(2, "file-rename", "rename a file in the file manager",
       "The Thunar file manager is open on the Documents folder. Rename the file draft.txt to report-final.txt. Change nothing else.", ["Thunar"]),
    mk(3, "folder-move", "new folder + move a file",
       "The Thunar file manager is open on the Documents folder. Create a new folder named archive inside Documents and move the file old.log into it. Leave keep.txt where it is.", ["Thunar"]),
    mk(4, "settings-toggle", "turn a setting off",
       "The Appearance settings window is open. Turn OFF font anti-aliasing (it is on the Fonts tab).", ["Appearance"]),
    mk(5, "browser-form", "fill and submit a web form",
       "A web browser is open on a signup form. Enter the name Ada Lovelace, choose the colour green, tick the subscribe box, and press Submit.", ["Signup form"]),
    mk(6, "spreadsheet-cell", "formula in a spreadsheet cell + save",
       "Gnumeric (a spreadsheet) is open with numbers in A1, B1 and C1. In cell D1 enter a formula that adds A1, B1 and C1 (=SUM(A1:C1)), then save the file with Ctrl+S, keeping the same name and format.", ["Gnumeric"]),
    mk(7, "terminal-count", "one command in a terminal",
       "A terminal window is open. Find out how many regular files (not folders) are directly inside /home/pn/inbox, and write just that number into the file /home/pn/count.txt.", ["Terminal"]),
    mk(8, "copy-across-windows", "copy a code from one app into another",
       "Two windows are open: a text editor showing a support ticket, and a web browser with a form. Copy the ticket code from the editor into the form's Code field and press Submit.", ["Mousepad", "Ticket form"]),
]


def prove(c):
    """two controls per scorer on the live desktop container: the fresh seed must FAIL, the hand solution must PASS (the exact-equality scorers reject near misses);
    prints one line per task and returns the number of wrong controls."""
    bad = 0
    for t in TASKS:
        reset(c)
        t["seed"](c)
        time.sleep(1.5)
        fresh, fd = t["score"](c)
        t["hand"](c)
        time.sleep(1.0)
        hand, hd = t["score"](c)
        verdict = "OK" if (not fresh and hand) else "BAD"
        bad += verdict == "BAD"
        print("%s %-22s fresh=%s hand=%s  fresh_detail=%s hand_detail=%s" % (verdict, t["id"] + " " + t["title"], fresh, hand, str(fd)[:50], str(hd)[:60]))
        reset(c)
    return bad


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--prove":
        sys.exit(1 if prove(L.cid(sys.argv[2])) else 0)
    print(json.dumps([{k: v for k, v in t.items() if k in ("id", "title", "short", "prompt")} for t in TASKS], indent=1))
