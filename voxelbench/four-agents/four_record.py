#!/usr/bin/env python3
"""four_record.py: one Hermes Agent session in a pty, recorded for the item-80 race clip (capsule only).

usage: four_record.py --agent a --out events-a.jsonl --prompt prompt.txt --workdir /tmp/arena80/a --home /tmp/arena80/hm-a --hermes-home <dir>
                      --hermes-bin <checkout>/.hermes/bin/hermes [--cols 78 --rows 18] [--wall-s 2400] [--go FILE] [--ready FILE] [--state-url http://127.0.0.1:8191/_state]
                      [--quiet-s 20] [--toolsets terminal,file] [--max-turns 80] [--final FILE] [--result FILE]

Same recorder idea as the 2026-10-27 two-way race (pty + pyte, events with timestamps), changed for four concurrent agents:
* events are JSONL lines {"t": <epoch s>, "d": <base64 chunk>} or {"t":..,"mark":name}: ABSOLUTE epoch time, so four recordings and the proxy's hud.jsonl share one
  clock, and a crash keeps everything written so far.
* fair start: each recorder boots Hermes, types the prompt (chunked, then waits until the line's tail is on the pyte screen + 0.6 s, the L31 rule: an
  Enter <50 ms after the last buffer change is read as a pasted newline), touches --ready, then WAITS for the --go file the item creates once all four
  are ready, and presses Enter. The race clock starts at that Enter ("enter" mark).
* the process is mirrored to our own stdout, so a tmux session running this script shows the live pane (`tmux attach -t f80-<agent>`).
* done = the screen minus the status bar row unchanged for --quiet-s AND the agent's proxy inflight count 0 for that long (when --state-url is given).
  wall = --wall-s after Enter: two Ctrl-C, the mark, kill. Exit code 0 done, 3 wall, 4 hermes exited early, 5 setup failed.
  a quiet pane whose agent's LAST proxy request was not HTTP 200 ends as "server_error" (exit 0, no last_change mark), never "done" (80b).
* HOME for the agent is a throwaway directory (the clip must not show a username or a home path). The agent runs the install's minted launcher with ITS OWN
  HERMES_HOME (never `-p <profile>`, so the main hermes-gateway.service never serves it). A new data root would otherwise build its own dependency environment
  (measured 2026-11-03: python-olm cmake, minutes, then a failure) and the stale checkout venv lacks modules (snowballstemmer: every tool call died), so four_mkhome.py
  symlinks the main root's `installs/` and `tools/` into the new home; HERMES_DISABLE_LAZY_INSTALLS=1 keeps it read-only. Proven: a full mock task in a symlinked home.
"""
import argparse
import base64
import fcntl
import json
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
import urllib.request

import pyte

ap = argparse.ArgumentParser()
ap.add_argument("--agent", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--workdir", required=True)
ap.add_argument("--home", required=True)
ap.add_argument("--hermes-home", required=True)
ap.add_argument("--hermes-bin", required=True, help="the minted launcher <checkout>/.hermes/bin/hermes (runs the install's own store interpreter + dependency generation)")
ap.add_argument("--cols", type=int, default=78)
ap.add_argument("--rows", type=int, default=18)
ap.add_argument("--wall-s", type=float, default=2400)
ap.add_argument("--go", default=None)
ap.add_argument("--ready", default=None)
ap.add_argument("--state-url", default=None)
ap.add_argument("--quiet-s", type=float, default=20.0)
ap.add_argument("--toolsets", default="terminal,file")
ap.add_argument("--max-turns", type=int, default=80)
ap.add_argument("--final", default=None)
ap.add_argument("--result", default=None)
ap.add_argument("--go-timeout", type=float, default=300.0)
ap.add_argument("--no-mirror", action="store_true")
a = ap.parse_args()

prompt = " ".join(open(a.prompt).read().split())
os.makedirs(a.workdir, exist_ok=True)
os.makedirs(a.home, exist_ok=True)
env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": a.home, "HERMES_HOME": a.hermes_home, "HERMES_DISABLE_LAZY_INSTALLS": "1",
       "TERM": "xterm-256color", "LANG": "C.UTF-8", "COLORTERM": "truecolor", "COLUMNS": str(a.cols), "LINES": str(a.rows), "TERMINAL_CWD": a.workdir,
       "TMPDIR": a.home + "/tmp", "PS1": "$ "}
os.makedirs(a.home + "/tmp", exist_ok=True)
cmd = [a.hermes_bin, "chat", "--cli", "--toolsets", a.toolsets, "--max-turns", str(a.max_turns)]

out = open(a.out, "w", buffering=1)
scr = pyte.Screen(a.cols, a.rows)
stm = pyte.ByteStream(scr)


def log(o):
    out.write(json.dumps(o) + "\n")


pid, fd = pty.fork()
if pid == 0:
    os.chdir(a.workdir)
    os.execve(cmd[0], cmd, env)
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", a.rows, a.cols, 0, 0))
log({"hdr": True, "agent": a.agent, "cols": a.cols, "rows": a.rows, "t": round(time.time(), 3)})
last_out = time.time()
mirror = None if a.no_mirror else sys.stdout.buffer


def pump(timeout=0.05):
    global last_out
    r, _, _ = select.select([fd], [], [], timeout)
    if r:
        try:
            d = os.read(fd, 65536)
        except OSError:
            return False
        if not d:
            return False
        log({"t": round(time.time(), 3), "d": base64.b64encode(d).decode()})
        last_out = time.time()
        stm.feed(d)
        if mirror:
            try:
                mirror.write(d)
                mirror.flush()
            except Exception:
                pass
    return True


def mark(name, **kw):
    log({"t": round(time.time(), 3), "mark": name, **kw})
    sys.stderr.write("[rec %s] %s %s\n" % (a.agent, time.strftime("%H:%M:%S"), name))
    sys.stderr.flush()


def screen_text():
    return "\n".join(scr.display)


def body_sig():
    rows = [l for l in scr.display[:-1] if " ctx " not in l and "⏲" not in l]
    return hash(tuple(rows))


def finish(code, end):
    mark("end", reason=end)
    if a.final:
        open(a.final, "w").write(screen_text() + "\n")
    if a.result:
        json.dump({"agent": a.agent, "end": end, "t": round(time.time(), 3)}, open(a.result, "w"))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except Exception:
            pass
        time.sleep(0.5)
    try:
        os.waitpid(pid, os.WNOHANG)
    except Exception:
        pass
    out.close()
    sys.exit(code)


# 1. UI up: the input prompt line is on screen and has been for 3 s
t_ui0 = time.time()
seen_at = None
while time.time() - t_ui0 < 120:
    if not pump(0.1):
        finish(4, "hermes_exited_before_ui")
    if any("❯" in l for l in scr.display):
        seen_at = seen_at or time.time()
        if time.time() - seen_at > 3.0:
            break
else:
    finish(5, "ui_timeout")
mark("ui_ready")
for _ in range(10):
    pump(0.1)

# 2. type the prompt (chunks), wait for its tail on screen, then +0.6 s
tl = "".join(prompt.split())[-32:]
for i in range(0, len(prompt), 24):
    os.write(fd, prompt[i:i + 24].encode())
    t_end = time.time() + 0.03
    while time.time() < t_end:
        pump(0.01)
ok = False
for _ in range(150):
    pump(0.2)
    if tl in "".join(screen_text().split()):
        ok = True
        break
mark("typed" if ok else "typed_tail_missing")
if not ok:
    finish(5, "prompt_not_on_screen")
t_end = time.time() + 0.8
while time.time() < t_end:
    pump(0.05)

# 3. barrier: ready -> wait for GO
if a.ready:
    open(a.ready, "w").write(str(time.time()))
if a.go:
    t_w = time.time()
    while not os.path.exists(a.go):
        pump(0.05)
        if time.time() - t_w > a.go_timeout:
            finish(5, "go_timeout")
mark("typed_waiting_done")
os.write(fd, b"\r")
t_enter = time.time()
mark("enter")

# 4. run until done / wall / exit
last_sig = body_sig()
last_change = time.time()
idle_infl = None
state_err = 0
last_status = 200
while True:
    alive = pump(0.2)
    if not alive:
        finish(4, "hermes_exited")
    now = time.time()
    sg = body_sig()
    if sg != last_sig:
        last_sig = sg
        last_change = now
    infl = None
    if a.state_url and int(now) % 3 == 0:
        try:
            st = json.loads(urllib.request.urlopen(a.state_url, timeout=1.5).read())
            infl = st["agents"][a.agent]["inflight"]
            last_status = st["agents"][a.agent].get("last_status", 200)
            state_err = 0
        except Exception:
            state_err += 1
    if infl is not None:
        if infl > 0:
            idle_infl = None
        elif idle_infl is None:
            idle_infl = now
    quiet = now - last_change
    infl_ok = (not a.state_url) or (idle_infl is not None and now - idle_infl >= a.quiet_s) or state_err > 20
    if now - t_enter > 15 and quiet >= a.quiet_s and infl_ok and any("❯" in l for l in scr.display[-8:]):
        if last_status != 200:
            # the agent's LAST request was rejected or broke (a server error): the pane going quiet is a stop, not a finish (item 80 counted three of these as "done")
            mark("server_error_last_request", status=last_status)
            finish(0, "server_error")
        log({"t": round(last_change, 3), "mark": "last_change"})
        finish(0, "done")
    if now - t_enter > a.wall_s:
        mark("wall")
        os.write(fd, b"\x03")
        t_w = time.time() + 2
        while time.time() < t_w:
            pump(0.1)
        os.write(fd, b"\x03")
        t_w = time.time() + 4
        while time.time() < t_w:
            pump(0.1)
        finish(3, "wall")
