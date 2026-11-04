#!/usr/bin/env python3
"""pane_rec.py <tmux-session> <out.jsonl> <stop-file> [interval_s]

Snapshots the visible Hermes pane (ANSI colours kept, `capture-pane -e`) every interval seconds while the tmux session exists, and appends one json line
{"t": <epoch>, "text": <screen>} only when the screen changed. The renderer replays these at 8x for the cutaway and masks paths / the user name / the host
name at render time (the raw frames are kept private, never published). Stops when <stop-file> exists or the session is gone."""
import json
import os
import subprocess
import sys
import time

sess, out, stop = sys.argv[1:4]
iv = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
last, n = None, 0
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "a") as f:
    while not os.path.exists(stop):
        t = time.time()
        r = subprocess.run(["tmux", "capture-pane", "-t", "=%s:" % sess, "-p", "-e"], capture_output=True, text=True)
        if r.returncode != 0:
            time.sleep(iv)
            if subprocess.run(["tmux", "has-session", "-t", "=%s" % sess], capture_output=True).returncode != 0 and n > 0:
                break
            continue
        if r.stdout != last:
            f.write(json.dumps({"t": round(t, 2), "text": r.stdout}) + "\n")
            f.flush()
            last, n = r.stdout, n + 1
        time.sleep(max(0.05, iv - (time.time() - t)))
