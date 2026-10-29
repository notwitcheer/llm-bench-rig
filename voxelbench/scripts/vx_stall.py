#!/usr/bin/env python3
"""stall detector for run_agent.sh (HARDENING L37). prints one line: `stall <idle_s>` | `ok <idle_s> <why>` | `unknown <why>`.
usage: vx_stall.py <profile state.db> <llama-server port> <idle_threshold_s>

stall = the newest message row in the run profile's state.db is older than the threshold AND the model server is not processing a
request right now (/slots). 72a (Ornith, 2026-10-29): Hermes stopped a repetition-looping response ("Response Stopped - Repetition
Detected", the partial response discarded), the CLI went back to its prompt with the goal still active, and nothing continued for
32 of the 60 minutes. a long single generation keeps /slots busy and a foreground tool call is capped at 600 s by Hermes, so the
default threshold of 720 s does not fire on normal work. fail-safe: any error prints `unknown` and the driver does nothing."""
import json, sqlite3, sys, time, urllib.request

try:
    db, port, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    row = c.execute("select max(timestamp) from messages").fetchone()
    last = row[0] if row and row[0] else None
    if last is None:
        print("unknown no-messages-yet"); sys.exit(0)
    idle = time.time() - float(last)
    if idle < thr:
        print(f"ok {int(idle)} recent-message"); sys.exit(0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=4) as r:
            slots = json.loads(r.read().decode())
    except Exception as e:
        print(f"unknown slots-unavailable {type(e).__name__}"); sys.exit(0)
    busy = any(bool(s.get("is_processing")) for s in slots) if isinstance(slots, list) else None
    if busy is None:
        print("unknown slots-shape"); sys.exit(0)
    print(f"ok {int(idle)} server-busy" if busy else f"stall {int(idle)}")
except Exception as e:
    print(f"unknown {type(e).__name__}")
