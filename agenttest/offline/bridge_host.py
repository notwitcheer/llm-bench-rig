#!/usr/bin/env python3
"""bridge_host.py: HOST side of the offline bridge. Listens on a UNIX socket and relays every connection to the model server on 127.0.0.1:<port>.
stdlib only (capsule has no socat). The socket sits in a private directory that is bind-mounted into a `--network none` container; that file is the only channel
out of the container. Nothing here binds a TCP port.

usage: bridge_host.py --sock DIR/model.sock --target 127.0.0.1:8178 [--stats FILE]
stats (json, rewritten every 2 s and at exit): connections, bytes_up (container -> model server), bytes_down, started_epoch.
"""
import argparse
import asyncio
import json
import os
import signal
import time

STATS = {"connections": 0, "bytes_up": 0, "bytes_down": 0, "started_epoch": int(time.time()), "target": None}


async def pipe(r, w, key):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            STATS[key] += len(b)
            w.write(b)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass


async def handle(cr, cw, host, port):
    STATS["connections"] += 1
    try:
        sr, sw = await asyncio.open_connection(host, port)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, sw, "bytes_up"), pipe(sr, cw, "bytes_down"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    host, port = a.target.rsplit(":", 1)
    if host != "127.0.0.1":
        raise SystemExit("target must be 127.0.0.1 (the model server is loopback-only)")
    STATS["target"] = a.target
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    srv = await asyncio.start_unix_server(lambda r, w: handle(r, w, host, int(port)), path=a.sock)
    os.chmod(a.sock, 0o660)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, stop.set)

    def dump():
        if a.stats:
            tmp = a.stats + ".tmp"
            json.dump(dict(STATS, updated_epoch=int(time.time())), open(tmp, "w"))
            os.replace(tmp, a.stats)

    print("bridge_host up:", a.sock, "->", a.target, flush=True)
    while not stop.is_set():
        dump()
        try:
            await asyncio.wait_for(stop.wait(), 2)
        except asyncio.TimeoutError:
            pass
    dump()
    srv.close()
    if os.path.exists(a.sock):
        os.unlink(a.sock)


asyncio.run(main())
