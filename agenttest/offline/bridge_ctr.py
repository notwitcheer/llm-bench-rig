#!/usr/bin/env python3
"""bridge_ctr.py: CONTAINER side of the offline bridge (PID 1 of the container). Listens on 127.0.0.1:<port> INSIDE the container (loopback only, `lo` exists even with
--network none) and relays every connection to the bind-mounted UNIX socket from bridge_host.py. Hermes' base_url is http://127.0.0.1:<port>/v1.
stdlib only. Counters go to /tmp/bridge_ctr_stats.json (rewritten every 2 s).

usage: bridge_ctr.py --listen 127.0.0.1:18080 --sock /bridge/model.sock
"""
import argparse
import asyncio
import json
import time

STATS = {"connections": 0, "bytes_up": 0, "bytes_down": 0, "failed_connects": 0, "started_epoch": int(time.time())}


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


async def handle(cr, cw, sock):
    STATS["connections"] += 1
    try:
        sr, sw = await asyncio.open_unix_connection(sock)
    except OSError:
        STATS["failed_connects"] += 1
        cw.close()
        return
    await asyncio.gather(pipe(cr, sw, "bytes_up"), pipe(sr, cw, "bytes_down"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:18080")
    ap.add_argument("--sock", default="/bridge/model.sock")
    a = ap.parse_args()
    host, port = a.listen.rsplit(":", 1)
    if host != "127.0.0.1":
        raise SystemExit("listen must be 127.0.0.1")
    await asyncio.start_server(lambda r, w: handle(r, w, a.sock), host, int(port))
    print("bridge_ctr up:", a.listen, "->", a.sock, flush=True)
    while True:
        json.dump(dict(STATS, updated_epoch=int(time.time())), open("/tmp/bridge_ctr_stats.json", "w"))
        await asyncio.sleep(2)


asyncio.run(main())
