#!/usr/bin/env python3
"""desk_lib.py: container-side helpers for the desktop arena (all run by the HARNESS through docker exec, never by the agent).
cid(home) -> the sandbox container id Hermes recorded for this HERMES_HOME (bot-desktop/sandbox.json)
dx(cid, cmd, user='pn', detach=False, env=True) -> run a shell command in the container with the desktop's DISPLAY/XAUTHORITY/DBUS env (published by Hermes in
  /tmp/hermes-bot-desktop/default/env inside the container)
grab(cid, path) -> one full-screen PNG of the agent's desktop via ImageMagick import (the same X display the model sees), written on the host
"""
import json
import os
import subprocess

ENVFILE = "/tmp/hermes-bot-desktop/default/env"


def cid(home):
    return json.load(open(os.path.join(home, "bot-desktop", "sandbox.json")))["container"]


def dx(c, cmd, user="pn", detach=False, env=True, timeout=60):
    pre = "set -a; . %s; set +a; " % ENVFILE if env else ""
    argv = ["docker", "exec"] + (["-d"] if detach else []) + ["-u", user, c, "bash", "-lc", pre + cmd]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def grab(c, path):
    fmt = "jpg:- " if path.endswith(".jpg") else "png:-"
    q = "-quality 80 " if path.endswith(".jpg") else ""
    argv = ["docker", "exec", "-u", "pn", c, "bash", "-lc", "set -a; . %s; set +a; import -window root %s%s" % (ENVFILE, q, fmt)]
    r = subprocess.run(argv, capture_output=True, timeout=30)
    if r.returncode != 0 or not r.stdout:
        return False
    open(path, "wb").write(r.stdout)
    return True
