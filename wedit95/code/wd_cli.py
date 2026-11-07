#!/usr/bin/env python3
"""wd_cli.py: the `wd` command the editor types. A thin client of wd_service.py (port in $WD_PORT). Prints the service's JSON answer.
  wd info | wd transcripts | wd status
  wd acts FILE [--tail] [--out acts.npy]
  wd weights LAYER [--out W.npy]
  wd apply SPEC.json --name NAME
  wd eval NAME | wd result NAME [--wait 90] | wd submit NAME
Paths are relative to the working directory; nothing outside it can be read or written through this tool."""
import json, os, sys, urllib.parse, urllib.request

port = os.environ.get("WD_PORT")
if not port:
    sys.exit("wd: not configured")
argv = sys.argv[1:]
if not argv or argv[0] in ("-h", "--help", "help"):
    print(__doc__)
    sys.exit(0)
verb, rest = argv[0], argv[1:]
q = {}
pos = []
i = 0
while i < len(rest):
    x = rest[i]
    if x == "--tail":
        q["tail"] = "1"
    elif x.startswith("--") and i + 1 < len(rest):
        q[x[2:]] = rest[i + 1]
        i += 1
    else:
        pos.append(x)
    i += 1
if verb == "acts" and pos:
    q["file"] = pos[0]
elif verb == "weights" and pos:
    q["layer"] = pos[0]
elif verb == "apply" and pos:
    q["spec"] = pos[0]
elif verb in ("eval", "result", "submit") and pos:
    q["name"] = pos[0]
cwd = os.getcwd()
for k in ("file", "spec", "out"):
    if k in q and not os.path.isabs(q[k]):
        q[k] = os.path.join(cwd, q[k])
try:
    r = urllib.request.urlopen("http://127.0.0.1:%s/%s?%s" % (port, verb, urllib.parse.urlencode(q)), timeout=240).read().decode()
except Exception as e:
    sys.exit("wd: the tool service did not answer (%s)" % str(e)[:100])
print(r)
