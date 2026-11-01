"""isolation.py: the proof legs for "Hermes Agent with the network cable pulled". Every leg is recorded raw (command, rc, output) in <run>/isolation/proof.json + proof.txt.

legs (all must hold for verdict PASS; the task run refuses to start on FAIL):
  L1 docker inspect: HostConfig.NetworkMode == "none", the only network is "none", no published ports, mounts listed.
  L2 failing probes INSIDE the container (docker exec): curl https://example.com, curl a bare IP (no DNS needed), curl the docker0 gateway on :8090 (= Donald, must be
     unreachable), getent hosts, a python socket connect to 1.1.1.1:443, /proc/net/route. Each must fail (rc != 0 / errno ENETUNREACH / no default route).
  L3 egress counters INSIDE: /sys/class/net/*/statistics for every interface, after the probes and again after the tasks. With --network none the only interface is `lo`:
     zero non-loopback interfaces, zero non-loopback packets. `lo` carries the Hermes -> in-container relay traffic (the unix-socket bridge), reported separately and never as egress.
  C0 control, same image + same flags, no bridge, idle + the same probes: EVERY counter incl. lo must read zero.
  C1 positive control (optional): same image, same curl, DEFAULT bridge network: must succeed with eth0 counters > 0, so the failing curl is the network flag and not a broken tool.
"""
import json
import subprocess
import time

IMG_USER = "1000:1000"


def dk(args, timeout=60, inp=None):
    try:
        p = subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=timeout, input=inp)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def inspect(name):
    rc, out = dk(["inspect", name])
    d = json.loads(out)[0]
    hc, ns = d["HostConfig"], d["NetworkSettings"]
    return {
        "network_mode": hc.get("NetworkMode"), "networks": sorted((ns.get("Networks") or {}).keys()), "ports": ns.get("Ports") or {},
        "port_bindings": hc.get("PortBindings") or {}, "publish_all": hc.get("PublishAllPorts"), "image": d.get("Image"),
        "user": d["Config"].get("User"), "cap_drop": hc.get("CapDrop"), "security_opt": hc.get("SecurityOpt"), "pids_limit": hc.get("PidsLimit"),
        "mounts": [{"src": m.get("Source"), "dst": m.get("Destination"), "rw": m.get("RW")} for m in d.get("Mounts", [])],
        "entrypoint": d["Config"].get("Entrypoint"), "cmd": d["Config"].get("Cmd"), "dns": hc.get("Dns"), "extra_hosts": hc.get("ExtraHosts"),
        "ip_addresses": [v.get("IPAddress") for v in (ns.get("Networks") or {}).values()],
    }


COUNTER_SH = ('for i in /sys/class/net/*; do n=$(basename "$i"); for f in rx_bytes rx_packets rx_errors rx_dropped tx_bytes tx_packets tx_errors tx_dropped; '
              'do echo "$n $f $(cat "$i/statistics/$f")"; done; done')
PY_CONNECT = ("import socket,errno,sys\ns=socket.socket();s.settimeout(5)\n"
              "try:\n s.connect(('1.1.1.1',443)); print('CONNECTED'); sys.exit(0)\n"
              "except OSError as e: print('connect failed errno',e.errno,errno.errorcode.get(e.errno),e.strerror); sys.exit(1)")
PROBES = [
    ("curl https://example.com", ["curl", "-sS", "-m", "6", "-o", "/dev/null", "-w", "%{http_code}", "https://example.com"]),
    ("curl bare IP 1.1.1.1 (no DNS needed)", ["curl", "-sS", "-m", "6", "--noproxy", "*", "-o", "/dev/null", "-w", "%{http_code}", "http://1.1.1.1/"]),
    ("curl docker0 gateway :8090 (Donald)", ["curl", "-sS", "-m", "6", "-o", "/dev/null", "-w", "%{http_code}", "http://172.17.0.1:8090/health"]),
    ("getent hosts example.com", ["getent", "hosts", "example.com"]),
    ("python socket connect 1.1.1.1:443", ["python3", "-c", PY_CONNECT.replace("\\n", "\n")]),
    ("cat /proc/net/route", ["cat", "/proc/net/route"]),
]


def parse_counters(txt):
    out = {}
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) == 3 and p[2].isdigit():
            out.setdefault(p[0], {})[p[1]] = int(p[2])
    return out


def counters_exec(name):
    rc, out = dk(["exec", name, "sh", "-c", COUNTER_SH])
    return parse_counters(out)


def run_probes(exec_prefix):
    res = []
    for label, argv in PROBES:
        rc, out = dk(exec_prefix + argv, timeout=40)
        res.append({"probe": label, "argv": argv if "python3" not in argv[0] else ["python3", "-c", "<socket connect 1.1.1.1:443>"], "rc": rc, "output": out[:400]})
    return res


def probe_failed(p):
    if p["probe"] == "cat /proc/net/route":  # only the header line, i.e. no routes at all
        return len([l for l in p["output"].splitlines() if l.strip()]) <= 1
    return p["rc"] != 0 and "CONNECTED" not in p["output"] and not p["output"].startswith("200")


def nonlo(counters):
    ifs = [i for i in counters if i != "lo"]
    pk = sum(counters[i].get("tx_packets", 0) + counters[i].get("rx_packets", 0) for i in ifs)
    by = sum(counters[i].get("tx_bytes", 0) + counters[i].get("rx_bytes", 0) for i in ifs)
    return {"interfaces": ifs, "packets": pk, "bytes": by}


def control_zero(image, extra=None):
    """C0: a throwaway container, same flags, no bridge: probes then counters."""
    cmd = ["run", "--rm", "--network", "none", "--user", IMG_USER, "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "sh", image, "-c",
           "; ".join(["curl -sS -m 6 -o /dev/null https://example.com >/dev/null 2>&1", "curl -sS -m 6 --noproxy '*' -o /dev/null http://1.1.1.1/ >/dev/null 2>&1",
                      "getent hosts example.com >/dev/null 2>&1", COUNTER_SH])]
    rc, out = dk(cmd, timeout=90)
    c = parse_counters(out)
    allzero = bool(c) and all(v == 0 for i in c.values() for v in i.values())
    return {"rc": rc, "counters": c, "all_zero": allzero, "interfaces": sorted(c)}


def control_positive(image):
    """C1: same image + curl on the default bridge network."""
    cmd = ["run", "--rm", "--user", IMG_USER, "--entrypoint", "sh", image, "-c",
           'curl -sS -m 10 -o /dev/null -w "http=%{http_code}\n" https://example.com; ' + COUNTER_SH]
    rc, out = dk(cmd, timeout=90)
    c = parse_counters(out)
    http = next((l for l in out.splitlines() if l.startswith("http=")), "")
    eth = {i: v for i, v in c.items() if i != "lo"}
    return {"rc": rc, "http": http, "interfaces": sorted(c), "non_lo_packets": nonlo(c)["packets"], "ok": http == "http=200" and nonlo(c)["packets"] > 0, "available": http != ""}


def prove(name, image, out_json, out_txt, want_positive=True, phase="pre"):
    """full proof against a RUNNING container `name`. returns the proof dict (verdict in ['verdict'])."""
    t0 = time.time()
    ins = inspect(name)
    l1 = ins["network_mode"] == "none" and ins["networks"] == ["none"] and not ins["ports"] and not ins["port_bindings"] and not ins["publish_all"] and not any(ins["ip_addresses"])
    probes = run_probes(["exec", name])
    l2 = all(probe_failed(p) for p in probes)
    cnt = counters_exec(name)
    nl = nonlo(cnt)
    l3 = nl["interfaces"] == [] and nl["packets"] == 0
    c0 = control_zero(image)
    c1 = control_positive(image) if want_positive else {"available": False, "skipped": True}
    verdict = "PASS" if (l1 and l2 and l3 and c0["all_zero"]) else "FAIL"
    proof = {"phase": phase, "verdict": verdict, "legs": {"L1_inspect_none": l1, "L2_probes_all_failed": l2, "L3_no_nonloopback_egress": l3, "C0_control_all_counters_zero": c0["all_zero"],
                                                 "C1_positive_control_ok": c1.get("ok")}, "inspect": ins, "probes": probes, "counters_after_probes": cnt, "non_loopback": nl,
             "control_zero": c0, "control_positive_default_network": c1, "elapsed_s": round(time.time() - t0, 1), "stamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(proof, open(out_json, "w"), indent=1)
    L = [f"ISOLATION PROOF ({phase}) verdict {verdict} {proof['stamp_utc']}", "", "L1 docker inspect " + name,
         f"  NetworkMode={ins['network_mode']} Networks={ins['networks']} Ports={ins['ports']} PortBindings={ins['port_bindings']} IPs={ins['ip_addresses']}",
         f"  user={ins['user']} cap_drop={ins['cap_drop']} security_opt={ins['security_opt']} pids_limit={ins['pids_limit']} image={ins['image'][:19]}"]
    L += [f"  mount {m['src']} -> {m['dst']} rw={m['rw']}" for m in ins["mounts"]]
    L += ["", "L2 failing probes inside the container (every one must fail)"]
    for p in probes:
        L.append(f"  [{'FAILED as required' if probe_failed(p) else 'DID NOT FAIL'}] {p['probe']}: rc={p['rc']} {p['output'][:160]!r}")
    L += ["", "L3 /sys/class/net/*/statistics inside, after the probes", f"  interfaces: {sorted(cnt)}  non-loopback interfaces: {nl['interfaces']}  non-loopback packets: {nl['packets']}"]
    for i, v in sorted(cnt.items()):
        L.append(f"  {i}: " + " ".join(f"{k}={x}" for k, x in v.items()))
    L += ["", f"C0 control (same image + flags, no bridge, idle + probes): interfaces {c0['interfaces']} all counters zero: {c0['all_zero']}",
          f"C1 positive control (same curl, default bridge network): {c1}"]
    open(out_txt, "w").write("\n".join(L) + "\n")
    return proof
