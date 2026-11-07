#!/usr/bin/env python3
"""wd_service.py: the SANDBOX TOOL of item 95. A small localhost HTTP service the editor's `wd` command talks to (stdlib only).
It holds everything the editor may do to the 0.8B model, and nothing else:
  info        model facts, limits, remaining budget (evals, minutes)
  transcripts the base model's dev-set thinking texts (saved by the harness before the take): looping tails and closed heads, written into the agent's workdir
  acts FILE   a forward pass of Qwen3.5-0.8B (base weights, GPU/CPU) over a text file the editor wrote: the mean MLP intermediate (the input of down_proj) per layer -> npy (24 x 3584)
  weights L   layer L's down_proj weight of the BASE model as float32 npy (1024 x 3584)
  apply SPEC  validate (limits in wd_edit.py) and build a candidate GGUF copy; the base file is never touched
  eval NAME   start a dev-set run (20 GSM8K train problems, 16,384-token cap, thinking on) of candidate NAME on a private llama-server; non-blocking
  result NAME progress / the dev loop rate; blocks up to --wait seconds for completion
  submit NAME mark the editor's final choice (the harness also scores the best-by-dev candidate on its own; submit is a courtesy)
Held-out problems are never reachable from here. Every call is appended to calls.jsonl (time, verb, args, outcome): the gate and the renderer read it.
usage: wd_service.py --port P --state DIR --work AGENT_WORKDIR --base-rows base-dev.json --tables DIR --evals 10 --minutes 100 [--stub]
--stub: no GPU, no model: eval returns synthetic rows (CPU proof of the plumbing; never a result).
"""
import argparse, json, os, signal, subprocess, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wd_common as C  # noqa: E402
import wd_edit as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--base-rows", required=True)
ap.add_argument("--tables", required=True)
ap.add_argument("--evals", type=int, default=10)
ap.add_argument("--minutes", type=float, default=100)
ap.add_argument("--eval-port", type=int, default=8283)
ap.add_argument("--slots", type=int, default=20)
ap.add_argument("--cap", type=int, default=C.CAP)
ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N dev problems")
ap.add_argument("--acts-device", default="cuda")
ap.add_argument("--eval-ngl", type=int, default=99, help="test knob: 0 = run the dev evaluation on CPU (proof only)")
ap.add_argument("--stub", action="store_true")
a = ap.parse_args()

os.makedirs(a.state, exist_ok=True)
os.makedirs(os.path.join(a.state, "cand"), exist_ok=True)
os.makedirs(os.path.join(a.state, "evals"), exist_ok=True)
WORK = os.path.realpath(a.work)
T0 = time.time()
LOCK = threading.Lock()          # one GPU job at a time: eval OR acts
JOBS = {}                        # name -> dict(status, done, total, summary, t0, t1)
CALLS = open(os.path.join(a.state, "calls.jsonl"), "a", buffering=1)
SUBMITTED = []
BASE = json.load(open(a.base_rows))
SRV = os.environ.get("WD_SRV", os.path.expanduser("~/llama.cpp-donald/build/bin/llama-server"))
PY = sys.executable


def log(verb, args, out):
    CALLS.write(json.dumps(dict(t=round(time.time(), 2), dt=round(time.time() - T0, 1), verb=verb, args=args, out=out)) + "\n")


def mins_left():
    return a.minutes - (time.time() - T0) / 60.0


def evals_used():
    return sum(1 for j in JOBS.values() if j.get("counted"))


def under_work(p):
    rp = os.path.realpath(p if os.path.isabs(p) else os.path.join(WORK, p))
    if rp != WORK and not rp.startswith(WORK + os.sep):
        raise ValueError("path is outside the working directory")
    return rp


def v_info(q):
    layers = ["linear-attention" if (i + 1) % 4 else "full-attention" for i in range(C.LAYERS)]
    s = BASE["summary"]
    return dict(model="Qwen3.5-0.8B (instruct, thinking mode on), bf16", layers=C.LAYERS, layer_types=layers, hidden=C.D_MODEL, mlp_intermediate=C.D_FF,
                edit_site="each layer's mlp.down_proj weight, shape (1024, 3584): columns are the 3584 MLP neurons, the output of a neuron is its column",
                edit_rule="for each neuron j you list in a layer: W[:, j] <- W[:, j] - strength * (W[:, j] . d) * d, with d = your direction / |direction| (1024 numbers). a spec may list up to 2 layers",
                limits=dict(max_layers=E.MAX_LAYERS, max_neurons_per_layer=E.MAX_NEURONS, max_abs_strength=E.MAX_STRENGTH, layers="0..%d" % (C.LAYERS - 1)),
                spec_format={"edits": [{"layer": 15, "direction_file": "d15.npy  (or \"direction\": [1024 floats])", "neurons": [0, 1, 2], "strength": 6.0}]},
                failure="with thinking on, the unedited model usually never closes its thinking: it re-derives the same step in different words until the token cap (a 'thinking loop').",
                dev_set=dict(problems=s["n"], cap_tokens=a.cap, base_loops=s["loops"], base_loop_rate=s["loop_rate"], sampler="temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5"),
                budget=dict(evals_total=a.evals, evals_used=evals_used(), minutes_left=round(mins_left(), 1)),
                commands=["wd info", "wd transcripts", "wd acts FILE [--tail] --out acts.npy", "wd weights LAYER --out W.npy", "wd apply SPEC.json --name NAME", "wd eval NAME", "wd result NAME", "wd submit NAME", "wd status"])


def v_transcripts(q):
    from wd_acts import pick_rows
    d = os.path.join(WORK, "transcripts")
    os.makedirs(d, exist_ok=True)
    loops, prods = pick_rows(BASE["rows"])
    for i, r in enumerate(loops, 1):
        open(os.path.join(d, "loop_%d_tail.txt" % i), "w").write(r["think"][-5000:])
    for i, r in enumerate(prods, 1):
        open(os.path.join(d, "start_%d_head.txt" % i), "w").write(r["think"][:2500])
    return dict(written=sorted(os.listdir(d)), note="loop_N_tail.txt = the last 5,000 characters of a thinking that never closed; start_N_head.txt = the first 2,500 characters of a thinking (the productive start)")


def v_acts(q):
    f = under_work(q["file"][0])
    out = under_work(q["out"][0]) if "out" in q else os.path.join(WORK, "acts.npy")
    if mins_left() <= 0:
        return dict(error="time budget used up")
    tail = "--tail" in q
    txt = open(f).read()
    if tail:
        tmp = os.path.join(a.state, "acts_in.txt")
        open(tmp, "w").write(txt[-7000:])
        f = tmp
    if a.stub:
        arr = np.random.RandomState(1).randn(C.LAYERS, C.D_FF).astype(np.float32)
        np.save(out, arr)
        return dict(saved=os.path.relpath(out, WORK), shape=list(arr.shape), note="STUB")
    if True:
        p = subprocess.run([PY, os.path.join(HERE, "wd_acts.py"), "text", "--file", f, "--layer", "-1", "--out", out, "--device", a.acts_device], capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return dict(error=(p.stderr or p.stdout)[-300:])
    arr = np.load(out)
    return dict(saved=os.path.relpath(out, WORK), shape=list(arr.shape), note="row L = mean over the text's tokens of layer L's MLP intermediate (the input of down_proj), 3584 numbers per layer")


def v_weights(q):
    L = int(q["layer"][0])
    if not 0 <= L < C.LAYERS:
        return dict(error="layer 0..%d" % (C.LAYERS - 1))
    out = under_work(q["out"][0]) if "out" in q else os.path.join(WORK, "W%d.npy" % L)
    if a.stub:
        W = np.random.RandomState(L).randn(C.D_MODEL, C.D_FF).astype(np.float32)
    else:
        sys.path.insert(0, os.path.expanduser("~/llama.cpp-qwen4/gguf-py"))
        import gguf
        r = gguf.GGUFReader(E.BASE_GGUF)
        t = {x.name: x for x in r.tensors}["blk.%d.ffn_down.weight" % L]
        W = E.bf16_to_f32(np.asarray(t.data).view(np.uint16).reshape(C.D_MODEL, C.D_FF)).astype(np.float32)
    np.save(out, W)
    return dict(saved=os.path.relpath(out, WORK), shape=list(W.shape), note="float32 copy of the BASE down_proj weight; columns = neurons")


def v_apply(q):
    spec_path = under_work(q["spec"][0])
    name = q["name"][0]
    if not name.replace("_", "").replace("-", "").isalnum() or len(name) > 24:
        return dict(error="name: letters, digits, - and _ only, up to 24 characters")
    if len(os.listdir(os.path.join(a.state, "cand"))) >= 10 and not os.path.exists(os.path.join(a.state, "cand", name + ".gguf")):
        return dict(error="at most 10 candidates; evaluate or reuse one")
    spec = json.load(open(spec_path))
    out = os.path.join(a.state, "cand", name + ".gguf")
    os.makedirs(os.path.join(a.state, "specs"), exist_ok=True)
    try:
        if a.stub:
            E.validate(spec, os.path.dirname(spec_path))
            rep = [dict(layer=int(e["layer"]), neurons=len(e["neurons"]), stub=True, changed_entries=1000) for e in spec["edits"]]
            open(out, "w").write("stub")
        else:
            rep = E.apply_spec(spec, out, E.BASE_GGUF, os.path.dirname(spec_path))
    except Exception as e:
        return dict(error="spec rejected: %s" % str(e)[:300])
    # keep a self-contained copy of the spec (directions inlined) for the report
    plan = E.validate(spec, os.path.dirname(spec_path))
    json.dump(dict(edits=[dict(layer=L, neurons=nr, strength=s, direction=[round(float(x), 6) for x in d]) for L, nr, s, d in plan]), open(os.path.join(a.state, "specs", name + ".json"), "w"))
    return dict(candidate=name, report=rep, next="wd eval %s" % name)


def eval_thread(name):
    j = JOBS[name]
    gg = os.path.join(a.state, "cand", name + ".gguf")
    probs = C.problems("dev")
    if a.limit:
        probs = probs[:a.limit]
    j["total"] = len(probs)
    with LOCK:
        j["status"] = "running"
        j["t0"] = time.time()
        p = None
        try:
            if a.stub:
                time.sleep(3)
                rows = []
                for i, pr in enumerate(probs):
                    rows.append(dict(id=pr["id"], gold=pr["gold"], closed=(i % 3 == 0), correct=(i % 6 == 0), finish="stop", tokens=1000, think="stub", answer="stub", s=1.0))
                    j["done"] = i + 1
            else:
                ctx = a.slots * (a.cap + 1200)
                log_f = open(os.path.join(a.state, "evals", name + ".server.log"), "w")
                p = subprocess.Popen([SRV, "-m", gg, "--port", str(a.eval_port), "--host", "127.0.0.1", "-ngl", str(a.eval_ngl), "-c", str(ctx), "-np", str(a.slots), "--jinja", "--threads", "4", "--no-webui"],
                                     stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
                import urllib.request
                for _ in range(120):
                    if p.poll() is not None:
                        raise RuntimeError("0.8B server exited at load")
                    try:
                        if b"ok" in urllib.request.urlopen("http://127.0.0.1:%d/health" % a.eval_port, timeout=2).read().lower():
                            break
                    except Exception:
                        time.sleep(1)
                def on_done(row):
                    j["done"] = j.get("done", 0) + 1
                rows = C.run_problems(a.eval_port, probs, a.slots, cap=a.cap, seed=1, on_done=on_done)
            s = C.summarise(rows)
            json.dump(dict(summary=s, rows=rows), open(os.path.join(a.state, "evals", name + ".json"), "w"))
            j["summary"] = s
            j["status"] = "done"
        except Exception as e:
            j["status"] = "error"
            j["error"] = str(e)[:300]
            traceback.print_exc()
        finally:
            if p is not None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=30)
                except Exception:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
            j["t1"] = time.time()
            log("eval_done", dict(name=name), dict(status=j["status"], summary=j.get("summary"), seconds=round(j["t1"] - j["t0"], 1)))


def v_eval(q):
    name = q["name"][0]
    gg = os.path.join(a.state, "cand", name + ".gguf")
    if not os.path.exists(gg):
        return dict(error="no candidate %r: run wd apply first" % name)
    if name in JOBS and JOBS[name]["status"] in ("queued", "running", "done"):
        return dict(error="candidate %r was already evaluated or is running (dev set, one try per problem); build a new candidate" % name)
    if evals_used() >= a.evals:
        return dict(error="eval budget used up (%d of %d)" % (evals_used(), a.evals))
    if mins_left() < 5:
        return dict(error="less than 5 minutes left")
    JOBS[name] = dict(status="queued", done=0, total=0, counted=True)
    threading.Thread(target=eval_thread, args=(name,), daemon=True).start()
    return dict(started=name, evals_left=a.evals - evals_used(), next="wd result %s   (takes about 3-5 minutes)" % name)


def v_result(q):
    name = q["name"][0]
    wait = min(float(q.get("wait", ["90"])[0]), 120.0)
    j = JOBS.get(name)
    if not j:
        return dict(error="no evaluation named %r" % name)
    t = time.time()
    while j["status"] in ("queued", "running") and time.time() - t < wait:
        time.sleep(1)
    if j["status"] == "done":
        s = j["summary"]
        return dict(candidate=name, status="done", dev_problems=s["n"], thinking_never_closed=s["loops"], loop_rate=s["loop_rate"], closed=s["closed"], answered_correctly=s["correct"],
                    mean_tokens=s["mean_tokens"], base_loop_rate=BASE["summary"]["loop_rate"], evals_left=a.evals - evals_used(), minutes_left=round(mins_left(), 1))
    return dict(candidate=name, status=j["status"], progress="%d of %d finished" % (j.get("done", 0), j.get("total", 0)), error=j.get("error"), note="call wd result %s again" % name)


def v_submit(q):
    name = q["name"][0]
    if not os.path.exists(os.path.join(a.state, "cand", name + ".gguf")):
        return dict(error="no candidate %r" % name)
    SUBMITTED.append(name)
    json.dump(SUBMITTED, open(os.path.join(a.state, "submitted.json"), "w"))
    return dict(submitted=name, note="recorded; the harness scores the best-by-dev candidate on its own as well")


def v_status(q):
    return dict(evals_used=evals_used(), evals_total=a.evals, minutes_left=round(mins_left(), 1), candidates=sorted(os.listdir(os.path.join(a.state, "cand"))),
                evaluations={k: dict(status=v["status"], loop_rate=(v.get("summary") or {}).get("loop_rate")) for k, v in JOBS.items()})


VERBS = dict(info=v_info, transcripts=v_transcripts, acts=v_acts, weights=v_weights, apply=v_apply, eval=v_eval, result=v_result, submit=v_submit, status=v_status)


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        verb = u.path.strip("/")
        q = parse_qs(u.query)
        if verb == "health":
            out = dict(ok=True)
        elif verb in VERBS:
            try:
                out = VERBS[verb](q)
            except Exception as e:
                out = dict(error="%s: %s" % (type(e).__name__, str(e)[:300]))
            if verb not in ("result", "status"):
                log(verb, {k: v[0] for k, v in q.items()}, out if len(json.dumps(out)) < 1500 else {"keys": list(out)})
        else:
            out = dict(error="unknown command %r" % verb)
        b = json.dumps(out, indent=1).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


print("wd_service up on %d (state %s, work %s, evals %d, minutes %g, stub=%s)" % (a.port, a.state, WORK, a.evals, a.minutes, a.stub), flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
