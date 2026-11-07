#!/usr/bin/env python3
"""gen_eval.py: generation leg of item 96-arete-eval. One answer per Propaganda_Eval prompt from ONE already-running llama-server, STREAMED, 4 requests in flight.

usage: gen_eval.py --port P --tag base|arete --items items.jsonl --out RUNDIR [--limit N] [--workers 4] [--max-tokens 30000]
                   [--deadline EPOCH] [--stop-new-s 420] [--abort-s 60]

Protocol (dataset README, pinned revision): the prompt is ONE user turn, no system prompt, no tools; thinking off through chat_template_kwargs.enable_thinking=false;
temperature 0; max_tokens 30000. The graded answer is the final answer only: delta.content, with any think block stripped (everything up to the last </think>);
an unfinished think block, an empty content or finish_reason=length gives response "" (= unanswered, counted 0/0 by score.py). The dropped partial text of a
length-capped row is kept in `partial_chars` / the stream file, never in `response`.

Files (append-only, resumable per row; a row is written only when its request finished):
  RUNDIR/completions-<tag>.jsonl : {id, response, finish_reason, completion_tokens, prompt_tokens, n_chunks, t_first_ms, t_total_ms, timings, r_chars, partial_chars, tag}
  RUNDIR/stream-<tag>.jsonl      : {id, ch: [[ms_since_request_start, text], ...]} content chunks, for the clip replay at real speed
  RUNDIR/gen-status-<tag>.json   : {n_total, n_done, complete, ...} rewritten every 20 rows and at exit
exit 0 = every prompt answered (file complete), 3 = prompts remain (deadline), 1 = server/harness fault (the request error is printed, nothing is guessed).
"""
import argparse, http.client, json, os, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--items", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=30000)
ap.add_argument("--deadline", type=float, default=0)
ap.add_argument("--stop-new-s", type=float, default=420.0, help="start no new prompt when fewer than this many seconds remain before --deadline")
ap.add_argument("--abort-s", type=float, default=60.0, help="cancel requests still in flight this many seconds before --deadline (their rows are not written)")
ap.add_argument("--model", default="eval")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
CP = os.path.join(a.out, "completions-%s.jsonl" % a.tag)
SP = os.path.join(a.out, "stream-%s.jsonl" % a.tag)
ST = os.path.join(a.out, "gen-status-%s.json" % a.tag)
items = [json.loads(l) for l in open(a.items)]
if a.limit:
    items = items[:a.limit]
N = len(items)


def done_ids():
    d = set()
    if os.path.exists(CP):
        for l in open(CP):
            try:
                r = json.loads(l)
                if "id" in r and "error" not in r:
                    d.add(r["id"])
            except Exception:
                pass
    return d


def final_answer(content, finish):
    """-> (response, partial_chars). Final answer only; '' when there is none."""
    text = content or ""
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    elif text.lstrip().startswith("<think>"):
        text = ""                      # think block never closed: no final answer
    text = text.strip()
    if finish == "length":
        return "", len(text)           # reasoning/answer ran into the token cap: unanswered (README)
    return text, 0


class Aborted(Exception):
    pass


def request(prompt, abort_at):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        r = conn.getresponse()
        if r.status != 200:
            raise RuntimeError("HTTP %d: %s" % (r.status, r.read(300).decode("utf-8", "replace")))
        ch, content, rchars = [], [], 0
        finish, usage, timings = None, {}, {}
        t_first = None
        while True:
            if abort_at and time.time() > abort_at:
                raise Aborted()
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            o = json.loads(data)
            if o.get("usage"):
                usage = o["usage"]
            if o.get("timings"):
                timings = o["timings"]
            for c in o.get("choices") or []:
                dl = c.get("delta") or {}
                if dl.get("reasoning_content"):
                    rchars += len(dl["reasoning_content"])
                if dl.get("content"):
                    ms = int((time.monotonic() - t0) * 1000)
                    if t_first is None:
                        t_first = ms
                    ch.append([ms, dl["content"]])
                    content.append(dl["content"])
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
        t_total = int((time.monotonic() - t0) * 1000)
        if finish is None:
            raise RuntimeError("stream ended without a finish_reason (%d chunks)" % len(ch))
        return dict(content="".join(content), ch=ch, finish=finish, usage=usage, timings=timings, rchars=rchars, t_first=t_first or t_total, t_total=t_total)
    finally:
        conn.close()


lock = threading.Lock()
state = dict(next=0, fault=None, aborted=False, written=0, tok=0, t_start=time.time())
have = done_ids()
todo = [it for it in items if it["id"] not in have]
print("[gen_eval] %s: %d prompts, %d already done, %d to do (port %d, %d workers, max_tokens %d)" % (a.tag, N, len(have), len(todo), a.port, a.workers, a.max_tokens), flush=True)
agg = dict(n_length=0, n_empty=0, tokens=0, rows=0)
if os.path.exists(CP):
    for l in open(CP):
        try:
            r = json.loads(l)
        except Exception:
            continue
        agg["rows"] += 1
        agg["tokens"] += int(r.get("completion_tokens") or 0)
        agg["n_length"] += 1 if r.get("finish_reason") == "length" else 0
        agg["n_empty"] += 1 if not (r.get("response") or "").strip() else 0


def write_status(final=False):
    done = len(done_ids())
    wall = time.time() - state["t_start"]
    st = dict(tag=a.tag, n_total=N, n_done=done, complete=(done >= N), n_length=agg["n_length"], n_empty=agg["n_empty"], completion_tokens_total=agg["tokens"],
              mean_tokens=round(agg["tokens"] / max(agg["rows"], 1), 1), this_run_wall_s=round(wall, 1),
              this_run_agg_tok_s=round(state["tok"] / max(wall, 1e-9), 1), max_tokens=a.max_tokens, workers=a.workers, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    tmp = ST + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, ST)
    return st


def worker():
    while True:
        with lock:
            if state["fault"] or state["aborted"]:
                return
            if a.deadline and time.time() > a.deadline - a.stop_new_s:
                return
            if state["next"] >= len(todo):
                return
            it = todo[state["next"]]
            state["next"] += 1
        err = None
        for attempt in range(3):
            try:
                res = request(it["prompt"], (a.deadline - a.abort_s) if a.deadline else 0)
                err = None
                break
            except Aborted:
                with lock:
                    state["aborted"] = True
                return
            except Exception as e:  # logged with its repr: a bare `continue` here once hid a shadowed-name bug in another harness
                err = repr(e)[:300]
                print("[gen_eval] request fault id=%s attempt %d: %s" % (it["id"], attempt + 1, err), flush=True)
                time.sleep(2 * (attempt + 1))
        if err:
            with lock:
                state["fault"] = "id %s: %s" % (it["id"], err)
            return
        resp, partial = final_answer(res["content"], res["finish"])
        u, tm = res["usage"], res["timings"]
        if u.get("completion_tokens") is None and tm.get("predicted_n") is not None:
            u = dict(u, completion_tokens=tm["predicted_n"])   # server without stream usage: its own predicted_n
        row = dict(id=it["id"], tag=a.tag, response=resp, finish_reason=res["finish"], completion_tokens=u.get("completion_tokens"), prompt_tokens=u.get("prompt_tokens"),
                   n_chunks=len(res["ch"]), t_first_ms=res["t_first"], t_total_ms=res["t_total"], r_chars=res["rchars"], partial_chars=partial,
                   timings={k: tm[k] for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second", "prompt_per_second", "draft_n", "draft_n_accepted") if k in tm})
        with lock:
            with open(SP, "a") as f:
                f.write(json.dumps(dict(id=it["id"], ch=res["ch"]), ensure_ascii=False) + "\n")
            with open(CP, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            state["written"] += 1
            state["tok"] += int(u.get("completion_tokens") or 0)
            agg["rows"] += 1
            agg["tokens"] += int(u.get("completion_tokens") or 0)
            agg["n_length"] += 1 if res["finish"] == "length" else 0
            agg["n_empty"] += 1 if not resp else 0
            n = state["written"]
            if n % 20 == 0:
                write_status()
            if n % 50 == 0:
                w = time.time() - state["t_start"]
                print("[gen_eval] %s %d/%d  %.0f tok/s aggregate  %.1f s/prompt  (length-capped %d, empty %d)" % (a.tag, len(have) + n, N, state["tok"] / w, w / n, agg["n_length"], agg["n_empty"]), flush=True)


ths = [threading.Thread(target=worker) for _ in range(a.workers)]
[t.start() for t in ths]
[t.join() for t in ths]
st = write_status(final=True)
print("[gen_eval] leg end %s: %d/%d rows, %d length-capped, %d empty, %.0f tok/s aggregate this run, %.0f s" % (a.tag, st["n_done"], N, st["n_length"], st["n_empty"], st["this_run_agg_tok_s"], st["this_run_wall_s"]), flush=True)
if state["fault"]:
    print("[gen_eval] FAULT %s" % state["fault"], flush=True)
    sys.exit(1)
sys.exit(0 if st["complete"] else 3)
