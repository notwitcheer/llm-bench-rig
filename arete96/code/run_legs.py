#!/usr/bin/env python3
"""run_legs.py: run the day-0 board items of Qwen3.8-27B against ONE already-running llama-server, one item at a time (as the day-0 harness did), saving every answer.
usage: run_legs.py --port P --build old|new --file q6k|q4km --tasks gpqa,he,arc,mmlu --out RESDIR --deadline EPOCH [--limit N]
Same settings as day 0 (benchmark-rig commit 6c7ecd8, copied unchanged to nerfevals/): think-off, temperature 0, max_tokens 2048 (4096 for HumanEval), the same few-shot, the same parser.
Resumable per item: <out>/<build>-<file>/<task>.jsonl gets one line per finished item {k, p (answer), e (expected), ok, tok, s}. No new item starts in the last 4 min before --deadline.
exit 0 = every requested task complete, 3 = items remain (deadline), 1 = server/harness fault."""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from nerf_common import load_items, read_arm, expected_n  # noqa: E402
from nerfevals.base import LLMClient, parse_choice  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--build", required=True, choices=["old", "new"])
ap.add_argument("--file", required=True)
ap.add_argument("--tasks", default="gpqa,he,arc,mmlu")
ap.add_argument("--out", required=True)
ap.add_argument("--deadline", type=float, default=0)
ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N items of each task (never counted)")
a = ap.parse_args()

d = os.path.join(a.out, "%s-%s" % (a.build, a.file))
os.makedirs(d, exist_ok=True)
client = LLMClient("http://127.0.0.1:%d/v1" % a.port, "nerf", timeout=1800, think=False)
remain = 0
t_leg = time.time()
for task in a.tasks.split(","):
    done = read_arm(a.out, a.build, a.file, task)
    items = list(load_items(task))
    if a.limit:
        items = items[:a.limit]
    todo = [it for it in items if it["key"] not in done]
    print("[run_legs] %s-%s %s: %d items, %d done, %d to do" % (a.build, a.file, task, len(items), len(done), len(todo)), flush=True)
    t0 = time.time()
    with open(os.path.join(d, task + ".jsonl"), "a") as fh:
        for n, it in enumerate(todo):
            if a.deadline and time.time() > a.deadline - 240:
                remain += len(todo) - n
                print("[run_legs] deadline: %d %s items left (resumable)" % (len(todo) - n, task), flush=True)
                break
            t1 = time.time()
            try:
                if it["kind"] == "mc":
                    resp = client.chat(it["messages"], max_tokens=it["max_tokens"])
                    pred = parse_choice(resp, valid=it["valid"])
                    rec = dict(k=it["key"], p=pred, e=it["expected"], ok=(pred == it["expected"]))
                else:
                    from nerfevals.humaneval import build_executable_program, _execute
                    resp = client.chat(it["messages"], max_tokens=it["max_tokens"], preserve_indent=True)
                    prog = build_executable_program(it["prompt"], resp, it["test"], it["entry"])
                    ok, err = _execute(prog, timeout=10)
                    rec = dict(k=it["key"], p="pass" if ok else "fail", e="pass", ok=bool(ok), err=(err or "")[-80:])
            except Exception as e:  # a request fault is recorded, never silently dropped: it is re-run on resume only if absent from the file
                print("[run_legs] FAULT %s %s: %s" % (task, it["key"], str(e)[:200]), flush=True)
                sys.exit(1)
            rec["tok"] = client.last_completion_tokens
            rec["s"] = round(time.time() - t1, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (n + 1) % 100 == 0:
                print("[run_legs] %s %d/%d  %.1f s/item" % (task, n + 1, len(todo), (time.time() - t0) / (n + 1)), flush=True)
    got = read_arm(a.out, a.build, a.file, task)
    print("[run_legs] leg end %s-%s %s: %d/%d items, %d correct, %.0f s" % (a.build, a.file, task, len(got), len(items), sum(1 for v in got.values() if v["ok"]), time.time() - t0), flush=True)
print("[run_legs] leg end %s-%s all tasks, %.0f s, remain %d" % (a.build, a.file, time.time() - t_leg, remain), flush=True)
sys.exit(3 if remain else 0)
