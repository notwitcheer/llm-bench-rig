#!/usr/bin/env python3
"""arm9a1.py gsm8k <upstream> <alias> <out.json> <arm>   item 9a1: first 20 rows of openai/gsm8k main test, greedy, thinking OFF.
stdlib + pyarrow (parquet read from the local HF cache). The tool-call smoke and the decode probe live inline in the item script."""
import glob
import json
import os
import re
import sys
import urllib.request


def post(up, body, timeout=600):
    req = urllib.request.Request(up + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def gold_rows(n=20):
    import pyarrow.parquet as pq
    f = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")))[0]
    t = pq.read_table(f).slice(0, n).to_pylist()
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in t]


def last_number(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(nums[-1].rstrip(".")) if nums else None


def gsm8k(up, alias, out, arm):
    res = []
    for i, (q, g) in enumerate(gold_rows(20)):
        body = {"model": alias, "messages": [{"role": "user", "content": q + "\nGive the final answer as a number on the last line."}],
                "max_tokens": 512, "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = post(up, body)
            txt = r["choices"][0]["message"].get("content") or ""
            p = last_number(txt)
            ok = p is not None and abs(p - float(g)) < 1e-9
        except Exception as e:
            p, ok = None, False
            txt = "ERR %s" % e
        res.append({"i": i, "gold": g, "pred": p, "ok": ok})
    d = {"arm": arm, "n": len(res), "correct": sum(r["ok"] for r in res), "rows": res}
    json.dump(d, open(out, "w"), indent=1)
    print("%s gsm8k %d/%d" % (arm, d["correct"], d["n"]))


if __name__ == "__main__":
    if sys.argv[1] == "gsm8k":
        gsm8k(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        sys.exit("usage: arm9a1.py gsm8k <upstream> <alias> <out.json> <arm>")
