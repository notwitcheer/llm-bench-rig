#!/usr/bin/env python3
"""wd_common.py: shared constants + helpers for item 95 (weight editor). Frozen by PREREG.md (sha256 in FROZEN.sha256).

The model under edit is Qwen/Qwen3.5-0.8B (revision 2fc06364715b967f1860aea9cf38778875588b17), served from a bf16 GGUF by llama-server; an edit is a patch of the bf16 bytes of
blk.<L>.ffn_down.weight inside a COPY of that GGUF (the bytes equal the safetensors tensor model.language_model.layers.<L>.mlp.down_proj.weight, checked at staging).
Loop = the thinking never closes (no </think>, the answer field is empty) inside the token cap. Problem sets are GSM8K, fixed by seed (disjoint sources: dev = train split, held-out = test split).
"""
import json, os, random, re, struct, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
LAYERS = 24
D_FF = 3584
D_MODEL = 1024
SAMP = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)    # the replication's sampler (Qwen's thinking-mode recommendation + presence penalty 1.5)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CAP = 16384
N_DEV, N_HELD = 20, 40
SEED_SET = 20261106
Z95 = 1.959963984540054


def gsm_gold(ans):
    m = re.search(r"####\s*(.*)", ans)
    return (m.group(1) if m else ans).strip().replace(",", "")


def problems(kind):
    """kind dev|held -> list of dict(id, problem, gold). dev = GSM8K train, held-out = GSM8K test, random.Random(SEED_SET)."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    split, n = ("train", N_DEV) if kind == "dev" else ("test", N_HELD)
    rows = list(ds[split])
    idx = random.Random(SEED_SET).sample(range(len(rows)), n)
    return [dict(id="%s%d" % ("tr" if kind == "dev" else "te", i), problem=rows[i]["question"], gold=gsm_gold(rows[i]["answer"])) for i in idx]


def boxed(text):
    i = (text or "").rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(text) and d:
        d += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1] if d == 0 else None


def norm_num(s):
    s = (s or "").replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").replace("\\!", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m[-1])
    return str(int(v)) if v == int(v) else str(v)


def is_correct(content, gold):
    b = boxed(content)
    return b is not None and norm_num(b) == norm_num(gold)


def post(url, obj, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def wilson(c, n, z=Z95):
    import math
    if n <= 0:
        return 0.0, 1.0
    p = c / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_less(a1, n1, a2, n2):
    """one-sided exact p that arm 1 (a1 of n1 loops) has a LOWER loop rate than arm 2 (a2 of n2), hypergeometric."""
    import math
    tot, k = a1 + a2, n1 + n2
    p = 0.0
    for x in range(0, a1 + 1):
        if x > n1 or tot - x > n2:
            continue
        p += math.comb(n1, x) * math.comb(n2, tot - x) / math.comb(k, tot)
    return min(1.0, p)


def run_problems(port, probs, nslots, cap=CAP, seed=1, on_done=None):
    """sample one try per problem against a running llama-server (OpenAI chat route, thinking on); returns list of rows in problem order. nslots tries run at once."""
    import concurrent.futures as cf
    def one(i):
        p = probs[i]
        body = dict(SAMP)
        body.update(model="wd", messages=[{"role": "user", "content": p["problem"] + SUFFIX}], max_tokens=cap, seed=seed * 1000 + i, stream=False,
                    chat_template_kwargs={"enable_thinking": True})
        t = time.time()
        r = post("http://127.0.0.1:%d/v1/chat/completions" % port, body)
        m = r["choices"][0]["message"]
        content, think = (m.get("content") or ""), (m.get("reasoning_content") or "")
        row = dict(id=p["id"], gold=p["gold"], closed=bool(content.strip()), correct=is_correct(content, p["gold"]), finish=r["choices"][0].get("finish_reason"),
                   tokens=(r.get("usage") or {}).get("completion_tokens"), think=think, answer=content, s=round(time.time() - t, 1))
        if on_done:
            on_done(row)
        return row
    with cf.ThreadPoolExecutor(max_workers=nslots) as ex:
        return list(ex.map(one, range(len(probs))))


def summarise(rows):
    n = len(rows)
    loops = sum(1 for r in rows if not r["closed"])
    return dict(n=n, loops=loops, loop_rate=round(loops / n, 4) if n else None, closed=n - loops, correct=sum(1 for r in rows if r["correct"]),
                mean_tokens=round(sum(r["tokens"] or 0 for r in rows) / n, 1) if n else None)
