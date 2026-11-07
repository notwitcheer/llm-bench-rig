#!/usr/bin/env python3
"""nerf_common.py: shared constants + helpers for the Qwen3.8-27B 'did they nerf it' audit (item 90a-qwen38-nerf-audit). Frozen by PREREG.md (sha256 in FROZEN.sha256).

The item sets are the day-0 board's own items (same datasets, same sampling seed, same few-shot, same prompts, same parser), taken from the day-0 harness
source (benchmark-rig commit 6c7ecd8, copied to nerfevals/ unchanged). Only the tasks that have a day-0 comparator on exactly these items are run:
  gpqa  : 198 items (day-0 per-item predicted letters exist)             -> paired per item
  he    : 164 HumanEval tasks (day-0 failing task ids exist)             -> paired per item
  arc   : 1172 ARC-Challenge test items (day-0 total only)               -> aggregate
  mmlu  : 19 of 57 subjects (sorted, every 3rd), the day-0 50% sample    -> aggregate over the same items (day-0 per-subject counts)
"""
import json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FILES = {  # tag -> (gguf file name, day-0 board slug, day-0 dir name under ../day0)
    "q6k": "Qwen3.8-27B-Q6_K.gguf",
    "q4km": "Qwen3.8-27B-Q4_K_M.gguf",
}
TASKS = ["gpqa", "he", "arc", "mmlu"]          # run order inside a pass: the short, paired ones first
TASK_LABEL = {"gpqa": "GPQA-diamond", "he": "HumanEval", "arc": "ARC-Challenge", "mmlu": "MMLU (19 subjects)"}
Z95 = 1.959963984540054
DAY0 = os.path.join(os.path.dirname(HERE), "day0")


def wilson(c, n, z=Z95):
    if n <= 0:
        raise ValueError("n")
    p = c / n
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    lo, hi = centre - half, centre + half
    if c == 0:
        lo = 0.0
    if c == n:
        hi = 1.0
    return max(0.0, lo), min(1.0, hi)


def mcnemar_exact(b, c):
    """two-sided exact binomial p on the discordant pairs (b = A right/B wrong, c = A wrong/B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def mmlu_subjects():
    import nerfevals.mmlu as m
    return sorted(m.SUBJECTS)[::3]


# ------------------------------------------------------------------------------------------------ day-0 comparators (read from the board's own json)
def d0(tag, name):
    return json.load(open(os.path.join(DAY0, tag, name)))


def day0_task(tag, task):
    """-> dict(c, n, items) ; items = {key: correct(bool)} when the day-0 record has per-item data, else None. tag = q6k | q4km."""
    if task == "gpqa":
        comp = d0(tag, "gpqa_progress.json")["completed"]
        return dict(c=sum(1 for v in comp.values() if v["correct"]), n=len(comp),
                    items={k: bool(v["correct"]) for k, v in comp.items()}, pred={k: v["predicted"] for k, v in comp.items()}, exp={k: v["expected"] for k, v in comp.items()})
    if task == "he":
        det = d0(tag, "humaneval_detail.json")
        fails = {e["task_id"] for e in det["errors"]}
        assert len(fails) == det["total"] - det["passed"], "day-0 humaneval error list is capped: per-item pairing impossible"
        return dict(c=det["passed"], n=det["total"], items={"HumanEval/%d" % i: ("HumanEval/%d" % i) not in fails for i in range(det["total"])})
    if task == "arc":
        det = d0(tag, "arc_challenge_detail.json")
        return dict(c=det["correct"], n=det["total"], items=None)
    if task == "mmlu":
        ps = d0(tag, "mmlu_detail.json")["per_subject"]
        subs = mmlu_subjects()
        return dict(c=sum(ps[s]["correct"] for s in subs), n=sum(ps[s]["total"] for s in subs), items=None, per_subject={s: ps[s] for s in subs})
    raise ValueError(task)


# ------------------------------------------------------------------------------------------------ the item sets (identical to the day-0 harness)
def load_items(task):
    """yield dicts: key, messages, kind('mc'|'code'), expected, valid, extra. Needs the cached datasets (HF_DATASETS_OFFLINE=1 is fine)."""
    from datasets import load_dataset
    if task == "gpqa":
        import nerfevals.gpqa as g
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for i, raw in enumerate(ds):
            it = g._build_item(raw, i)
            yield dict(key=str(i), messages=g._build_messages(it), kind="mc", expected=g.LETTERS[it["answer"]], valid="ABCD", max_tokens=2048)
    elif task == "arc":
        import nerfevals.arc as a
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        few = list(ds["train"])[:25]
        for i, it in enumerate(list(ds["test"])):
            yield dict(key=str(i), messages=a._build_messages(it, few), kind="mc", expected=a._normalize_label(it["answerKey"]), valid=a._valid_letters(it), max_tokens=2048)
    elif task == "mmlu":
        import nerfevals.mmlu as m
        for subj in mmlu_subjects():
            ds = load_dataset("cais/mmlu", subj)
            few = list(ds["dev"])[:5]
            test = list(ds["test"])
            rng = random.Random(42)
            k = max(1, int(len(test) * 0.5))
            idx = sorted(rng.sample(range(len(test)), k))
            for i in idx:
                yield dict(key="%s:%d" % (subj, i), subject=subj, messages=m._build_messages(subj, test[i], few), kind="mc", expected=m.LETTERS[test[i]["answer"]], valid="ABCD", max_tokens=2048)
    elif task == "he":
        import nerfevals.humaneval as h
        ds = load_dataset("openai/openai_humaneval")
        for it in list(ds["test"]):
            yield dict(key=it["task_id"], messages=h._build_messages(it["prompt"]), kind="code", prompt=it["prompt"], test=it["test"], entry=it["entry_point"], max_tokens=4096)
    else:
        raise ValueError(task)


def identity_check():
    """CPU proof that our item sets ARE the day-0 item sets. Returns a list of problems (empty = identical)."""
    bad = []
    for tag in FILES:
        pg = day0_task(tag, "gpqa")
        for it in load_items("gpqa"):
            if pg["exp"].get(it["key"]) != it["expected"]:
                bad.append("%s gpqa item %s expected letter differs from day 0" % (tag, it["key"]))
                break
    cnt = {}
    for it in load_items("mmlu"):
        cnt[it["subject"]] = cnt.get(it["subject"], 0) + 1
    for tag in FILES:
        ps = d0(tag, "mmlu_detail.json")["per_subject"]
        for s, n in cnt.items():
            if ps[s]["total"] != n:
                bad.append("%s mmlu %s: %d items now vs %d day-0" % (tag, s, n, ps[s]["total"]))
    n_arc = sum(1 for _ in load_items("arc"))
    for tag in FILES:
        if day0_task(tag, "arc")["n"] != n_arc:
            bad.append("%s arc size %d vs day-0 %d" % (tag, n_arc, day0_task(tag, "arc")["n"]))
    ids = {it["key"] for it in load_items("he")}
    if len(ids) != 164 or any(("HumanEval/%d" % i) not in ids for i in range(164)):
        bad.append("humaneval ids")
    return bad


# ------------------------------------------------------------------------------------------------ per-arm results on disk
def read_arm(resdir, arm, tag, task):
    """-> dict key -> record for <resdir>/<arm>-<tag>/<task>.jsonl (last record per key wins)."""
    p = os.path.join(resdir, "%s-%s" % (arm, tag), task + ".jsonl")
    out = {}
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
                out[r["k"]] = r
            except Exception:
                pass
    return out


TASK_N = {"gpqa": 198, "he": 164, "arc": 1172, "mmlu": None}   # mmlu n comes from the day-0 sample (computed)


def expected_n(tag, task):
    return day0_task(tag, task)["n"]


def pass_pending(resdir, build, tag, limit=0):
    """items still to do for one pass (4 tasks); `limit` is the smoke knob (first N items per task)."""
    miss = 0
    for t in TASKS:
        want = expected_n(tag, t)
        if limit:
            want = min(want, limit)
        have = read_arm(resdir, build, tag, t)
        if limit:
            miss += max(0, want - len(have))
        else:
            miss += max(0, want - len(have))
    return miss


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pending":
        print(pass_pending(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0))
    elif len(sys.argv) > 1 and sys.argv[1] == "identity":
        b = identity_check()
        print("IDENTITY " + ("OK" if not b else "BAD " + "; ".join(b[:3])))
