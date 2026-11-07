#!/usr/bin/env python3
"""make_mock_results.py <outdir> <scenario> : synthetic per-item results + facts.json for the verdict self-test and the mock render (never counted; facts.json carries "mock": true and every frame gets a MOCK banner).
scenarios: same edge down up mixed changed partial buildshift nohead"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nerf_common as N  # noqa: E402

LET = "ABCD"


def keys_for(tag, task):
    d = N.day0_task(tag, task)
    if task in ("gpqa", "he"):
        return sorted(d["items"], key=lambda k: int(k.split("/")[-1]) if "/" in k else int(k)), d
    if task == "arc":
        return [str(i) for i in range(d["n"])], d
    subs = N.d0(tag, "mmlu_detail.json")["per_subject"]
    ks = []
    for s in N.mmlu_subjects():
        ks += ["%s:%d" % (s, i) for i in range(subs[s]["total"])]
    return ks, d


def write(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def gen(tag, task, mode, rng):
    """mode: 'same' (day-0 outcomes with a few symmetric flips) | ('shift', delta_correct) | 'edge'"""
    ks, d = keys_for(tag, task)
    n = len(ks)
    paired = d["items"] is not None
    if paired:
        ok = {k: d["items"][k] for k in ks}
    else:
        c0 = d["c"]
        order = list(range(n)); rng.shuffle(order)
        ok = {ks[i]: (j < c0) for j, i in enumerate(order)}
    def flip(want_net, extra):  # flip `extra` right->wrong and extra+want_net... implemented as: b wrong-ings and c righting
        right = [k for k in ks if ok[k]]; wrong = [k for k in ks if not ok[k]]
        rng.shuffle(right); rng.shuffle(wrong)
        b = max(0, extra + (-want_net if want_net < 0 else 0)); c = max(0, extra + (want_net if want_net > 0 else 0))
        for k in right[:min(b, len(right))]: ok[k] = False
        for k in wrong[:min(c, len(wrong))]: ok[k] = True
    if mode == "same":
        flip(0, max(1, n // 40))
    elif mode == "edge":
        if task == "gpqa":
            flip(-15, 15)
        else:
            flip(0, max(1, n // 40))
    elif isinstance(mode, tuple):
        flip(mode[1], max(1, n // 40))
    recs = []
    for k in ks:
        if task == "gpqa":
            exp = d["exp"][k]; pred = exp if ok[k] else rng.choice([x for x in LET if x != exp])
        elif task == "he":
            exp, pred = "pass", "pass" if ok[k] else "fail"
        else:
            exp = "A"; pred = "A" if ok[k] else "B"
        recs.append(dict(k=k, p=pred, e=exp, ok=bool(ok[k]), tok=2, s=0.1))
    return recs


def make(outdir, scenario):
    rng = random.Random(sum(ord(ch) * (i + 1) for i, ch in enumerate(scenario)))
    os.makedirs(outdir, exist_ok=True)
    res = os.path.join(outdir, "results")
    shifts = {"down": {"gpqa": -40, "he": -25, "arc": -90, "mmlu": -220}, "up": {"gpqa": 40, "he": 6, "arc": 30, "mmlu": 200},
              "mixed": {"gpqa": 40, "he": -25, "arc": 0, "mmlu": 0}}
    for tag in N.FILES:
        for build in ("new", "old"):
            for t in N.TASKS:
                if scenario == "nohead" and build == "new" and tag == "q6k":
                    continue
                if scenario == "partial" and not (build == "new" and tag == "q6k"):
                    continue
                if scenario == "buildshift":
                    mode = (("shift", shifts["down"][t]) if (build == "new") else "same")
                elif scenario in shifts:
                    mode = ("shift", shifts[scenario][t]) if build == "new" else "same"
                elif scenario == "edge":
                    mode = "edge" if build == "new" else "same"
                else:
                    mode = "same"
                recs = gen(tag, t, mode, rng)
                if scenario == "partial" and t == "arc":
                    recs = recs[:300]
                write(os.path.join(res, "%s-%s" % (build, tag), t + ".jsonl"), recs)
    q6 = "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727"
    q4 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
    facts = dict(mock=True, hash_ok=dict(q6k=(scenario != "changed"), q4km=True), oid_aug14_q6k=q6, oid_aug14_q4km=q4,
                 sha_q6k=(q6 if scenario != "changed" else q6[:20] + "0" * 44), sha_q4km=q4, hf_weight_files_changed=0, hf_weight_files_total=18)
    json.dump(facts, open(os.path.join(outdir, "facts.json"), "w"))


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
    print("mock", sys.argv[2], "->", sys.argv[1])
