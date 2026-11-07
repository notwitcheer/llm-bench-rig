#!/usr/bin/env python3
"""wd_make_mock.py <RUNDIR> <scenario> [--take TAKEDIR] : synthetic arm files for the verdict self-test and the mock render (never counted; summary.json carries "mock": true, every frame a MOCK banner).
scenarios: cured (editor ~ script) | editorfail (editor does not cure, script does) | nocand. Looping rows get a repeated 'wait, let me re-check' thinking text; closed rows a short answer."""
import json, os, random, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wd_common as C  # noqa: E402

LOOP = ("Wait, let me re-check the first step. The problem says the total is made of the parts, so I should add the parts again. "
        "Wait, actually, let me re-read: if I add them in a different order I should get the same total. Hmm, let me verify once more. ")


def rows(n, loops, rng, setname):
    idx = list(range(n))
    rng.shuffle(idx)
    lp = set(idx[:loops])
    out = []
    for i in range(n):
        if i in lp:
            think = ("Here's my thought process for solving this word problem: step %d. " % i) + LOOP * 60
            out.append(dict(id="%s%d" % ("te" if setname == "held" else "tr", i), gold="1", closed=False, correct=False, finish="length", tokens=C.CAP, think=think, answer="", s=300.0))
        else:
            think = "Let me work it out. 3 + 4 = 7, then 7 * 2 = 14. So the answer is 14. Check: 14 / 2 = 7, 7 - 4 = 3. Good."
            out.append(dict(id="%s%d" % ("te" if setname == "held" else "tr", i), gold="14", closed=True, correct=True, finish="stop", tokens=900 + 37 * i, think=think, answer="The answer is \\boxed{14}.", s=30.0))
    return out


def write_arm(rd, arm, st, loops, rng):
    n = C.N_HELD if st == "held" else C.N_DEV
    r = rows(n, loops, rng, st)
    s = C.summarise(r)
    s.update(set=st, tag=arm)
    os.makedirs(os.path.join(rd, "arms"), exist_ok=True)
    json.dump(dict(summary=s, rows=r), open(os.path.join(rd, "arms", "%s-%s.json" % (arm, st)), "w"))


def make(rd, scen, take=None):
    rng = random.Random(sum(ord(c) for c in scen))
    H = {"cured": dict(base=36, published=8, scripted=10, editor=12), "editorfail": dict(base=36, published=8, scripted=10, editor=35), "nocand": dict(base=36, published=8, scripted=10, editor=12)}[scen]
    for arm, loops in H.items():
        if scen == "nocand" and arm == "editor":
            continue
        write_arm(rd, arm, "held", loops, rng)
        write_arm(rd, arm, "dev", max(1, round(loops * C.N_DEV / C.N_HELD)), rng)
    os.makedirs(os.path.join(rd, "take", "state"), exist_ok=True)
    if take:
        for sub in ("rec", "hud"):
            if os.path.isdir(os.path.join(take, sub)):
                shutil.copytree(os.path.join(take, sub), os.path.join(rd, "take", sub), dirs_exist_ok=True)
        shutil.copy(os.path.join(take, "state", "calls.jsonl"), os.path.join(rd, "take", "state", "calls.jsonl"))
    gate = dict(scored=0 if scen == "nocand" else 3, best=None if scen == "nocand" else "m3", flags=[], why="no wd eval finished" if scen == "nocand" else "", calls={"eval": 3})
    json.dump(gate, open(os.path.join(rd, "take", "gate.json"), "w"))
    json.dump(dict(mock=True), open(os.path.join(rd, "facts.json"), "w"))


if __name__ == "__main__":
    t = sys.argv[sys.argv.index("--take") + 1] if "--take" in sys.argv else None
    make(sys.argv[1], sys.argv[2], t)
    print("mock", sys.argv[2], "->", sys.argv[1])
