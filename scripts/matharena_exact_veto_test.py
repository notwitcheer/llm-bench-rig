import collections, signal, sys
import sympy
from datasets import load_dataset
from lib.evals.matharena import grade_with_timeout
from latex2sympy2_extended import latex2sympy

class TO(Exception): pass
def _h(*a): raise TO()
signal.signal(signal.SIGALRM, _h)

def to_exact(s):
    s = str(s).strip().strip("$")
    try:
        e = latex2sympy(s)
    except Exception:
        try: e = sympy.sympify(s)
        except Exception: return None
    try:
        if not getattr(e, "is_number", False) or e.atoms(sympy.Float): return None
    except Exception: return None
    return e

def exact_veto(pred, gold, secs=5):
    """True = our tolerant verdict must be vetoed: both sides exact numbers and provably different."""
    if pred is None: return False
    signal.alarm(secs)
    try:
        g = to_exact(gold); p = to_exact(pred)
        if g is None or p is None: return False
        d = sympy.nsimplify(sympy.simplify(g - p)) if False else sympy.simplify(g - p)
        return bool(d != 0) and d.is_zero is False
    except TO: return False
    except Exception: return False
    finally: signal.alarm(0)

for s in (sys.argv[1:] or ["hmmt_feb_2026", "aime_2026"]):
    ds = load_dataset(f"MathArena/{s}_outputs", split="train")
    conf = collections.Counter(); vetoes = collections.Counter()
    for r in ds:
        lab = bool(r["correct"]); gold = str(r["gold_answer"])
        pa = r.get("parsed_answer"); pa = None if pa in (None, "", "None") else str(pa)
        ours, _ = grade_with_timeout(pa, gold, secs=10)
        if ours and exact_veto(pa, gold):
            vetoes[(lab, r.get("problem_idx"), gold[:40], (pa or "")[:40])] += 1
            ours = False
        conf[(lab, ours)] += 1
    tot = sum(conf.values()); agree = conf[(True, True)] + conf[(False, False)]
    print(f"{s} WITH VETO: rows {tot}, agree {agree/tot:.1%}, judge-true/ours-false {conf[(True, False)]}, judge-false/ours-true {conf[(False, True)]}", flush=True)
    print("  vetoes (judge_label, idx, gold, parsed): ", flush=True)
    for k, v in vetoes.most_common(12): print("   ", v, k, flush=True)
