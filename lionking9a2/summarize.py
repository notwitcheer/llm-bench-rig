#!/usr/bin/env python3
"""Applies PREREG.md to finished run dirs. Nothing is worded by hand.

usage: summarize.py --headline-dir RUN/headline --ext-dir RUN/ext20 --headline lk --out RUN/summary.json
  headline-dir: one run dir (results.json + <chain>.npz + forced.json)
  ext-dir:      a dir of per-chain run dirs (each with results.json); optional
Writes summary.json and readout.md next to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K = 12
N = 1 << K
NAMES = ["Z_zero", "Z_resample", "Z_random", "P_hop1", "P_other", "P_random"]
CTRL = "P_random"
DATA = Path("/home/witcheer/esoteric/lionking9a2/repo/data/chains.jsonl")


def minimal_flipping(F: np.ndarray) -> list[int]:
    """flipping subsets with no flipping proper subset (strict inclusion)."""
    mins = []
    for s in np.nonzero(F)[0]:
        s = int(s)
        t = (s - 1) & s
        minimal = True
        while t > 0:
            if F[t]:
                minimal = False
                break
            t = (t - 1) & s
        if minimal:
            mins.append(s)
    return mins


def bits(s: int) -> list[int]:
    return [i for i in range(K) if s >> i & 1]


def analyse(run: Path, res: dict, forced: set[str]) -> dict:
    c = res["chain"]
    npz = np.load(run / f"{c}.npz")
    d = {k: npz[k].astype(float) for k in NAMES}
    assert d["P_hop1"].shape == (N,), d["P_hop1"].shape
    m0 = float(res["m0"])
    # score.py: flipped = margin < 0. patched margin = m0 - delta, so flipped iff m0 - delta < 0.
    F = (m0 - d["P_hop1"]) < 0
    F[0] = False  # row 0 is the empty set (S=empty)
    mins = minimal_flipping(F)
    disj = [(a, b) for i, a in enumerate(mins) for b in mins[i + 1:] if a & b == 0]
    full = {k: float(d[k][N - 1]) for k in NAMES}
    flip = {k: bool((m0 - d[k][N - 1]) < 0) for k in NAMES}
    gen = res["generation"]["P_hop1"]["full"]
    noise = float(res["noise"]["P_hop1"])
    gate_ok = m0 >= 3 * noise
    if not gate_ok:
        verdict = "INVALID-NOISE"
    elif flip["P_hop1"] and not flip[CTRL] and gen["starts_with_bridge"]:
        verdict = "REPRODUCED ON ONE CHAIN"
    elif flip["P_hop1"]:
        verdict = "PARTIAL"
    else:
        verdict = "NOT REPRODUCED"
    if F.sum() == 0:
        uniq = "NO FLIP"
    elif disj:
        uniq = "NOT UNIQUE"
    else:
        uniq = "UNIQUE"
    ex = disj[0] if disj else None
    return {
        "chain": c,
        "answer": res["answer"],
        "bridge": res["bridge"],
        "m0": m0,
        "gate": "forced" if c in forced else "pass",
        "noise_P_hop1": noise,
        "full_delta": full,
        "full_flipped": flip,
        "gen_clean": res["generation"]["P_hop1"]["none"]["text"],
        "gen_patched": gen["text"],
        "gen_starts_with_bridge": bool(gen["starts_with_bridge"]),
        "P_hop1_flip_subsets": int(F.sum()),
        "P_hop1_minimal_flipping": len(mins),
        "P_hop1_min_size": min((len(bits(s)) for s in mins), default=None),
        "P_hop1_disjoint_pairs": len(disj),
        "P_hop1_disjoint_example": [bits(ex[0]), bits(ex[1])] if ex else None,
        "verdict": verdict,
        "uniqueness": uniq,
        "margin_patched_full": m0 - full["P_hop1"],
        "margin_random_full": m0 - full["P_random"],
    }


def line1(h: dict) -> str | None:
    v, F, D = h["verdict"], h["P_hop1_flip_subsets"], h["P_hop1_disjoint_pairs"]
    if v == "REPRODUCED ON ONE CHAIN":
        alt = "no disjoint pair flips it" if D == 0 else f"{D} disjoint pairs flip it too"
        return (f"the lion king's answer flips when we patch 12 neurons from another question, "
                f"and {F:,} of 4,096 subsets flip it. {alt}. one chain on qwen3-1.7B-base, the paper's model")
    if v == "PARTIAL":
        return (f"the lion king patch flips the answer on qwen3-1.7B-base, but the random control flips it too "
                f"or the greedy answer does not change: {F:,} of 4,096 subsets flip it")
    if v == "NOT REPRODUCED":
        return f"the lion king patch did not flip the answer on qwen3-1.7B-base: {F:,} of 4,096 subsets flip it"
    return None


def load_dir(d: Path) -> tuple[dict, set[str]]:
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    fp = d / "forced.json"
    forced = set(json.loads(fp.read_text(encoding="utf-8"))["forced"]) if fp.exists() else set()
    return res, forced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline-dir", required=True)
    ap.add_argument("--ext-dir")
    ap.add_argument("--headline", default="lk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    hdir = Path(a.headline_dir)
    res, forced = load_dir(hdir)
    head_rows = [analyse(hdir, c, forced) for c in res["chains"] if c["chain"] == a.headline]
    if not head_rows:
        sys.exit(f"headline chain {a.headline} not in {hdir}")
    h = head_rows[0]

    ext = []
    if a.ext_dir and Path(a.ext_dir).is_dir():
        for sub in sorted(p for p in Path(a.ext_dir).iterdir() if (p / "results.json").exists()):
            r, f = load_dir(sub)
            for c in r["chains"]:
                ext.append(analyse(sub, c, f))
    ext_by = {e["chain"]: e for e in ext}
    if ext and a.headline in ext_by:
        same = ext_by[a.headline]["P_hop1_flip_subsets"] == h["P_hop1_flip_subsets"]
    else:
        same = None

    # margin split (paper's exploratory claim: disjoint alternatives more common at low margin)
    flipped_ext = [e for e in ext if e["P_hop1_flip_subsets"] > 0]
    lo = [e for e in flipped_ext if e["m0"] < 4]
    hi = [e for e in flipped_ext if e["m0"] >= 4]

    summ = {
        "mock": a.mock,
        "headline_chain": a.headline,
        "prompt": next(json.loads(l)["prompt"] for l in DATA.read_text(encoding="utf-8").splitlines()
                       if l.strip() and json.loads(l)["id"] == f"{a.headline}_comp1"),
        "headline": h,
        "line1": line1(h),
        "ext_n": len(ext),
        "ext_reproduced": sum(e["verdict"] == "REPRODUCED ON ONE CHAIN" for e in ext),
        "ext_flip": sum(e["P_hop1_flip_subsets"] > 0 for e in ext),
        "ext_not_unique": sum(e["uniqueness"] == "NOT UNIQUE" for e in ext),
        "ext_margin_split": {
            "m0_lt_4": {"n": len(lo), "not_unique": sum(e["uniqueness"] == "NOT UNIQUE" for e in lo)},
            "m0_ge_4": {"n": len(hi), "not_unique": sum(e["uniqueness"] == "NOT UNIQUE" for e in hi)},
        },
        "headline_matches_ext_row": same,
        "chains": ext,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")

    L = [f"# 9a2 readout ({'MOCK, not counted' if a.mock else 'counted'})", "",
         f"headline chain `{a.headline}` on Qwen/Qwen3-1.7B-Base (gate: {h['gate']}, m0 {h['m0']:.3f})", "",
         f"- verdict: **{h['verdict']}** (rule: PREREG.md)",
         f"- full 12-neuron patch, P_hop1: delta {h['full_delta']['P_hop1']:.2f}, flips: {h['full_flipped']['P_hop1']}",
         f"- random 12 (layer-matched): delta {h['full_delta']['P_random']:.2f}, flips: {h['full_flipped'][CTRL]}",
         f"- greedy patched answer: {h['gen_patched']!r} (starts with bridge: {h['gen_starts_with_bridge']})",
         f"- noise floor (bf16): {h['noise_P_hop1']:.3f}",
         f"- flipping subsets: {h['P_hop1_flip_subsets']:,} of 4,096; minimal flipping sets: {h['P_hop1_minimal_flipping']}; "
         f"smallest {h['P_hop1_min_size']} neurons",
         f"- disjoint pairs among minimal flipping sets: {h['P_hop1_disjoint_pairs']} -> {h['uniqueness']}",
         "", f"line 1 (as it would post): {summ['line1']}", ""]
    if ext:
        L += ["## extension: all chains (forced inclusion, labelled)", "",
              f"- chains run: {len(ext)}; flip (any subset): {summ['ext_flip']}; "
              f"verdict REPRODUCED ON ONE CHAIN: {summ['ext_reproduced']}; NOT UNIQUE: {summ['ext_not_unique']}",
              f"- disjoint-pair share by margin (paper's exploratory cut, post hoc): "
              f"m0<4 {summ['ext_margin_split']['m0_lt_4']['not_unique']}/{summ['ext_margin_split']['m0_lt_4']['n']} not unique; "
              f"m0>=4 {summ['ext_margin_split']['m0_ge_4']['not_unique']}/{summ['ext_margin_split']['m0_ge_4']['n']}",
              "", "| chain | m0 | gate | verdict | flip subsets | disjoint pairs | greedy patched |",
              "|---|---|---|---|---|---|---|"]
        for e in ext:
            L.append(f"| {e['chain']} | {e['m0']:.2f} | {e['gate']} | {e['verdict']} | {e['P_hop1_flip_subsets']} | "
                     f"{e['P_hop1_disjoint_pairs']} | {e['gen_patched'][:20]!r} |")
    (out.parent / "readout.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"verdict={h['verdict']} line1={summ['line1']!r}")


if __name__ == "__main__":
    main()
