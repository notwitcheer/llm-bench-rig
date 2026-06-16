"""Run the full microcosm sweep on capsule: {action_only, echo} x {6,7,8} x 3 seeds.
Writes results/echo_maze/results.json (raw per-seed solve rates + mean/std + the gap by size).
Each arm trains one model per seed on a MIX of all sizes; eval is per size on a fixed maze set.

OBS_MODE=walls_only drops the goal bearing (the agent must build an internal map instead of
following the handed-over direction) and writes results_walls_only.json — the decisive test of
whether the full-obs null is real or a bearing artifact."""
import json
import os
import statistics as st
from pathlib import Path
import torch

from lib.echo_maze.train import train
from lib.echo_maze.rollout import solve_rate

ARMS = {"action_only": 0.0, "echo": 1.0}
SIZES = [6, 7, 8]
SEEDS = [0, 1, 2]
EVAL_SEED = 999          # fixed => identical eval mazes for every arm/seed
N_EVAL = 300
N_TRAIN_PER_SIZE = 4000
STEPS = 3000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    walls_only = os.environ.get("OBS_MODE") == "walls_only"
    include_bearing = not walls_only
    out_path = "results/echo_maze/results_walls_only.json" if walls_only else "results/echo_maze/results.json"
    print(f"OBS_MODE={'walls_only' if walls_only else 'full'} include_bearing={include_bearing} -> {out_path}",
          flush=True)
    raw = {arm: {s: [] for s in SIZES} for arm in ARMS}
    for arm, lam in ARMS.items():
        for seed in SEEDS:
            model, log = train(lam=lam, seed=seed, sizes=SIZES,
                               n_train_per_size=N_TRAIN_PER_SIZE, steps=STEPS, device=device,
                               include_bearing=include_bearing)
            for size in SIZES:
                sr = solve_rate(model, size, N_EVAL, EVAL_SEED, device=device,
                                include_bearing=include_bearing)
                raw[arm][size].append(sr)
                print(f"{arm} seed={seed} size={size} loss={log[-1]['loss']:.3f} solve={sr:.3f}",
                      flush=True)
    agg = {arm: {s: {"mean": st.mean(raw[arm][s]),
                     "std": st.pstdev(raw[arm][s]) if len(raw[arm][s]) > 1 else 0.0,
                     "raw": raw[arm][s]} for s in SIZES} for arm in ARMS}
    gap = {s: agg["echo"][s]["mean"] - agg["action_only"][s]["mean"] for s in SIZES}
    out = {"arms": list(ARMS), "sizes": SIZES, "seeds": SEEDS, "n_eval": N_EVAL,
           "steps": STEPS, "n_train_per_size": N_TRAIN_PER_SIZE, "agg": agg, "gap": gap}
    out["obs_mode"] = "walls_only" if walls_only else "full"
    Path("results/echo_maze").mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    for s in SIZES:                                   # sanity gate
        for arm in ARMS:
            assert all(0.0 <= v <= 1.0 for v in agg[arm][s]["raw"])
    print("GAP by size (echo - action_only):", {s: round(gap[s], 3) for s in SIZES})


if __name__ == "__main__":
    main()
