# ECHO has no maze, so I built one: the "free world model" loss is inert in pure imitation

**Rig:** one RTX 5090 32GB · torch 2.10+cu128 (sm_120) · a 10M-param decoder-only transformer trained from scratch (minutes per run, ~5.7GB peak, runs alongside a live 27B server)
**Paper:** ECHO — *Terminal Agents Learn World Models for Free* ([arXiv 2605.24517](https://arxiv.org/abs/2605.24517), code [microsoft/echo-rl](https://github.com/microsoft/echo-rl)). The idea: standard agent RL (GRPO) trains only on action tokens and masks the environment output; ECHO adds a free cross-entropy on those env-observation tokens (already in the rollout, same forward pass) so the policy learns an implicit world model. `world_model_coeff=0.0` recovers vanilla GRPO; `>0` is ECHO.
**First, a correction:** the paper and repo contain **no maze microcosm** — I grepped both (`maze`=0, `microcosm`=0, `toy`=0). ECHO is exclusively a full-scale terminal-agent RL paper (Qwen3-8B/14B). So this isn't a reproduction — it's a microcosm I designed to isolate ECHO's *mechanism* on hardware anyone can run, before committing GPU-weeks to the real thing.

## The setup

A 10M transformer, behavior-cloned on BFS-optimal maze trajectories encoded as a token "terminal": interleaved observation tokens (the 4-neighbour wall pattern + a coarse goal bearing) and the optimal action. The **entire A/B is the loss mask**: action-only (`λ=0`) computes loss on action tokens only; ECHO-style (`λ=1`) adds the free CE on the observation tokens too. Same data, same model, same forward pass.

Eval: greedy rollout, the environment supplies the true next observation (the model's predicted obs is training-only), solve = reach goal within 4x the optimal path. 300 held-out mazes per size, 3 seeds, sizes 6x6 / 7x7 / 8x8. Then the decisive ablation — **walls-only**, dropping the goal bearing so the agent can't just follow a handed-over direction and *must* build an internal map (exactly where predicting walls should help).

## The numbers

Solve rate, 3-seed mean ± spread. Gap = (λ1 − λ0) in points.

| regime | 6x6 | 7x7 | 8x8 |
|---|---|---|---|
| **full obs** action-only | 0.788 ± 0.017 | 0.713 ± 0.009 | 0.633 ± 0.012 |
| **full obs** + env-token | 0.807 ± 0.021 | 0.706 ± 0.006 | 0.623 ± 0.022 |
| → gap | +1.9 | −0.8 | −1.0 |
| **walls-only** action-only | 0.606 ± 0.010 | 0.492 ± 0.011 | 0.491 ± 0.012 |
| **walls-only** + env-token | 0.622 ± 0.011 | 0.502 ± 0.018 | 0.482 ± 0.011 |
| → gap | +1.7 | +1.0 | −0.9 |

**A clean null, twice.** Every gap sits inside the seed spread, with no consistent sign and — the tell that matters — **no growth with maze size** in either regime. The hypothesis was that a world model should help *more* as mazes get bigger; the opposite of that is "no trend," and that's what we get. The walls-only ablation kills the obvious objection: even when the agent is denied the goal direction and has to map the maze itself, predicting observations buys nothing.

## Why it's inert (and what it says about ECHO)

In behavior cloning the policy already gets a clean, dense supervision signal — the optimal action at every step. Predicting the next observation is a representation-learning side-quest that doesn't change the action decision the model is being graded on. The world-model tokens are "free" to add, and free is exactly what they're worth here.

This **doesn't refute ECHO** — it *locates* its gain. ECHO is on-policy RL, where the reward signal is sparse and the rollouts are the model's own. In that setting the env-token loss can shape the *rollout distribution* and credit assignment — it has a job to do that simply doesn't exist in imitation. So the lesson the microcosm actually teaches: the "for free" gain is not a generic auxiliary-loss free lunch you can bolt onto any objective. It lives in the RL coupling. Which is the whole point of testing it on-policy next.

## Honest caveats

- This is behavior cloning, not RL — by design. It tests whether the *auxiliary loss alone* helps; it cannot speak to the on-policy regime ECHO actually uses.
- 10M from scratch, not an 8–14B pretrained LLM. A microcosm isolates a mechanism; it doesn't transfer magnitudes.
- The +1.7 to +1.9 pt at 6x6 is within seed spread — reported, not claimed.
- One λ (0 vs 1) and one architecture. The point was a clean mechanism read, not a sweep.

## Reproduce

`lib/echo_maze/` (maze + BFS, partial-obs env, encoding, the loss-mask A/B, the 10M GPT, BC training, rollout) + `scripts/echo_maze_sweep.py` (`OBS_MODE=walls_only` for the ablation) + `scripts/chart_echo_maze.py`. 15 unit tests (stdlib logic on CPU, torch on GPU). Two sweeps, ~35 min each on one 5090.
