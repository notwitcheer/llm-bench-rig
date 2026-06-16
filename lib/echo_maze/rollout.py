"""Greedy evaluation rollout. The model chooses actions; the ENVIRONMENT supplies the true next
observation (the model's predicted observation is ignored at eval — the world-model loss is
training-only). The eval maze set is fixed by eval_seed so every arm/seed is scored on identical
mazes. Torch; runs on capsule."""
import random
import torch

from lib.echo_maze.maze import (
    gen_maze, bfs_path, env_reset, env_observe, rollout_episode, random_endpoints,
)
from lib.echo_maze.encode import BOS, wall_tok, bear_tok, act_token, decode_action, ACT_TOKEN_IDS


def make_model_policy(model, device, block_size):
    """Fresh closure per episode: maintains the running token sequence and returns argmax over
    ACTION tokens only at the current step."""
    seq = [BOS]
    act_ids = torch.tensor(ACT_TOKEN_IDS, device=device)

    def policy_fn(env):
        wb, br = env_observe(env)
        seq.extend([wall_tok(wb), bear_tok(br)])
        idx = torch.tensor([seq[-block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(idx)[0, -1]            # last position predicts the action
        best = int(torch.argmax(logits[act_ids]))
        action = decode_action(ACT_TOKEN_IDS[best])
        seq.append(act_token(action))
        return action

    return policy_fn


def solve_rate(model, size, n_eval, eval_seed, device="cuda", block_size=512,
               budget_factor=4, hard_cap=400):
    rng = random.Random(eval_seed)
    model.eval()
    solved = 0
    for _ in range(n_eval):
        maze = gen_maze(size, rng)
        start, goal = random_endpoints(size, rng)
        opt = max(len(bfs_path(maze, start, goal)) - 1, 1)
        budget = min(budget_factor * opt, hard_cap)
        env = env_reset(maze, start, goal)
        policy = make_model_policy(model, device, block_size)
        if rollout_episode(env, policy, budget):
            solved += 1
    return solved / n_eval
