"""Token vocabulary and trajectory encoding for the maze 'terminal'. A training sequence
interleaves observation tokens (wall pattern, goal bearing) with the optimal action, so that
predicting an action = the policy and predicting the next observation = an implicit world model.
Pure stdlib. The loss mask (build_loss_mask) is the entire A/B."""
from lib.echo_maze.maze import (
    env_reset, env_observe, bfs_path, path_to_actions, DIR_ORDER,
)

PAD, BOS, EOS = 0, 1, 2
ACT_BASE = 3    # actions N,E,S,W -> 3..6
WALL_BASE = 7   # wall patterns 0..15 -> 7..22
BEAR_BASE = 23  # bearings 0..7 -> 23..30
VOCAB_SIZE = 31


def act_token(d):
    return ACT_BASE + DIR_ORDER.index(d)


def wall_tok(bits):
    return WALL_BASE + bits


def bear_tok(b):
    return BEAR_BASE + b


def decode_action(tok):
    return DIR_ORDER[tok - ACT_BASE]


ACT_TOKEN_IDS = [ACT_BASE + i for i in range(4)]


def encode_trajectory(maze, start, goal, include_bearing=True):
    """Walk the BFS-optimal path: BOS, then (wall[,bear],act) per step, (wall[,bear],EOS) at goal.
    Returns (tokens, roles) of equal length; roles in {bos,wall,bear,act,eos}. With
    include_bearing=False the goal bearing is dropped (walls-only ablation: the agent must build
    an internal map instead of following the handed-over goal direction)."""
    path = bfs_path(maze, start, goal)
    acts = path_to_actions(path)
    env = env_reset(maze, start, goal)
    tokens, roles = [BOS], ["bos"]
    for i, cell in enumerate(path):
        env.pos = cell
        wb, br = env_observe(env)
        tokens.append(wall_tok(wb))
        roles.append("wall")
        if include_bearing:
            tokens.append(bear_tok(br))
            roles.append("bear")
        if i < len(acts):
            tokens.append(act_token(acts[i]))
            roles.append("act")
        else:
            tokens.append(EOS)
            roles.append("eos")
    return tokens, roles


def build_loss_mask(roles, lam):
    """Per-token TARGET weight. Action/EOS always 1.0 (the policy + stop); observation tokens
    (wall, bear) weighted lam — lam=0 is the action-only arm, lam>0 the ECHO-style arm; the aux
    loss is 'free' (those tokens are already in the sequence). bos/pad = 0.0."""
    out = []
    for role in roles:
        if role in ("act", "eos"):
            out.append(1.0)
        elif role in ("wall", "bear"):
            out.append(float(lam))
        else:
            out.append(0.0)
    return out
