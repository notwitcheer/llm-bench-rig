"""ECHO maze microcosm — stdlib logic (Mac, no GPU)."""
import random
from lib.echo_maze.maze import gen_maze, bfs_path, path_to_actions, Maze, DIRS, OPP


def test_maze_is_perfect_and_connected():
    rng = random.Random(0)
    m = gen_maze(6, rng)
    assert isinstance(m, Maze) and m.size == 6
    # passages are symmetric (if A opens to B, B opens back to A)
    for (r, c), dirs in m.passages.items():
        for d in dirs:
            dr, dc = DIRS[d]
            assert OPP[d] in m.passages[(r + dr, c + dc)]
    # perfect maze: exactly n_cells-1 undirected edges (a spanning tree)
    edges = sum(len(v) for v in m.passages.values()) // 2
    assert edges == 6 * 6 - 1


def test_bfs_returns_valid_shortest_path():
    rng = random.Random(1)
    m = gen_maze(7, rng)
    path = bfs_path(m, (0, 0), (6, 6))
    assert path[0] == (0, 0) and path[-1] == (6, 6)
    # consecutive cells are adjacent through an open passage
    for a, b in zip(path, path[1:]):
        d = next(k for k, (dr, dc) in DIRS.items() if (a[0] + dr, a[1] + dc) == b)
        assert d in m.passages[a]


def test_path_to_actions_matches_moves():
    acts = path_to_actions([(0, 0), (0, 1), (1, 1)])
    assert acts == ["E", "S"]


from lib.echo_maze.maze import (
    MazeEnv, env_reset, env_observe, env_step, at_goal, wall_bits, bearing, rollout_episode,
)


def _corridor():
    # 1-row "maze": cells (0,0)-(0,1)-(0,2) all open E/W along the row
    passages = {(0, 0): {"E"}, (0, 1): {"E", "W"}, (0, 2): {"W"}}
    return Maze(size=3, passages=passages)


def test_wall_bits_and_step_respect_walls():
    m = _corridor()
    env = env_reset(m, (0, 0), (0, 2))
    # at (0,0): open E only -> walls N,S,W set, E clear. bits: N=1,E=2,S=4,W=8 -> 1+4+8=13
    assert wall_bits(m, (0, 0)) == 13
    assert env_step(env, "N") is False and env.pos == (0, 0)   # blocked
    assert env_step(env, "E") is True and env.pos == (0, 1)    # moved


def test_bearing_distinguishes_cardinals_and_zero_at_goal():
    vals = {d: bearing((1, 1), tgt) for d, tgt in
            {"N": (0, 1), "E": (1, 2), "S": (2, 1), "W": (1, 0)}.items()}
    assert len(set(vals.values())) == 4          # four distinct octants
    assert bearing((1, 1), (1, 1)) == 0          # goal-at-cell


def test_rollout_episode_oracle_solves_random_policy_does_not():
    m = _corridor()
    goal = (0, 2)
    def oracle(env):
        return path_to_actions(bfs_path(m, env.pos, goal))[0]
    assert rollout_episode(env_reset(m, (0, 0), goal), oracle, max_steps=10) is True
    def stuck(env):
        return "N"   # always into a wall in the corridor
    assert rollout_episode(env_reset(m, (0, 0), goal), stuck, max_steps=10) is False


from lib.echo_maze.encode import (
    encode_trajectory, decode_action, act_token, VOCAB_SIZE, ACT_TOKEN_IDS,
)


def test_action_token_roundtrip():
    for d in ["N", "E", "S", "W"]:
        assert decode_action(act_token(d)) == d
    assert ACT_TOKEN_IDS == [act_token(d) for d in ["N", "E", "S", "W"]]
    assert VOCAB_SIZE == 31


def test_encode_trajectory_structure():
    rng = random.Random(2)
    m = gen_maze(6, rng)
    path = bfs_path(m, (0, 0), (5, 5))
    tokens, roles = encode_trajectory(m, (0, 0), (5, 5))
    assert len(tokens) == len(roles) == 1 + 3 * len(path)   # BOS + (wall,bear,act/eos) per cell
    assert roles[0] == "bos" and roles[-1] == "eos"
    assert roles.count("act") == len(path) - 1 and roles.count("eos") == 1


def test_encode_trajectory_walls_only_drops_bearing():
    rng = random.Random(2)
    m = gen_maze(6, rng)
    path = bfs_path(m, (0, 0), (5, 5))
    tokens, roles = encode_trajectory(m, (0, 0), (5, 5), include_bearing=False)
    assert "bear" not in roles
    assert len(tokens) == len(roles) == 1 + 2 * len(path)   # BOS + (wall, act/eos) per cell
    assert roles[0] == "bos" and roles[-1] == "eos"


from lib.echo_maze.encode import build_loss_mask


def test_loss_mask_action_only_vs_echo():
    roles = ["bos", "wall", "bear", "act", "wall", "bear", "eos"]
    assert build_loss_mask(roles, 0.0) == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]  # action-only
    assert build_loss_mask(roles, 1.0) == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # +env tokens
    assert build_loss_mask(roles, 0.5)[1] == 0.5                                # obs scaled by lam
