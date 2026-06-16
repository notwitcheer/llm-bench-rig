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
