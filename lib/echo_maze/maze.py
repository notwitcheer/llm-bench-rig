"""Perfect-maze generation (randomized DFS), BFS shortest path, and a partial-observability
grid environment. Pure stdlib so the logic is unit-testable on the Mac with no GPU/torch.

A 'perfect' maze has exactly one simple path between any two cells. The agent sees only its
4-neighbour wall pattern + a coarse 8-way bearing to the goal — no global view — so solving
requires integrating history into an internal map."""
from collections import deque
from dataclasses import dataclass
import math

# direction -> (drow, dcol); N is up. The N,E,S,W order is canonical for wall bits everywhere.
DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
DIR_ORDER = ["N", "E", "S", "W"]
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass
class Maze:
    size: int
    passages: dict  # (r, c) -> set of open directions (no wall that way)


def gen_maze(size, rng):
    """Recursive-backtracker (randomized DFS). `rng` is a random.Random for determinism."""
    passages = {(r, c): set() for r in range(size) for c in range(size)}
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        nbrs = []
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                nbrs.append((d, (nr, nc)))
        if not nbrs:
            stack.pop()
            continue
        d, (nr, nc) = rng.choice(nbrs)
        passages[(r, c)].add(d)
        passages[(nr, nc)].add(OPP[d])
        visited.add((nr, nc))
        stack.append((nr, nc))
    return Maze(size=size, passages=passages)


def bfs_path(maze, start, goal):
    """Shortest path as a list of cells from start to goal inclusive (unique in a perfect maze)."""
    prev = {start: None}
    q = deque([start])
    while q:
        cell = q.popleft()
        if cell == goal:
            break
        r, c = cell
        for d in maze.passages[cell]:
            dr, dc = DIRS[d]
            nxt = (r + dr, c + dc)
            if nxt not in prev:
                prev[nxt] = cell
                q.append(nxt)
    if goal not in prev:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def path_to_actions(path):
    """Directions taken between consecutive cells of a path."""
    acts = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        for d, (dr, dc) in DIRS.items():
            if (r0 + dr, c0 + dc) == (r1, c1):
                acts.append(d)
                break
    return acts


def random_endpoints(size, rng):
    """Distinct start/goal cells."""
    a = (rng.randrange(size), rng.randrange(size))
    b = (rng.randrange(size), rng.randrange(size))
    while b == a:
        b = (rng.randrange(size), rng.randrange(size))
    return a, b
