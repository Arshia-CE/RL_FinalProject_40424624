"""Seeded maze generation, BFS validation and map persistence.

The base map is generated from the student-specific seed on a 17x17 grid
(``base_seed = int("40424624"[-2]) = 2``, ``size = 15 + (2 % 4) = 17``):

- at least 15% of all cells are walls (we target ~18% for a maze-like feel),
- at least 5 penalty cells (6 by default),
- start, key, locked door, goal and one periodic-gate cell,
- the goal sits inside a walled chamber whose only entrance is the locked
  door, so the mission order start -> key -> door -> goal is structural and
  not just a reward artifact,
- the periodic gate (the chosen dynamic feature) is placed on the corridor
  cell at the chamber entrance -- the only cell from which the door can be
  reached -- so every successful episode must interact with it and the agent
  genuinely has to reason about the gate phase (wait, or time its arrival).

Validation uses deterministic BFS: a path must exist start -> key with the
door closed, and key -> goal with the door open. The gate is treated as
passable for reachability because a closed gate only delays the agent (it
behaves like a wall bump), it never disconnects the maze. If a candidate map
violates any constraint it is regenerated reproducibly with
``effective_seed = base_seed * 1000 + attempt``.

The final validated map is saved to ``environments/maps/`` and is shared by
all three algorithms, as the specification requires.

Usage:
    python environments/generator.py      # generate + validate + save + preview
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

# Cell type characters used in the grid.
WALL = "#"
NORMAL = "."
PENALTY = "P"
START = "S"
KEY = "K"
DOOR = "D"
GOAL = "G"
GATE = "T"

CELL_LEGEND = {
    WALL: "wall",
    NORMAL: "normal",
    PENALTY: "penalty",
    START: "start",
    KEY: "key",
    DOOR: "locked door",
    GOAL: "goal",
    GATE: "periodic gate",
}

ACTIONS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = Path(__file__).resolve().parent / "maps"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "default.json"

Position = tuple[int, int]


@dataclass
class MazeMap:
    """A generated, validated maze map plus its generation metadata."""

    grid: list[list[str]]
    size: int
    base_seed: int
    attempt: int
    effective_seed: int
    start: Position
    key: Position
    door: Position
    goal: Position
    gate: Position
    penalty_cells: list[Position]
    gate_period: int
    gate_open_phases: list[int]

    # queries

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def cell(self, pos: Position) -> str:
        return self.grid[pos[0]][pos[1]]

    def is_wall(self, pos: Position) -> bool:
        """Out-of-bounds counts as a wall (the grid border is solid)."""
        return not self.in_bounds(pos) or self.cell(pos) == WALL

    @property
    def wall_count(self) -> int:
        return sum(row.count(WALL) for row in self.grid)

    @property
    def wall_fraction(self) -> float:
        return self.wall_count / (self.size * self.size)

    @property
    def passable_count(self) -> int:
        """Number of non-wall cells; used for the episode step cap."""
        return self.size * self.size - self.wall_count

    # persistence

    def to_dict(self) -> dict:
        data = asdict(self)
        data["grid"] = ["".join(row) for row in self.grid]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MazeMap":
        data = dict(data)
        data["grid"] = [list(row) for row in data["grid"]]
        for field_name in ("start", "key", "door", "goal", "gate"):
            data[field_name] = tuple(data[field_name])
        data["penalty_cells"] = [tuple(p) for p in data["penalty_cells"]]
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MazeMap":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def ascii_render(self) -> str:
        return "\n".join("".join(row) for row in self.grid)


# BFS


def bfs_shortest_path(grid: list[list[str]], src: Position, dst: Position,
                      door_open: bool) -> list[Position] | None:
    """Deterministic BFS shortest path on the 4-connected grid.

    Walls block movement; the door blocks it unless ``door_open``. The gate
    and penalty cells are passable (a closed gate only delays the agent, and
    penalty cells merely cost reward). Returns the path including both
    endpoints, or None if unreachable.
    """
    n = len(grid)
    if src == dst:
        return [src]
    prev: dict[Position, Position | None] = {src: None}
    queue: deque[Position] = deque([src])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTIONS_4:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if not (0 <= nr < n and 0 <= nc < n) or nxt in prev:
                continue
            cell = grid[nr][nc]
            if cell == WALL or (cell == DOOR and not door_open):
                continue
            prev[nxt] = (r, c)
            if nxt == dst:
                path = [dst]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


def path_exists(grid: list[list[str]], src: Position, dst: Position,
                door_open: bool) -> bool:
    return bfs_shortest_path(grid, src, dst, door_open) is not None


# generation


def generate_map(config: dict, base_seed: int | None = None,
                 max_attempts: int = 200) -> MazeMap:
    """Generate a valid map, retrying deterministically until validation passes."""
    maze_cfg = config["maze"]
    if base_seed is None:
        base_seed = config["base_seed"]
    for attempt in range(max_attempts):
        effective_seed = base_seed * 1000 + attempt
        candidate = _try_generate(maze_cfg, base_seed, attempt, effective_seed)
        if candidate is not None and validate_map(candidate, maze_cfg)[0]:
            return candidate
    raise RuntimeError(f"No valid map found in {max_attempts} attempts "
                       f"(base_seed={base_seed}).")


def _try_generate(cfg: dict, base_seed: int, attempt: int,
                  effective_seed: int) -> MazeMap | None:
    """One reproducible generation attempt; returns None on any dead end."""
    n = cfg["size"]
    rng = random.Random(effective_seed)
    grid = [[NORMAL] * n for _ in range(n)]
    reserved: set[Position] = set()  # cells that must keep their current type

    # --- goal chamber in the bottom-right corner, entered only via the door.
    ring_r, ring_c = n - 4, n - 4
    ring = ([(ring_r, c) for c in range(ring_c, n)]
            + [(r, ring_c) for r in range(ring_r + 1, n)])
    for r, c in ring:
        grid[r][c] = WALL
    goal = (n - 1, n - 1)
    grid[goal[0]][goal[1]] = GOAL
    reserved.update(ring)
    reserved.update((r, c) for r in range(ring_r + 1, n)
                    for c in range(ring_c + 1, n))

    door_candidates = [p for p in ring if p != (ring_r, ring_c)]
    door = rng.choice(door_candidates)
    grid[door[0]][door[1]] = DOOR
    # The corridor cell just outside the door must stay open.
    outside = (ring_r - 1, door[1]) if door[0] == ring_r else (door[0], ring_c - 1)
    reserved.add(outside)

    # --- start in the top-left region, far from the goal chamber.
    start = (rng.randrange(3), rng.randrange(3))
    grid[start[0]][start[1]] = START
    reserved.add(start)

    # --- wall segments (length 2-4) until the target wall fraction is met.
    target_walls = math.ceil(cfg["target_wall_fraction"] * n * n)
    wall_count = sum(row.count(WALL) for row in grid)
    for _ in range(5000):
        if wall_count >= target_walls:
            break
        r, c = rng.randrange(n), rng.randrange(n)
        horizontal = rng.random() < 0.5
        for i in range(rng.randint(2, 4)):
            rr, cc = (r, c + i) if horizontal else (r + i, c)
            if not (0 <= rr < n and 0 <= cc < n) or (rr, cc) in reserved:
                break
            if grid[rr][cc] == NORMAL:
                grid[rr][cc] = WALL
                wall_count += 1

    def free_cells() -> list[Position]:
        return [(r, c) for r in range(n) for c in range(n)
                if grid[r][c] == NORMAL and (r, c) not in reserved]

    # --- key: far from the start (Manhattan distance >= maze size).
    far = [p for p in free_cells()
           if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= n]
    if not far:
        return None
    key = rng.choice(far)
    grid[key[0]][key[1]] = KEY

    # --- penalty cells, kept away from the immediate start neighborhood.
    candidates = [p for p in free_cells()
                  if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= 2]
    if len(candidates) < cfg["num_penalty_cells"]:
        return None
    for r, c in rng.sample(candidates, cfg["num_penalty_cells"]):
        grid[r][c] = PENALTY

    # --- periodic gate at the chamber entrance: by construction this is the
    #     only cell from which the door can be reached, so every successful
    #     episode must pass it and the gate phase truly shapes the policy.
    gate = outside
    grid[gate[0]][gate[1]] = GATE

    return MazeMap(
        grid=grid, size=n, base_seed=base_seed, attempt=attempt,
        effective_seed=effective_seed, start=start, key=key, door=door,
        goal=goal, gate=gate,
        penalty_cells=[(r, c) for r in range(n) for c in range(n)
                       if grid[r][c] == PENALTY],
        gate_period=cfg["gate_period"],
        gate_open_phases=list(cfg["gate_open_phases"]),
    )


# validation


def validate_map(maze: MazeMap, maze_cfg: dict) -> tuple[bool, list[str]]:
    """Check every specification constraint; returns (ok, list of problems)."""
    problems: list[str] = []
    if maze.wall_fraction < maze_cfg["min_wall_fraction"]:
        problems.append(f"wall fraction {maze.wall_fraction:.3f} below "
                        f"{maze_cfg['min_wall_fraction']}")
    if len(maze.penalty_cells) < maze_cfg["min_penalty_cells"]:
        problems.append(f"only {len(maze.penalty_cells)} penalty cells")
    if not path_exists(maze.grid, maze.start, maze.key, door_open=False):
        problems.append("no path start -> key with the door closed")
    if not path_exists(maze.grid, maze.key, maze.goal, door_open=True):
        problems.append("no path key -> goal with the door open")
    return (not problems, problems)


# CLI


def main() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    maze = generate_map(config)
    out_path = MAPS_DIR / "source.json"
    maze.save(out_path)

    reloaded = MazeMap.load(out_path)
    assert reloaded.to_dict() == maze.to_dict(), "save/load round-trip failed"

    print(f"Map generated with base_seed={maze.base_seed} "
          f"(effective_seed={maze.effective_seed}, attempt={maze.attempt})")
    print(f"Size: {maze.size}x{maze.size} | walls: {maze.wall_count} "
          f"({maze.wall_fraction:.1%}) | passable cells: {maze.passable_count}")
    print(f"Start {maze.start} -> Key {maze.key} -> Door {maze.door} "
          f"-> Goal {maze.goal}")
    print(f"Gate at {maze.gate} (period {maze.gate_period}, open phases "
          f"{maze.gate_open_phases}) | penalty cells: {maze.penalty_cells}")
    print(f"Saved to {out_path.relative_to(PROJECT_ROOT)}")
    print()
    legend = "  ".join(f"{ch}={name}" for ch, name in CELL_LEGEND.items())
    print(f"Legend: {legend}")
    print(maze.ascii_render())


if __name__ == "__main__":
    main()
