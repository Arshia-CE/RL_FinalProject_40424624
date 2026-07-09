"""Seeded maze generation, BFS validation and map persistence.

Maps are generated reproducibly from the student seed under the spec
constraints (>=15% walls, >=5 penalty cells; goal in a walled chamber whose
only entrance is the locked door, with the periodic gate on the single
corridor cell in front of it), BFS-validated for start -> key -> goal
solvability, and saved to environments/maps/. Transfer targets are derived
deterministically from the source map: "similar" moves ~18% of the walls,
"different" moves >=35%, relocates the key and adds penalty cells.
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


# transfer targets


def perturb_map(source: MazeMap, maze_cfg: dict, *, change_fraction: float,
                relocate_key: bool, extra_penalty_cells: int,
                seed_offset: int, max_attempts: int = 200) -> MazeMap:
    """Derive a BFS-valid target map by *moving* ceil(fraction * walls) walls
    (wall count is preserved); optionally relocate the key and add penalty
    cells. The goal chamber ring, door, gate, goal and start never change."""
    n = source.size
    ring_r, ring_c = n - 4, n - 4
    ring = set([(ring_r, c) for c in range(ring_c, n)]
               + [(r, ring_c) for r in range(ring_r + 1, n)])
    moves = math.ceil(change_fraction * source.wall_count)

    for attempt in range(max_attempts):
        seed = source.base_seed * 100_000 + seed_offset + attempt
        rng = random.Random(seed)
        grid = [row[:] for row in source.grid]

        movable = [(r, c) for r in range(n) for c in range(n)
                   if grid[r][c] == WALL and (r, c) not in ring]
        free = [(r, c) for r in range(n) for c in range(n)
                if grid[r][c] == NORMAL]
        if moves > min(len(movable), len(free)):
            raise RuntimeError("not enough movable walls / free cells")
        for r, c in rng.sample(movable, moves):
            grid[r][c] = NORMAL
        for r, c in rng.sample(free, moves):
            grid[r][c] = WALL

        key = tuple(source.key)
        if relocate_key:
            candidates = [
                (r, c) for r in range(n) for c in range(n)
                if grid[r][c] == NORMAL
                and abs(r - source.start[0]) + abs(c - source.start[1]) >= n
                and abs(r - source.key[0]) + abs(c - source.key[1]) >= 5]
            if not candidates:
                continue
            key = rng.choice(candidates)
            grid[source.key[0]][source.key[1]] = NORMAL
            grid[key[0]][key[1]] = KEY

        penalty_cells = [tuple(p) for p in source.penalty_cells]
        if extra_penalty_cells:
            candidates = [
                (r, c) for r in range(n) for c in range(n)
                if grid[r][c] == NORMAL
                and abs(r - source.start[0]) + abs(c - source.start[1]) >= 2]
            if len(candidates) < extra_penalty_cells:
                continue
            new_cells = rng.sample(candidates, extra_penalty_cells)
            for r, c in new_cells:
                grid[r][c] = PENALTY
            penalty_cells += new_cells

        candidate = MazeMap(
            grid=grid, size=n, base_seed=source.base_seed, attempt=attempt,
            effective_seed=seed, start=tuple(source.start), key=key,
            door=tuple(source.door), goal=tuple(source.goal),
            gate=tuple(source.gate), penalty_cells=penalty_cells,
            gate_period=source.gate_period,
            gate_open_phases=list(source.gate_open_phases))
        if validate_map(candidate, maze_cfg)[0]:
            return candidate
    raise RuntimeError(f"no valid target map in {max_attempts} attempts")


TARGET_SEED_OFFSETS = {"similar": 1000, "different": 2000}


def make_target_map(source: MazeMap, config: dict, kind: str) -> MazeMap:
    tcfg = config["transfer"][f"{kind}_target"]
    return perturb_map(
        source, config["maze"],
        change_fraction=tcfg["obstacle_change_fraction"],
        relocate_key=tcfg["move_key_or_goal"],
        extra_penalty_cells=tcfg["extra_penalty_cells"],
        seed_offset=TARGET_SEED_OFFSETS[kind])


def walls_changed_fraction(source: MazeMap, target: MazeMap) -> float:
    """Share of the source's walls that were removed (== added elsewhere)."""
    removed = sum(1 for r in range(source.size) for c in range(source.size)
                  if source.grid[r][c] == WALL and target.grid[r][c] != WALL)
    return removed / source.wall_count


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

    for kind in ("similar", "different"):
        target = make_target_map(maze, config, kind)
        target_path = MAPS_DIR / f"target_{kind}.json"
        target.save(target_path)
        changed = walls_changed_fraction(maze, target)
        print()
        print(f"Transfer target '{kind}': {changed:.1%} of walls moved "
              f"(attempt {target.attempt}), key {tuple(target.key)}, "
              f"{len(target.penalty_cells)} penalty cells")
        print(f"Saved to {target_path.relative_to(PROJECT_ROOT)}")
        print(target.ascii_render())


if __name__ == "__main__":
    main()
