"""Seeded maze generation and transfer-target derivation.

The source map is generated reproducibly from the student seed under the
spec constraints and BFS-validated; transfer targets are derived from it by
moving walls ("similar" ~18%, "different" >=35% plus key relocation and
extra penalty cells). All maps are saved to environments/maps/.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import (CELL_LEGEND, DEFAULT_CONFIG_PATH, DOOR,
                                   GATE, GOAL, KEY, MAPS_DIR, NORMAL, PENALTY,
                                   START, WALL, MazeMap, Position,
                                   validate_map)


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

    # goal chamber in the bottom-right corner, entered only via the door
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
    # the corridor cell just outside the door must stay open
    outside = (ring_r - 1, door[1]) if door[0] == ring_r else (door[0], ring_c - 1)
    reserved.add(outside)

    # start in the top-left region, far from the goal chamber
    start = (rng.randrange(3), rng.randrange(3))
    grid[start[0]][start[1]] = START
    reserved.add(start)

    # wall segments (length 2-4) until the target wall fraction is met
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

    # key: far from the start (Manhattan distance >= maze size)
    far = [p for p in free_cells()
           if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= n]
    if not far:
        return None
    key = rng.choice(far)
    grid[key[0]][key[1]] = KEY

    # penalty cells, kept away from the immediate start neighborhood
    candidates = [p for p in free_cells()
                  if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= 2]
    if len(candidates) < cfg["num_penalty_cells"]:
        return None
    for r, c in rng.sample(candidates, cfg["num_penalty_cells"]):
        grid[r][c] = PENALTY

    # periodic gate at the chamber entrance: by construction this is the
    # only cell from which the door can be reached, so every successful
    # episode must pass it and the gate phase truly shapes the policy
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
