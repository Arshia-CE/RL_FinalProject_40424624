"""Transfer learning for Q-Learning: initial-Q construction (full, scaled by
beta, selective by unchanged local neighborhood) between source and target
mazes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.generator import (DEFAULT_CONFIG_PATH, MAPS_DIR, WALL,
                                    MazeMap, Position)
from environments.maze import State

NEIGHBORHOOD_RADIUS = 1  # 3x3 window around the cell


def neighborhood_signature(maze: MazeMap, pos: Position) -> tuple:
    """Cell types in the (2r+1)^2 window around pos; off-grid counts as wall."""
    r0, c0 = pos
    return tuple(
        maze.grid[r][c] if maze.in_bounds((r, c)) else WALL
        for r in range(r0 - NEIGHBORHOOD_RADIUS, r0 + NEIGHBORHOOD_RADIUS + 1)
        for c in range(c0 - NEIGHBORHOOD_RADIUS, c0 + NEIGHBORHOOD_RADIUS + 1))


def unchanged_positions(source: MazeMap, target: MazeMap) -> set[Position]:
    """Cells whose local neighborhood is identical in both maps."""
    return {(r, c) for r in range(source.size) for c in range(source.size)
            if not source.is_wall((r, c)) and not target.is_wall((r, c))
            and neighborhood_signature(source, (r, c))
            == neighborhood_signature(target, (r, c))}


def initial_q_table(source_q: dict[State, np.ndarray], scenario: str, *,
                    beta: float | None = None,
                    unchanged: set[Position] | None = None
                    ) -> dict[State, np.ndarray]:
    """Initial Q-table for one of the four spec scenarios."""
    if scenario == "scratch":
        return {}
    if scenario == "full":
        return {s: q.copy() for s, q in source_q.items()}
    if scenario == "scaled":
        return {s: beta * q for s, q in source_q.items()}
    if scenario == "selective":
        return {s: q.copy() for s, q in source_q.items()
                if (s.r, s.c) in unchanged}
    raise ValueError(f"unknown transfer scenario {scenario!r}")


def main() -> None:
    json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    source = MazeMap.load(MAPS_DIR / "source.json")
    for kind in ("similar", "different"):
        target = MazeMap.load(MAPS_DIR / f"target_{kind}.json")
        unchanged = unchanged_positions(source, target)
        both_passable = sum(
            1 for r in range(source.size) for c in range(source.size)
            if not source.is_wall((r, c)) and not target.is_wall((r, c)))
        print(f"{kind}: {len(unchanged)}/{both_passable} passable cells "
              f"({len(unchanged) / both_passable:.1%}) have an unchanged "
              f"3x3 neighborhood")


if __name__ == "__main__":
    main()
