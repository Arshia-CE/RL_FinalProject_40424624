"""Maze map data model: cell types, the MazeMap container, BFS reachability
and specification validation."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

# cell type characters used in the grid
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
