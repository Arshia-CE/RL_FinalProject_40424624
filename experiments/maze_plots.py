"""Maze-rendered figures: value heatmaps, policy arrows, visit maps and
policy-disagreement maps."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import (DOOR, GATE, GOAL, KEY, PENALTY, START,
                                   MazeMap)
from environments.maze import EV_DOOR_PASS, EV_KEY_PICKUP, State
from experiments.analysis import (ARROW, DIVERGING_CMAP, GRID_COLOR, INK,
                                  INK_2, SEQ_RAMP, SURFACE, VALUE_CMAP,
                                  WALL_COLOR, hex_to_rgb, save_figure)

CELL_LETTER = {START: "S", KEY: "K", DOOR: "D", GOAL: "G", GATE: "T",
               PENALTY: "P"}
CELL_WASH = {START: "#2a78d6", KEY: "#eda100", DOOR: "#e87ba4",
             GOAL: "#1baf7a", GATE: "#4a3aa7", PENALTY: "#e34948"}
# darker end of the sequential ramp: readable as a line on the light surface
PATH_CMAP = LinearSegmentedColormap.from_list("path_blue", SEQ_RAMP[2:])


def _grid_lines(ax, n: int) -> None:
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=0.5)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(0, n, 2))
    ax.set_yticks(range(0, n, 2))
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def _cell_letters(ax, maze: MazeMap, stroke: bool = True,
                  fontsize: float = 7.0) -> None:
    effects = ([path_effects.withStroke(linewidth=1.6, foreground="white")]
               if stroke else None)
    for r in range(maze.size):
        for c in range(maze.size):
            cell = maze.grid[r][c]
            if cell in CELL_LETTER:
                ax.text(c, r, CELL_LETTER[cell], color=INK,
                        fontsize=fontsize, ha="center", va="center",
                        path_effects=effects)


def draw_maze(ax, maze: MazeMap, wash_alpha: float = 0.35,
              letter_size: float = 6.0) -> None:
    """Recessive maze background: dark walls, washed + lettered special cells."""
    n = maze.size
    img = np.tile(hex_to_rgb(SURFACE), (n, n, 1))
    for r in range(n):
        for c in range(n):
            cell = maze.grid[r][c]
            if maze.is_wall((r, c)):
                img[r, c] = hex_to_rgb(WALL_COLOR)
            elif cell in CELL_WASH:
                img[r, c] = ((1 - wash_alpha) * hex_to_rgb(SURFACE)
                             + wash_alpha * hex_to_rgb(CELL_WASH[cell]))
    ax.imshow(img)
    _grid_lines(ax, n)
    for r in range(n):
        for c in range(n):
            cell = maze.grid[r][c]
            if cell in CELL_LETTER:
                ax.text(c - 0.32, r - 0.28, CELL_LETTER[cell], color=INK_2,
                        fontsize=letter_size, ha="left", va="top")


def value_grid(maze: MazeMap, V: dict[State, float], has_key: int,
               reduce_phases=max) -> np.ma.MaskedArray:
    """V reduced over gate phases for one key status; walls masked."""
    grid = np.full((maze.size, maze.size), np.nan)
    for r in range(maze.size):
        for c in range(maze.size):
            if not maze.is_wall((r, c)):
                grid[r, c] = reduce_phases(
                    V[State(r, c, has_key, p)] for p in range(maze.gate_period))
    return np.ma.masked_invalid(grid)


def plot_value_heatmap(maze: MazeMap, V: dict[State, float], title: str,
                       path: Path) -> None:
    grids = [value_grid(maze, V, k) for k in (0, 1)]
    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, grid, k in zip(axes, grids, (0, 1)):
        im = ax.imshow(grid, cmap=VALUE_CMAP, vmin=vmin, vmax=vmax)
        _grid_lines(ax, maze.size)
        _cell_letters(ax, maze)
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("V (max over gate phases)", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(title, color=INK)
    save_figure(fig, path)


def plot_policy_arrows(maze: MazeMap, policy: dict[State, int | None],
                       title: str, path: Path, phase: int = 0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, k in zip(axes, (0, 1)):
        draw_maze(ax, maze)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    continue
                action = policy[State(r, c, k, phase)]
                if action is None:
                    continue  # terminal; the G letter already marks it
                ax.text(c, r + 0.05, ARROW[action], color=INK, fontsize=9,
                        ha="center", va="center")
        ax.set_title(f"{'without key (k=0)' if k == 0 else 'with key (k=1)'}"
                     f", gate phase {phase}")
    fig.suptitle(title, color=INK)
    save_figure(fig, path)


def plot_policy_phase_grid(maze: MazeMap, policy: dict[State, int | None],
                           gate_open_phases: list[int], title: str,
                           path: Path, has_key: int = 1) -> None:
    """Small multiples: the greedy policy at every gate phase (fixed key)."""
    period = maze.gate_period
    rows = 2
    cols = (period + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows))
    for phase, ax in zip(range(period), axes.flat):
        draw_maze(ax, maze, letter_size=5.0)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    continue
                action = policy[State(r, c, has_key, phase)]
                if action is not None:
                    ax.text(c, r + 0.05, ARROW[action], color=INK,
                            fontsize=6.5, ha="center", va="center")
        state_txt = "open" if phase in gate_open_phases else "closed"
        ax.set_title(f"phase {phase} — gate {state_txt}")
    for ax in axes.flat[period:]:
        ax.axis("off")
    fig.suptitle(f"{title} (has_key={has_key})", color=INK)
    save_figure(fig, path)


def plot_visit_map(maze: MazeMap, visits: dict[State, int], title: str,
                   path: Path) -> None:
    """State-visit heatmap on log scale, split by key possession."""
    grids = []
    for k in (0, 1):
        grid = np.full((maze.size, maze.size), np.nan)
        for r in range(maze.size):
            for c in range(maze.size):
                if not maze.is_wall((r, c)):
                    total = sum(visits.get(State(r, c, k, p), 0)
                                for p in range(maze.gate_period))
                    grid[r, c] = np.log10(total + 1)
        grids.append(np.ma.masked_invalid(grid))
    vmax = max(g.max() for g in grids)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, grid, k in zip(axes, grids, (0, 1)):
        im = ax.imshow(grid, cmap=VALUE_CMAP, vmin=0, vmax=vmax)
        _grid_lines(ax, maze.size)
        _cell_letters(ax, maze)
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("log₁₀(visits + 1), summed over gate phases",
                   color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(title, color=INK)
    save_figure(fig, path)


def plot_final_paths(maze: MazeMap, rollouts: list[tuple[str, dict]],
                     title: str, path: Path) -> None:
    """One greedy episode per agent (from greedy_rollout), drawn through
    cell centers and colored by step order; key pickup and door passage
    marked."""
    fig, axes = plt.subplots(1, len(rollouts),
                             figsize=(5.1 * len(rollouts), 5.6))
    for ax, (label, roll) in zip(np.atleast_1d(axes), rollouts):
        draw_maze(ax, maze)
        pts = np.array([(s.c, s.r) for s in roll["states"]], dtype=float)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segments, cmap=PATH_CMAP,
                            array=np.arange(len(segments)),
                            linewidth=2.4, capstyle="round", zorder=2.6)
        ax.add_collection(lc)
        for event, marker, size in ((EV_KEY_PICKUP, "*", 13),
                                    (EV_DOOR_PASS, "D", 8)):
            step = roll["event_steps"].get(event)
            if step is not None:
                s = roll["states"][step]
                ax.plot(s.c, s.r, marker,
                        color=CELL_WASH[KEY if event == EV_KEY_PICKUP
                                        else DOOR],
                        markersize=size, markeredgecolor="white",
                        markeredgewidth=0.8, zorder=3.5)
        outcome = "goal" if roll["terminated"] else "timeout"
        ax.set_title(f"{label} — {roll['steps']} steps, "
                     f"return {roll['return']:.0f} ({outcome})")
    cbar = fig.colorbar(lc, ax=axes, shrink=0.85)
    cbar.set_label("step along the episode", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(f"{title} — ★ key pickup, ◆ door passage", color=INK)
    save_figure(fig, path)


def plot_disagreement_map(maze: MazeMap, agent_policy: dict,
                          agent_defined: set, vi_policy: dict, title: str,
                          path: Path) -> None:
    """Per-cell share of gate phases whose greedy action matches VI.

    Blue = agrees in all phases, red = disagrees in all; cells the agent
    never visited stay surface-colored.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, k in zip(axes, (0, 1)):
        base = np.tile(hex_to_rgb(SURFACE), (maze.size, maze.size, 1))
        frac = np.full((maze.size, maze.size), np.nan)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    base[r, c] = hex_to_rgb(WALL_COLOR)
                    continue
                defined = [State(r, c, k, p) for p in range(maze.gate_period)
                           if State(r, c, k, p) in agent_defined
                           and vi_policy.get(State(r, c, k, p)) is not None]
                if defined:
                    frac[r, c] = (sum(agent_policy[s] == vi_policy[s]
                                      for s in defined) / len(defined))
        ax.imshow(base)
        im = ax.imshow(np.ma.masked_invalid(frac), cmap=DIVERGING_CMAP,
                       vmin=0.0, vmax=1.0)
        _grid_lines(ax, maze.size)
        _cell_letters(ax, maze)
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("share of gate phases agreeing with VI", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(f"{title} — unvisited cells left blank", color=INK)
    save_figure(fig, path)
