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

from environments.maze_map import DOOR, GOAL, KEY, PENALTY, START, MazeMap
from environments.maze import EV_DOOR_PASS, EV_KEY_PICKUP, State
from experiments.analysis import (ARROW, DIVERGING_CMAP, GRID_COLOR, INK,
                                  INK_2, SEQ_RAMP, SURFACE, VALUE_CMAP,
                                  WALL_COLOR, hex_to_rgb, save_figure)

# the map's old gate cell ("T") is plain floor under the energy dynamics,
# so it gets no letter or wash
CELL_LETTER = {START: "S", KEY: "K", DOOR: "D", GOAL: "G", PENALTY: "P"}
CELL_WASH = {START: "#2a78d6", KEY: "#eda100", DOOR: "#e87ba4",
             GOAL: "#1baf7a", PENALTY: "#e34948"}
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
               reduce_energy=max) -> np.ma.MaskedArray:
    """V reduced over non-exhausted energy levels for one key status; walls
    masked (energy 0 is terminal, its zeros would pollute the reduction)."""
    by_cell: dict[tuple[int, int], list[float]] = {}
    for s, v in V.items():
        if s.has_key == has_key and s.energy > 0:
            by_cell.setdefault((s.r, s.c), []).append(v)
    grid = np.full((maze.size, maze.size), np.nan)
    for (r, c), vals in by_cell.items():
        grid[r, c] = reduce_energy(vals)
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
    cbar.set_label("V (max over energy levels)", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(title, color=INK)
    save_figure(fig, path)


def plot_policy_arrows(maze: MazeMap, policy: dict[State, int | None],
                       title: str, path: Path,
                       energy: int | None = None) -> None:
    if energy is None:
        energy = max(s.energy for s in policy)  # the full starting budget
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, k in zip(axes, (0, 1)):
        draw_maze(ax, maze)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    continue
                action = policy[State(r, c, k, energy)]
                if action is None:
                    continue  # terminal; the G letter already marks it
                ax.text(c, r + 0.05, ARROW[action], color=INK, fontsize=9,
                        ha="center", va="center")
        ax.set_title(f"{'without key (k=0)' if k == 0 else 'with key (k=1)'}"
                     f", energy {energy}")
    fig.suptitle(title, color=INK)
    save_figure(fig, path)


def plot_policy_energy_grid(maze: MazeMap, policy: dict[State, int | None],
                            energy_levels: list[int], title: str,
                            path: Path, has_key: int = 1) -> None:
    """Small multiples: the greedy policy at chosen energy levels (fixed
    key); where low-budget desperation shortcuts show up."""
    n = len(energy_levels)
    rows = 2
    cols = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows))
    for energy, ax in zip(energy_levels, axes.flat):
        draw_maze(ax, maze, letter_size=5.0)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    continue
                action = policy[State(r, c, has_key, energy)]
                if action is not None:
                    ax.text(c, r + 0.05, ARROW[action], color=INK,
                            fontsize=6.5, ha="center", va="center")
        ax.set_title(f"energy {energy}")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{title} (has_key={has_key})", color=INK)
    save_figure(fig, path)


def plot_visit_map(maze: MazeMap, visits: dict[State, int], title: str,
                   path: Path) -> None:
    """State-visit heatmap on log scale, split by key possession."""
    totals: dict[tuple[int, int, int], int] = {}
    for s, v in visits.items():
        cell = (s.r, s.c, s.has_key)
        totals[cell] = totals.get(cell, 0) + v
    grids = []
    for k in (0, 1):
        grid = np.full((maze.size, maze.size), np.nan)
        for r in range(maze.size):
            for c in range(maze.size):
                if not maze.is_wall((r, c)):
                    grid[r, c] = np.log10(totals.get((r, c, k), 0) + 1)
        grids.append(np.ma.masked_invalid(grid))
    vmax = max(g.max() for g in grids)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, grid, k in zip(axes, grids, (0, 1)):
        im = ax.imshow(grid, cmap=VALUE_CMAP, vmin=0, vmax=vmax)
        _grid_lines(ax, maze.size)
        _cell_letters(ax, maze)
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("log₁₀(visits + 1), summed over energy levels",
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
        ax.set_title(f"{label} — {roll['steps']} steps, "
                     f"return {roll['return']:.0f} ({roll['outcome']})")
    cbar = fig.colorbar(lc, ax=axes, shrink=0.85)
    cbar.set_label("step along the episode", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(f"{title} — ★ key pickup, ◆ door passage", color=INK)
    save_figure(fig, path)


def plot_transfer_q_diff(maze: MazeMap, q_before: dict, q_after: dict,
                         title: str, path: Path) -> None:
    """Before/after target training on a transferred table: per-cell
    max-action |ΔQ| (top row) and the share of energy levels keeping the
    transferred greedy action (bottom row; levels with no data blank)."""
    zeros = np.zeros(len(ARROW))
    per_cell: dict[tuple[int, int, int], list[State]] = {}
    for s in set(q_before) | set(q_after):
        per_cell.setdefault((s.r, s.c, s.has_key), []).append(s)
    dq_grids, keep_grids = [], []
    for k in (0, 1):
        dq = np.full((maze.size, maze.size), np.nan)
        keep = np.full((maze.size, maze.size), np.nan)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    continue
                diffs, kept, counted = [0.0], 0, 0
                for s in per_cell.get((r, c, k), []):
                    qb = q_before.get(s, zeros)
                    qa = q_after.get(s, zeros)
                    diffs.append(float(np.abs(qa - qb).max()))
                    if qb.any() or qa.any():
                        counted += 1
                        kept += int(np.argmax(qa) == np.argmax(qb))
                dq[r, c] = max(diffs)
                if counted:
                    keep[r, c] = kept / counted
        dq_grids.append(np.ma.masked_invalid(dq))
        keep_grids.append(np.ma.masked_invalid(keep))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.6))
    vmax = max(g.max() for g in dq_grids)
    for ax, grid, k in zip(axes[0], dq_grids, (0, 1)):
        im_dq = ax.imshow(grid, cmap=VALUE_CMAP, vmin=0, vmax=vmax)
        _grid_lines(ax, maze.size)
        _cell_letters(ax, maze)
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im_dq, ax=axes[0], shrink=0.9)
    cbar.set_label("max-action |ΔQ| (max over energy levels)", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    for ax, grid, k in zip(axes[1], keep_grids, (0, 1)):
        base = np.tile(hex_to_rgb(SURFACE), (maze.size, maze.size, 1))
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    base[r, c] = hex_to_rgb(WALL_COLOR)
        ax.imshow(base)
        im_keep = ax.imshow(grid, cmap=DIVERGING_CMAP, vmin=0.0, vmax=1.0)
        _grid_lines(ax, maze.size)
        _cell_letters(ax, maze)
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im_keep, ax=axes[1], shrink=0.9)
    cbar.set_label("share of energy levels keeping the transferred greedy "
                   "action", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(title, color=INK)
    save_figure(fig, path)


def plot_disagreement_map(maze: MazeMap, agent_policy: dict,
                          agent_defined: set, vi_policy: dict, title: str,
                          path: Path) -> None:
    """Per-cell share of energy levels whose greedy action matches VI.

    Blue = agrees at every level, red = disagrees at every level; cells the
    agent never visited stay surface-colored.
    """
    per_cell: dict[tuple[int, int, int], list[State]] = {}
    for s in agent_defined:
        if vi_policy.get(s) is not None:
            per_cell.setdefault((s.r, s.c, s.has_key), []).append(s)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, k in zip(axes, (0, 1)):
        base = np.tile(hex_to_rgb(SURFACE), (maze.size, maze.size, 1))
        frac = np.full((maze.size, maze.size), np.nan)
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    base[r, c] = hex_to_rgb(WALL_COLOR)
                    continue
                defined = per_cell.get((r, c, k))
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
    cbar.set_label("share of energy levels agreeing with VI", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(f"{title} — unvisited cells left blank", color=INK)
    save_figure(fig, path)
