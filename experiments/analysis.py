"""Shared figure-generation and analysis helpers for the experiment scripts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.generator import (DOOR, GATE, GOAL, KEY, PENALTY, START,
                                    MazeMap)
from environments.maze import State

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RAW_DATA_DIR = PROJECT_ROOT / "results" / "raw_data"

# light-mode palette (validated reference palette; see report)
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]
VALUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
VALUE_CMAP.set_bad("#383835")  # walls
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]  # fixed slot order
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
WALL_COLOR = "#383835"

CELL_LETTER = {START: "S", KEY: "K", DOOR: "D", GOAL: "G", GATE: "T",
               PENALTY: "P"}
CELL_WASH = {START: "#2a78d6", KEY: "#eda100", DOOR: "#e87ba4",
             GOAL: "#1baf7a", GATE: "#4a3aa7", PENALTY: "#e34948"}
ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}  # U D L R

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.titlesize": 10,
})


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


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


def draw_maze(ax, maze: MazeMap, wash_alpha: float = 0.35,
              letter_size: float = 6.0) -> None:
    """Recessive maze background: dark walls, washed + lettered special cells."""
    n = maze.size
    img = np.tile(_rgb(SURFACE), (n, n, 1))
    for r in range(n):
        for c in range(n):
            cell = maze.grid[r][c]
            if maze.is_wall((r, c)):
                img[r, c] = _rgb(WALL_COLOR)
            elif cell in CELL_WASH:
                img[r, c] = ((1 - wash_alpha) * _rgb(SURFACE)
                             + wash_alpha * _rgb(CELL_WASH[cell]))
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
        for (r, c), ch in ((pos, CELL_LETTER[maze.grid[pos[0]][pos[1]]])
                           for pos in [tuple(maze.start), tuple(maze.key),
                                       tuple(maze.door), tuple(maze.goal),
                                       tuple(maze.gate)]
                           + [tuple(p) for p in maze.penalty_cells]):
            ax.text(c, r, ch, color=INK, fontsize=7, ha="center", va="center",
                    path_effects=[path_effects.withStroke(linewidth=1.6,
                                                          foreground="white")])
        ax.set_title(f"{'without key (k=0)' if k == 0 else 'with key (k=1)'}")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("V (max over gate phases)", color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(title, color=INK)
    _save(fig, path)


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
    _save(fig, path)


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
    _save(fig, path)


def plot_convergence(deltas_by_gamma: dict[float, list[float]],
                     threshold: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (gamma, deltas), color in zip(sorted(deltas_by_gamma.items()),
                                      CATEGORICAL):
        xs = range(1, len(deltas) + 1)
        ax.semilogy(xs, deltas, color=color, linewidth=2,
                    label=f"γ = {gamma:g}")
        ax.annotate(f"γ = {gamma:g}", (len(deltas), deltas[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=INK_2)
    ax.axhline(threshold, color=BASELINE, linewidth=1, linestyle="--")
    ax.annotate(f"threshold {threshold:g}", (1, threshold),
                textcoords="offset points", xytext=(2, 4),
                fontsize=8, color=MUTED)
    ax.set_xlabel("sweep")
    ax.set_ylabel("max |Vₖ₊₁ − Vₖ|")
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_title(title, color=INK)
    ax.margins(x=0.08)
    _save(fig, path)


def rolling_mean(values, window: int) -> np.ndarray:
    return np.convolve(np.asarray(values, dtype=float),
                       np.ones(window) / window, mode="valid")


def plot_training_curves(runs: dict[str, list[list[dict]]],
                         metrics: list[tuple[str, str]], title: str,
                         path: Path, window: int = 100) -> None:
    """Seed-averaged rolling curves; runs = {label: [per-seed history rows]}."""
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(4.4 * len(metrics), 3.5))
    for (key, ylabel), ax in zip(metrics, np.atleast_1d(axes)):
        for (label, seed_histories), color in zip(runs.items(), CATEGORICAL):
            curves = np.array([[row[key] for row in h]
                               for h in seed_histories], dtype=float)
            smoothed = np.array([rolling_mean(c, window) for c in curves])
            xs = np.arange(window - 1, curves.shape[1])
            mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
            ax.plot(xs, mean, color=color, linewidth=2, label=label)
            if len(seed_histories) > 1:
                ax.fill_between(xs, mean - std, mean + std, color=color,
                                alpha=0.18, linewidth=0)
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8,
                                  labelcolor=INK_2)
    fig.suptitle(f"{title} (rolling mean, window {window})", color=INK)
    fig.tight_layout()
    _save(fig, path)


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
        for pos in ([tuple(maze.start), tuple(maze.key), tuple(maze.door),
                     tuple(maze.goal), tuple(maze.gate)]
                    + [tuple(p) for p in maze.penalty_cells]):
            ch = CELL_LETTER[maze.grid[pos[0]][pos[1]]]
            ax.text(pos[1], pos[0], ch, color=INK, fontsize=7, ha="center",
                    va="center",
                    path_effects=[path_effects.withStroke(
                        linewidth=1.6, foreground="white")])
        ax.set_title("without key (k=0)" if k == 0 else "with key (k=1)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("log₁₀(visits + 1), summed over gate phases",
                   color=INK_2)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle(title, color=INK)
    _save(fig, path)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
