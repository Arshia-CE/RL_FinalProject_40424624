"""Shared analysis utilities: palette/style tokens, CSV helpers and the
training-curve / convergence / trace figures. Maze-rendered figures live in
experiments/maze_plots.py."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RAW_DATA_DIR = PROJECT_ROOT / "results" / "raw_data"

# light-mode palette (validated reference palette; see report)
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]
VALUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
VALUE_CMAP.set_bad("#383835")  # walls
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#e34948", "#f0efec", "#2a78d6"])
DIVERGING_CMAP.set_bad(alpha=0.0)
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]  # fixed slot order
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
WALL_COLOR = "#383835"
ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}  # U D L R

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.titlesize": 10,
})


def hex_to_rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:  # union of keys, first-seen order (rows may differ)
        fieldnames += [k for k in row if k not in fieldnames]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values, window: int) -> np.ndarray:
    return np.convolve(np.asarray(values, dtype=float),
                       np.ones(window) / window, mode="valid")


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
    save_figure(fig, path)


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
    save_figure(fig, path)


def plot_sarsa_trace(step_trace: list[dict], trace_dump: list[dict],
                     gamma: float, lam: float, title: str,
                     path: Path) -> None:
    """One traced episode: per-step TD error and eligibility decay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.7))
    steps = [row["step"] for row in step_trace]
    ax1.axhline(0, color=BASELINE, linewidth=1)
    ax1.plot(steps, [row["delta"] for row in step_trace],
             color=CATEGORICAL[0], linewidth=2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("TD error δ")

    # eligibility of the pairs bumped in the first steps: straight lines on
    # a log axis confirm the (gamma*lambda)^t geometric decay
    pairs: list[tuple] = []
    for row in step_trace:
        key = (row["r"], row["c"], row["has_key"], row["phase"],
               row["action"])
        if key not in pairs:
            pairs.append(key)
        if len(pairs) == 4:
            break
    for key, color in zip(pairs, CATEGORICAL):
        series = [(row["step"], row["E"]) for row in trace_dump
                  if (row["r"], row["c"], row["has_key"], row["phase"],
                      row["action"]) == key]
        ax2.semilogy([s for s, _ in series], [e for _, e in series],
                     color=color, linewidth=2,
                     label=f"s=({key[0]},{key[1]}) p={key[3]} "
                           f"a={ARROW[key[4]]}")
    ax2.set_xlabel("step")
    ax2.set_ylabel("eligibility E (log scale)")
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    for ax in (ax1, ax2):
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{title} (γλ = {gamma * lam:g})", color=INK)
    fig.tight_layout()
    save_figure(fig, path)
