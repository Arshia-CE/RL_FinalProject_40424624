"""Batch experiment runner; regenerates all raw data, models and figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.value_iteration import (MODELS_DIR, ValueIteration,
                                    greedy_rollouts)
from environments.generator import DEFAULT_CONFIG_PATH, MAPS_DIR, MazeMap
from environments.maze import MazeEnv, State
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR, plot_convergence,
                                  plot_policy_arrows, plot_policy_phase_grid,
                                  plot_value_heatmap, write_csv)

EVAL_EPISODES = 500
EVAL_SEED = 999


def run_value_iteration(config: dict) -> None:
    """Gamma sweep + reference-policy figures (report section: Value Iteration)."""
    maze = MazeMap.load(MAPS_DIR / "source.json")
    env = MazeEnv(maze, config, reward_mode="sparse")
    vi_cfg = config["value_iteration"]
    ref_gamma = vi_cfg["gamma"]
    start = State(maze.start[0], maze.start[1], 0, 0)

    results = {}
    for gamma in vi_cfg["gamma_sweep"]:
        vi = ValueIteration(env, gamma,
                            threshold=vi_cfg["convergence_threshold"],
                            max_iterations=vi_cfg["max_iterations"])
        results[gamma] = vi.solve()
        results[gamma].save(MODELS_DIR / f"vi_sparse_gamma{gamma:g}.json")
        print(f"  gamma={gamma:g}: {results[gamma].iterations} sweeps, "
              f"{results[gamma].runtime_seconds:.2f}s, "
              f"V(start)={results[gamma].V[start]:.2f}")

    reference = results[ref_gamma]
    nonterminal = [s for s, a in reference.policy.items() if a is not None]
    rows = []
    for gamma, res in sorted(results.items()):
        stats = greedy_rollouts(MazeEnv(maze, config, reward_mode="sparse"),
                                res.policy, episodes=EVAL_EPISODES,
                                seed=EVAL_SEED)
        agreement = sum(res.policy[s] == reference.policy[s]
                        for s in nonterminal) / len(nonterminal)
        rows.append({
            "gamma": gamma,
            "iterations": res.iterations,
            "runtime_seconds": round(res.runtime_seconds, 4),
            "converged": res.converged,
            "v_start": round(res.V[start], 4),
            "eval_episodes": stats["episodes"],
            "success_rate": stats["success_rate"],
            "mean_return": round(stats["mean_return"], 2),
            "mean_steps_when_successful":
                round(stats["mean_steps_when_successful"], 2),
            "policy_agreement_vs_ref": round(agreement, 4),
            "ref_gamma": ref_gamma,
        })
    write_csv(rows, RAW_DATA_DIR / "vi_gamma_sweep.csv")

    plot_convergence({g: r.deltas for g, r in results.items()},
                     vi_cfg["convergence_threshold"],
                     "Value Iteration convergence per discount factor",
                     FIGURES_DIR / "vi_convergence.png")
    plot_value_heatmap(maze, reference.V,
                       f"Optimal value function (γ={ref_gamma:g}, sparse reward)",
                       FIGURES_DIR / f"vi_value_heatmap_gamma{ref_gamma:g}.png")
    plot_policy_arrows(maze, reference.policy,
                       f"Optimal policy (γ={ref_gamma:g}, sparse reward)",
                       FIGURES_DIR / f"vi_policy_gamma{ref_gamma:g}.png")
    plot_policy_phase_grid(maze, reference.policy, maze.gate_open_phases,
                           f"Optimal policy per gate phase (γ={ref_gamma:g})",
                           FIGURES_DIR / f"vi_policy_by_phase_gamma{ref_gamma:g}.png")
    print(f"  wrote {RAW_DATA_DIR / 'vi_gamma_sweep.csv'} and 4 figures")


EXPERIMENTS = {
    "vi": run_value_iteration,
    # added step by step: q_learning, sarsa_lambda, comparison, transfer
}


def main() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; "
                             f"available: {', '.join(EXPERIMENTS)}")
        print(f"[{name}]")
        EXPERIMENTS[name](config)
    print("done")


if __name__ == "__main__":
    main()
