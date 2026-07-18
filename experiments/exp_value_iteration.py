"""Value Iteration experiment: gamma sweep + reference-policy figures."""

from __future__ import annotations

from agents.value_iteration import MODELS_DIR, ValueIteration, greedy_rollouts
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import MazeEnv, State
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR, plot_convergence,
                                  write_csv)
from experiments.common import EVAL_EPISODES, EVAL_SEED
from experiments.maze_plots import (plot_policy_arrows,
                                    plot_policy_energy_grid,
                                    plot_value_heatmap)

# panel budgets for the per-energy policy figure: full budget down into the
# desperation band (e <= 14) where the policy starts detouring around pits
ENERGY_PANEL_LEVELS = [57, 40, 30, 20, 12, 6]

RAW_DIR = RAW_DATA_DIR / "vi"
FIG_DIR = FIGURES_DIR / "vi"
MODEL_DIR = MODELS_DIR / "vi"


def run_value_iteration(config: dict) -> None:
    maze = MazeMap.load(MAPS_DIR / "source.json")
    env = MazeEnv(maze, config, reward_mode="sparse")
    vi_cfg = config["value_iteration"]
    ref_gamma = vi_cfg["gamma"]
    start = State(maze.start[0], maze.start[1], 0, env.energy_initial)

    results = {}
    for gamma in vi_cfg["gamma_sweep"]:
        vi = ValueIteration(env, gamma,
                            threshold=vi_cfg["convergence_threshold"],
                            max_iterations=vi_cfg["max_iterations"])
        results[gamma] = vi.solve()
        results[gamma].save(MODEL_DIR / f"vi_sparse_gamma{gamma:g}.json")
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
            "death_rate": stats["death_rate"],
            "mean_return": round(stats["mean_return"], 2),
            "mean_steps_when_successful":
                round(stats["mean_steps_when_successful"], 2),
            "policy_agreement_vs_ref": round(agreement, 4),
            "ref_gamma": ref_gamma,
        })
    write_csv(rows, RAW_DIR / "vi_gamma_sweep.csv")

    plot_convergence({g: r.deltas for g, r in results.items()},
                     vi_cfg["convergence_threshold"],
                     "Value Iteration convergence per discount factor",
                     FIG_DIR / "vi_convergence.png")
    plot_value_heatmap(maze, reference.V,
                       f"Optimal value function (γ={ref_gamma:g}, sparse reward)",
                       FIG_DIR / f"vi_value_heatmap_gamma{ref_gamma:g}.png")
    plot_policy_arrows(maze, reference.policy,
                       f"Optimal policy (γ={ref_gamma:g}, sparse reward)",
                       FIG_DIR / f"vi_policy_gamma{ref_gamma:g}.png")
    plot_policy_energy_grid(maze, reference.policy, ENERGY_PANEL_LEVELS,
                            f"Optimal policy per energy level (γ={ref_gamma:g})",
                            FIG_DIR / f"vi_policy_by_energy_gamma{ref_gamma:g}.png")
    print(f"  wrote {RAW_DIR / 'vi_gamma_sweep.csv'} and 4 figures")
