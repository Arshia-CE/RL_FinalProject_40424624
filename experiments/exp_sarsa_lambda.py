"""SARSA(lambda) experiment: lambda sweep with speed/stability metrics."""

from __future__ import annotations

import numpy as np

from agents.q_learning import epsilon_schedule
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import MODELS_DIR, VIResult
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import MazeEnv
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR,
                                  plot_sarsa_trace, plot_training_curves,
                                  rolling_mean, write_csv)
from experiments.common import (EVAL_EPISODES, EVAL_SEED, evaluate_greedy,
                                table_policy)

RAW_DIR = RAW_DATA_DIR / "sarsa"
FIG_DIR = FIGURES_DIR / "sarsa"
MODEL_DIR = MODELS_DIR / "sarsa"


def run_sarsa_lambda(config: dict) -> None:
    maze = MazeMap.load(MAPS_DIR / "source.json")
    scfg = config["sarsa_lambda"]
    episodes = scfg["episodes"]
    seeds = scfg["seeds"]
    reference = VIResult.load(
        MODELS_DIR / "vi"
        / f"vi_sparse_gamma{config['value_iteration']['gamma']:g}.json")
    schedule = epsilon_schedule(scfg["epsilon_decay_schedule"],
                                scfg["epsilon_start"], scfg["epsilon_end"],
                                scfg["epsilon_decay_episodes"])
    trace_lambda = scfg["lambda_sweep"][-1]

    histories: dict[float, list[list[dict]]] = {}
    training_rows, summary_rows = [], []
    for lam in scfg["lambda_sweep"]:
        per_seed = []
        for seed in seeds:
            env = MazeEnv(maze, config, reward_mode="sparse", seed=seed)
            agent = SarsaLambdaAgent(env, scfg["alpha"], scfg["gamma"], lam,
                                     trace_type=scfg["trace_type"],
                                     trace_prune=scfg["trace_prune"],
                                     seed=seed)
            is_canonical = seed == seeds[0]
            trace_eps = (frozenset({scfg["trace_episode"]})
                         if is_canonical and lam == trace_lambda
                         else frozenset())
            history, step_trace, dump = agent.train(episodes, schedule,
                                                    trace_episodes=trace_eps)
            per_seed.append(history)
            training_rows += [{"lambda": lam, "seed": seed, **row}
                              for row in history]
            if step_trace:
                write_csv(step_trace, RAW_DIR / "sarsa_step_trace.csv")
                write_csv(dump, RAW_DIR / "sarsa_trace_dump.csv")
                plot_sarsa_trace(step_trace, dump, scfg["gamma"], lam,
                                 f"SARSA(λ={lam:g}) traced episode "
                                 f"{scfg['trace_episode']}",
                                 FIG_DIR / "sarsa_delta_trace.png")
            if is_canonical:
                agent.save(MODEL_DIR / f"sarsa_lambda{lam:g}_sparse.json",
                           {"reward_mode": "sparse", "seed": seed,
                            "episodes": episodes,
                            "schedule": scfg["epsilon_decay_schedule"],
                            "trace_prune": scfg["trace_prune"]})

            eval_stats = evaluate_greedy(
                MazeEnv(maze, config, reward_mode="sparse"),
                table_policy(agent.Q), EVAL_EPISODES, EVAL_SEED)
            success = [row["success"] for row in history]
            roll = rolling_mean(success, 100)
            sustained = next((i + 99 for i, v in enumerate(roll) if v >= 0.9),
                             None)
            late_returns = [row["return"] for row in history[-1000:]]
            policy = agent.greedy_policy()
            visited = [s for s in agent.visited_states()
                       if policy[s] is not None]
            agreement = (sum(policy[s] == reference.policy[s]
                             for s in visited) / len(visited))
            summary_rows.append({
                "lambda": lam, "seed": seed,
                "first_success_episode":
                    next((r["episode"] for r in history if r["success"]),
                         None),
                "episodes_to_90pct_success": sustained,
                "final_train_success_rate": round(float(roll[-1]), 3),
                "late_return_std": round(float(np.std(late_returns)), 2),
                **{f"eval_{k}": v for k, v in eval_stats.items()},
                "visited_states": len(visited),
                "vi_policy_agreement": round(agreement, 4),
            })
            print(f"  lambda={lam:g} seed {seed}: 90% success at ep "
                  f"{sustained}, late return std "
                  f"{summary_rows[-1]['late_return_std']}, "
                  f"eval return {eval_stats['mean_return']}")
        histories[lam] = per_seed

    write_csv(training_rows, RAW_DIR / "sarsa_training.csv")
    write_csv(summary_rows, RAW_DIR / "sarsa_summary.csv")
    plot_training_curves(
        {f"λ = {lam:g}": histories[lam] for lam in scfg["lambda_sweep"]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "SARSA(λ) lambda sweep, sparse reward",
        FIG_DIR / "sarsa_lambda_sweep.png")
    print("  wrote 4 CSVs and 2 figures")
