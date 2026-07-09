"""Q-Learning experiments: epsilon-decay schedules and reward shaping."""

from __future__ import annotations

from agents.q_learning import QLearningAgent, epsilon_schedule
from agents.value_iteration import MODELS_DIR, VIResult
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import MazeEnv
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR,
                                  plot_training_curves, rolling_mean,
                                  write_csv)
from experiments.common import (EVAL_EPISODES, EVAL_SEED, evaluate_greedy,
                                table_policy)
from experiments.maze_plots import plot_visit_map

RAW_DIR = RAW_DATA_DIR / "q_learning"
FIG_DIR = FIGURES_DIR / "q_learning"
MODEL_DIR = MODELS_DIR / "q_learning"


def run_q_learning(config: dict) -> None:
    maze = MazeMap.load(MAPS_DIR / "source.json")
    qcfg = config["q_learning"]
    episodes = qcfg["episodes"]
    seeds = qcfg["seeds"]
    reference = VIResult.load(
        MODELS_DIR / "vi"
        / f"vi_sparse_gamma{config['value_iteration']['gamma']:g}.json")

    histories: dict[tuple[str, str], list[list[dict]]] = {}
    canonical: dict[tuple[str, str], QLearningAgent] = {}
    training_rows, summary_rows = [], []
    for mode in ("sparse", "shaped"):
        for schedule_name in qcfg["epsilon_decay_schedules"]:
            per_seed = []
            for seed in seeds:
                env = MazeEnv(maze, config, reward_mode=mode, seed=seed)
                agent = QLearningAgent(env, qcfg["alpha"], qcfg["gamma"],
                                       seed=seed)
                schedule = epsilon_schedule(schedule_name,
                                            qcfg["epsilon_start"],
                                            qcfg["epsilon_end"],
                                            qcfg["epsilon_decay_episodes"])
                is_canonical = seed == seeds[0]
                trace_eps = (frozenset({qcfg["trace_episode"]})
                             if is_canonical and mode == "sparse"
                             and schedule_name == "exponential"
                             else frozenset())
                history, trace = agent.train(episodes, schedule,
                                             trace_episodes=trace_eps)
                per_seed.append(history)
                training_rows += [{"reward_mode": mode,
                                   "schedule": schedule_name, "seed": seed,
                                   **row} for row in history]
                if trace:
                    write_csv(trace, RAW_DIR / "q_update_trace.csv")
                if is_canonical:
                    canonical[(mode, schedule_name)] = agent
                    agent.save(
                        MODEL_DIR / f"q_learning_{mode}_{schedule_name}.json",
                        {"reward_mode": mode, "schedule": schedule_name,
                         "episodes": episodes, "seed": seed,
                         "epsilon_start": qcfg["epsilon_start"],
                         "epsilon_end": qcfg["epsilon_end"],
                         "epsilon_decay_episodes":
                             qcfg["epsilon_decay_episodes"]})

                # summary metrics (evaluation always on the sparse env)
                eval_stats = evaluate_greedy(
                    MazeEnv(maze, config, reward_mode="sparse"),
                    table_policy(agent.Q), EVAL_EPISODES, EVAL_SEED)
                success = [row["success"] for row in history]
                roll = rolling_mean(success, 100)
                sustained = next((i + 99 for i, v in enumerate(roll)
                                  if v >= 0.9), None)
                policy = agent.greedy_policy()
                visited = [s for s in agent.visited_states()
                           if policy[s] is not None]
                agreement = (sum(policy[s] == reference.policy[s]
                                 for s in visited) / len(visited))
                summary_rows.append({
                    "reward_mode": mode, "schedule": schedule_name,
                    "seed": seed,
                    "first_success_episode":
                        next((r["episode"] for r in history if r["success"]),
                             None),
                    "episodes_to_90pct_success": sustained,
                    "final_train_success_rate": round(float(roll[-1]), 3),
                    **{f"eval_{k}": v for k, v in eval_stats.items()},
                    "visited_states": len(visited),
                    "vi_policy_agreement": round(agreement, 4),
                })
                print(f"  {mode}/{schedule_name} seed {seed}: "
                      f"90% success at ep {sustained}, "
                      f"eval return {eval_stats['mean_return']}, "
                      f"agreement {agreement:.1%}")
            histories[(mode, schedule_name)] = per_seed

    write_csv(training_rows, RAW_DIR / "q_learning_training.csv")
    write_csv(summary_rows, RAW_DIR / "q_learning_summary.csv")

    plot_training_curves(
        {"linear": histories[("sparse", "linear")],
         "exponential": histories[("sparse", "exponential")]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "ε-decay schedules, sparse reward",
        FIG_DIR / "q_learning_decay_schedules.png")
    plot_training_curves(
        {"sparse": histories[("sparse", "exponential")],
         "shaped": histories[("shaped", "exponential")]},
        [("success", "success rate"), ("key_picked", "key found rate"),
         ("steps", "steps per episode")],
        "sparse vs shaped reward, exponential decay",
        FIG_DIR / "q_learning_reward_shaping.png")
    plot_visit_map(maze, canonical[("sparse", "exponential")].visits,
                   "State visits during training (sparse, exponential decay)",
                   FIG_DIR / "q_learning_visit_map.png")

    # did shaping change the final policy? compare on jointly-visited states
    sparse_agent = canonical[("sparse", "exponential")]
    shaped_agent = canonical[("shaped", "exponential")]
    sparse_policy = sparse_agent.greedy_policy()
    shaped_policy = shaped_agent.greedy_policy()
    both = [s for s in sparse_agent.visited_states()
            if sparse_policy.get(s) is not None
            and s in shaped_agent.Q and shaped_agent.Q[s].any()]
    same = sum(sparse_policy[s] == shaped_policy[s] for s in both) / len(both)
    print(f"  sparse vs shaped greedy policies agree on {same:.1%} "
          f"of {len(both)} jointly-visited states")
    print("  wrote 2 CSVs, q_update_trace.csv and 3 figures")
