"""Three-algorithm comparison on the same map and sparse reward."""

from __future__ import annotations

import time

import numpy as np

from agents.q_learning import QLearningAgent, epsilon_schedule
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import MODELS_DIR, ValueIteration
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import MazeEnv, State
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR,
                                  plot_training_curves, rolling_mean,
                                  write_csv)
from experiments.common import (EVAL_EPISODES, EVAL_SEED, dict_policy,
                                evaluate_greedy, table_policy)
from experiments.maze_plots import plot_disagreement_map

RAW_DIR = RAW_DATA_DIR / "comparison"
FIG_DIR = FIGURES_DIR / "comparison"

BEST_LAMBDA = 0.7  # best speed/stability balance from the lambda sweep


def run_comparison(config: dict) -> None:
    maze = MazeMap.load(MAPS_DIR / "source.json")
    gamma = config["value_iteration"]["gamma"]
    qcfg, scfg = config["q_learning"], config["sarsa_lambda"]
    seed = qcfg["seeds"][0]

    # solve/train the three canonical agents with wall-clock timing
    t0 = time.perf_counter()
    vi = ValueIteration(MazeEnv(maze, config, reward_mode="sparse"), gamma,
                        threshold=config["value_iteration"]["convergence_threshold"],
                        max_iterations=config["value_iteration"]["max_iterations"])
    vi_res = vi.solve()
    vi_time = time.perf_counter() - t0  # includes model construction

    t0 = time.perf_counter()
    ql = QLearningAgent(MazeEnv(maze, config, reward_mode="sparse", seed=seed),
                        qcfg["alpha"], qcfg["gamma"], seed=seed)
    ql_hist, _ = ql.train(qcfg["episodes"],
                          epsilon_schedule("exponential",
                                           qcfg["epsilon_start"],
                                           qcfg["epsilon_end"],
                                           qcfg["epsilon_decay_episodes"]))
    ql_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sarsa = SarsaLambdaAgent(MazeEnv(maze, config, reward_mode="sparse",
                                     seed=seed),
                             scfg["alpha"], scfg["gamma"], BEST_LAMBDA,
                             trace_type=scfg["trace_type"],
                             trace_prune=scfg["trace_prune"], seed=seed)
    sarsa_hist, _, _ = sarsa.train(scfg["episodes"],
                                   epsilon_schedule(
                                       scfg["epsilon_decay_schedule"],
                                       scfg["epsilon_start"],
                                       scfg["epsilon_end"],
                                       scfg["epsilon_decay_episodes"]))
    sarsa_time = time.perf_counter() - t0

    # exact Q* for action-gap analysis
    v_star = np.array([vi_res.V[s] for s in vi.states])
    q_star = vi._q_from(v_star)
    v_pi = {"value_iteration": vi_res.V,
            "q_learning": vi.evaluate_policy(ql.greedy_policy()),
            "sarsa_lambda": vi.evaluate_policy(sarsa.greedy_policy())}
    start = State(maze.start[0], maze.start[1], 0, 0)

    penalty_adjacent = {(p[0] + dr, p[1] + dc) for p in maze.penalty_cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))}

    agents = {
        "value_iteration": (dict_policy(vi_res.policy), vi_time, None,
                            vi_res.policy,
                            {s for s, a in vi_res.policy.items()
                             if a is not None}),
        "q_learning": (table_policy(ql.Q), ql_time, ql_hist,
                       ql.greedy_policy(), set(ql.visited_states())),
        "sarsa_lambda": (table_policy(sarsa.Q), sarsa_time, sarsa_hist,
                         sarsa.greedy_policy(), set(sarsa.visited_states())),
    }
    visit_counts = {"q_learning": ql.visits, "sarsa_lambda": sarsa.visits}

    summary_rows, gap_records = [], {}
    for name, (act, runtime, hist, policy, defined) in agents.items():
        eval_stats = evaluate_greedy(MazeEnv(maze, config,
                                             reward_mode="sparse"),
                                     act, EVAL_EPISODES, EVAL_SEED)
        row = {"algorithm": name, "runtime_seconds": round(runtime, 2),
               "table_states": len(defined),
               "model_file_kb": None, "episodes_to_90pct_success": None,
               "env_steps_to_90pct": None, "total_env_steps": None,
               **{f"eval_{k}": v for k, v in eval_stats.items()},
               "v_pi_start": round(v_pi[name][start], 4),
               "v_star_start": round(v_pi["value_iteration"][start], 4)}
        if hist is not None:
            success = [r["success"] for r in hist]
            roll = rolling_mean(success, 100)
            sustained = next((i + 99 for i, v in enumerate(roll) if v >= 0.9),
                             None)
            row["episodes_to_90pct_success"] = sustained
            row["env_steps_to_90pct"] = sum(r["steps"]
                                            for r in hist[:sustained])
            row["total_env_steps"] = sum(r["steps"] for r in hist)
            # agreement + action gaps on visited, non-terminal states
            states = [s for s in defined if policy.get(s) is not None]
            agree = sum(policy[s] == vi_res.policy[s] for s in states)
            row["vi_agreement"] = round(agree / len(states), 4)
            gaps = []
            for s in states:
                if policy[s] != vi_res.policy[s]:
                    i = vi.index[s]
                    gaps.append((float(q_star[i, vi_res.policy[s]]
                                       - q_star[i, policy[s]]), s))
            gap_values = np.array([g for g, _ in gaps])
            gap_records[name] = gaps
            row["disagreements"] = len(gaps)
            row["median_action_gap"] = round(float(np.median(gap_values)), 3)
            row["gap_below_0.5"] = round(float((gap_values < 0.5).mean()), 4)
            pen = [s for s in states if (s.r, s.c) in penalty_adjacent]
            row["penalty_adjacent_agreement"] = round(
                sum(policy[s] == vi_res.policy[s] for s in pen) / len(pen), 4)
        summary_rows.append(row)
        print(f"  {name}: runtime {row['runtime_seconds']}s, "
              f"eval return {eval_stats['mean_return']}, "
              f"V^pi(start) {row['v_pi_start']}")

    model_files = {
        "value_iteration": MODELS_DIR / "vi" / f"vi_sparse_gamma{gamma:g}.json",
        "q_learning": MODELS_DIR / "q_learning"
                      / "q_learning_sparse_exponential.json",
        "sarsa_lambda": MODELS_DIR / "sarsa"
                        / f"sarsa_lambda{BEST_LAMBDA:g}_sparse.json",
    }
    for row in summary_rows:
        row["model_file_kb"] = round(
            model_files[row["algorithm"]].stat().st_size / 1024, 1)
    write_csv(summary_rows, RAW_DIR / "comparison_summary.csv")

    # three sample disagreement states, chosen by distinct mechanisms
    samples = []
    for label, source, pick in (
            ("penalty_adjacent", "sarsa_lambda",
             lambda gaps: max((g for g in gaps
                               if (g[1].r, g[1].c) in penalty_adjacent
                               and g[0] > 0.5),
                              key=lambda g: visit_counts["sarsa_lambda"]
                              .get(g[1], 0), default=None)),
            ("near_tie", "q_learning",
             lambda gaps: max((g for g in gaps if g[0] < 0.05),
                              key=lambda g: visit_counts["q_learning"]
                              .get(g[1], 0), default=None)),
            ("large_gap", "q_learning",
             lambda gaps: max(gaps, key=lambda g: g[0], default=None))):
        found = pick(gap_records[source])
        if found is None:
            continue
        gap, s = found
        i = vi.index[s]
        agent = ql if source == "q_learning" else sarsa
        agent_policy = agent.greedy_policy()
        samples.append({
            "label": label, "algorithm": source,
            "r": s.r, "c": s.c, "has_key": s.has_key, "phase": s.phase,
            "visits": visit_counts[source].get(s, 0),
            "vi_action": vi_res.policy[s], "agent_action": agent_policy[s],
            "q_star_vi_action": round(float(q_star[i, vi_res.policy[s]]), 3),
            "q_star_agent_action": round(float(q_star[i, agent_policy[s]]), 3),
            "action_gap": round(gap, 3),
            "agent_q_values": " ".join(f"{v:.2f}" for v in agent.Q[s]),
        })
        print(f"  sample [{label}] {source}: s=({s.r},{s.c},k={s.has_key},"
              f"p={s.phase}) visits={samples[-1]['visits']} "
              f"VI={samples[-1]['vi_action']} agent="
              f"{samples[-1]['agent_action']} gap={gap:.3f}")
    write_csv(samples, RAW_DIR / "comparison_sample_states.csv")

    plot_disagreement_map(maze, ql.greedy_policy(),
                          set(ql.visited_states()), vi_res.policy,
                          "Q-Learning greedy policy vs Value Iteration",
                          FIG_DIR / "comparison_disagreement_qlearning.png")
    plot_disagreement_map(maze, sarsa.greedy_policy(),
                          set(sarsa.visited_states()), vi_res.policy,
                          f"SARSA(λ={BEST_LAMBDA:g}) greedy policy vs "
                          f"Value Iteration",
                          FIG_DIR / "comparison_disagreement_sarsa.png")
    plot_training_curves(
        {"Q-Learning": [ql_hist], f"SARSA(λ={BEST_LAMBDA:g})": [sarsa_hist]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "Model-free methods, sparse reward, exponential decay",
        FIG_DIR / "comparison_learning_curves.png")
    print("  wrote 2 CSVs and 3 figures")
