"""Batch experiment runner; regenerates all raw data, models and figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from agents.q_learning import QLearningAgent, epsilon_schedule
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import (MODELS_DIR, ValueIteration, VIResult,
                                    greedy_rollouts)
from environments.generator import DEFAULT_CONFIG_PATH, MAPS_DIR, MazeMap
from environments.maze import (EV_GATE_BLOCKED, EV_PENALTY, EV_WALL_HIT,
                               MazeEnv, State)
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR, plot_convergence,
                                  plot_policy_arrows, plot_policy_phase_grid,
                                  plot_sarsa_trace, plot_training_curves,
                                  plot_value_heatmap, plot_visit_map,
                                  rolling_mean, write_csv)

EVAL_EPISODES = 500
EVAL_SEED = 999


def evaluate_greedy(env: MazeEnv, table: dict[State, np.ndarray],
                    episodes: int, seed: int) -> dict:
    """Greedy rollouts from a (possibly partial) Q-table on the sparse env."""
    from collections import Counter
    env.reset(seed=seed)
    events, returns, steps, wins = Counter(), [], [], 0
    for _ in range(episodes):
        state = env.reset()
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            q = table.get(state)
            action = 0 if q is None else int(np.argmax(q))
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            events.update(info["events"])
        wins += terminated
        returns.append(total)
        steps.append(env.steps)
    return {"success_rate": wins / episodes,
            "mean_return": round(sum(returns) / episodes, 2),
            "mean_steps": round(sum(steps) / episodes, 2),
            "penalty_entries_per_ep": round(events[EV_PENALTY] / episodes, 3),
            "wall_hits_per_ep": round(events[EV_WALL_HIT] / episodes, 3),
            "gate_blocked_per_ep": round(events[EV_GATE_BLOCKED] / episodes, 3)}


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


def run_q_learning(config: dict) -> None:
    """Decay-schedule and reward-shaping studies (report section: Q-Learning)."""
    maze = MazeMap.load(MAPS_DIR / "source.json")
    qcfg = config["q_learning"]
    episodes = qcfg["episodes"]
    seeds = qcfg["seeds"]
    reference = VIResult.load(
        MODELS_DIR / f"vi_sparse_gamma{config['value_iteration']['gamma']:g}.json")

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
                    write_csv(trace, RAW_DATA_DIR / "q_update_trace.csv")
                if is_canonical:
                    canonical[(mode, schedule_name)] = agent
                    agent.save(
                        MODELS_DIR / f"q_learning_{mode}_{schedule_name}.json",
                        {"reward_mode": mode, "schedule": schedule_name,
                         "episodes": episodes, "seed": seed,
                         "epsilon_start": qcfg["epsilon_start"],
                         "epsilon_end": qcfg["epsilon_end"],
                         "epsilon_decay_episodes":
                             qcfg["epsilon_decay_episodes"]})

                # summary metrics (evaluation always on the sparse env)
                eval_stats = evaluate_greedy(
                    MazeEnv(maze, config, reward_mode="sparse"),
                    agent.Q, EVAL_EPISODES, EVAL_SEED)
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

    write_csv(training_rows, RAW_DATA_DIR / "q_learning_training.csv")
    write_csv(summary_rows, RAW_DATA_DIR / "q_learning_summary.csv")

    plot_training_curves(
        {"linear": histories[("sparse", "linear")],
         "exponential": histories[("sparse", "exponential")]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "ε-decay schedules, sparse reward",
        FIGURES_DIR / "q_learning_decay_schedules.png")
    plot_training_curves(
        {"sparse": histories[("sparse", "exponential")],
         "shaped": histories[("shaped", "exponential")]},
        [("success", "success rate"), ("key_picked", "key found rate"),
         ("steps", "steps per episode")],
        "sparse vs shaped reward, exponential decay",
        FIGURES_DIR / "q_learning_reward_shaping.png")
    plot_visit_map(maze, canonical[("sparse", "exponential")].visits,
                   "State visits during training (sparse, exponential decay)",
                   FIGURES_DIR / "q_learning_visit_map.png")

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
    print(f"  wrote 2 CSVs, q_update_trace.csv and 3 figures")


def run_sarsa_lambda(config: dict) -> None:
    """Lambda sweep with speed/stability metrics (report section: SARSA(λ))."""
    maze = MazeMap.load(MAPS_DIR / "source.json")
    scfg = config["sarsa_lambda"]
    episodes = scfg["episodes"]
    seeds = scfg["seeds"]
    reference = VIResult.load(
        MODELS_DIR / f"vi_sparse_gamma{config['value_iteration']['gamma']:g}.json")
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
                write_csv(step_trace, RAW_DATA_DIR / "sarsa_step_trace.csv")
                write_csv(dump, RAW_DATA_DIR / "sarsa_trace_dump.csv")
                plot_sarsa_trace(step_trace, dump, scfg["gamma"], lam,
                                 f"SARSA(λ={lam:g}) traced episode "
                                 f"{scfg['trace_episode']}",
                                 FIGURES_DIR / "sarsa_delta_trace.png")
            if is_canonical:
                agent.save(MODELS_DIR / f"sarsa_lambda{lam:g}_sparse.json",
                           {"reward_mode": "sparse", "seed": seed,
                            "episodes": episodes,
                            "schedule": scfg["epsilon_decay_schedule"],
                            "trace_prune": scfg["trace_prune"]})

            eval_stats = evaluate_greedy(
                MazeEnv(maze, config, reward_mode="sparse"),
                agent.Q, EVAL_EPISODES, EVAL_SEED)
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

    write_csv(training_rows, RAW_DATA_DIR / "sarsa_training.csv")
    write_csv(summary_rows, RAW_DATA_DIR / "sarsa_summary.csv")
    plot_training_curves(
        {f"λ = {lam:g}": histories[lam] for lam in scfg["lambda_sweep"]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "SARSA(λ) lambda sweep, sparse reward",
        FIGURES_DIR / "sarsa_lambda_sweep.png")
    print("  wrote 4 CSVs and 2 figures")


EXPERIMENTS = {
    "vi": run_value_iteration,
    "q_learning": run_q_learning,
    "sarsa_lambda": run_sarsa_lambda,
    # added step by step: comparison, transfer
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
