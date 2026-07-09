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
from transfer.transfer_learning import initial_q_table, unchanged_positions
from environments.maze import (EV_GATE_BLOCKED, EV_PENALTY, EV_WALL_HIT,
                               MazeEnv, State)
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR, plot_convergence,
                                  plot_disagreement_map, plot_policy_arrows,
                                  plot_policy_phase_grid, plot_sarsa_trace,
                                  plot_training_curves, plot_value_heatmap,
                                  plot_visit_map, rolling_mean, write_csv)

EVAL_EPISODES = 500
EVAL_SEED = 999


def table_policy(table: dict[State, np.ndarray]):
    """Greedy action function over a (possibly partial) Q-table."""
    return lambda s: 0 if s not in table else int(np.argmax(table[s]))


def dict_policy(policy: dict[State, int | None]):
    """Action function over an explicit policy dict (e.g. from VI)."""
    return lambda s: policy.get(s) if policy.get(s) is not None else 0


def evaluate_greedy(env: MazeEnv, action_fn, episodes: int,
                    seed: int) -> dict:
    """Deterministic rollouts of a policy on the sparse env."""
    from collections import Counter
    env.reset(seed=seed)
    events, returns, steps, wins = Counter(), [], [], 0
    for _ in range(episodes):
        state = env.reset()
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            state, reward, terminated, truncated, info = env.step(
                action_fn(state))
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

    write_csv(training_rows, RAW_DATA_DIR / "sarsa_training.csv")
    write_csv(summary_rows, RAW_DATA_DIR / "sarsa_summary.csv")
    plot_training_curves(
        {f"λ = {lam:g}": histories[lam] for lam in scfg["lambda_sweep"]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "SARSA(λ) lambda sweep, sparse reward",
        FIGURES_DIR / "sarsa_lambda_sweep.png")
    print("  wrote 4 CSVs and 2 figures")


def run_comparison(config: dict) -> None:
    """Three-algorithm comparison on the same map and sparse reward."""
    import time
    maze = MazeMap.load(MAPS_DIR / "source.json")
    gamma = config["value_iteration"]["gamma"]
    qcfg, scfg = config["q_learning"], config["sarsa_lambda"]
    best_lambda = 0.7  # best speed/stability balance from the lambda sweep
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
                             scfg["alpha"], scfg["gamma"], best_lambda,
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

    for row in summary_rows:
        name = {"value_iteration": f"vi_sparse_gamma{gamma:g}.json",
                "q_learning": "q_learning_sparse_exponential.json",
                "sarsa_lambda": f"sarsa_lambda{best_lambda:g}_sparse.json"}
        row["model_file_kb"] = round(
            (MODELS_DIR / name[row["algorithm"]]).stat().st_size / 1024, 1)
    write_csv(summary_rows, RAW_DATA_DIR / "comparison_summary.csv")

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
    write_csv(samples, RAW_DATA_DIR / "comparison_sample_states.csv")

    plot_disagreement_map(maze, ql.greedy_policy(),
                          set(ql.visited_states()), vi_res.policy,
                          "Q-Learning greedy policy vs Value Iteration",
                          FIGURES_DIR / "comparison_disagreement_qlearning.png")
    plot_disagreement_map(maze, sarsa.greedy_policy(),
                          set(sarsa.visited_states()), vi_res.policy,
                          f"SARSA(λ={best_lambda:g}) greedy policy vs "
                          f"Value Iteration",
                          FIGURES_DIR / "comparison_disagreement_sarsa.png")
    plot_training_curves(
        {"Q-Learning": [ql_hist], f"SARSA(λ={best_lambda:g})": [sarsa_hist]},
        [("success", "success rate"), ("return", "episode return"),
         ("steps", "steps per episode")],
        "Model-free methods, sparse reward, exponential decay",
        FIGURES_DIR / "comparison_learning_curves.png")
    print("  wrote 2 CSVs and 3 figures")


def _train_with_snapshots(agent: QLearningAgent, episodes: int, schedule,
                          state: State, checkpoints: list[int]
                          ) -> tuple[list[dict], list[dict]]:
    """Train in chunks, snapshotting Q(state) at each checkpoint episode.
    Chunking preserves the exact run (env/agent RNGs continue across chunks)."""
    history, snapshots, done = [], [], 0
    marks = sorted({cp for cp in checkpoints if cp <= episodes} | {episodes})
    for checkpoint in marks:
        if checkpoint > done:
            chunk, _ = agent.train(checkpoint - done,
                                   lambda ep, off=done: schedule(ep + off))
            for i, row in enumerate(chunk):
                row["episode"] = done + i
            history += chunk
            done = checkpoint
        q = agent.Q.get(state)
        snapshots.append({
            "episode": checkpoint,
            **{f"q_{name}": (round(float(q[a]), 4) if q is not None else 0.0)
               for a, name in enumerate(("up", "down", "left", "right"))},
            "greedy_action": (int(np.argmax(q)) if q is not None
                              and q.any() else None)})
    return history, snapshots


def run_transfer(config: dict) -> None:
    """Four transfer scenarios on both target maps (report section: transfer)."""
    tcfg = config["transfer"]
    source_maze = MazeMap.load(MAPS_DIR / "source.json")
    source_q, _ = QLearningAgent.load_table(
        MODELS_DIR / "q_learning_sparse_exponential.json")
    schedule = epsilon_schedule(tcfg["epsilon_decay_schedule"],
                                tcfg["epsilon_start"], tcfg["epsilon_end"],
                                tcfg["epsilon_decay_episodes"])
    episodes = tcfg["episodes"]
    seeds = tcfg["seeds"]

    training_rows, summary_rows = [], []
    negative_state: State | None = None
    for kind in ("similar", "different"):
        target = MazeMap.load(MAPS_DIR / f"target_{kind}.json")
        unchanged = unchanged_positions(source_maze, target)
        print(f"  [{kind}] {len(unchanged)} cells with unchanged "
              f"3x3 neighborhood")

        # exact optimum of the target as the final-performance reference
        vi_target = ValueIteration(
            MazeEnv(target, config, reward_mode="sparse"),
            config["value_iteration"]["gamma"],
            threshold=config["value_iteration"]["convergence_threshold"],
            max_iterations=config["value_iteration"]["max_iterations"])
        vi_res = vi_target.solve()
        vi_eval = evaluate_greedy(MazeEnv(target, config, reward_mode="sparse"),
                                  dict_policy(vi_res.policy), EVAL_EPISODES,
                                  EVAL_SEED)
        print(f"  [{kind}] target optimum: return {vi_eval['mean_return']}, "
              f"steps {vi_eval['mean_steps']}")

        if kind == "different":
            # negative-transfer candidate: keyless state near the OLD key
            # where the transferred greedy action is worst under target Q*
            v_star = np.array([vi_res.V[s] for s in vi_target.states])
            q_star = vi_target._q_from(v_star)
            best_gap = -1.0
            for s, q in source_q.items():
                near_old = (abs(s.r - source_maze.key[0])
                            + abs(s.c - source_maze.key[1])) <= 2
                if not (near_old and s.has_key == 0 and q.any()
                        and s in vi_target.index):
                    continue
                i = vi_target.index[s]
                gap = float(q_star[i, vi_res.policy[s]]
                            - q_star[i, int(np.argmax(q))])
                if gap > best_gap:
                    best_gap, negative_state = gap, s
            print(f"  [{kind}] negative-transfer case: "
                  f"s={tuple(negative_state)} gap={best_gap:.2f}")

        scenarios: dict[str, dict] = {
            "scratch": initial_q_table(source_q, "scratch"),
            "full": initial_q_table(source_q, "full"),
            **{f"scaled_{beta:g}": initial_q_table(source_q, "scaled",
                                                   beta=beta)
               for beta in tcfg["beta_values"]},
            "selective": initial_q_table(source_q, "selective",
                                         unchanged=unchanged),
        }

        histories: dict[str, list[list[dict]]] = {}
        for name, init_q in scenarios.items():
            jumpstart = evaluate_greedy(
                MazeEnv(target, config, reward_mode="sparse"),
                table_policy(init_q), tcfg["jumpstart_episodes"], EVAL_SEED)
            per_seed = []
            for seed in seeds:
                agent = QLearningAgent(
                    MazeEnv(target, config, reward_mode="sparse", seed=seed),
                    tcfg["alpha"], tcfg["gamma"], seed=seed)
                agent.Q = {s: q.copy() for s, q in init_q.items()}
                if (kind == "different" and name == "full"
                        and seed == seeds[0]):
                    history, snapshots = _train_with_snapshots(
                        agent, episodes, schedule, negative_state,
                        tcfg["negative_case_checkpoints"])
                    v_row = {"episode": "target_optimal",
                             **{f"q_{n}": round(float(
                                 q_star[vi_target.index[negative_state], a]), 4)
                                for a, n in enumerate(("up", "down", "left",
                                                       "right"))},
                             "greedy_action": vi_res.policy[negative_state]}
                    write_csv(snapshots + [v_row],
                              RAW_DATA_DIR / "transfer_negative_case.csv")
                else:
                    history, _ = agent.train(episodes, schedule)
                per_seed.append(history)
                training_rows += [{"target": kind, "scenario": name,
                                   "seed": seed, **row} for row in history]

                final_eval = evaluate_greedy(
                    MazeEnv(target, config, reward_mode="sparse"),
                    table_policy(agent.Q), EVAL_EPISODES, EVAL_SEED)
                success = [row["success"] for row in history]
                roll = rolling_mean(success, 100)
                sustained = next((i + 99 for i, v in enumerate(roll)
                                  if v >= 0.9), None)
                summary_rows.append({
                    "target": kind, "scenario": name, "seed": seed,
                    "initial_states": len(init_q),
                    "jumpstart_success": jumpstart["success_rate"],
                    "jumpstart_return": jumpstart["mean_return"],
                    "first_success_episode":
                        next((r["episode"] for r in history if r["success"]),
                             None),
                    "episodes_to_90pct_success": sustained,
                    "early_mean_return": round(sum(
                        r["return"] for r in history[:100]) / 100, 2),
                    **{f"final_{k}": v for k, v in final_eval.items()},
                    "vi_optimal_return": vi_eval["mean_return"],
                })
            histories[name] = per_seed
            print(f"  [{kind}] {name}: jumpstart return "
                  f"{jumpstart['mean_return']}, 90% at ep "
                  f"{summary_rows[-1]['episodes_to_90pct_success']}, "
                  f"final return {summary_rows[-1]['final_mean_return']}")

        plot_training_curves(
            {label: histories[label]
             for label in ("scratch", "full", "scaled_0.5", "selective")},
            [("success", "success rate"), ("return", "episode return")],
            f"Transfer scenarios on the {kind} target",
            FIGURES_DIR / f"transfer_curves_{kind}.png")
        plot_training_curves(
            {"scratch": histories["scratch"],
             **{f"β = {beta:g}": histories[f"scaled_{beta:g}"]
                for beta in tcfg["beta_values"]}},
            [("success", "success rate"), ("return", "episode return")],
            f"Scaled transfer (beta sweep) on the {kind} target",
            FIGURES_DIR / f"transfer_beta_{kind}.png")

    write_csv(training_rows, RAW_DATA_DIR / "transfer_training.csv")
    write_csv(summary_rows, RAW_DATA_DIR / "transfer_summary.csv")
    print("  wrote 3 CSVs and 4 figures")


EXPERIMENTS = {
    "vi": run_value_iteration,
    "q_learning": run_q_learning,
    "sarsa_lambda": run_sarsa_lambda,
    "comparison": run_comparison,
    "transfer": run_transfer,
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
