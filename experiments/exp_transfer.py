"""Transfer-learning experiments: four scenarios on both target maps."""

from __future__ import annotations

import numpy as np

from agents.q_learning import QLearningAgent, epsilon_schedule
from agents.value_iteration import MODELS_DIR, ValueIteration
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import MazeEnv, State
from experiments.analysis import (FIGURES_DIR, RAW_DATA_DIR,
                                  plot_training_curves, rolling_mean,
                                  write_csv)
from experiments.common import (EVAL_EPISODES, EVAL_SEED, dict_policy,
                                evaluate_greedy, table_policy)
from experiments.maze_plots import plot_transfer_q_diff
from transfer.transfer_learning import initial_q_table, unchanged_positions

RAW_DIR = RAW_DATA_DIR / "transfer"
FIG_DIR = FIGURES_DIR / "transfer"


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
    tcfg = config["transfer"]
    source_maze = MazeMap.load(MAPS_DIR / "source.json")
    # the shaped table is the canonical source: the sparse one never learns.
    # Shaped Q converges to Q_sparse - phi(s), so cross-map transfer must
    # de-shape with the SOURCE potential and re-base into the TARGET's, or
    # the action-independent offset turns into phantom TD errors that
    # poison bootstrapping (full transfer collapsed to 2% train success on
    # the different target without this).
    source_q, _ = QLearningAgent.load_table(
        MODELS_DIR / "q_learning" / "q_learning_shaped_exponential.json")
    src_phi = MazeEnv(source_maze, config, reward_mode="shaped")._phi
    sparse_equiv = {s: q + src_phi(s) for s, q in source_q.items()}
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

        tgt_phi = MazeEnv(target, config, reward_mode="shaped")._phi
        scenarios: dict[str, dict] = {
            name: {s: q - tgt_phi(s) for s, q in table.items()}
            for name, table in {
                "scratch": initial_q_table(sparse_equiv, "scratch"),
                "full": initial_q_table(sparse_equiv, "full"),
                **{f"scaled_{beta:g}": initial_q_table(sparse_equiv,
                                                       "scaled", beta=beta)
                   for beta in tcfg["beta_values"]},
                "selective": initial_q_table(sparse_equiv, "selective",
                                             unchanged=unchanged),
            }.items()}

        histories: dict[str, list[list[dict]]] = {}
        for name, init_q in scenarios.items():
            jumpstart = evaluate_greedy(
                MazeEnv(target, config, reward_mode="sparse"),
                table_policy(init_q), tcfg["jumpstart_episodes"], EVAL_SEED)
            per_seed = []
            for seed in seeds:
                # shaped training (sparse never ignites); evals stay sparse
                agent = QLearningAgent(
                    MazeEnv(target, config, reward_mode="shaped", seed=seed),
                    tcfg["alpha"], tcfg["gamma"], seed=seed)
                agent.Q = {s: q.copy() for s, q in init_q.items()}
                if (kind == "different" and name == "full"
                        and seed == seeds[0]):
                    history, snapshots = _train_with_snapshots(
                        agent, episodes, schedule, negative_state,
                        tcfg["negative_case_checkpoints"])
                    # report Q* in the same target-shaped coordinates as
                    # the snapshots (subtract the state's potential)
                    phi_ns = tgt_phi(negative_state)
                    v_row = {"episode": "target_optimal",
                             **{f"q_{n}": round(float(
                                 q_star[vi_target.index[negative_state], a]
                                 - phi_ns), 4)
                                for a, n in enumerate(("up", "down", "left",
                                                       "right"))},
                             "greedy_action": vi_res.policy[negative_state]}
                    write_csv(snapshots + [v_row],
                              RAW_DIR / "transfer_negative_case.csv")
                else:
                    history, _ = agent.train(episodes, schedule)
                if name == "full" and seed == seeds[0]:
                    # before/after artifacts: the "before" table is the
                    # committed source Q-table (full transfer copies it)
                    agent.save(MODELS_DIR / "transfer"
                               / f"transfer_full_{kind}.json",
                               {"target": kind, "scenario": name,
                                "seed": seed, "episodes": episodes})
                    plot_transfer_q_diff(
                        target, init_q, agent.Q,
                        f"Full transfer on the {kind} target — Q before vs "
                        f"after {episodes} episodes (seed {seed})",
                        FIG_DIR / f"transfer_q_diff_{kind}.png")
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
            FIG_DIR / f"transfer_curves_{kind}.png")
        plot_training_curves(
            {"scratch": histories["scratch"],
             **{f"β = {beta:g}": histories[f"scaled_{beta:g}"]
                for beta in tcfg["beta_values"]}},
            [("success", "success rate"), ("return", "episode return")],
            f"Scaled transfer (beta sweep) on the {kind} target",
            FIG_DIR / f"transfer_beta_{kind}.png")

    write_csv(training_rows, RAW_DIR / "transfer_training.csv")
    write_csv(summary_rows, RAW_DIR / "transfer_summary.csv")
    print("  wrote 3 CSVs, 6 figures and 2 transferred Q-tables")
