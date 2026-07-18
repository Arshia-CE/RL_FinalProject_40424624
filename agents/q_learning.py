"""Tabular Q-Learning (model-free, off-policy): epsilon-greedy behavior with
linear or exponential decay, per-episode metrics, an optional per-update
trace, and Q-table persistence."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import DEFAULT_CONFIG_PATH, MAPS_DIR, MazeMap
from environments.maze import (ACTIONS, EV_DOOR_LOCKED, EV_DOOR_PASS,
                               EV_ENERGY_OUT, EV_KEY_PICKUP, EV_PENALTY,
                               EV_TIMEOUT, EV_WALL_HIT, MazeEnv, State)

MODELS_DIR = PROJECT_ROOT / "results" / "models"


def epsilon_schedule(kind: str, start: float, end: float,
                     decay_episodes: int):
    """epsilon(episode): linear or exponential interpolation start -> end."""
    if kind == "linear":
        return lambda ep: max(end, start - (start - end) * ep / decay_episodes)
    if kind == "exponential":
        rate = (end / start) ** (1.0 / decay_episodes)
        return lambda ep: max(end, start * rate ** ep)
    raise ValueError(f"unknown epsilon schedule {kind!r}")


class QLearningAgent:
    """Off-policy TD control:  Q(s,a) += alpha * (r + gamma*max Q(s',.) - Q(s,a))."""

    def __init__(self, env: MazeEnv, alpha: float, gamma: float, seed: int = 0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.Q: dict[State, np.ndarray] = {}
        self.visits: dict[State, int] = {}
        self._rng = random.Random(seed)

    def q_values(self, state: State) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(len(ACTIONS))
        return self.Q[state]

    def _act(self, state: State, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return self._rng.randrange(len(ACTIONS))
        q = self.q_values(state)
        best = q.max()  # random tie-breaking matters with zero-initialized Q
        return self._rng.choice([a for a in ACTIONS if q[a] == best])

    def update(self, state: State, action: int, reward: float,
               next_state: State, terminated: bool) -> dict:
        """One tabular backup; the step cap only truncates, so the bootstrap
        term is zeroed solely on true termination."""
        q = self.q_values(state)
        q_before = float(q[action])
        max_next = (0.0 if terminated
                    else float(self.q_values(next_state).max()))
        target = reward + self.gamma * max_next
        q[action] = q_before + self.alpha * (target - q_before)
        return {"q_before": round(q_before, 6),
                "max_next_q": round(max_next, 6),
                "td_target": round(target, 6),
                "td_error": round(target - q_before, 6),
                "q_after": round(float(q[action]), 6)}

    def train(self, episodes: int, schedule,
              trace_episodes: frozenset[int] = frozenset(),
              env_seed: int | None = None) -> tuple[list[dict], list[dict]]:
        """Returns (per-episode history rows, per-update trace rows)."""
        if env_seed is not None:
            self.env.reset(seed=env_seed)
        history: list[dict] = []
        trace: list[dict] = []
        for episode in range(episodes):
            eps = schedule(episode)
            state = self.env.reset()
            events: Counter = Counter()
            ep_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                self.visits[state] = self.visits.get(state, 0) + 1
                action = self._act(state, eps)
                nxt, reward, terminated, truncated, info = self.env.step(action)
                stats = self.update(state, action, reward, nxt, terminated)
                ep_return += reward
                events.update(info["events"])
                if episode in trace_episodes:
                    trace.append({
                        "episode": episode, "step": info["step"],
                        "r": state.r, "c": state.c, "has_key": state.has_key,
                        "energy": state.energy, "action": action,
                        "reward": reward, "next_r": nxt.r, "next_c": nxt.c,
                        "next_has_key": nxt.has_key,
                        "next_energy": nxt.energy,
                        "events": "|".join(info["events"]),
                        **stats,
                        "alpha": self.alpha, "gamma": self.gamma,
                        "epsilon": round(eps, 4),
                    })
                state = nxt
            death = int(events[EV_ENERGY_OUT] > 0)
            history.append({
                "episode": episode, "epsilon": round(eps, 4),
                "steps": self.env.steps, "return": round(ep_return, 2),
                "success": int(terminated and not death),
                "wall_hits": events[EV_WALL_HIT],
                "penalty_entries": events[EV_PENALTY],
                "energy_left": state.energy, "death": death,
                "locked_door_attempts": events[EV_DOOR_LOCKED],
                "door_passes": events[EV_DOOR_PASS],
                "key_picked": int(events[EV_KEY_PICKUP] > 0),
                "timeout": int(events[EV_TIMEOUT] > 0),
            })
        return history, trace

    def greedy_policy(self) -> dict[State, int | None]:
        return {s: (None if self.env.is_terminal(s) else int(np.argmax(q)))
                for s, q in self.Q.items()}

    def visited_states(self) -> list[State]:
        """States with at least one updated action value."""
        return [s for s, q in self.Q.items() if q.any()]

    def save(self, path: Path, metadata: dict) -> None:
        states = list(self.Q)
        data = {"algorithm": "q_learning", "alpha": self.alpha,
                "gamma": self.gamma, **metadata,
                "states": [list(s) for s in states],
                "Q": [self.Q[s].tolist() for s in states],
                "visits": [self.visits.get(s, 0) for s in states]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def load_table(path: Path) -> tuple[dict[State, np.ndarray], dict]:
        data = json.loads(path.read_text(encoding="utf-8"))
        table = {State(*s): np.array(q)
                 for s, q in zip(data["states"], data["Q"])}
        meta = {k: v for k, v in data.items() if k not in ("states", "Q")}
        return table, meta


def main() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    maze = MazeMap.load(MAPS_DIR / "source.json")
    qcfg = config["q_learning"]

    env = MazeEnv(maze, config, reward_mode="sparse", seed=7)
    agent = QLearningAgent(env, qcfg["alpha"], qcfg["gamma"], seed=7)
    schedule = epsilon_schedule("exponential", qcfg["epsilon_start"],
                                qcfg["epsilon_end"],
                                qcfg["epsilon_decay_episodes"])
    history, trace = agent.train(qcfg["episodes"], schedule,
                                 trace_episodes=frozenset({qcfg["episodes"] // 2}))

    last = history[-100:]
    print(f"trained {len(history)} episodes "
          f"(alpha={qcfg['alpha']}, gamma={qcfg['gamma']}, exponential decay)")
    print(f"last 100 episodes: success {sum(r['success'] for r in last)}%, "
          f"mean return {sum(r['return'] for r in last) / 100:.1f}, "
          f"mean steps {sum(r['steps'] for r in last) / 100:.1f}")
    print(f"visited {len(agent.visited_states())} of "
          f"{len(env.enumerate_states())} states")

    row = trace[0]
    print("sample real Q-update (reconstructable by hand):")
    print(f"  s=({row['r']},{row['c']},k={row['has_key']},e={row['energy']}) "
          f"a={row['action']} r={row['reward']} -> "
          f"s'=({row['next_r']},{row['next_c']},k={row['next_has_key']},"
          f"e={row['next_energy']})")
    print(f"  Q before {row['q_before']}, max_a' Q(s') {row['max_next_q']}, "
          f"target {row['td_target']}, TD error {row['td_error']}, "
          f"Q after {row['q_after']}")

    from agents.value_iteration import VIResult
    ref = VIResult.load(MODELS_DIR / "vi" / "vi_sparse_gamma0.95.json")
    policy = agent.greedy_policy()
    for min_visits in (1, 10, 100):
        states = [s for s in agent.visited_states()
                  if policy[s] is not None
                  and agent.visits.get(s, 0) >= min_visits]
        agreement = sum(policy[s] == ref.policy[s] for s in states) / len(states)
        print(f"agreement with VI on states visited >= {min_visits:3d}: "
              f"{agreement:5.1%}  ({len(states)} states)")

    out = MODELS_DIR / "q_learning" / "q_learning_sparse_exponential.json"
    agent.save(out, {"reward_mode": "sparse", "schedule": "exponential",
                     "episodes": qcfg["episodes"],
                     "epsilon_start": qcfg["epsilon_start"],
                     "epsilon_end": qcfg["epsilon_end"],
                     "epsilon_decay_episodes": qcfg["epsilon_decay_episodes"]})
    table, _ = QLearningAgent.load_table(out)
    assert all(np.array_equal(table[s], agent.Q[s]) for s in agent.Q)
    print(f"saved Q-table to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
