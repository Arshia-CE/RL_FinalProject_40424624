"""Value Iteration over the exact MDP model, implemented from scratch.

Bellman optimality backups until max_s |V_{k+1}(s) - V_k(s)| < threshold,
then greedy policy extraction; per-sweep deltas, iteration count and runtime
are stored with the result. NumPy only vectorizes our own backup over padded
transition arrays.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.generator import DEFAULT_CONFIG_PATH, MAPS_DIR, MazeMap
from environments.maze import ACTION_NAMES, ACTIONS, MazeEnv, State

MODELS_DIR = PROJECT_ROOT / "results" / "models"


@dataclass
class VIResult:
    """Converged value function, greedy policy and solve statistics."""

    gamma: float
    threshold: float
    reward_mode: str
    iterations: int
    runtime_seconds: float
    converged: bool
    deltas: list[float]
    V: dict[State, float]
    policy: dict[State, int | None]  # None for terminal states

    def save(self, path: Path) -> None:
        states = list(self.V.keys())
        data = {
            "algorithm": "value_iteration",
            "gamma": self.gamma,
            "threshold": self.threshold,
            "reward_mode": self.reward_mode,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "converged": self.converged,
            "deltas": self.deltas,
            "states": [list(s) for s in states],
            "V": [self.V[s] for s in states],
            "policy": [-1 if self.policy[s] is None else self.policy[s]
                       for s in states],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VIResult":
        data = json.loads(path.read_text(encoding="utf-8"))
        states = [State(*s) for s in data["states"]]
        return cls(
            gamma=data["gamma"], threshold=data["threshold"],
            reward_mode=data["reward_mode"], iterations=data["iterations"],
            runtime_seconds=data["runtime_seconds"],
            converged=data["converged"], deltas=data["deltas"],
            V=dict(zip(states, data["V"])),
            policy={s: (None if a == -1 else a)
                    for s, a in zip(states, data["policy"])},
        )


class ValueIteration:
    """Exact planner over the full MDP model of a :class:`MazeEnv`."""

    MAX_OUTCOMES = 3  # intended + two perpendicular deviations (pre-merge)

    def __init__(self, env: MazeEnv, gamma: float, threshold: float = 1e-6,
                 max_iterations: int = 10_000):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.states = env.enumerate_states()
        self.index = {s: i for i, s in enumerate(self.states)}
        self._build_model()

    def _build_model(self) -> None:
        """Cache the transition model as padded arrays for fast sweeps."""
        n_s, n_a, n_o = len(self.states), len(ACTIONS), self.MAX_OUTCOMES
        self._prob = np.zeros((n_s, n_a, n_o))
        self._reward = np.zeros((n_s, n_a, n_o))
        self._next = np.zeros((n_s, n_a, n_o), dtype=np.int64)
        self._cont = np.zeros((n_s, n_a, n_o))  # 0 where s' is terminal
        for i, state in enumerate(self.states):
            for action in ACTIONS:
                outcomes = self.env.transitions(state, action)
                assert len(outcomes) <= n_o
                for j, (p, nxt, reward, done) in enumerate(outcomes):
                    self._prob[i, action, j] = p
                    self._reward[i, action, j] = reward
                    self._next[i, action, j] = self.index[nxt]
                    self._cont[i, action, j] = 0.0 if done else 1.0

    def _q_from(self, V: np.ndarray) -> np.ndarray:
        """One Bellman backup: Q(s,a) = sum_o P * (R + gamma * V(s'))."""
        return (self._prob
                * (self._reward + self.gamma * self._cont * V[self._next])
                ).sum(axis=2)

    def solve(self) -> VIResult:
        V = np.zeros(len(self.states))
        deltas: list[float] = []
        converged = False
        start = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            V_new = self._q_from(V).max(axis=1)
            delta = float(np.max(np.abs(V_new - V)))
            deltas.append(delta)
            V = V_new
            if delta < self.threshold:
                converged = True
                break
        runtime = time.perf_counter() - start

        greedy = np.argmax(self._q_from(V), axis=1)
        return VIResult(
            gamma=self.gamma, threshold=self.threshold,
            reward_mode=self.env.reward_mode, iterations=iteration,
            runtime_seconds=runtime, converged=converged, deltas=deltas,
            V={s: float(V[i]) for i, s in enumerate(self.states)},
            policy={s: (None if self.env.is_terminal(s) else int(greedy[i]))
                    for i, s in enumerate(self.states)},
        )


def greedy_rollouts(env: MazeEnv, policy: dict[State, int | None],
                    episodes: int, seed: int) -> dict:
    """Run the greedy policy on the stochastic env; summary statistics."""
    env.reset(seed=seed)
    wins, returns, win_steps = 0, [], []
    for _ in range(episodes):
        state = env.reset()
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            state, reward, terminated, truncated, _ = env.step(policy[state])
            total += reward
        wins += terminated
        returns.append(total)
        if terminated:
            win_steps.append(env.steps)
    return {"episodes": episodes, "success_rate": wins / episodes,
            "mean_return": sum(returns) / episodes,
            "mean_steps_when_successful":
                sum(win_steps) / len(win_steps) if win_steps else None}


def main() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    maze = MazeMap.load(MAPS_DIR / "source.json")
    env = MazeEnv(maze, config, reward_mode="sparse")
    vi_cfg = config["value_iteration"]

    vi = ValueIteration(env, gamma=vi_cfg["gamma"],
                        threshold=vi_cfg["convergence_threshold"],
                        max_iterations=vi_cfg["max_iterations"])
    result = vi.solve()
    print(f"gamma={result.gamma}: {'converged' if result.converged else 'NOT converged'}"
          f" after {result.iterations} sweeps in {result.runtime_seconds:.2f}s"
          f" (final delta {result.deltas[-1]:.2e}, threshold {result.threshold:g})")

    start_state = State(maze.start[0], maze.start[1], 0, 0)
    print(f"V(start) = {result.V[start_state]:.2f}")

    out = MODELS_DIR / f"vi_sparse_gamma{result.gamma:g}.json"
    result.save(out)
    assert VIResult.load(out).V == result.V
    print(f"saved reference model to {out.relative_to(PROJECT_ROOT)}")

    # the optimal action in front of the gate should depend on the phase
    gr, gc = maze.gate
    wait_cell = next((gr + dr, gc + dc) for dr, dc in
                     ((-1, 0), (1, 0), (0, -1), (0, 1))
                     if not maze.is_wall((gr + dr, gc + dc))
                     and (gr + dr, gc + dc) != tuple(maze.door))
    print(f"policy at {wait_cell} (holding the key), by gate phase:")
    for phase in range(maze.gate_period):
        s = State(wait_cell[0], wait_cell[1], 1, phase)
        gate = "open " if env.gate_open(phase) else "closed"
        print(f"  phase {phase} (gate {gate}): {ACTION_NAMES[result.policy[s]]}"
              f"   V={result.V[s]:.2f}")

    stats = greedy_rollouts(MazeEnv(maze, config, reward_mode="sparse"),
                            result.policy, episodes=500, seed=999)
    print(f"greedy policy on stochastic env: success {stats['success_rate']:.1%}, "
          f"mean return {stats['mean_return']:.1f}, "
          f"mean steps (successes) {stats['mean_steps_when_successful']:.1f}")


if __name__ == "__main__":
    main()
