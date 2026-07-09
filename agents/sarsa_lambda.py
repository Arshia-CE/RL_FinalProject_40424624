"""Tabular SARSA(lambda) (model-free, on-policy) with replacing or
accumulating eligibility traces, per-step delta/E tracing, and Q-table
persistence. lambda = 0 reduces exactly to one-step SARSA."""

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

from agents.q_learning import epsilon_schedule
from environments.generator import DEFAULT_CONFIG_PATH, MAPS_DIR, MazeMap
from environments.maze import (ACTIONS, EV_DOOR_LOCKED, EV_GATE_BLOCKED,
                               EV_KEY_PICKUP, EV_PENALTY, EV_WALL_HIT,
                               MazeEnv, State)

MODELS_DIR = PROJECT_ROOT / "results" / "models"


class SarsaLambdaAgent:
    """On-policy TD control with eligibility traces:

        delta = r + gamma * Q(s',a') - Q(s,a)
        E(s,a) bumped (replacing: =1, accumulating: +=1), then
        Q += alpha * delta * E   for every traced pair,   E *= gamma*lambda
    """

    def __init__(self, env: MazeEnv, alpha: float, gamma: float, lam: float,
                 trace_type: str = "replacing", trace_prune: float = 1e-4,
                 seed: int = 0):
        if trace_type not in ("replacing", "accumulating"):
            raise ValueError(f"unknown trace_type {trace_type!r}")
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.trace_type = trace_type
        self.trace_prune = trace_prune
        self.Q: dict[State, np.ndarray] = {}
        self.E: dict[State, np.ndarray] = {}
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
        best = q.max()  # random tie-breaking, as in Q-Learning
        return self._rng.choice([a for a in ACTIONS if q[a] == best])

    def begin_episode(self) -> None:
        self.E = {}

    def update(self, state: State, action: int, reward: float,
               next_state: State, next_action: int,
               terminated: bool) -> dict:
        """One SARSA(lambda) backup over all traced pairs."""
        q_before = float(self.q_values(state)[action])
        next_q = (0.0 if terminated
                  else float(self.q_values(next_state)[next_action]))
        target = reward + self.gamma * next_q
        delta = target - q_before

        trace = self.E.setdefault(state, np.zeros(len(ACTIONS)))
        if self.trace_type == "replacing":
            trace[action] = 1.0
        else:
            trace[action] += 1.0

        decay = self.gamma * self.lam
        pruned = []
        for traced_state, evec in self.E.items():
            self.q_values(traced_state)
            self.Q[traced_state] += self.alpha * delta * evec
            evec *= decay
            if evec.max() < self.trace_prune:
                pruned.append(traced_state)
        for traced_state in pruned:
            del self.E[traced_state]

        return {"q_before": round(q_before, 6),
                "next_q": round(next_q, 6),
                "td_target": round(target, 6),
                "delta": round(delta, 6),
                "q_after": round(float(self.Q[state][action]), 6),
                "active_traces": len(self.E)}

    def train(self, episodes: int, schedule,
              trace_episodes: frozenset[int] = frozenset(),
              env_seed: int | None = None
              ) -> tuple[list[dict], list[dict], list[dict]]:
        """Returns (per-episode history, per-step trace, trace dump rows)."""
        if env_seed is not None:
            self.env.reset(seed=env_seed)
        history: list[dict] = []
        step_trace: list[dict] = []
        trace_dump: list[dict] = []
        for episode in range(episodes):
            eps = schedule(episode)
            self.begin_episode()
            state = self.env.reset()
            action = self._act(state, eps)
            events: Counter = Counter()
            ep_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                self.visits[state] = self.visits.get(state, 0) + 1
                nxt, reward, terminated, truncated, info = self.env.step(action)
                next_action = self._act(nxt, eps)
                stats = self.update(state, action, reward, nxt, next_action,
                                    terminated)
                ep_return += reward
                events.update(info["events"])
                if episode in trace_episodes:
                    step_trace.append({
                        "episode": episode, "step": info["step"],
                        "r": state.r, "c": state.c, "has_key": state.has_key,
                        "phase": state.phase, "action": action,
                        "reward": reward, "next_r": nxt.r, "next_c": nxt.c,
                        "next_has_key": nxt.has_key,
                        "next_phase": nxt.phase, "next_action": next_action,
                        **stats,
                        "alpha": self.alpha, "gamma": self.gamma,
                        "lambda": self.lam, "epsilon": round(eps, 4),
                    })
                    trace_dump += [{
                        "episode": episode, "step": info["step"],
                        "r": s.r, "c": s.c, "has_key": s.has_key,
                        "phase": s.phase, "action": a,
                        "E": round(float(evec[a]), 6)}
                        for s, evec in self.E.items()
                        for a in ACTIONS if evec[a] > 0]
                state, action = nxt, next_action
            history.append({
                "episode": episode, "epsilon": round(eps, 4),
                "steps": self.env.steps, "return": round(ep_return, 2),
                "success": int(terminated),
                "wall_hits": events[EV_WALL_HIT],
                "penalty_entries": events[EV_PENALTY],
                "gate_blocked": events[EV_GATE_BLOCKED],
                "locked_door_attempts": events[EV_DOOR_LOCKED],
                "key_picked": int(events[EV_KEY_PICKUP] > 0),
            })
        return history, step_trace, trace_dump

    def greedy_policy(self) -> dict[State, int | None]:
        return {s: (None if self.env.is_terminal(s) else int(np.argmax(q)))
                for s, q in self.Q.items()}

    def visited_states(self) -> list[State]:
        return [s for s, q in self.Q.items() if q.any()]

    def save(self, path: Path, metadata: dict) -> None:
        states = list(self.Q)
        data = {"algorithm": "sarsa_lambda", "alpha": self.alpha,
                "gamma": self.gamma, "lambda": self.lam,
                "trace_type": self.trace_type, **metadata,
                "states": [list(s) for s in states],
                "Q": [self.Q[s].tolist() for s in states],
                "visits": [self.visits.get(s, 0) for s in states]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")


def main() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    maze = MazeMap.load(MAPS_DIR / "source.json")
    scfg = config["sarsa_lambda"]
    lam = scfg["lambda_sweep"][-1]

    env = MazeEnv(maze, config, reward_mode="sparse", seed=7)
    agent = SarsaLambdaAgent(env, scfg["alpha"], scfg["gamma"], lam,
                             trace_type=scfg["trace_type"],
                             trace_prune=scfg["trace_prune"], seed=7)
    schedule = epsilon_schedule(scfg["epsilon_decay_schedule"],
                                scfg["epsilon_start"], scfg["epsilon_end"],
                                scfg["epsilon_decay_episodes"])
    history, step_trace, trace_dump = agent.train(
        scfg["episodes"], schedule,
        trace_episodes=frozenset({scfg["trace_episode"]}))

    last = history[-100:]
    print(f"SARSA(lambda={lam}, {scfg['trace_type']} traces): "
          f"{len(history)} episodes")
    print(f"last 100: success {sum(r['success'] for r in last)}%, "
          f"mean return {sum(r['return'] for r in last) / 100:.1f}, "
          f"mean steps {sum(r['steps'] for r in last) / 100:.1f}")
    print(f"visited {len(agent.visited_states())} of 2796 states")

    print(f"traced episode {scfg['trace_episode']} "
          f"({len(step_trace)} steps): delta and trace evolution")
    for row in step_trace[:6]:
        print(f"  step {row['step']}: s=({row['r']},{row['c']},"
              f"k={row['has_key']},p={row['phase']}) a={row['action']} "
              f"r={row['reward']:+.0f} delta={row['delta']:+.4f} "
              f"active traces={row['active_traces']}")
    first = step_trace[0]
    origin = (first["r"], first["c"], first["has_key"], first["phase"],
              first["action"])
    decay = [row["E"] for row in trace_dump
             if (row["r"], row["c"], row["has_key"], row["phase"],
                 row["action"]) == origin]
    print(f"E(s_0,a_0) over successive steps: "
          f"{[round(e, 4) for e in decay[:6]]} (gamma*lambda = "
          f"{scfg['gamma'] * lam:.3f} per step)")

    out = MODELS_DIR / f"sarsa_lambda{lam:g}_sparse.json"
    agent.save(out, {"reward_mode": "sparse",
                     "schedule": scfg["epsilon_decay_schedule"],
                     "episodes": scfg["episodes"], "seed": 7})
    print(f"saved Q-table to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
