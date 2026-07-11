"""Shared experiment utilities: config loading, greedy policy wrappers and
deterministic policy evaluation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import DEFAULT_CONFIG_PATH
from environments.maze import (EV_GATE_BLOCKED, EV_PENALTY, EV_WALL_HIT,
                               MazeEnv, State)

EVAL_EPISODES = 500
EVAL_SEED = 999


def load_config() -> dict:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def table_policy(table: dict[State, np.ndarray]):
    """Greedy action function over a (possibly partial) Q-table."""
    return lambda s: 0 if s not in table else int(np.argmax(table[s]))


def dict_policy(policy: dict[State, int | None]):
    """Action function over an explicit policy dict (e.g. from VI)."""
    return lambda s: policy.get(s) if policy.get(s) is not None else 0


def greedy_rollout(env: MazeEnv, action_fn, seed: int) -> dict:
    """One deterministic greedy episode: the full state path, plus the path
    index of each event's first occurrence (for figure markers)."""
    env.reset(seed=seed)
    state = env.reset()
    states, event_steps, total = [state], {}, 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        state, reward, terminated, truncated, info = env.step(
            action_fn(state))
        total += reward
        states.append(state)
        for event in info["events"]:
            event_steps.setdefault(event, len(states) - 1)
    return {"states": states, "return": total, "steps": env.steps,
            "terminated": terminated, "event_steps": event_steps}


def evaluate_greedy(env: MazeEnv, action_fn, episodes: int,
                    seed: int) -> dict:
    """Deterministic rollouts of a policy on the sparse env."""
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
