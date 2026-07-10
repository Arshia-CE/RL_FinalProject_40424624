"""Game sessions for the GUI: WATCH mode (trained policies acting greedily)
and TRAIN mode (Q-Learning / SARSA(λ) learning live) on the real MazeEnv."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.q_learning import QLearningAgent, epsilon_schedule
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import MODELS_DIR, ValueIteration
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import (EV_DOOR_LOCKED, EV_DOOR_PASS, EV_GATE_BLOCKED,
                               EV_GOAL, EV_KEY_PICKUP, EV_PENALTY,
                               EV_WALL_HIT, MazeEnv, State)
from experiments.common import load_config

WORLDS = {
    "source": {"file": "source.json", "label": "WORLD 1 · SOURCE",
               "desc": "THE SEEDED MAZE"},
    "similar": {"file": "target_similar.json", "label": "WORLD 2 · SIMILAR",
                "desc": "18% WALLS MOVED"},
    "different": {"file": "target_different.json",
                  "label": "WORLD 3 · DIFFERENT",
                  "desc": "KEY MOVED · +3 PITS"},
}

BRAINS = {
    "vi": {"label": "VALUE ITERATION", "desc": "EXACT MDP PLAN"},
    "q_learning": {"label": "Q-LEARNING", "desc": "TRAINED ON WORLD 1"},
    "sarsa": {"label": "SARSA λ=0.7", "desc": "TRAINED ON WORLD 1"},
}

MODES = {
    "watch": {"label": "WATCH", "desc": "AGENT PLAYS"},
    "train": {"label": "TRAIN", "desc": "LEARNS FROM ZERO"},
}

MODEL_FILES = {
    "q_learning": MODELS_DIR / "q_learning"
                  / "q_learning_sparse_exponential.json",
    "sarsa": MODELS_DIR / "sarsa" / "sarsa_lambda0.7_sparse.json",
}


class GameSession:
    """Owns the env + acting/learning agent; stepped by the app's game loop."""

    def __init__(self):
        self.config = load_config()
        self._vi_policies: dict[str, dict] = {}
        self._tables: dict[str, dict] = {}
        self._mazes: dict[str, MazeMap] = {}
        self.world = "source"
        self.brain = "vi"
        self.mode = "watch"
        self.episode = 0          # display counter (watch: run #)
        self.trained_episodes = 0  # train mode: completed episodes
        self.recent = deque(maxlen=100)
        self.load_world(self.world)

    # selection

    def load_world(self, key: str) -> None:
        self.world = key
        if key not in self._mazes:
            self._mazes[key] = MazeMap.load(MAPS_DIR / WORLDS[key]["file"])
        self.maze = self._mazes[key]
        self.restart()

    def set_brain(self, key: str) -> None:
        if self.mode == "train" and key == "vi":
            key = "q_learning"
        self.brain = key
        self.restart()

    def set_mode(self, key: str) -> None:
        self.mode = key
        if key == "train" and self.brain == "vi":
            self.brain = "q_learning"
        self.restart()

    def restart(self) -> None:
        """Fresh run: new episode (watch) or brand-new learner (train)."""
        self.episode = 0
        self.trained_episodes = 0
        self.recent = deque(maxlen=100)
        self.outcome: str | None = None
        self.score = 0.0
        self._pending_action: int | None = None
        if self.mode == "watch":
            self._watch_policy = self._brain_policy()
            self.agent = None
            self._new_watch_episode()
        else:
            acfg = self.config["q_learning" if self.brain == "q_learning"
                               else "sarsa_lambda"]
            env = MazeEnv(self.maze, self.config, reward_mode="sparse",
                          seed=7)
            if self.brain == "q_learning":
                self.agent = QLearningAgent(env, acfg["alpha"],
                                            acfg["gamma"], seed=7)
            else:
                self.agent = SarsaLambdaAgent(
                    env, acfg["alpha"], acfg["gamma"], 0.7,
                    trace_type=acfg["trace_type"],
                    trace_prune=acfg["trace_prune"], seed=7)
                self.agent.begin_episode()
            self._schedule = epsilon_schedule(
                "exponential", acfg["epsilon_start"], acfg["epsilon_end"],
                acfg["epsilon_decay_episodes"])
            self.episode_budget = acfg["episodes"]
            self.env = env
            self.state = env.reset()

    episode_budget = 0

    @property
    def training_complete(self) -> bool:
        """Training freezes after the config's episode budget; the hero then
        plays its learned table greedily."""
        return (self.mode == "train"
                and self.trained_episodes >= self.episode_budget)

    def _brain_policy(self):
        if self.brain == "vi":
            if self.world not in self._vi_policies:
                vi = ValueIteration(
                    MazeEnv(self.maze, self.config, reward_mode="sparse"),
                    self.config["value_iteration"]["gamma"],
                    threshold=self.config["value_iteration"]
                    ["convergence_threshold"],
                    max_iterations=self.config["value_iteration"]
                    ["max_iterations"])
                self._vi_policies[self.world] = vi.solve().policy
            policy = self._vi_policies[self.world]
            return lambda s: policy.get(s) if policy.get(s) is not None else 0
        if self.brain not in self._tables:
            table, _ = QLearningAgent.load_table(MODEL_FILES[self.brain])
            self._tables[self.brain] = table
        table = self._tables[self.brain]
        return lambda s: (int(np.argmax(table[s]))
                          if s in table and table[s].any() else 0)

    def _new_watch_episode(self) -> None:
        self.episode += 1
        self.env = MazeEnv(self.maze, self.config, reward_mode="sparse",
                           seed=1234 + self.episode * 7919)
        self.state = self.env.reset()
        self.score = 0.0
        self.outcome = None

    def begin_next_episode(self) -> None:
        """After a training episode ends: continue with the same learner."""
        if self.mode == "watch":
            self._new_watch_episode()
            return
        self.outcome = None
        self.score = 0.0
        self.state = self.env.reset()
        self._pending_action = None
        if self.brain == "sarsa":
            self.agent.begin_episode()

    # stepping

    def step(self) -> dict | None:
        """One animated environment step (both modes)."""
        if self.outcome:
            return None
        prev = self.state
        eps = self.current_epsilon()
        learning = self.mode == "train" and not self.training_complete
        if self.mode == "watch":
            action = self._watch_policy(prev)
        elif not learning:  # trained: play the learned table greedily
            action = self.agent._act(prev, 0.0)
        elif self.brain == "q_learning":
            action = self.agent._act(prev, eps)
        else:  # sarsa: on-policy — execute the action chosen last update
            if self._pending_action is None:
                self._pending_action = self.agent._act(prev, eps)
            action = self._pending_action

        nxt, reward, terminated, truncated, info = self.env.step(action)
        if learning:
            self.agent.visits[prev] = self.agent.visits.get(prev, 0) + 1
            if self.brain == "q_learning":
                self.agent.update(prev, action, reward, nxt, terminated)
            else:
                next_action = self.agent._act(nxt, eps)
                self.agent.update(prev, action, reward, nxt, next_action,
                                  terminated)
                self._pending_action = next_action
        self.state = nxt
        self.score += reward
        if terminated:
            self.outcome = "clear"
        elif truncated:
            self.outcome = "timeout"
        if self.outcome:
            self.recent.append(int(terminated))
            if learning:
                self.trained_episodes += 1
        events = info["events"]
        return {
            "prev": prev, "next": nxt, "reward": reward,
            "direction": info["executed_direction"],
            "moved": (nxt.r, nxt.c) != (prev.r, prev.c),
            "wall": EV_WALL_HIT in events,
            "gate_blocked": EV_GATE_BLOCKED in events,
            "door_locked": EV_DOOR_LOCKED in events,
            "key": EV_KEY_PICKUP in events,
            "door": EV_DOOR_PASS in events,
            "pit": EV_PENALTY in events,
            "goal": EV_GOAL in events,
            "outcome": self.outcome,
            "episode_end": self.outcome if self.mode == "train" else None,
        }

    def run_batch(self, episodes: int) -> None:
        """Headless training episodes (fast-forward at high speed)."""
        if self.mode != "train" or self.training_complete:
            return
        episodes = min(episodes, self.episode_budget - self.trained_episodes)
        offset = self.trained_episodes
        history = self.agent.train(
            episodes, lambda ep, off=offset: self._schedule(ep + off))[0]
        self.trained_episodes += episodes
        for row in history:
            self.recent.append(row["success"])
        self.score = history[-1]["return"]
        self.begin_next_episode()

    # HUD / overlay helpers

    def current_epsilon(self) -> float | None:
        if self.mode != "train":
            return None
        if self.training_complete:
            return 0.0
        return self._schedule(self.trained_episodes)

    def win_rate(self) -> float | None:
        """Success rate over the recent episodes (both modes)."""
        if not self.recent:
            return None
        return sum(self.recent) / len(self.recent)

    def episode_number(self) -> int:
        if self.mode != "train":
            return self.episode
        return min(self.trained_episodes + 1, self.episode_budget)

    def policy_action(self, r: int, c: int) -> int | None:
        """Greedy action at (r, c) for the agent's current key/phase."""
        s = State(r, c, self.state.has_key, self.state.phase)
        if self.mode == "watch":
            if self.brain == "vi":
                policy = self._vi_policies[self.world]
                return policy.get(s)
            table = self._tables[self.brain]
            q = table.get(s)
        else:
            q = self.agent.Q.get(s)
        if q is None or not q.any():
            return None
        return int(np.argmax(q))

    def gate_open(self) -> bool:
        return self.env.gate_open(self.state.phase)

    def gate_countdown(self) -> int:
        phases = self.maze.gate_open_phases
        if self.gate_open():
            return len(phases) - self.state.phase
        return self.maze.gate_period - self.state.phase

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def step_cap(self) -> int:
        return self.env.max_steps

    @property
    def has_key(self) -> bool:
        return bool(self.state.has_key)

    def rewards(self) -> dict:
        return self.config["rewards"]["sparse"]
