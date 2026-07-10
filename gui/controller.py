"""Game session: the trained agent playing one of the project maps with the
real MazeEnv dynamics; emits events the renderer animates."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.value_iteration import ValueIteration
from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import (EV_DOOR_LOCKED, EV_DOOR_PASS, EV_GATE_BLOCKED,
                               EV_GOAL, EV_KEY_PICKUP, EV_PENALTY,
                               EV_WALL_HIT, MazeEnv)
from experiments.common import load_config

WORLDS = {
    "source": {"file": "source.json", "label": "WORLD 1 · SOURCE",
               "desc": "THE ORIGINAL SEEDED MAZE"},
    "similar": {"file": "target_similar.json", "label": "WORLD 2 · SIMILAR",
                "desc": "TRANSFER TARGET — 18% CHANGED"},
    "different": {"file": "target_different.json",
                  "label": "WORLD 3 · DIFFERENT",
                  "desc": "TRANSFER TARGET — KEY MOVED, +3 PITS"},
}


class GameSession:
    """Environment + optimal policy for one world; stepped by the app loop."""

    def __init__(self):
        self.config = load_config()
        self._policies: dict[str, dict] = {}
        self._mazes: dict[str, MazeMap] = {}
        self.world = "source"
        self.episode = 0
        self.load_world(self.world)

    # world management

    def load_world(self, key: str) -> None:
        self.world = key
        if key not in self._mazes:
            self._mazes[key] = MazeMap.load(MAPS_DIR / WORLDS[key]["file"])
        self.maze = self._mazes[key]
        if key not in self._policies:
            vi = ValueIteration(
                MazeEnv(self.maze, self.config, reward_mode="sparse"),
                self.config["value_iteration"]["gamma"],
                threshold=self.config["value_iteration"]
                ["convergence_threshold"],
                max_iterations=self.config["value_iteration"]
                ["max_iterations"])
            self._policies[key] = vi.solve().policy
        self.policy = self._policies[key]
        self.reset()

    def reset(self) -> None:
        self.episode += 1
        self.env = MazeEnv(self.maze, self.config, reward_mode="sparse",
                           seed=1234 + self.episode * 7919)
        self.state = self.env.reset()
        self.score = 0.0
        self.outcome: str | None = None

    # stepping

    def step(self) -> dict | None:
        """One environment step; returns an event record for the renderer."""
        if self.outcome:
            return None
        prev = self.state
        action = self.policy.get(prev)
        if action is None:
            action = 0
        nxt, reward, terminated, truncated, info = self.env.step(action)
        self.state = nxt
        self.score += reward
        events = info["events"]
        if terminated:
            self.outcome = "clear"
        elif truncated:
            self.outcome = "timeout"
        moved = (nxt.r, nxt.c) != (prev.r, prev.c)
        return {
            "prev": prev, "next": nxt, "reward": reward,
            "direction": info["executed_direction"], "moved": moved,
            "wall": EV_WALL_HIT in events,
            "gate_blocked": EV_GATE_BLOCKED in events,
            "door_locked": EV_DOOR_LOCKED in events,
            "key": EV_KEY_PICKUP in events,
            "door": EV_DOOR_PASS in events,
            "pit": EV_PENALTY in events,
            "goal": EV_GOAL in events,
            "outcome": self.outcome,
        }

    # HUD helpers

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
