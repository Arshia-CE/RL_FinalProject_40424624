"""Dynamic maze environment modeled as an MDP.

State s = (r, c, has_key, gate_phase); actions U/D/L/R with 0.8/0.1/0.1
stochasticity. Walls, the locked door (keyless) and the gate block by
keeping the agent in place; the gate phase advances every step, and gate
entry requires the gate to be open on ARRIVAL (the phase after the step),
not at departure. Rewards are
sparse or potential-based shaped; the goal terminates, the step cap
truncates. transitions() exposes the exact model for Value Iteration, built
from the same resolution code that step() samples from.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import (DEFAULT_CONFIG_PATH, DOOR, GOAL, KEY,
                                   MAPS_DIR, PENALTY, MazeMap, Position)

# actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT),
                 LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

# events (the spec's minimum loggable set, plus gate_blocked for our feature)
EV_MOVE = "move"
EV_WALL_HIT = "wall_hit"
EV_PENALTY = "penalty_cell"
EV_KEY_PICKUP = "key_pickup"
EV_DOOR_LOCKED = "locked_door_attempt"
EV_DOOR_PASS = "door_pass"
EV_GATE_BLOCKED = "gate_blocked"
EV_GOAL = "goal_reached"
EV_TIMEOUT = "timeout"

# event -> key in the reward config (EV_MOVE/EV_TIMEOUT carry no extra reward)
EVENT_REWARD_KEY = {
    EV_WALL_HIT: "wall_hit",
    EV_PENALTY: "penalty_cell",
    EV_KEY_PICKUP: "key_pickup",
    EV_DOOR_LOCKED: "locked_door_attempt",
    EV_DOOR_PASS: "door_pass",
    EV_GATE_BLOCKED: "gate_blocked",
    EV_GOAL: "goal",
}


class State(NamedTuple):
    r: int
    c: int
    has_key: int
    phase: int


def _bfs_distances(maze: MazeMap, source: Position,
                   door_open: bool) -> dict[Position, int]:
    """Distance from every reachable cell to ``source`` (4-connected BFS)."""
    dist = {source: 0}
    queue: deque[Position] = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (r + dr, c + dc)
            if nxt in dist or maze.is_wall(nxt):
                continue
            if maze.cell(nxt) == DOOR and not door_open:
                continue
            dist[nxt] = dist[(r, c)] + 1
            queue.append(nxt)
    return dist


class MazeEnv:
    """Stochastic dynamic-maze MDP over a validated :class:`MazeMap`."""

    def __init__(self, maze: MazeMap, config: dict,
                 reward_mode: str = "sparse", seed: int | None = None):
        if reward_mode not in ("sparse", "shaped"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.maze = maze
        self.reward_mode = reward_mode
        self.rewards = config["rewards"]["sparse"]
        self.shaping_gamma = config["rewards"]["shaped"]["shaping_gamma"]
        self.shaping_scale = config["rewards"]["shaped"]["shaping_scale"]
        self.p_intended = config["transition"]["p_intended"]
        self.p_perpendicular = config["transition"]["p_perpendicular"]
        self.max_steps = int(config["episode"]["step_cap_multiplier"]
                             * maze.passable_count)
        # distance maps for the shaping potential (and later analysis)
        self.dist_to_key = _bfs_distances(maze, maze.key, door_open=False)
        self.dist_to_goal = _bfs_distances(maze, maze.goal, door_open=True)
        self._key_to_goal = self.dist_to_goal[maze.key]
        self._rng = random.Random(maze.effective_seed if seed is None else seed)
        self._state: State | None = None
        self._steps = 0
        self._terminated = False
        self._truncated = False

    # episode interface

    def reset(self, seed: int | None = None,
              state: State | None = None) -> State:
        """Start a new episode; ``state`` overrides the start state (used by
        tests, evaluation and the GUI)."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._state = (State(self.maze.start[0], self.maze.start[1], 0, 0)
                       if state is None else state)
        self._steps = 0
        self._terminated = False
        self._truncated = False
        return self._state

    def step(self, action: int) -> tuple[State, float, bool, bool, dict]:
        """Sample one transition; returns (s', reward, terminated, truncated, info)."""
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("episode has ended; call reset()")
        u = self._rng.random()
        if u < self.p_intended:
            direction = action
        elif u < self.p_intended + self.p_perpendicular:
            direction = PERPENDICULAR[action][0]
        else:
            direction = PERPENDICULAR[action][1]
        prev = self._state
        nxt, reward, events, terminated = self._resolve(prev, direction)
        self._steps += 1
        truncated = not terminated and self._steps >= self.max_steps
        if truncated:
            events = events + [EV_TIMEOUT]
        self._state = nxt
        self._terminated, self._truncated = terminated, truncated
        info = {"events": events, "intended_action": action,
                "executed_direction": direction, "step": self._steps,
                "prev_state": prev}
        return nxt, reward, terminated, truncated, info

    @property
    def state(self) -> State | None:
        return self._state

    @property
    def steps(self) -> int:
        return self._steps

    # MDP model

    def is_terminal(self, state: State) -> bool:
        return (state.r, state.c) == self.maze.goal

    def gate_open(self, phase: int) -> bool:
        return phase in self.maze.gate_open_phases

    def enumerate_states(self) -> list[State]:
        """All states: non-wall cells x has_key x gate phase."""
        return [State(r, c, k, p)
                for r in range(self.maze.size) for c in range(self.maze.size)
                if not self.maze.is_wall((r, c))
                for k in (0, 1) for p in range(self.maze.gate_period)]

    def transitions(self, state: State,
                    action: int) -> list[tuple[float, State, float, bool]]:
        """Exact model P(s'|s,a): list of (prob, s', reward, done). Empty for
        terminal states."""
        if self.is_terminal(state):
            return []
        outcomes: dict[tuple[State, float, bool], float] = {}
        branches = ((action, self.p_intended),
                    (PERPENDICULAR[action][0], self.p_perpendicular),
                    (PERPENDICULAR[action][1], self.p_perpendicular))
        for direction, prob in branches:
            nxt, reward, _, done = self._resolve(state, direction)
            key = (nxt, reward, done)
            outcomes[key] = outcomes.get(key, 0.0) + prob
        return [(p, nxt, reward, done)
                for (nxt, reward, done), p in outcomes.items()]

    # dynamics

    def _resolve(self, state: State,
                 direction: int) -> tuple[State, float, list[str], bool]:
        """Deterministic outcome of moving in ``direction`` (no sampling,
        no step-cap logic); shared by step() and transitions()."""
        r, c, k, phase = state
        dr, dc = ACTION_DELTAS[direction]
        target = (r + dr, c + dc)
        next_phase = (phase + 1) % self.maze.gate_period
        nr, nc, nk = r, c, k
        if self.maze.is_wall(target):
            events = [EV_WALL_HIT]
        elif target == self.maze.door and not k:
            events = [EV_DOOR_LOCKED]
        elif target == self.maze.gate and not self.gate_open(next_phase):
            events = [EV_GATE_BLOCKED]  # must be open when you arrive
        else:
            nr, nc = target
            cell = self.maze.cell(target)
            if cell == KEY and not k:
                nk = 1
                events = [EV_KEY_PICKUP]
            elif cell == DOOR:
                events = [EV_DOOR_PASS]  # only reachable with the key
            elif cell == PENALTY:
                events = [EV_PENALTY]
            elif cell == GOAL:
                events = [EV_GOAL]
            else:
                events = [EV_MOVE]
        nxt = State(nr, nc, nk, next_phase)
        done = (nr, nc) == self.maze.goal
        reward = float(self.rewards["step"])
        for ev in events:
            if ev in EVENT_REWARD_KEY:
                reward += self.rewards[EVENT_REWARD_KEY[ev]]
        if self.reward_mode == "shaped":
            reward += (self.shaping_gamma * self._phi(nxt) - self._phi(state))
        return nxt, reward, events, done

    def _phi(self, state: State) -> float:
        """Shaping potential: negative remaining mission distance, 0 at goal."""
        pos = (state.r, state.c)
        if pos == self.maze.goal:
            return 0.0
        unreachable = self.maze.size ** 2
        if state.has_key:
            d = self.dist_to_goal.get(pos, unreachable)
        else:
            d = self.dist_to_key.get(pos, unreachable) + self._key_to_goal
        return -self.shaping_scale * d


class EventLogger:
    """Collects per-step records (s, a, r, s', events) and writes CSV.

    The record is complete enough to reconstruct any Q-update by hand, as the
    report requires. Files go under results/raw_data/ (CSV, since *.log is
    git-ignored).
    """

    FIELDS = ["episode", "step", "r", "c", "has_key", "phase",
              "intended_action", "executed_direction", "reward",
              "next_r", "next_c", "next_has_key", "next_phase",
              "events", "terminated", "truncated"]

    def __init__(self):
        self.rows: list[dict] = []

    def log_step(self, episode: int, state: State, reward: float,
                 next_state: State, terminated: bool, truncated: bool,
                 info: dict) -> None:
        self.rows.append({
            "episode": episode, "step": info["step"],
            "r": state.r, "c": state.c, "has_key": state.has_key,
            "phase": state.phase,
            "intended_action": info["intended_action"],
            "executed_direction": info["executed_direction"],
            "reward": reward,
            "next_r": next_state.r, "next_c": next_state.c,
            "next_has_key": next_state.has_key, "next_phase": next_state.phase,
            "events": "|".join(info["events"]),
            "terminated": int(terminated), "truncated": int(truncated),
        })

    def save_csv(self, path: Path) -> None:
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def clear(self) -> None:
        self.rows = []


def main() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    maze = MazeMap.load(MAPS_DIR / "source.json")

    env = MazeEnv(maze, config)
    states = env.enumerate_states()
    print(f"|S| = {len(states)} states "
          f"({maze.passable_count} cells x 2 key x {maze.gate_period} phases), "
          f"|A| = 4, step cap = {env.max_steps}")

    for state in states:
        for action in ACTIONS:
            probs = sum(p for p, *_ in env.transitions(state, action))
            assert env.is_terminal(state) or abs(probs - 1.0) < 1e-12
    print("transition model: probabilities sum to 1 for every (s, a)")

    # the gate phase visibly changes the model: entry needs the gate OPEN ON
    # ARRIVAL, so the boundary phases show the rule (open {0,1,2}, period 6):
    gr, gc = maze.gate
    enters = State(gr, gc - 1, 1, 5)   # closed now, open on arrival
    blocked = State(gr, gc - 1, 1, 2)  # open now, closed on arrival
    for s in (enters, blocked):
        outs = env.transitions(s, RIGHT)
        move = next((p for p, ns, *_ in outs if (ns.r, ns.c) == (gr, gc)), 0.0)
        arrival = (s.phase + 1) % maze.gate_period
        print(f"  from {tuple(s)} action=right: "
              f"P(enter gate cell) = {move:.1f} (gate "
              f"{'open' if env.gate_open(arrival) else 'closed'} on arrival)")

    for mode in ("sparse", "shaped"):
        env = MazeEnv(maze, config, reward_mode=mode, seed=123)
        logger = EventLogger()
        state = env.reset()
        policy_rng = random.Random(0)
        total, counts = 0.0, Counter()
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_rng.randrange(4)
            nxt, reward, terminated, truncated, info = env.step(action)
            logger.log_step(0, state, reward, nxt, terminated, truncated, info)
            total += reward
            counts.update(info["events"])
            state = nxt
        outcome = "goal" if terminated else "timeout"
        print(f"random episode [{mode:6s}]: {env.steps} steps, "
              f"return {total:8.1f}, outcome {outcome}, "
              f"log rows {len(logger.rows)}")
        print(f"  events: {dict(counts)}")


if __name__ == "__main__":
    main()
