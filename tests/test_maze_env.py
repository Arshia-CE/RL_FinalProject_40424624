"""Unit tests for the maze MDP: transition model, dynamics, rewards,
episode handling and event logging."""

import copy
import csv
import random
from collections import Counter

import pytest

from environments.maze_map import NORMAL
from environments.maze import (ACTION_DELTAS, ACTIONS, EV_DOOR_LOCKED,
                               EV_DOOR_PASS, EV_GATE_BLOCKED, EV_GOAL,
                               EV_KEY_PICKUP, EV_MOVE, EV_PENALTY,
                               EV_TIMEOUT, EV_WALL_HIT, EventLogger, MazeEnv,
                               State)

DELTA_TO_ACTION = {delta: a for a, delta in ACTION_DELTAS.items()}


def action_towards(src, dst):
    return DELTA_TO_ACTION[(dst[0] - src[0], dst[1] - src[1])]


def normal_neighbor(maze, pos):
    """A plain corridor cell adjacent to ``pos`` (for clean single-event tests)."""
    for dr, dc in ACTION_DELTAS.values():
        nxt = (pos[0] + dr, pos[1] + dc)
        if maze.in_bounds(nxt) and maze.grid[nxt[0]][nxt[1]] == NORMAL:
            return nxt
    raise AssertionError(f"no normal neighbor around {pos}")


def det_step(env, state, action):
    """One deterministic step from an arbitrary state."""
    env.reset(state=state)
    return env.step(action)


@pytest.fixture()
def det_env(maze, det_config):
    return MazeEnv(maze, det_config, seed=0)


@pytest.fixture()
def closed_phase(maze):
    return next(p for p in range(maze.gate_period)
                if p not in maze.gate_open_phases)


class TestTransitionModel:
    def test_probabilities_sum_to_one_everywhere(self, maze, config):
        env = MazeEnv(maze, config)
        for state in env.enumerate_states():
            for action in ACTIONS:
                total = sum(p for p, *_ in env.transitions(state, action))
                if env.is_terminal(state):
                    assert env.transitions(state, action) == []
                else:
                    assert total == pytest.approx(1.0)

    def test_sampled_steps_stay_inside_model_support(self, maze, config):
        env = MazeEnv(maze, config, seed=7)
        rng = random.Random(7)
        state = env.reset()
        for _ in range(300):
            action = rng.randrange(4)
            support = {(nxt, reward, done): p
                       for p, nxt, reward, done in env.transitions(state, action)}
            nxt, reward, terminated, truncated, _ = env.step(action)
            assert support.get((nxt, reward, terminated), 0.0) > 0.0
            state = env.reset() if (terminated or truncated) else nxt

    def test_action_noise_frequencies(self, maze, config):
        env = MazeEnv(maze, config, seed=11)
        env.reset()
        counts = Counter()
        for _ in range(3000):
            _, _, terminated, truncated, info = env.step(0)
            counts[info["executed_direction"]] += 1
            if terminated or truncated:
                env.reset()
        assert counts[0] / 3000 == pytest.approx(0.8, abs=0.03)
        for perp in (2, 3):
            assert counts[perp] / 3000 == pytest.approx(0.1, abs=0.03)


class TestDynamics:
    def test_free_move_advances_position_and_phase(self, maze, det_env):
        # any normal -> normal move found on the map
        src = normal_neighbor(maze, maze.key)
        dst = normal_neighbor(maze, src)
        state = State(src[0], src[1], 0, 0)
        nxt, reward, terminated, _, info = det_step(det_env, state,
                                                    action_towards(src, dst))
        assert (nxt.r, nxt.c) == dst
        assert nxt.phase == 1 and not terminated
        assert info["events"] == [EV_MOVE]
        assert reward == det_env.rewards["step"]

    def test_wall_bump_stays_and_penalizes(self, maze, det_env):
        state = State(maze.start[0], maze.start[1], 0, 0)
        wall_action = next(
            a for a, (dr, dc) in ACTION_DELTAS.items()
            if maze.is_wall((maze.start[0] + dr, maze.start[1] + dc)))
        nxt, reward, _, _, info = det_step(det_env, state, wall_action)
        assert (nxt.r, nxt.c) == tuple(maze.start)
        assert nxt.phase == 1  # time moves on even when blocked
        assert info["events"] == [EV_WALL_HIT]
        assert reward == det_env.rewards["step"] + det_env.rewards["wall_hit"]

    def test_locked_door_blocks_without_key(self, maze, det_env):
        gate = tuple(maze.gate)  # the only cell in front of the door
        state = State(gate[0], gate[1], 0, 0)
        action = action_towards(gate, maze.door)
        nxt, reward, _, _, info = det_step(det_env, state, action)
        assert (nxt.r, nxt.c) == gate
        assert info["events"] == [EV_DOOR_LOCKED]
        assert reward == (det_env.rewards["step"]
                          + det_env.rewards["locked_door_attempt"])

    def test_door_passes_with_key(self, maze, det_env):
        gate = tuple(maze.gate)
        state = State(gate[0], gate[1], 1, 0)
        nxt, reward, _, _, info = det_step(det_env, state,
                                           action_towards(gate, maze.door))
        assert (nxt.r, nxt.c) == tuple(maze.door)
        assert info["events"] == [EV_DOOR_PASS]
        assert reward == det_env.rewards["step"] + det_env.rewards["door_pass"]

    def test_gate_blocked_when_closed_open_when_not(self, maze, det_env,
                                                    closed_phase):
        src = normal_neighbor(maze, maze.gate)
        action = action_towards(src, maze.gate)
        blocked = State(src[0], src[1], 1, closed_phase)
        nxt, reward, _, _, info = det_step(det_env, blocked, action)
        assert (nxt.r, nxt.c) == src
        assert info["events"] == [EV_GATE_BLOCKED]
        assert reward == (det_env.rewards["step"]
                          + det_env.rewards["gate_blocked"])

        open_state = State(src[0], src[1], 1, maze.gate_open_phases[0])
        nxt, _, _, _, info = det_step(det_env, open_state, action)
        assert (nxt.r, nxt.c) == tuple(maze.gate)
        assert info["events"] == [EV_MOVE]

    def test_gate_entry_uses_arrival_phase(self, maze, det_env):
        """Boundary phases pin the rule: the gate must be open on ARRIVAL."""
        period, opens = maze.gate_period, set(maze.gate_open_phases)
        last_open = next(p for p in opens if (p + 1) % period not in opens)
        last_closed = next(p for p in range(period) if p not in opens
                           and (p + 1) % period in opens)
        src = normal_neighbor(maze, maze.gate)
        action = action_towards(src, maze.gate)

        # open now, closed on arrival -> blocked
        nxt, _, _, _, info = det_step(
            det_env, State(src[0], src[1], 1, last_open), action)
        assert (nxt.r, nxt.c) == src
        assert info["events"] == [EV_GATE_BLOCKED]

        # closed now, open on arrival -> enters
        nxt, _, _, _, info = det_step(
            det_env, State(src[0], src[1], 1, last_closed), action)
        assert (nxt.r, nxt.c) == tuple(maze.gate)
        assert info["events"] == [EV_MOVE]

    def test_key_pickup_once(self, maze, det_env):
        src = normal_neighbor(maze, maze.key)
        action = action_towards(src, maze.key)
        nxt, reward, _, _, info = det_step(det_env, State(src[0], src[1], 0, 0),
                                           action)
        assert nxt.has_key == 1
        assert info["events"] == [EV_KEY_PICKUP]
        assert reward == det_env.rewards["step"] + det_env.rewards["key_pickup"]
        # revisiting the key cell with the key is a plain move
        nxt, reward, _, _, info = det_step(det_env, State(src[0], src[1], 1, 0),
                                           action)
        assert info["events"] == [EV_MOVE]
        assert reward == det_env.rewards["step"]

    def test_penalty_cell_costs(self, maze, det_env):
        pen = tuple(maze.penalty_cells[0])
        src = normal_neighbor(maze, pen)
        nxt, reward, _, _, info = det_step(det_env, State(src[0], src[1], 0, 0),
                                           action_towards(src, pen))
        assert (nxt.r, nxt.c) == pen
        assert info["events"] == [EV_PENALTY]
        assert reward == (det_env.rewards["step"]
                          + det_env.rewards["penalty_cell"])

    def test_goal_terminates(self, maze, det_env):
        src = normal_neighbor(maze, maze.goal)
        state = State(src[0], src[1], 1, 0)
        nxt, reward, terminated, _, info = det_step(det_env, state,
                                                    action_towards(src, maze.goal))
        assert terminated
        assert info["events"] == [EV_GOAL]
        assert reward == det_env.rewards["step"] + det_env.rewards["goal"]
        assert det_env.is_terminal(nxt)
        assert det_env.transitions(nxt, 0) == []


class TestEpisode:
    def test_step_cap_truncates(self, maze, det_config):
        cfg = copy.deepcopy(det_config)
        cfg["episode"]["step_cap_multiplier"] = 0.05
        env = MazeEnv(maze, cfg, seed=0)
        env.reset()
        wall_action = next(
            a for a, (dr, dc) in ACTION_DELTAS.items()
            if maze.is_wall((maze.start[0] + dr, maze.start[1] + dc)))
        for step in range(1, env.max_steps + 1):
            _, _, terminated, truncated, info = env.step(wall_action)
        assert not terminated and truncated
        assert EV_TIMEOUT in info["events"]
        with pytest.raises(RuntimeError):
            env.step(wall_action)

    def test_step_before_reset_raises(self, maze, config):
        with pytest.raises(RuntimeError):
            MazeEnv(maze, config).step(0)

    def test_same_seed_same_trajectory(self, maze, config):
        def rollout():
            env = MazeEnv(maze, config, seed=42)
            rng = random.Random(1)
            state, out = env.reset(), []
            for _ in range(200):
                state, reward, terminated, truncated, _ = env.step(rng.randrange(4))
                out.append((state, reward))
                if terminated or truncated:
                    break
            return out

        assert rollout() == rollout()


class TestRewards:
    def test_shaping_is_potential_based(self, maze, config):
        sparse = MazeEnv(maze, config, reward_mode="sparse")
        shaped = MazeEnv(maze, config, reward_mode="shaped")
        gamma = config["rewards"]["shaped"]["shaping_gamma"]
        rng = random.Random(3)
        states = rng.sample(sparse.enumerate_states(), 100)
        for state in states:
            if sparse.is_terminal(state):
                continue
            for action in ACTIONS:
                # the same s' can occur with different rewards (e.g. wall hit
                # vs gate bump), so compare the full outcome distributions
                expected = sorted(
                    (round(p, 9), nxt, done,
                     round(r + gamma * shaped._phi(nxt) - shaped._phi(state), 9))
                    for p, nxt, r, done in sparse.transitions(state, action))
                got = sorted(
                    (round(p, 9), nxt, done, round(r, 9))
                    for p, nxt, r, done in shaped.transitions(state, action))
                assert got == expected

    def test_phi_zero_at_goal(self, maze, config):
        env = MazeEnv(maze, config, reward_mode="shaped")
        assert env._phi(State(maze.goal[0], maze.goal[1], 1, 0)) == 0.0

    def test_phi_continuous_at_key_pickup(self, maze, config):
        env = MazeEnv(maze, config, reward_mode="shaped")
        without = env._phi(State(maze.key[0], maze.key[1], 0, 0))
        with_key = env._phi(State(maze.key[0], maze.key[1], 1, 0))
        assert without == pytest.approx(with_key)

    def test_shaping_does_not_change_dynamics(self, maze, config):
        def event_stream(mode):
            env = MazeEnv(maze, config, reward_mode=mode, seed=5)
            rng = random.Random(2)
            env.reset()
            events = []
            for _ in range(300):
                _, _, terminated, truncated, info = env.step(rng.randrange(4))
                events.append(tuple(info["events"]))
                if terminated or truncated:
                    break
            return events

        assert event_stream("sparse") == event_stream("shaped")


class TestEventLogger:
    def test_csv_round_trip(self, maze, config, tmp_path):
        env = MazeEnv(maze, config, seed=9)
        logger = EventLogger()
        state = env.reset()
        for _ in range(25):
            nxt, reward, terminated, truncated, info = env.step(0)
            logger.log_step(0, state, reward, nxt, terminated, truncated, info)
            state = nxt
            if terminated or truncated:
                break
        path = tmp_path / "log.csv"
        logger.save_csv(path)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(logger.rows)
        first = rows[0]
        assert first["episode"] == "0" and first["step"] == "1"
        # the row carries the full (s, a, r, s') needed to redo a Q-update
        assert {"r", "c", "has_key", "phase", "intended_action", "reward",
                "next_r", "next_c", "next_has_key",
                "next_phase"} <= set(first)
