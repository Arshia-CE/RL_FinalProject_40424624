"""Unit tests for the learning updates: Q-Learning backups and
SARSA(lambda) eligibility-trace mechanics."""

import numpy as np
import pytest

from agents.q_learning import QLearningAgent, epsilon_schedule
from agents.sarsa_lambda import SarsaLambdaAgent
from environments.maze import MazeEnv, State


@pytest.fixture()
def env(maze, config):
    return MazeEnv(maze, config, seed=0)


def two_states(maze):
    s1 = State(maze.start[0], maze.start[1], 0, 0)
    s2 = State(maze.start[0], maze.start[1], 0, 1)
    return s1, s2


class TestQLearningUpdate:
    def test_backup_math(self, env, maze):
        agent = QLearningAgent(env, alpha=0.5, gamma=0.9)
        s1, s2 = two_states(maze)
        agent.q_values(s1)[:] = [1.0, 2.0, 3.0, 4.0]
        agent.q_values(s2)[:] = [0.0, 10.0, 0.0, 0.0]
        stats = agent.update(s1, 0, reward=5.0, next_state=s2,
                             terminated=False)
        # Q <- 1 + 0.5 * (5 + 0.9*10 - 1) = 7.5
        assert stats["td_target"] == pytest.approx(14.0)
        assert stats["td_error"] == pytest.approx(13.0)
        assert agent.Q[s1][0] == pytest.approx(7.5)
        assert list(agent.Q[s1][1:]) == [2.0, 3.0, 4.0]  # others untouched

    def test_terminal_has_no_bootstrap(self, env, maze):
        agent = QLearningAgent(env, alpha=0.5, gamma=0.9)
        s1, s2 = two_states(maze)
        agent.q_values(s1)[:] = [1.0, 0.0, 0.0, 0.0]
        agent.q_values(s2)[:] = [99.0, 99.0, 99.0, 99.0]
        agent.update(s1, 0, reward=5.0, next_state=s2, terminated=True)
        assert agent.Q[s1][0] == pytest.approx(1 + 0.5 * (5 - 1))

    def test_epsilon_schedules_hit_endpoints(self):
        for kind in ("linear", "exponential"):
            sched = epsilon_schedule(kind, 1.0, 0.05, 100)
            assert sched(0) == pytest.approx(1.0)
            assert sched(100) == pytest.approx(0.05, abs=1e-9)
            assert sched(500) == pytest.approx(0.05)


class TestSarsaLambda:
    def make(self, env, lam, trace_type="replacing"):
        return SarsaLambdaAgent(env, alpha=0.5, gamma=0.9, lam=lam,
                                trace_type=trace_type)

    def test_lambda0_is_one_step_sarsa(self, env, maze):
        agent = self.make(env, lam=0.0)
        s1, s2 = two_states(maze)
        agent.q_values(s1)[:] = [1.0, 2.0, 3.0, 4.0]
        agent.q_values(s2)[:] = [0.0, 10.0, 0.0, 7.0]
        agent.begin_episode()
        stats = agent.update(s1, 0, reward=5.0, next_state=s2,
                             next_action=3, terminated=False)
        # on-policy target uses Q(s',a'=3)=7, not max=10:
        # delta = 5 + 0.9*7 - 1 = 10.3 ; Q <- 1 + 0.5*10.3 = 6.15
        assert stats["delta"] == pytest.approx(10.3)
        assert agent.Q[s1][0] == pytest.approx(6.15)
        assert list(agent.Q[s1][1:]) == [2.0, 3.0, 4.0]
        # with lambda = 0 the trace dies immediately: nothing else is traced
        assert agent.E == {}

    def test_trace_propagates_delta_backwards(self, env, maze):
        agent = self.make(env, lam=0.5)
        s1, s2 = two_states(maze)
        s3 = State(s1.r, s1.c, 0, 2)
        agent.begin_episode()
        agent.update(s1, 0, reward=0.0, next_state=s2, next_action=1,
                     terminated=False)
        stats = agent.update(s2, 1, reward=10.0, next_state=s3,
                             next_action=0, terminated=False)
        # after step 1, E(s1,0) decayed to gamma*lam = 0.45; step-2 delta
        # (=10) must update s1 through the trace: 0.5 * 10 * 0.45 = 2.25
        delta2 = stats["delta"]
        assert agent.Q[s1][0] == pytest.approx(0.5 * delta2 * 0.45)
        assert agent.Q[s2][1] == pytest.approx(0.5 * delta2 * 1.0)

    def test_replacing_vs_accumulating_on_revisit(self, env, maze):
        s1, s2 = two_states(maze)
        for trace_type, expected in (("replacing", 1.0),
                                     ("accumulating", 1.45)):
            agent = self.make(env, lam=0.5, trace_type=trace_type)
            agent.begin_episode()
            agent.update(s1, 0, 0.0, s2, 1, terminated=False)
            # revisit the same pair: E was decayed to 0.45, then bumped
            agent.update(s1, 0, 0.0, s2, 1, terminated=False)
            bumped = [e for e in (agent.E[s1][0] / (0.9 * 0.5),)]
            assert bumped[0] == pytest.approx(expected)

    def test_traces_reset_between_episodes(self, env, maze):
        agent = self.make(env, lam=0.9)
        s1, s2 = two_states(maze)
        agent.begin_episode()
        agent.update(s1, 0, 1.0, s2, 1, terminated=False)
        assert agent.E
        agent.begin_episode()
        assert agent.E == {}

    def test_terminal_zeroes_bootstrap(self, env, maze):
        agent = self.make(env, lam=0.9)
        s1, s2 = two_states(maze)
        agent.q_values(s1)[:] = [2.0, 0.0, 0.0, 0.0]
        agent.q_values(s2)[:] = [50.0, 50.0, 50.0, 50.0]
        agent.begin_episode()
        stats = agent.update(s1, 0, reward=10.0, next_state=s2,
                             next_action=0, terminated=True)
        assert stats["delta"] == pytest.approx(10 - 2)
        assert agent.Q[s1][0] == pytest.approx(2 + 0.5 * 8)
