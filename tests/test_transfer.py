"""Unit tests for transfer-learning initial-Q construction and
neighborhood-change detection."""

import numpy as np
import pytest

from environments.maze_map import MAPS_DIR, MazeMap
from environments.maze import State
from transfer.transfer_learning import (initial_q_table,
                                        neighborhood_signature,
                                        unchanged_positions)


@pytest.fixture(scope="module")
def different():
    return MazeMap.load(MAPS_DIR / "target_different.json")


@pytest.fixture(scope="module")
def source_q():
    return {
        State(0, 1, 0, 0): np.array([1.0, -2.0, 3.0, 4.0]),
        State(12, 10, 0, 3): np.array([-5.0, 10.0, 0.5, -0.5]),
        State(14, 11, 1, 5): np.array([0.0, 0.0, 7.0, 2.0]),
    }


class TestInitialQ:
    def test_scratch_is_empty(self, source_q):
        assert initial_q_table(source_q, "scratch") == {}

    def test_full_is_an_independent_copy(self, source_q):
        table = initial_q_table(source_q, "full")
        state = State(0, 1, 0, 0)
        assert np.array_equal(table[state], source_q[state])
        table[state][0] = 99.0
        assert source_q[state][0] == 1.0  # source must stay untouched

    @pytest.mark.parametrize("beta", [0.25, 0.5, 0.75])
    def test_scaled_multiplies_every_value(self, source_q, beta):
        table = initial_q_table(source_q, "scaled", beta=beta)
        assert set(table) == set(source_q)
        for state, q in source_q.items():
            assert np.allclose(table[state], beta * q)
            # scaling never changes the initial greedy action
            assert np.argmax(table[state]) == np.argmax(q)

    def test_selective_filters_by_unchanged_positions(self, source_q):
        unchanged = {(0, 1), (14, 11)}
        table = initial_q_table(source_q, "selective", unchanged=unchanged)
        assert set(table) == {State(0, 1, 0, 0), State(14, 11, 1, 5)}
        assert np.array_equal(table[State(0, 1, 0, 0)],
                              source_q[State(0, 1, 0, 0)])

    def test_unknown_scenario_raises(self, source_q):
        with pytest.raises(ValueError):
            initial_q_table(source_q, "half-hearted")


class TestNeighborhoods:
    def test_signature_covers_3x3_and_offgrid_is_wall(self, maze):
        signature = neighborhood_signature(maze, (0, 0))
        assert len(signature) == 9
        assert signature[0] == "#"  # (-1,-1) is off-grid -> wall

    def test_map_vs_itself_leaves_everything_unchanged(self, maze):
        unchanged = unchanged_positions(maze, maze)
        passable = {(r, c) for r in range(maze.size) for c in range(maze.size)
                    if not maze.is_wall((r, c))}
        assert unchanged == passable

    def test_moved_key_neighborhood_counts_as_changed(self, maze, different):
        unchanged = unchanged_positions(maze, different)
        assert tuple(maze.key) not in unchanged        # key removed here
        assert tuple(different.key) not in unchanged   # key appeared here
        # and the set only ever contains cells passable in both maps
        for pos in unchanged:
            assert not maze.is_wall(pos) and not different.is_wall(pos)