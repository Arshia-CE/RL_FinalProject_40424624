"""Unit tests for the seeded maze generator and BFS validation."""

import copy

import pytest

from environments.generator import (DOOR, GATE, GOAL, KEY, PENALTY, START,
                                    WALL, MazeMap, bfs_shortest_path,
                                    generate_map, path_exists, validate_map)


def cells_of_type(maze, cell_type):
    return [(r, c) for r in range(maze.size) for c in range(maze.size)
            if maze.grid[r][c] == cell_type]


class TestGeneration:
    def test_deterministic(self, config):
        assert generate_map(config).to_dict() == generate_map(config).to_dict()

    def test_reproduces_committed_map(self, config, maze):
        # guards against silent drift between the code and the saved artifact
        assert generate_map(config).to_dict() == maze.to_dict()

    def test_seed_and_size_derivation(self, config, maze):
        assert config["base_seed"] == int("40424624"[-2]) == 2
        assert maze.size == 15 + (config["base_seed"] % 4) == 17

    def test_wall_fraction_constraint(self, config, maze):
        assert maze.wall_fraction >= config["maze"]["min_wall_fraction"]

    def test_penalty_cell_constraint(self, config, maze):
        assert len(maze.penalty_cells) >= config["maze"]["min_penalty_cells"]
        assert sorted(cells_of_type(maze, PENALTY)) == sorted(
            tuple(p) for p in maze.penalty_cells)

    def test_special_cells_unique_and_consistent(self, maze):
        for cell_type, pos in ((START, maze.start), (KEY, maze.key),
                               (DOOR, maze.door), (GOAL, maze.goal),
                               (GATE, maze.gate)):
            assert cells_of_type(maze, cell_type) == [tuple(pos)]

    def test_generated_map_validates(self, config, maze):
        ok, problems = validate_map(maze, config["maze"])
        assert ok, problems

    def test_other_seeds_also_produce_valid_maps(self, config):
        for seed in range(10):
            m = generate_map(config, base_seed=seed)
            assert validate_map(m, config["maze"])[0]
            assert m.size == config["maze"]["size"]

    def test_impossible_constraints_raise(self, config):
        cfg = copy.deepcopy(config)
        cfg["maze"]["min_wall_fraction"] = 0.99  # can never be satisfied
        with pytest.raises(RuntimeError):
            generate_map(cfg, max_attempts=5)

    def test_save_load_round_trip(self, maze, tmp_path):
        path = tmp_path / "map.json"
        maze.save(path)
        assert MazeMap.load(path).to_dict() == maze.to_dict()


class TestPaths:
    def test_start_to_key_with_door_closed(self, maze):
        assert path_exists(maze.grid, maze.start, maze.key, door_open=False)

    def test_key_to_goal_with_door_open(self, maze):
        assert path_exists(maze.grid, maze.key, maze.goal, door_open=True)

    def test_goal_unreachable_without_door(self, maze):
        # the chamber must be sealed: the locked door is the only entrance
        assert not path_exists(maze.grid, maze.start, maze.goal,
                               door_open=False)

    def test_gate_is_the_only_approach_to_the_door(self, maze):
        r, c = maze.door
        open_neighbors = [(r + dr, c + dc)
                          for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                          if not maze.is_wall((r + dr, c + dc))]
        # exactly two: the gate outside and one chamber cell inside
        assert len(open_neighbors) == 2
        assert tuple(maze.gate) in open_neighbors

    def test_bfs_path_is_a_valid_walk(self, maze):
        path = bfs_shortest_path(maze.grid, maze.start, maze.key,
                                 door_open=False)
        assert path[0] == tuple(maze.start) and path[-1] == tuple(maze.key)
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            assert abs(r1 - r2) + abs(c1 - c2) == 1
            assert not maze.is_wall((r2, c2))
            assert maze.grid[r2][c2] != DOOR
