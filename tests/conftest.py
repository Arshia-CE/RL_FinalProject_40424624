"""Shared fixtures: project config, the committed source map, and a
deterministic config variant (p_intended = 1) for exact dynamics tests."""

import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import DEFAULT_CONFIG_PATH, MAPS_DIR, MazeMap


@pytest.fixture(scope="session")
def config():
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def maze():
    return MazeMap.load(MAPS_DIR / "source.json")


@pytest.fixture()
def det_config(config):
    cfg = copy.deepcopy(config)
    cfg["transition"]["p_intended"] = 1.0
    cfg["transition"]["p_perpendicular"] = 0.0
    return cfg
