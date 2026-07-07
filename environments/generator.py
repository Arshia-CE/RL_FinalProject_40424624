"""Seeded maze generation and BFS validation.

Planned contents (implemented in the generator step of the plan):

- Reproducible 17x17 map generation from ``base_seed = 2`` (derived from the
  student ID per the specification).
- Constraints: at least 15% wall cells, at least 5 penalty cells, plus start,
  key, locked door, goal and the periodic-gate cell.
- Deterministic BFS validation that a path exists start -> key -> goal
  (treating the door as passable once the key is held); reproducible repair /
  regeneration if a candidate map is invalid.
- Saving/loading the final map file under ``environments/maps/`` so all three
  algorithms run on the exact same environment.
- Generation of the transfer-learning target maps (similar: 15-20% obstacles
  moved; different: >=35% changed, key/goal relocated, extra penalty cells).
"""
