"""Dynamic maze environment modeled as a Markov Decision Process.

Planned contents (implemented in the environment step of the plan):

- State ``s = (x, y, has_key, gate_phase)`` — position, key possession and the
  phase of the periodic gate (the chosen dynamic feature), so the Markov
  property holds without inspecting history.
- Actions UP / DOWN / LEFT / RIGHT with stochastic transitions: probability
  0.8 for the intended direction and 0.1 for each perpendicular direction.
  Hitting a wall keeps the agent in place and incurs a penalty.
- Cell types: wall, normal, penalty, start, key, locked door, goal and the
  periodic gate.
- Two reward functions: sparse and shaped (both configurable via
  ``experiments/configs/``).
- Event logging for: normal move, wall hit, penalty-cell entry, key pickup,
  locked-door attempt, successful door pass, goal reached, step-cap timeout.
- Episode termination: goal reached or step cap exceeded
  (cap = 3 x number of passable cells, recorded in the config).
"""
