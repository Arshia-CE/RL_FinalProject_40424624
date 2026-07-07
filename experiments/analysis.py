"""Statistical analysis and figure generation from raw experiment data.

Planned contents (implemented alongside the experiment steps):

- Learning curves (reward / steps / success rate per episode, smoothed).
- Value-function heatmaps and policy-arrow maps.
- State-visit-count maps and final trajectories.
- Policy-agreement percentage against the Value Iteration reference and the
  colored disagreement map.
- Transfer-learning comparisons: jumpstart, learning speed, final
  performance, and Q/policy differences before vs. after transfer.

Every figure is produced strictly from the CSV/JSON raw data committed under
``results/raw_data/`` — no manually entered numbers.
"""
