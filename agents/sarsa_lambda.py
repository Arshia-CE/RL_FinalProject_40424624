"""SARSA(lambda) (model-free, on-policy, with eligibility traces).

Planned contents (implemented in the SARSA(lambda) step of the plan):

- Tabular SARSA with eligibility traces (replacing traces; the choice is
  justified in the report).
- Lambda studied over {0, 0.3, 0.7, 0.9} via the experiment config.
- Per-step logging of the TD error (delta) and the eligibility trace values
  for a short episode, as required for the report analysis.
- Same per-episode CSV metrics as Q-Learning for a fair comparison.
"""
