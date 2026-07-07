"""Q-Learning (model-free, off-policy, from scratch).

Planned contents (implemented in the Q-Learning step of the plan):

- Tabular Q-Learning with an epsilon-greedy behavior policy.
- Two epsilon decay schedules — linear and exponential — selectable from the
  experiment config for direct comparison.
- Per-episode metric logging to CSV: total reward, steps, success flag, wall
  hits, penalty-cell entries and current epsilon.
- Persistence of the learned Q-table (also reused by the transfer-learning
  part) and a logged single-step Q-update so one real update can be
  reconstructed by hand in the report.
"""
