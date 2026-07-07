"""Value Iteration (model-based, from scratch — no library RL implementations).

Planned contents (implemented in the Value Iteration step of the plan):

- Bellman optimality backups over the exact transition model exposed by the
  environment (0.8 / 0.1 / 0.1 stochasticity included).
- Convergence criterion: maximum absolute change of the value function
  between two consecutive sweeps falls below a configured threshold.
- Greedy policy extraction from the converged value function.
- Persistence of V, the final policy, iteration count, runtime and the
  convergence threshold to ``results/models`` and ``results/raw_data``.
- The resulting policy is the reference for evaluating Q-Learning and
  SARSA(lambda).
"""
