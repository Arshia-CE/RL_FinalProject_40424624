"""Transfer-learning scenarios for Q-Learning.

Planned contents (implemented in the transfer steps of the plan):

- Source-task training and Q-table persistence.
- Two BFS-validated target environments: "similar" (15-20% of obstacles moved,
  start/key/goal fixed) and "different" (>=35% of obstacles changed, key or
  goal relocated, new penalty cells).
- Four training scenarios per target:
    1. from scratch (zero Q-table, baseline),
    2. full transfer of the source Q-table,
    3. scaled transfer  Q_T = beta * Q_S  with beta in {0.25, 0.50, 0.75},
    4. selective transfer of only those states whose local neighborhood is
       unchanged between source and target.
- Metrics per scenario: jumpstart (initial performance), learning speed and
  final performance, plus at least one documented negative-transfer case.
"""
