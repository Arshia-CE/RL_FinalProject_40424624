"""Batch experiment runner — reproduces every result in the report.

Planned contents (filled in incrementally as each algorithm lands):

- Runs all experiments defined by the JSON configs in ``experiments/configs/``
  with fixed seeds, writing raw per-episode CSVs to ``results/raw_data/``,
  trained models to ``results/models/`` and figures to ``results/figures/``.
- Covered experiments: Value Iteration gamma sweep, Q-Learning decay-schedule
  and reward-shaping comparisons, SARSA(lambda) lambda sweep, three-algorithm
  comparison, and the four transfer-learning scenarios on both target maps.

Usage:
    python experiments/run_experiments.py
"""


def main() -> None:
    print("No experiments implemented yet - they are added step by step "
          "alongside each algorithm. See README.md.")


if __name__ == "__main__":
    main()
