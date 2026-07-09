"""Batch experiment dispatcher; regenerates all raw data, models and figures.

Run all experiments or a subset:
    python experiments/run_experiments.py [vi|q_learning|sarsa_lambda|comparison|transfer ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import load_config
from experiments.exp_comparison import run_comparison
from experiments.exp_q_learning import run_q_learning
from experiments.exp_sarsa_lambda import run_sarsa_lambda
from experiments.exp_transfer import run_transfer
from experiments.exp_value_iteration import run_value_iteration

EXPERIMENTS = {
    "vi": run_value_iteration,
    "q_learning": run_q_learning,
    "sarsa_lambda": run_sarsa_lambda,
    "comparison": run_comparison,
    "transfer": run_transfer,
}


def main() -> None:
    config = load_config()
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; "
                             f"available: {', '.join(EXPERIMENTS)}")
        print(f"[{name}]")
        EXPERIMENTS[name](config)
    print("done")


if __name__ == "__main__":
    main()
