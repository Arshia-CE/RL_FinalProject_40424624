# RL_FinalProject_40424624

**Design and Analysis of an Intelligent Agent in a Dynamic Maze**
Reinforcement Learning — Final Project

- **Student ID:** 40424624

## Project parameters (derived from the student ID)

Per the project specification, the base seed and maze size are derived from the
student ID:

```python
student_id = "40424624"
base_seed  = int(student_id[-2])        # = 2
maze_size  = 15 + (base_seed % 4)       # = 17  ->  17x17 maze
```

| Parameter | Value |
|---|---|
| Base seed | `2` |
| Maze size | `17 x 17` |
| Dynamic feature | **Periodic gate** (opens/closes on a fixed cycle) |
| State representation | `s = (x, y, has_key, gate_phase)` |

All experiment parameters live in [experiments/configs/default.json](experiments/configs/default.json)
so every result is reproducible from the committed configuration.

## Repository structure

```
RL_FinalProject_40424624/
├── environments/          # Maze MDP + seeded map generator
│   ├── maze.py
│   ├── generator.py
│   └── maps/              # Saved, BFS-validated map files
├── agents/                # From-scratch RL algorithms
│   ├── value_iteration.py
│   ├── q_learning.py
│   └── sarsa_lambda.py
├── transfer/              # Transfer-learning scenarios (Q-Learning only)
│   └── transfer_learning.py
├── gui/                   # Interactive Tkinter visualization
│   ├── app.py
│   └── renderer.py
├── experiments/           # Experiment runners + statistical analysis
│   ├── run_experiments.py
│   ├── analysis.py
│   └── configs/           # JSON config per experiment
├── results/
│   ├── raw_data/          # Per-episode CSV logs + event logs
│   ├── models/            # Saved V/Q tables and policies
│   ├── figures/           # Heatmaps, policy maps, learning curves
│   └── videos/
├── tests/                 # Unit tests (environment, updates)
├── report.pdf             # Final analytical report
├── requirements.txt
├── README.md
└── main.py                # Entry point (launches the GUI)
```

## Installation

Requires Python 3.10+. Tkinter ships with the standard CPython installer.

```bash
pip install -r requirements.txt
```

## Running

```bash
# Launch the interactive GUI
python main.py

# Reproduce all experiments, raw data and figures
python experiments/run_experiments.py

# Run the unit tests
python -m pytest tests/

# (Re)generate the seeded, BFS-validated map — deterministic, the committed
# environments/maps/source.json is reproduced byte-for-byte
python environments/generator.py
```

_(Sections below are filled in as the corresponding parts are implemented.)_

## Reproducing the results

_To be completed: exact commands and expected outputs for every figure in the
report. All figures are generated from the raw CSV data in `results/raw_data/`
using the committed configs — no manual editing._

## Attribution & AI assistance

_To be completed: sources used, plus the AI-assistance disclosure table required
by the specification (usage, received suggestion, student modification, reason)._
