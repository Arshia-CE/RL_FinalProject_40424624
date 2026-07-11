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
├── environments/          # Maze MDP + seeded map generation
│   ├── maze_map.py        # Cell types, MazeMap model, BFS, validation
│   ├── generator.py       # Seeded generation + transfer-target derivation
│   ├── maze.py            # The MDP environment (dynamics, rewards, logging)
│   └── maps/              # Saved, BFS-validated map files
├── agents/                # From-scratch RL algorithms
│   ├── value_iteration.py
│   ├── q_learning.py
│   └── sarsa_lambda.py
├── transfer/              # Transfer-learning scenarios (Q-Learning only)
│   └── transfer_learning.py
├── gui/                   # "MazeMario" — interactive Tkinter game GUI
│   ├── app.py             # Window assembly + game loop
│   ├── controller.py      # Game sessions: WATCH (trained) / TRAIN (live)
│   ├── renderer.py        # Pixel game board + animation engine
│   ├── hud.py             # HUD, control bar, menus, overlays
│   ├── sprites.py         # Pixel-art sprites (hero, dragon, princess, ...)
│   └── theme.py           # Palette, fonts, timing
├── experiments/
│   ├── run_experiments.py # Dispatcher: runs all or selected experiments
│   ├── common.py          # Config loading + policy evaluation helpers
│   ├── analysis.py        # Palette/style, CSV utils, curve/trace figures
│   ├── maze_plots.py      # Maze-rendered figures (heatmaps, policies, ...)
│   ├── exp_value_iteration.py
│   ├── exp_q_learning.py
│   ├── exp_sarsa_lambda.py
│   ├── exp_comparison.py
│   ├── exp_transfer.py
│   └── configs/           # JSON config (all experiment parameters)
├── results/               # One subfolder per experiment topic:
│   ├── raw_data/{vi,q_learning,sarsa,comparison,transfer}/
│   ├── models/{vi,q_learning,sarsa}/
│   ├── figures/{vi,q_learning,sarsa,comparison,transfer}/
│   └── videos/
├── tests/                 # Unit tests (generator, environment, updates)
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

## The GUI — MazeMario

`python main.py` launches a retro game-style visualization in which the agent
plays the maze. It boots into a title menu with three selectors:

- **SELECT WORLD** — the source maze or either transfer target
  (`environments/maps/*.json`, the exact maps used by all experiments).
- **HERO BRAIN** — Value Iteration (solved exactly on the fly for the chosen
  world), or the trained Q-Learning / SARSA(λ=0.7) tables loaded from
  `results/models/`. Picking a trained table on Worlds 2/3 lets you watch
  transfer behavior (including negative transfer) live.
- **MODE** — WATCH (the policy plays greedily) or TRAIN (a fresh Q-Learning /
  SARSA(λ) learner trains in real time; at speeds ≥ 2× it fast-forwards
  whole episodes per frame, ~60+ episodes/s).

In-game controls: PAUSE (menu), PLAY/PAUSE, STEP (single step), RESTART,
animation-speed slider (0.5–4×), and POLICY — an overlay of greedy-action
arrows for the agent's current key status and gate phase, updating live
during training. The HUD shows score (cumulative reward), steps/step-cap,
key status, the dragon-gate countdown, episode number, ε, and the recent
success rate.

Everything on the board is diegetic: walls are bricks, penalty cells are
thorn-ringed pits the hero falls into (with a floating −10), the key bobs
and sparkles, the locked door slides open, reaching the princess wins the
level — and the periodic gate is a dragon that emerges from its den on
closed phases and retreats on open ones, with bumps, popups and win/timeout
screens animating every logged environment event.

## Reproducing the results

Every number and figure in the report is generated by code from the committed
configuration ([experiments/configs/default.json](experiments/configs/default.json))
and the committed maps — nothing is entered by hand. The full pipeline:

```bash
pip install -r requirements.txt
python environments/generator.py           # reproduces all 3 maps byte-for-byte
python experiments/run_experiments.py      # all experiments (~60–90 min)
python -m pytest tests/                    # 58 unit tests
```

`run_experiments.py` also accepts a subset, e.g.
`python experiments/run_experiments.py vi comparison`. The experiments run in
dependency order (`transfer` needs the Q-table saved by `q_learning`;
`q_learning`/`sarsa_lambda` need the VI reference saved by `vi`):

| experiment | outputs (under `results/`) | ≈ time |
|---|---|---|
| `vi` | `raw_data/vi/` (γ sweep CSV), `models/vi/` (3), `figures/vi/` (4) | seconds |
| `q_learning` | training/summary/update-trace CSVs, 4 Q-tables, 3 figures | ~5 min |
| `sarsa_lambda` | training/summary/δ‑E trace CSVs, 4 Q-tables, 2 figures | ~25 min |
| `comparison` | summary + sample-state CSVs, 3 figures | ~2 min |
| `transfer` | training/summary/negative-case CSVs, 4 figures | ~30–60 min |

Determinism: map generation, Value Iteration and all agent training runs are
seeded (training seeds `{7, 21, 42}`, evaluation seed `999`, all recorded in
the config), so re-running reproduces the committed CSVs. Figures are drawn
exclusively from those CSVs / models; deleting everything under `results/`
and re-running the commands above rebuilds the complete set.


