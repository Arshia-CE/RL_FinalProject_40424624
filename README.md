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
| Dynamic feature | **Limited energy** (a 100-unit budget; every step burns 1, pits drain 6, empty = death) |
| State representation | `s = (x, y, has_key, energy)` — 47,066 states |

> **One task, several designs.** This branch replaces the periodic gate
> with an **observable limited-energy budget**: every step costs one
> unit (bumps included), penalty cells drain five extra, and running dry
> terminates the episode as a death (−50) — reaching the goal on the
> last unit still wins. The maps are byte-identical to the gate
> branches; the report closes with a retrospective comparing the clock
> designs (`main`: gate open at departure; `future_phase`: gate open on
> arrival) against this resource design, including the tabular scaling
> study the ×17 state space made possible.

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
│   ├── sprites.py         # Pixel-art sprites (hero, princess, key, ...)
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
│   ├── models/{vi,q_learning,sarsa,transfer}/
│   ├── figures/{vi,q_learning,sarsa,comparison,transfer}/
│   └── videos/
├── tests/                 # Unit tests (generator, environment, updates)
├── report.md              # Final analytical report
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
  world), or the trained Q-Learning (shaped) / SARSA(λ=0.7) tables loaded
  from `results/models/`. Picking the source-trained table on World 3 lets
  you watch the report's negative-transfer trap live.
- **MODE** — WATCH (the policy plays greedily) or TRAIN (a fresh learner
  trains in real time — shaped Q-Learning or sparse SARSA(λ), mirroring the
  canonical experiments; at speeds ≥ 2× it fast-forwards whole episodes per
  frame, ~60+ episodes/s).

In-game controls: PAUSE (menu), PLAY/PAUSE, STEP (single step), RESTART,
animation-speed slider (0.5–4×), and POLICY — an overlay of greedy-action
arrows for the agent's current key status and remaining energy, updating
live as the budget drains. The HUD shows score (cumulative reward), steps,
key status, the color-banded energy bar (green → gold → red), episode
number, ε, and the recent success rate.

Everything on the board is diegetic: walls are bricks, penalty cells are
thorn-ringed pits the hero falls into (a floating −10 plus a cyan −6 ENERGY
drain), the key bobs and sparkles, the locked door slides open, reaching
the princess wins the level — and running out of energy plays the branch's
signature animation: the hero's spirit flickers and rises off the board
under a −50 popup, followed by an OUT OF ENERGY screen.

## Reproducing the results

Every number and figure in the report is generated by code from the committed
configuration ([experiments/configs/default.json](experiments/configs/default.json))
and the committed maps — nothing is entered by hand. The full pipeline:

```bash
pip install -r requirements.txt
python environments/generator.py           # reproduces all 3 maps byte-for-byte
python experiments/run_experiments.py      # all experiments (~2.5–3 h)
python -m pytest tests/                    # 67 unit tests
```

`run_experiments.py` also accepts a subset, e.g.
`python experiments/run_experiments.py vi comparison`. The experiments run in
dependency order (`transfer` needs the Q-table saved by `q_learning`;
`q_learning`/`sarsa_lambda` need the VI reference saved by `vi`):

| experiment | outputs (under `results/`) | ≈ time |
|---|---|---|
| `vi` | `raw_data/vi/` (γ sweep CSV), `models/vi/` (3), `figures/vi/` (4) | ~15 s |
| `q_learning` | training/summary/update-trace CSVs, 4 Q-tables, 3 figures | ~15 min |
| `sarsa_lambda` | training/summary/δ‑E trace CSVs, 4 Q-tables, 2 figures | ~70 min |
| `comparison` | summary + sample-state CSVs, 4 figures | ~5 min |
| `transfer` | training/summary/negative-case CSVs, 2 Q-tables, 6 figures | ~90 min |

Determinism: map generation, Value Iteration and all agent training runs are
seeded (training seeds `{7, 21, 42}`, evaluation seed `999`, all recorded in
the config), so re-running reproduces the committed CSVs. Figures are drawn
exclusively from those CSVs / models; deleting everything under `results/`
and re-running the commands above rebuilds the complete set. One artifact is
deliberately not committed: `results/raw_data/transfer/transfer_training.csv`
(18 runs x 50,000 episodes, ~105 MB) exceeds GitHub's file-size limit and is
regenerated by the pipeline instead; the transfer summary CSV and figures
derived from it are committed.


