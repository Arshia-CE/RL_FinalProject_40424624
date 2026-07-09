# Report notes — working material for report.pdf

Running log of findings, design rationales and figure references, updated at
every project step. The final report (in whatever language) is assembled from
these notes; numbers here are copied from committed raw data in
`results/raw_data/`, never typed from memory.

## 1. Problem setup & derived parameters

- Student ID `40424624` → `base_seed = int(id[-2]) = 2`,
  `maze_size = 15 + (2 % 4) = 17`.
- Chosen dynamic feature: **periodic gate** (دروازه دوره‌ای), period 6,
  open in phases {0, 1, 2}, closed in {3, 4, 5}. Phase advances every time
  step whether or not the agent moves.
- State: `s = (r, c, has_key, gate_phase)` — the spec's minimal `(x, y, k)`
  plus exactly the variable the chosen feature requires.
  **Markov argument (report Q1):** door openness is a function of `has_key`,
  gate openness a function of `gate_phase`; with these in the state,
  P(s'|s,a) is well defined and no history is needed. Without `gate_phase`,
  the same (x, y, k) would sometimes be blocked and sometimes not, depending
  on arrival time → non-Markov.
- |S| = 233 passable cells × 2 key values × 6 phases = **2796 states**, |A| = 4.
- Step cap = 3 × 233 = **699** (multiplier recorded in config). Reaching the
  goal *terminates* the MDP; the step cap only *truncates* an episode —
  Value Iteration ignores it (it's not part of the stationary MDP), learning
  agents observe it as `truncated`. Worth a paragraph in the report.

## 2. Map generation (environments/generator.py)

- 17×17, walls 56/289 = **19.4%** (spec ≥ 15%), **6 penalty cells**
  (spec ≥ 5). Start (0,1), key (12,10), door (14,13), goal (16,16),
  gate (14,12). Penalty cells: (3,1), (3,6), (5,16), (9,8), (12,11), (15,0).
- Goal sits in a walled 3×3 chamber; the locked door is its **only**
  entrance, so the mission order start → key → door → goal is enforced by
  geometry, not just reward design.
- The gate occupies the single corridor cell in front of the door, so every
  successful episode must interact with it. (First draft placed the gate on
  *a* shortest key→door path; an equal-length detour existed, which would
  have let the optimal policy ignore the gate → moved to the structural
  bottleneck. Good example of "feature must not be decorative".)
- Deterministic BFS validation: start→key with door closed, key→goal with
  door open; the gate counts as passable because a closed gate only delays
  (wall-bump semantics), never disconnects. Invalid candidates are
  regenerated reproducibly (`effective_seed = base_seed*1000 + attempt`);
  seed 2 succeeds at attempt 0. Regeneration reproduces
  `environments/maps/source.json` byte-for-byte.
- Penalty cell (12,11) sits directly beside the key — a natural risk zone
  for the on-policy vs off-policy discussion (report Q2).

## 3. Rewards (experiments/configs/default.json)

- **Sparse:** step −1, wall hit −5, penalty cell −10, locked-door attempt −2,
  gate bump −2, key +50, goal +200, door pass 0.
  `door_pass = 0` on purpose: any repeatable positive bonus on a
  non-terminal cell can be farmed by stepping back and forth (+bonus − 2×step
  per loop). The event is still logged.
- **Shaped:** sparse + potential-based shaping F = γΦ(s′) − Φ(s),
  Φ = −(remaining BFS mission distance), i.e. dist-to-key + dist(key→goal)
  before pickup, dist-to-goal after. Φ is **continuous at key pickup**
  (naive per-subgoal distances would spike −γ·dist(key→goal) exactly when
  the agent achieves the subgoal). Φ(goal) = 0. shaping_gamma = 0.95.
- Since shaping is potential-based (Ng et al. 1999), the optimal policy is
  provably unchanged when the agent's γ equals shaping_gamma — to be
  demonstrated empirically in the Q-Learning step.
- Verified numerically: on the 33-step scripted optimal path, shaped − sparse
  return = 59.4 = exact telescoped potential sum.

## 4. Environment verification (tests/, 34 tests)

- Transition probabilities sum to 1 for **all** 2796×4 (s,a) pairs; sampled
  steps always lie in the support of `transitions()` (model ≡ simulation).
- Empirical noise ≈ 0.8/0.1/0.1 over 3000 steps.
- Every dynamics rule unit-tested (wall, door, gate by phase, key-once,
  penalty, goal, timeout). Shaping proven potential-based; dynamics
  identical under both reward modes.
- Generator: all base seeds 0–9 yield valid maps; goal provably sealed
  without the door; gate is the only approach to the door.

## 5. Value Iteration (agents/value_iteration.py)

Data: `results/raw_data/vi_gamma_sweep.csv`, models
`results/models/vi_sparse_gamma{0.9,0.95,0.99}.json`.
Evaluation protocol: greedy rollouts, 500 episodes, eval seed 999.

| γ | sweeps | runtime (s) | V(start) | success | mean return | mean steps | policy agreement vs γ=0.95 |
|---|---|---|---|---|---|---|---|
| 0.90 | 80 | 0.018 | −9.79 | 100% | 191.51 | 43.3 | 97.9% |
| 0.95 | 91 | 0.024 | 9.60 | 100% | 191.61 | 43.4 | 100% (ref) |
| 0.99 | 102 | 0.025 | 119.43 | 100% | 191.99 | 43.6 | 96.6% |

Analysis points:
- **Convergence has two regimes** (see `vi_convergence.png`): a ~45-sweep
  plateau while reward information propagates backward along the ~33-step
  optimal path (delta stays at the level set by step costs), then geometric
  decay with slope ln γ (contraction rate) — steeper for smaller γ.
  Iterations to threshold grow with γ: 80 / 91 / 102.
- **Value scale ≠ policy quality:** V(start) ranges from −9.8 (γ=0.9:
  discounted terminal rewards no longer cover accumulated step costs) to
  +119.4 (γ=0.99), yet all three policies reach 100% success with mean
  returns within 0.5 of each other.
- **Policy disagreements (2–3.4% of states)** concentrate where action
  values nearly tie (equidistant detours, far-from-goal states) — candidate
  states for the "three sample states" analysis (report Q5, vs model-free).
- **Phase-dependent optimal policy at the gate** (report: feature evidence;
  see `vi_policy_by_phase_gamma0.95.png` and the printout of
  `agents/value_iteration.py`): at the cell below the gate, holding the key —
  phases 0–2 (open): enter; phases 3–4 (closed): walk away and loop back
  (repeated −2 bumps are dearer than a detour); phase 5 (closed but about to
  open): bump once (−2) and enter next tick. Optimal action differs by phase
  at identical position/key → the feature genuinely shapes behavior.
- Footnote: in the k=0 heatmap panel the chamber interior shows high values —
  those states are unreachable (door blocks keyless entry), but a keyless
  agent *placed* there could still walk to G (the door gates entry, not the
  goal). Harmless; explain in report.

Figures so far: `vi_convergence.png`, `vi_value_heatmap_gamma0.95.png`,
`vi_policy_gamma0.95.png`, `vi_policy_by_phase_gamma0.95.png`.

## 6. Q-Learning (agents/q_learning.py)

Implementation notes (report material):
- Off-policy TD control, ε-greedy behavior; **random tie-breaking** in the
  greedy step (with zero-initialized Q, deterministic argmax would always
  pick action 0 and bias early exploration).
- **Truncation vs termination in the update:** the bootstrap term
  `max_a' Q(s',a')` is zeroed only on true termination (goal). A step-cap
  timeout still bootstraps — the cap is an episode-length device, not a
  property of the MDP.
- Per-state visit counts tracked during training (needed for the required
  visit-map figure and for visit-weighted analysis) and saved with the
  Q-table.
- Per-update trace rows (s, a, r, s', Q-before, max Q(s'), target, TD error,
  Q-after, α, γ, ε) recorded for selected episodes → source of the
  hand-reconstructed update the spec requires. Sample from the demo run
  (episode 2500, α=0.1, γ=0.95):
  s=(0,1,k=0,p=0), a=left, r=−1 → s'=(0,0,k=0,p=1);
  Q_before=−13.278179, max Q(s')=−11.006227,
  target = −1 + 0.95·(−11.006227) = −11.455916,
  TD error = 1.822263, Q_after = −13.278179 + 0.1·1.822263 = −13.095952. ✓
- Demo run (sparse, exponential decay, 5000 episodes, seeds env=7/agent=7):
  last-100 success **100%**, mean return **181.8** (VI optimum 191.6),
  mean steps **48.6** (VI 43.4); 2724/2796 states visited.

**Near-tie finding (feeds report Q5 and the comparison step):** raw greedy
agreement with the VI reference is only ~51–53% (barely rising with visit
threshold: 51.4% at ≥1, 53.0% at ≥100 visits), *yet* the learned policy
collects ~95% of the optimal return. Explanation, verified against exact VI
Q-values: 41.1% of non-terminal states have a second action within 1.0 of
optimal (28.6% within 0.5, 10.6% within 0.1) — corridor states where two
directions are equally good. Q-Learning's noisy estimates break these ties
arbitrarily. Consequence for the comparison section: report **raw agreement**
(spec metric) *plus* an action-gap-aware metric (agreement counting a_QL as
correct if Q*(s, a_QL) is within tolerance of V*(s)) and/or the policy's
value loss V*(start) − V^π(start); analyze 2–3 near-tie states explicitly.

## Report-question tracker

1. **MDP + Markov property** — material ready (§1, §3).
2. **on- vs off-policy near danger** — pending (Q-Learning/SARSA steps);
   penalty cell (12,11) beside the key is the designated observation zone.
3. **Why VI needs the model** — material forming (§5; `transitions()` vs
   sampled `step()`); finish after model-free steps for the contrast.
4. **Best λ** — pending (SARSA(λ) step).
5. **Three states where model-free ≠ VI** — pending (comparison step);
   near-tie states from §5 are candidates.
6. **Transfer: similar vs different target** — pending (transfer steps).

## Notable design corrections (report material)

- **Gate placement:** the first implementation put the gate on one shortest
  key→door path; an equal-length detour bypassed it, weakening the "real
  behavioral effect" requirement → moved to the chamber-entrance bottleneck,
  validated by VI's phase-dependent policy.
- **Shaping unit test:** the first version assumed each s′ appears once per
  (s,a); in fact the same s′ can occur with different rewards (wall −5 vs
  gate bump −2, both staying in place) → rewritten to compare full outcome
  distributions.
- **door_pass reward:** initially +20 per passage — farmable (+18 net per
  back-and-forth loop through the open door) → set to 0, event still logged.
