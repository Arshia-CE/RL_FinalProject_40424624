# Report notes — working material for report.pdf

Running log of findings, design rationales and figure references, updated at
every project step. The final report (in whatever language) is assembled from
these notes; numbers here are copied from committed raw data in
`results/raw_data/`, never typed from memory.

Layout note: results are organized per topic — raw_data/figures/models each
contain `vi/`, `q_learning/`, `sarsa/`, `comparison/`, `transfer/`
subfolders; file names referenced in these notes are unchanged.

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

## 7. Q-Learning experiments (run_experiments.py: q_learning)

Data: `q_learning_training.csv` (12 runs × 5000 episodes),
`q_learning_summary.csv`, `q_update_trace.csv` (full traced episode 2500);
models `q_learning_{sparse,shaped}_{linear,exponential}.json`.
Figures: `q_learning_decay_schedules.png`, `q_learning_reward_shaping.png`,
`q_learning_visit_map.png`. 2 schedules × 2 reward modes × 3 seeds
{7, 21, 42}; evaluation always on the sparse env (500 greedy episodes,
seed 999) so returns are comparable.

| run | episodes to 90% success (mean of seeds) | eval return | VI agreement |
|---|---|---|---|
| sparse / linear | ≈1546 | 190.8–191.6 | 57.9–58.6% |
| sparse / exponential | ≈1099 | 189.4–190.5 | 51.2–51.9% |
| shaped / linear | ≈890 | 190.4–190.8 | 73.0–75.0% |
| shaped / exponential | ≈380 | 189.2–190.6 | 67.4–67.8% |

Findings:
- **Decay schedules:** exponential reaches 90% success ~1.4× earlier than
  linear (ε collapses early, exploiting sooner), but linear's longer
  exploration buys visibly higher VI agreement (58% vs 51%) — broader,
  more uniform coverage. Both end at the same asymptote (100% success,
  ~190 return). Trade-off: learning speed vs state-space coverage.
- **Shaping = ~3× jumpstart, same destination:** shaped/exponential hits 90%
  success at ep ≈380 vs ≈1099 sparse — and the key-found rate hits 1.0
  almost immediately (potential gradient points at the key). Final eval
  returns are statistically identical across all 12 runs (189.2–191.6,
  VI optimum 191.6): the potential-based-invariance prediction confirmed
  empirically (report: shaping sped learning up *without* changing final
  performance).
- **No unwanted behavior from shaping:** eval penalty entries/ep 0.21–0.27,
  wall hits/ep 2.2–2.5, gate bumps/ep 0.22–0.45 — indistinguishable between
  sparse and shaped; steps/ep ~44–46 everywhere (no loops, no reward
  farming, no over-avoidance of penalty cells).
- **Nominal policies still differ:** sparse vs shaped greedy policies agree
  on only 49.3% of jointly-visited states while achieving identical
  returns — the near-tie effect from §6 again; nominal action agreement is
  a weak lens on policy quality.
- **Visit-map artifacts worth explaining in the report:**
  (a) the key cell has zero k=0 visits — entering it flips k to 1, so
  (12,10,k=0) is never a *decision* state; (b) the chamber interior has
  zero k=0 visits — unreachable without the key, matching the design;
  (c) in k=1 the gate/door cells are among the darkest — waiting and
  funneling at the bottleneck.

## 8. SARSA(λ) (agents/sarsa_lambda.py)

Implementation notes:
- On-policy: the update target uses Q(s', a') for the action *actually
  selected* by the ε-greedy policy (a' is then executed — no re-picking).
- **Replacing traces** (justify in report): with stochastic slips the agent
  revisits states within an episode; accumulating traces can push E > 1 and
  destabilize updates at high λ (double-counting), while replacing caps
  eligibility at 1 (Singh & Sutton 1996 favor replacing in tabular settings).
- Traces reset at episode start; pruned below 1e-4 (bounds the active set to
  ~≤60 pairs at γλ = 0.855 with negligible numerical effect — recorded in
  config as trace_prune).
- λ = 0 reduces *exactly* to one-step SARSA (unit-tested: trace dies
  immediately, single-pair update). Rising λ hands the TD error backward:
  unit-tested that a step-2 delta updates the step-1 pair by α·δ·(γλ).
- Same per-episode metrics/persistence as Q-Learning; update() extracted in
  both agents and unit-tested (8 update tests, 42 total).

Demo run (λ=0.9, replacing, sparse, exponential decay, seed 7):
last-100 success **100%**, mean return **182.3**, mean steps 47.2
(QL same setup: 181.8 / 48.6; VI optimum 191.6/43.4).

δ/E trace for report (episode 4900, 49 steps, `sarsa_step_trace.csv` +
`sarsa_trace_dump.csv` in step 10):
- steps 1–2: intended `left` slipped perpendicular into the top border →
  r = −6 wall bumps, **negative deltas** (−8.68, −4.28): outcomes worse than
  the state's expectation.
- steps 4–6: productive moves down the corridor, **positive deltas**
  (+1.28, +2.88, +1.40) — the value estimate along the path was still
  slightly pessimistic.
- E(s₀,a₀) after each subsequent step: 0.855, 0.731, 0.625, 0.534, 0.457,
  0.391 — exactly (γλ)ᵗ = 0.855ᵗ geometric decay; each later delta updates
  s₀ scaled by this factor. Interpretation: one wall bump at step 10 would
  still nudge the values of the previous ~2 dozen state-actions.

## 9. SARSA(λ) lambda sweep (run_experiments.py: sarsa_lambda)

Data: `sarsa_training.csv`, `sarsa_summary.csv`, `sarsa_step_trace.csv`,
`sarsa_trace_dump.csv`; models `sarsa_lambda{0,0.3,0.7,0.9}_sparse.json`.
Figures: `sarsa_lambda_sweep.png`, `sarsa_delta_trace.png`.
4 λ × 3 seeds, sparse reward, exponential decay, replacing traces.

| λ | episodes to 90% success (mean) | late return std (seed range) | eval return (seed range) |
|---|---|---|---|
| 0 | 1670 | 26.0 – 72.0 | 185.1 – 188.1 |
| 0.3 | 1334 | 22.6 – 45.4 | 187.6 – 190.7 |
| 0.7 | 982 | **16.5 – 24.4** | **189.5 – 190.8** |
| 0.9 | 805 | 16.5 – 31.6 | 187.0 – 189.1 |

**Answer to report Q4 (best λ):** learning speed rises monotonically with λ
(1670 → 805 episodes to 90%), but **λ = 0.7 gives the best balance**: ~1.7×
faster than λ=0 while showing the lowest and most consistent late-training
variance and the highest, tightest final returns. λ=0.9 is fastest to the
90% mark yet slightly noisier/lower at convergence — long traces also
propagate the deltas of exploratory slips backward, injecting noise. λ=0
(one-step SARSA) is slowest and has the largest across-seed spread in late
returns (a lingering-stale-values effect: single-step backups correct
distant states very slowly).

δ/E interpretation figure (`sarsa_delta_trace.png`, episode 4900):
- left panel: δ hovers near 0 (converged values) with sharp spikes at
  stochastic slips — worse-than-expected outcomes (wall/penalty, δ ≈ −43)
  and better-than-expected recoveries (δ ≈ +27);
- right panel: eligibilities of the first four state-actions are parallel
  straight lines on the log axis — exactly (γλ)ᵗ = 0.855ᵗ decay, each
  later δ updating them with that weight.

## 10. Three-algorithm comparison (run_experiments.py: comparison)

Data: `comparison_summary.csv`, `comparison_sample_states.csv`.
Figures: `comparison_disagreement_qlearning.png`,
`comparison_disagreement_sarsa.png`, `comparison_learning_curves.png`.
Canonical agents: VI γ=0.95; QL sparse/exponential seed 7;
SARSA λ=0.7 (best from §9) seed 7. Same map, same sparse reward.

| metric | VI | Q-Learning | SARSA(0.7) |
|---|---|---|---|
| runtime | **0.11 s** | 8.5 s | 54.8 s |
| env samples to 90% success | — (model access) | 612,908 | **526,282** |
| model file | **103 KB** (V+π) | 271 KB | 274 KB |
| eval return / steps | **191.6 / 43.4** | 190.0 / 45.6 | 190.6 / 44.4 |
| V^π(start) (exact, model) | **9.60 = V\*** | 5.93 | 7.83 |
| VI agreement (visited) | — | 51.4% | 51.9% |
| median action gap on disagreements | — | 4.23 | 4.39 |
| penalty entries / eval ep | 0.260 | 0.236 | **0.188** |
| penalty-adjacent agreement | — | 67.3% | 63.7% |

Analysis points:
- **Runtime vs samples (report Q3):** VI is ~77× faster than QL *because* it
  consumes the exact transition model (2796×4 outcome distributions);
  the model-free methods pay instead in experience: ~500–600k env steps.
  SARSA's trace loop costs ~6× QL's wall-clock per run but buys ~14% fewer
  samples to 90% — compute vs sample efficiency trade.
- **Exact policy values beat rollouts for grading policies:** value loss
  V*(start) − V^π(start) is 3.67 (QL) vs **1.77 (SARSA(0.7))** despite
  near-identical raw agreement (~51%) — agreement % is a weak metric
  (§6 near-ties); the λ=0.7 traces yield a genuinely better policy.
- **On-policy risk behavior (report Q2):** SARSA enters penalty cells
  least often in evaluation (0.188/ep vs QL 0.236, VI 0.260) and shows
  *lower* penalty-adjacent agreement with VI (63.7% vs 67.3%) — i.e. its
  deviations near danger are systematic (safer detours), not noise: during
  on-policy training its own ε-greedy slips into penalty cells are priced
  into Q. Both agents agree with VI *more* near penalties than globally
  (67/64% vs 51%) — the action gaps there are sharp, not near-ties.
- **Where disagreements live:** only ~19% of disagreements are near-ties
  (gap < 0.5); the rest sit in rarely-visited, off-path states whose large
  gaps barely affect V^π(start).

Three sample states (`comparison_sample_states.csv`, report Q5):
1. **(2,1), k=0, phase 4 — penalty-adjacent, 669 visits.** VI: right;
   SARSA: left (gap 3.13). The cell sits beside penalty (3,1); SARSA
   learned the detour that keeps slip-risk away from the −10 cell — the
   on-policy safety margin, exhibit A for Q2.
2. **(15,15), k=1, phase 1 — inside the chamber, 1418 visits.** VI: down;
   QL: right; gap **exactly 0.000** — two symmetric 2-step routes to the
   goal; the "disagreement" is a pure tie-break formality.
3. **(14,12), k=1, phase 0 — standing ON the gate, 5 visits.** VI: right
   (the door is one step away); QL: down; gap 19.7. Agents rarely *stand*
   on the gate (they pass through), so with 5 visits the estimate never
   learned the doorway — large errors concentrate where data is scarce.
- Disagreement maps: agreement (blue) hugs the mission corridors where
  visits concentrate; red scatters off-path; chamber blank at k=0
  (unreachable). Spatial pattern = the visit-density story in one figure.

## 11. Transfer target maps (environments/generator.py: perturb_map)

Maps: `environments/maps/target_similar.json`, `target_different.json` —
derived deterministically from the source (seed = base_seed·100000 +
offset {1000, 2000} + attempt; both succeed at attempt 0), BFS-validated,
chamber/door/gate/start never touched. Walls are *moved* (removed + added
elsewhere), so the wall count and the ≥15% constraint are preserved.

| | similar | different |
|---|---|---|
| walls moved | 11/56 = **19.6%** (spec 15–20%) | 20/56 = **35.7%** (spec ≥35%) |
| key | unchanged (12,10) | **moved to (5,15)** |
| penalty cells | unchanged (6) | **+3 → 9** (new: (3,10), (10,8), (15,3)) |
| start/door/goal/gate | unchanged | unchanged |

Notes:
- In the different target the new key (5,15) again sits beside a penalty
  cell (5,16) — the risky-pickup motif recurs, useful for the transfer
  analysis (does transferred knowledge steer the agent to the *old* key
  area = negative transfer?).
- Old key area (12,10) becomes a plain corridor in the different target —
  the transferred Q-table's "go to (12,10)" gradient is exactly the
  negative-transfer candidate for §12.
- 6 new unit tests (48 total): targets reproduce the committed files,
  fractions within spec bands, mission cells preserved/moved as required,
  BFS validity + chamber still sealed in both.

## 12. Transfer learning (run_experiments.py: transfer)

Data: `transfer_training.csv`, `transfer_summary.csv`,
`transfer_negative_case.csv`. Figures: `transfer_curves_{similar,different}.png`,
`transfer_beta_{similar,different}.png`. 6 scenarios × 2 targets × 3 seeds,
2500 episodes, ε 0.3→0.05 (all scenarios share the schedule; zero-init Q +
random tie-breaking explores like ε≈1 anyway, so scratch is not handicapped).
Jumpstart = greedy evaluation of the initial table *before* training
(200 episodes). Unchanged 3×3 neighborhoods: similar 119/222 cells (53.6%),
different 61/213 (28.6%) → selective transfers 1356 / 672 of 2724 states.

Seed-averaged results, all six scenarios per target (spec's four: scratch /
full / scaled with β ∈ {0.25, 0.5, 0.75} / selective). Target optimum:
similar 192.5, different 179.8:

| target | scenario | jumpstart | episodes to 90% | final return |
|---|---|---|---|---|
| similar | scratch | 0% / −3670 | 718 | 181.0 |
| similar | full | **100% / −139** | **99** | 184.9 |
| similar | scaled β=0.25 | 100% / −139 | **99** | **190.2** |
| similar | scaled β=0.50 | 100% / −139 | **99** | 188.9 |
| similar | scaled β=0.75 | 100% / −139 | **99** | 184.5 |
| similar | selective | 0% / −2777 | 132 | 185.4 |
| different | scratch | 0% / −3634 | 849 | 158.6 |
| different | full | **0% / −3573** | 691 | 161.2 |
| different | scaled β=0.25 | 0% / −3573 | 500 | 150.8 |
| different | scaled β=0.50 | 0% / −3573 | **478** | 162.5 |
| different | scaled β=0.75 | 0% / −3573 | 622 | 159.0 |
| different | selective | 0% / −3539 | 519 | 152.3 |

β=0.75 reading: closest to full transfer, as expected — same 99-episode
takeoff on similar but the lowest scaled final there (184.5 ≈ full's 184.9,
stale magnitudes persist), and on different it is the slowest of the three
β values (622 vs 478/500) because the stronger wrong prior resists
correction. Monotone in β on the different target's adaptation speed —
clean evidence that β controls "transfer intensity" through update
dynamics, not through the initial greedy policy (jumpstart identical for
all β).

Findings (report Q6):
- **Similar target: unambiguous positive transfer.** Full/scaled tables give
  100% jumpstart success (slow, detouring around the 11 moved walls, hence
  the −139 return) and hit the 90% success mark at the first measurable
  window (ep 99) — ~7× faster than scratch (718).
- **β does not change the jumpstart** — argmax is scale-invariant, so all
  β>0 share the transferred greedy policy exactly (identical −138.56 in the
  CSV). β only modulates *update dynamics*: smaller β = weaker prior that
  TD updates overwrite faster.
- **…which is why β=0.25 wins final performance on the similar target**
  (190.2 vs full's 184.9): full-magnitude stale values linger; the
  0.25-scaled table keeps the ordering (jumpstart) but re-fits cleanly.
- **Different target: transfer ≠ free lunch.** Full-transfer jumpstart is
  0% success, barely better than scratch (−3573 vs −3634) — the transferred
  policy marches to the *old* key location. Learning speed still improves
  modestly (691 vs 849), because the with-key half of the table and
  unchanged corridors remain useful; scaled β=0.5 is best (478, −44%).
- **Negative-transfer case, with the full arc** (`transfer_negative_case.csv`,
  state (12,12), k=0, phase 0 — two cells from the old key, behind the
  penalty cell (12,11)):
  1. ep 0: transferred greedy = **left**, straight into the penalty cell
     toward a key that no longer exists (target optimum: right, gap 14.3);
  2. ep 100: q_left has *risen* to +13.5 — early bootstrapping from stale,
     still-optimistic neighboring values briefly **amplifies** the error;
  3. ep 500: reality extinguishes it — q_left collapses to −4.0 and the
     greedy action flips to the target-optimal **right**;
  4. ep 1000+: the converged policy no longer passes here; values flatten
     into a stale near-tie (greedy drifts to `up`). Correction happens
     where it matters, then the state is simply abandoned.
- Honest limitation: with the fixed 2500-episode budget, final returns on
  the different target (150–163) sit below its optimum (179.8) for *every*
  scenario including scratch — adaptation, not full convergence, is what
  the budget measures there.

## 13. GUI — "MazeMario" (gui/)

Tkinter, custom pixel design; six modules (theme / sprites / renderer /
controller / hud / app). No threads: a cooperative `after()` game loop
(~30 fps) drives env steps, animations and HUD. WATCH mode replays trained
policies (VI solved live per world; QL/SARSA tables loaded from
results/models); TRAIN mode runs a fresh learner with the config's ε
schedule — paced (animated) below 2×, fast-forward (whole headless episodes
per frame via agent.train) at 2–4×.

Spec-requirement coverage (report table material):
| spec control | GUI element |
|---|---|
| algorithm + source/target env selection | HERO BRAIN + SELECT WORLD menus |
| train vs evaluate mode | MODE: TRAIN / WATCH |
| start/stop/resume/reset/re-run | START, PLAY/PAUSE, RESTART (+STEP) |
| animation speed | 0.5–4× slider (≥2× = training fast-forward) |
| policy display toggle | POLICY overlay (greedy arrows for the agent's *current* key/phase, live during training) |
| live info: episode, steps, reward, ε, key, recent success | HUD: EP · STEPS n/cap · SCORE · ε · KEY · WIN (last-100) |

Event visibility (all spec events are diegetic animations): wall hit =
bump + red −5 popup; penalty = hero falls into a thorn-ringed pit, red −10;
key = sparkle burst, +50, HUD GOT!, door keyhole appears unlockable; locked
door = bump −2; door pass = wooden panel slides open; goal = +200, hearts,
COURSE CLEAR screen; timeout = TIME UP screen; **periodic gate = a dragon
that rises from its den on closed phases and retreats on open ones** (HUD
shows IN·n/OUT·n countdown) — the chosen dynamic feature is thus the most
visually prominent object in the game, satisfying "must be clearly shown in
the GUI".
- Pedagogical extra: selecting the World-1-trained Q table on World 3 shows
  negative transfer live (the hero marches toward the removed key).
- Watching TRAIN mode: ~60+ eps/s fast-forward reached 56% win at ep ~880
  and 89% by ep ~1100 in a live session — matches the headless curves (§7).

## Report-question tracker

1. **MDP + Markov property** — material ready (§1, §3).
2. **on- vs off-policy near danger** — answered (§10): SARSA's measured
   safety margin (penalty entries 0.188 vs 0.236/ep; sample state 1).
3. **Why VI needs the model** — answered (§10): 0.11 s with the model vs
   ~500–600k env samples without; advantages/limits table material ready.
4. **Best λ** — answered (§9): λ=0.7, with numbers.
5. **Three states where model-free ≠ VI** — answered (§10): three
   mechanism-distinct samples with Q* gaps and local-structure analysis.
6. **Transfer: similar vs different target** — answered (§12): jumpstart /
   speed / final tables, β analysis, negative-transfer case with correction arc.

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
