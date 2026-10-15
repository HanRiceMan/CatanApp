# Step 3B-12 — Multi-Round On-Policy City PPO

## Scope and reproducibility

The parent is `Route-Isolated-City-Post1`. L1 Victory Plan, L2 Priority,
the base Plan–Action Actor, Champion, and Robber PPO are frozen. Only the
City-specific residual Adapter and City Critic are optimized. Investment
delegation stays at 0%; City multi-choice training delegation is 50%, while
evaluation delegation is 10%. No reward, executor, candidate mask, or
PlayerTrade/ImmediateWin arbitration changes were made.

Round 1 uses Rule/Champion/Mixed seed bases 36200000/36201000/36202000;
Round 2 uses 36300000/36301000/36302000. Independent fixed validation uses
36500000/36501000/36502000. The final 288-game paired evaluation uses
36400000/36401000/36402000 (96 seeds per profile, 24 per seat/profile).
Delegation and family-sampling RNG are external to GameState RNG.

Each round uses a distinct on-policy rollout, a preflight before constructing
optimizers, one two-epoch guarded PPO pass, and a one-shot update marker.
Round 2 carries the accepted Round 1 Adam state. An update rejected by the KL,
entropy, family-collapse, or nonfinite guards restores both model weights and
the prior optimizer state. The PPO configuration is actor LR 1e-4, critic LR
3e-4, batch 64, clip 0.1, entropy coefficient 0.005, gamma 0.99, and GAE
lambda 0.95. Existing terminal and variable-duration semi-MDP handling is
preserved. Validation critic error uses realized discounted macro returns to
terminal and is not an optimizer target.

## On-policy training

| Round | Games | City transitions | City | Road | BuyDev | BankTrade | EndTurn | Update |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 240 | 264 | 85 | 12 | 28 | 65 | 74 | Accepted |
| 2 | 228 | 267 | 70 | 15 | 31 | 59 | 92 | Accepted |

Both rounds balance Rule/Champion/Mixed and all four seats. The first has 80
games/profile and 60 games/seat; the second has 76 games/profile and 57
games/seat. There is no Round 1 data reuse in Round 2. All sampled states
have at least two executable City-route families. Preflight replayed frozen
latent, old action distribution, old log-probability, and old Critic value;
checked family masks, transition continuity, learner-step macro duration,
discounted macro reward, true terminal bootstrap, truncation exclusion, and
nonfinite rewards. Nondelegated and protected actions remain in reward spans
but are not Actor samples.

| Round | Mean KL | Max KL | L3 entropy before→after | Critic training MSE before→after | City Adapter Δ |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000379 | 0.0000769 | 1.22861→1.22894 | 0.66920→0.65141 | 0.03519 |
| 2 | 0.00000799 | 0.0000146 | 1.22989→1.23012 | 0.67660→0.67359 | 0.01459 |

Both rounds had clip fraction 0 and nonzero City Adapter gradients. L1/L2,
the base Actor, Champion, and Robber remained unchanged. Fixed Investment
probabilities over 218 states are exactly equal to the parent at each version.
The trainable City Adapter has 2,727 parameters and the City Critic 33,857.
Mean Adapter gradient norms were 0.414 and 0.307; mean actor losses were
0.03586 and 0.01381. The respective critic losses were 0.65106 and 0.66484.
Data SHA-256 values and the carried optimizer-state SHA-256 values are stored
in each Round manifest. The two rollout SHA values differ.

The 24-game independent validation set yielded 34 City decision states.
Realized-return Critic MSE was 0.78414 at Post1, 0.75700 after Round 1, and
0.75420 after Round 2. This small set is a diagnostic, not a performance
qualification. On these fixed states, City policy total-variation distance
was 0.00393 (Post1→Round 1) and 0.00182 (Round 1→Round 2), with one top-family
flip each. Post1→Round 2 TV was 0.00548 and two top families flipped. Critic
prediction versus realized-return correlation on the same 34 states was
0.157, 0.428, and 0.588; predictions remained narrowly ranged relative to
high-variance realized outcomes. The updates changed probabilities, but the
shift is small.

Observed (noncausal) City-selected versus City-deferred mean Advantage was
0.326 versus 0.026 in Round 1, and 0.034 versus 0.037 in Round 2. Terminal
transitions comprised 135/264 and 127/267; terminal absolute Advantage share
was 0.558 and 0.525. Neither comparison identifies the causal effect of
building City.

## Held-out evaluation

The precommitted 288 paired seeds were completed for every variant, with
Rule/Champion/Mixed 96 each and all four learner seats balanced. The 0%
variant matched the existing Safety controller on all 288 Action traces,
final GameStates, and GameState RNG counters. All 1,152 variant games were
terminal with no truncation; rollout assertions observed zero illegal Actions,
Executor failures, or arbitration RNG consumption. Protected PlayerTrade and
ImmediateWin precedence remained in place.

| Variant (City delegation) | Wins / 288 | Mean VP | Mean rank | Mean learner steps | City builds | Same-turn City / opportunities | City not built same turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safety (0%) | 137 | 7.9618 | 1.9948 | 73.625 | 458 | 376/412 (91.26%) | 36 |
| 3B-11 Post1 (10%) | 137 | 7.9410 | 2.0017 | 73.677 | 453 | 374/429 (87.18%) | 55 |
| 3B-12 Round 1 (10%) | 137 | 7.9410 | 2.0017 | 73.677 | 453 | 374/429 (87.18%) | 55 |
| 3B-12 Round 2 (10%) | 137 | 7.9410 | 2.0017 | 73.677 | 453 | 374/429 (87.18%) | 55 |

Each 10% variant made 42 actual delegated City decisions (39 distinct City
opportunities). All three selected the same family counts: City 13, EndTurn
14, BankTrade 9, BuyDev 5, and Road 1. Their full Action traces matched in
all 288 paired games. Construction/Investment totals likewise matched:
Road 1,669; BuyDev 597; BankTrade 981; EndTurn 4,876. The same-turn City
rate for delegated opportunities was 19/39 (48.72%). This does not establish
that delegation caused the difference from Safety because trajectories can
diverge and there are only 42 delegated decisions.

Profile-stratified paired bootstrap (5,000 resamples) gave Safety→Round 2
mean ΔVP = −0.0208, two-sided 95% CI [−0.0799, +0.0347]; mean Δrank =
+0.00694, CI [−0.0208, +0.0347]; and mean Δsteps = +0.0521, CI
[−0.399, +0.535]. Post1→Round 1, Round 1→Round 2, and Post1→Round 2 had
exactly zero paired differences for VP, rank, steps, and Action traces, so
their bootstrap intervals are [0, 0]. The identical outcomes are an empirical
result of this fixed 10% evaluation, not evidence that the learned
probabilities are identical.

## Decision and next step

Technical update: **accepted in both rounds**. Behavioral City improvement:
**not demonstrated**. The City Adapter changed on fixed states but too little
to alter any sampled Action in this 288-game 10% evaluation. The small
Safety–City VP gap is uncertain, but the lower same-turn City rate is a reason
not to expand City delegation. Do not promote Round 2 into the Champion or
increase delegation. Preserve Round 1/Round 2 as independent experimental
artifacts; retain 3B-11 Post1 as the comparison parent.

For Step 3B-13, inspect why the route-isolated gradient produces such small
policy movement despite 531 on-policy transitions: fixed-state action-logit
margins, terminal versus nonterminal credit, Critic calibration, and the
effective number of City choices after protected preemption. Decide whether
to change the credit-assignment/Actor training design or include more than
one Route, while continuing to guard Investment probabilities. Do not simply
repeat the same PPO update indefinitely, add a City bonus, or unfreeze L1/L2
without a new experiment design.

## Verification and artifacts

`run_multi_round_city_step3b12.py` is restartable by phase and seed shard.
`test_multi_round_city_step3b12.py` covers disjoint seed ranges, macro return
and truncation handling, and typed-Action Safety parity before serialization.
The evidence directory contains both training rollouts, preflights, one-shot
attempts/results, independent validation, manifests, weights, Adam states,
their SHA-256 manifest, and compact 288-game paired outcomes. Final Backend
test run: **420 passed, 1,172 subtests passed**, with two pre-existing
third-party deprecation warnings and no failures.
