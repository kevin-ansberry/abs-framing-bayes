# Did ABS change the value of framing and umpire zones?

A Bayesian look at called strikes before and after MLB's Automated Ball-Strike (ABS)
challenge system arrived in 2026.

> **Independent project.** Built from public data only (Statcast via `pybaseball` and the MLB
> Stats API). It uses no proprietary or club data and is not affiliated with any team or with MLB.

## The question

On pitches a batter takes, how likely is a called strike given where the pitch crossed the
plate, and how much of that comes from the umpire, the catcher (framing), the pitcher and
the batter? Did those umpire and catcher effects shrink between 2025 and 2026, when batters,
catchers and pitchers gained the ability to challenge calls? And, separately, who actually
wins challenges?

This matters for roster and development decisions: if the market price of framing was set in
the pre-ABS era, a change in how much framing is worth changes what a catcher is worth.

## Methods

| Technique | Where it is used |
|---|---|
| Bayesian inference (Stan) | `stan/called_strike.stan`, `stan/challenge.stan`; weakly informative priors, full posteriors |
| Mixed / hierarchical effects | Crossed random effects for umpire, catcher, pitcher, batter with partial pooling, with separate umpire and catcher scales per season |
| Spatiotemporal | A 2D strike-zone surface (Hilbert-space Gaussian process) plus a season-deviation surface and a month-to-month random-walk drift |
| Model checking | Posterior predictive checks, LOO-CV, simulation-based recovery of known parameters |

Model sketch (called strikes, `bernoulli_logit`):

```
a0 + (beta_d + [2026] * beta_dd) * edge_d + f_shared(x, z) + [2026] * f_delta(x, z) + tau[month]
   + u_umpire[season] + u_catcher[season] + u_pitcher + u_batter
```

`edge_d` is the distance outside the nominal zone edge. This parametric slope carries the coarse
shape of the zone so the Gaussian process only models smooth deviations from it; without it, the
sampler struggles because the surface must swing between large positive and negative values.

The estimands of interest are the difference in posterior scale of the catcher and umpire
effects (`d_sigma_c`, `d_sigma_u`), the change in edge sharpness (`beta_dd`), and the delta surface,
which shows where in the zone calls changed.

Pitch coordinates are rescaled so the nominal zone edge sits at +/-1 on both axes
(horizontal by plate width, vertical by each batter's own zone top and bottom).

## Status (read this first)

- **Tested:** the HSGP basis (checked against the exact kernel), coordinate normalization,
  and the data-prep pipeline, all on synthetic data (`pytest`).
- **Not yet run:** the Stan models have not been compiled or sampled, the Statcast and umpire
  pulls have not been run, and no results exist yet. The challenge model's input schema is an
  assumption about what ABS challenge data looks like; check what is publicly available.
- Run `scripts/simulate.py` and fit the model to the synthetic data first. The simulated
  catcher effect scale drops from 0.30 to 0.20, so `d_sigma_c` should be centered near -0.10.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

pytest                                                     # unit tests

# 1. Validate on synthetic data with known truth
python scripts/simulate.py --n 60000
python scripts/03_fit_called_strike.py --data data/processed/pitches_sim.parquet --n-max 30000 --chains 2

# 2. Real data (needs internet; confirm season dates first)
python scripts/01_pull_statcast.py
python scripts/02_build_dataset.py
python scripts/03_fit_called_strike.py --n-max 100000
python scripts/05_report.py

# 3. Challenge model, once data/raw/abs_challenges.csv exists (see script docstring)
python scripts/04_fit_challenge.py
```

## Layout

```
src/absbayes/   hsgp.py (spatial basis), features.py (cleaning, Stan data)
stan/           called_strike.stan, challenge.stan
scripts/        01 pull, 02 build, 03 fit, 04 challenge fit, 05 figures, simulate
tests/          basis accuracy, normalization, pipeline checks
```

## Modeling notes and limitations

- The spatial surface is evaluated once on a 51 x 51 grid and bilinearly interpolated to each pitch with a sparse matrix, so cost per pitch stays small as data grows (a dense per-pitch basis stalled the sampler at 60k pitches).
- The HSGP approximation needs the domain half-width `L` well beyond the data (here 4 vs. data
  within +/-2.5) and enough basis functions for the length-scale; both are covered by tests.
- Plate location is measured, not true, but it is the same measurement in both seasons, so the
  comparison is fair even if absolute accuracy is imperfect.
- Statcast stores the *final* call, after any ABS challenge (checked on ~320 overturned pitches:
  every overturned pitch inside the zone is a called strike, and overturned balls and strikes split
  at roughly a ball's width from the zone edge). The pipeline therefore fits two outcomes:
  `--outcome final` (the call that stood) and `--outcome original` (the umpire's first call,
  reconstructed as the final call flipped on overturned challenges). Comparing the two shows how
  much the challenge system corrects; the original-call model puts 2025 and 2026 on equal footing.
- Challenge flags come from MLB's game feed (`scripts/06_pull_challenges.py`); the feed schema was
  inspected on a handful of games only.
- 2026 is the first ABS season, so batter and catcher behavior is still settling. Treat the
  month drift term and any early-season differences with caution.
- A season is a poor "control": other changes between 2025 and 2026 are absorbed into the
  season effects. This is an association, not a clean causal estimate of ABS.

## Results

_To be added after running the models: surface plots, the posterior for `d_sigma_c` and
`d_sigma_u`, and a ranked, uncertainty-aware list of catchers._

## Results so far (exploratory)

Real-data run: 30,000 pitches, 2 chains, 600 warmup and 600 samples, original-call outcome.

- `d_sigma_c` +0.076 (90% interval -0.079 to +0.233) and `d_sigma_u` -0.090 (-0.231 to +0.037):
  intervals include zero, so no detectable change in the spread of catcher or umpire effects.
- The zone-change map (`figures/change_meaningful.png`) shows more called strikes at the top and bottom
  edges and fewer at the left and right edges. Cells are outlined only where there is at least a 90%
  probability that the change exceeds 0.05. Intervals are pointwise (no multiplicity correction), the GP
  smooths neighboring cells together, and the 2025 and 2026 samples differ in composition and timing,
  so treat this as a hypothesis for a larger sample, not a finding.

## Status and limitations

- The model was validated on simulated data with known parameters (`scripts/simulate.py`):
  catcher and umpire scales and the edge slope are recovered at 15k, 30k and 60k pitches.
- The real-data run so far uses a two-week 2026 sample plus 2025 comparison pitches, about 285
  pitches per umpire per season. Umpire and catcher scale estimates are therefore wide and
  should be read as a pipeline check, not a finding.
- Statcast stores the *final* call. The original call is reconstructed from the final call and
  the overturned flag from the game feed (`--outcome original`).
- The challenge model (`stan/challenge.stan`) is included but has not been run on real data; only
  12 pitcher challenges exist in the sample, so the pitcher role is not identifiable.
- An earlier shorter run (300 warmup and 300 samples): the effect-scale
  differences `d_sigma_c` (+0.06) and `d_sigma_u` (-0.08) have 90% intervals that include zero. Sampling
  is not converged: the GP length scale `rho_s` has R-hat 1.18 and an effective sample size near 10,
  nearly every iteration hits the treedepth cap, and 3.7% of pitches have a bad LOO Pareto-k. A longer
  run is the next step; treat the 2026-minus-2025 zone map as exploratory until then.
- A wider 2026 date range will tighten every estimate; the pipeline scales by changing the pull dates.

## Reproduce

```
python scripts/01_pull_statcast.py
python scripts/06_pull_challenges.py
python scripts/02_build_dataset.py
python scripts/03_fit_called_strike.py --outcome original --chains 2
python scripts/05_report.py --fit-dir outputs/called_strike_original
```
