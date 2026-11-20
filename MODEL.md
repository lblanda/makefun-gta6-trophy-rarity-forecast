# Method: forecasting GTA VI trophy earn rates at 2026-11-26T00:00:00Z

## What is forecast

For each of the 55 PlayStation 5 trophies of Grand Theft Auto VI, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase — the percent
of the players Exophase tracks for the game who have earned the trophy — at
2026-11-26T00:00:00Z, a week after launch. Scored by mean pinball loss on
the logit scale against the first valid read of the page in the hour from
the target.

## The model: a cohort mixture

An Exophase rate is a ratio, earners / players-tracked, and both sides move
after launch. A trend fit on the rate alone cannot tell "players are
progressing" from "new players are diluting the rate", so the program models
both sides separately.

**Players.** `N(t)`, the players tracked, is fitted as
`log N = a + b·ln h + c·(ln h)²` with `h` hours since the launch anchor
(New Zealand midnight, 2026-11-18T11:00:00Z — the first players; the house's
first read, 23:50Z that day, already shows 638 tracked). The fit is
extrapolated to the target, giving the join-time density `N'(s)` there.

**Earners.** A player of clock-age `a` (hours since they joined the tracked
population) has earned trophy `i` with probability

    F_i(a) = A_i · Φ((ln a − ln μ_i) / σ)

a lognormal CDF in clock age: `A_i` is the eventual share of players who
ever earn the trophy, `μ_i` the median clock-age at which an eventual
earner has earned it, `σ` shared across trophies (chosen from 0.7, 1.0,
1.4 by pooled fit). Clock age stands in for cumulative play: players play
a few hours a day, so a trophy that takes 10 play-hours has a median
clock-age of tens of hours.

**The rate.** The rate the page shows at time `t` is the mixture over the
tracked population's ages:

    r_i(t) = (1 / N(t)) · ∫ N'(s) · F_i(t − s) ds,   s in [t0, t]

computed on a 36-point log-spaced age grid. `(A_i, μ_i)` are fitted per
trophy by grid search (plus local refinement) on the weighted logit error
of the mixture against the case history — weights are binomial precision
`min(N·p·(1−p), 5000)` — with priors from `trophies.json` steadying trophies
whose history barely moves. The median at the target is the fitted mixture
read at the target.

## Why this, and not a trend

Launch night shows the dilution plainly: the page sat unchanged at 638
players tracked from 23:48 to 04:00, then one refresh took it to 3,868 and
every rate but the 100% one fell. A line fitted to `logit(rate)` against
log time cannot see that coming and cannot separate it from progression.
The mixture can: an early-story trophy whose old cohort is finishing
chapter 1 can still fall if the tracked population is growing faster than
its earners, and a late-story trophy can rise steeply once its cohort
reaches it.

## Priors (`trophies.json`)

Per-trophy `(A0, μ0)`, calibrated 2026-11-20 from the case files and the
public Exophase page at 2026-11-20T13:07Z: chapter 1 at 11.10%, chapter 2
at 1.44%, chapters 3–4 and the final mission at 0.02–0.03% (a handful of
players: the pre-launch cohort) with ~10k tracked. The story is long — the
platinum needs ~300 hours of play — so chapter medians sit at tens to
hundreds of clock-hours and the platinum / 100% trophies are effectively
unreachable in week one. The fit moves every trophy off its prior as the
case history allows; the priors only steady thin-data trophies. Prior
strength is worth a few reads (`λ = 0.05`, log-width 0.8).

## Spread

`s_i = sqrt(0.40² + s_fit² + (0.30·√d)²)`, floored at 0.55 and capped at
1.6, where `s_fit` is the trophy's fit residual on the logit scale and `d`
the horizon in days (4.52 for the final). Quantiles are
`logit⁻¹(logit(median) + z_τ·s_i)` with `z` the normal quantiles, rounded
to 4 decimals half-up.

## Fallbacks (validity first)

- Flat player counts (the practice cases, whose history never moves):
  no signal of any move, so the forecast is the house baseline — last
  rate with a random-walk spread. A curve fitted to a flat window is not
  identified: any slope passes through it and extrapolates wildly.
- A trophy at ≥ 99.5% in the last read (the opening mission, "Complete
  Gone Fishing", which the page shows at 100%): median 99.9, spread 0.3.
- Any unexpected failure in a case falls back to the house baseline, so
  the case stays valid.

## Validation (synthetic, `~/workspace/makefun/sim_test.py`)

Fake final-like cases (checkpoint 2026-11-21T11:30Z, target 2026-11-26)
built from a known ground truth, scored with the exact pinball loss:

| scenario | this program | house baseline | house curve baseline |
|---|---|---|---|
| A — matched family | **0.169** | 1.246 | 0.291 |
| B — wrong family (Weibull earn curves, saturating N) | **0.195** | 0.975 | 0.295 |
| C — player growth 3× faster after the checkpoint | **0.319** | 0.672 | 0.449 |

A direct (no-dilution) trend blend was tested and dropped: it scored
worse in all three scenarios (0.189 / 0.199 / 0.465), because it
extrapolates the rising pre-checkpoint rate trend without the dilution.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (else /input),
                  # writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Standard library only, no network, deterministic (two runs on the same
case are byte-identical), ~1.3 s for a 340-read final-like case (limit 8 s).
