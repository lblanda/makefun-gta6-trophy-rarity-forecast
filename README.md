# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lblanda/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lblanda/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — a week after launch — as five
quantiles per trophy, scored by pinball loss on the logit scale.

## Run it

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (else /input),
                  # writes $MAKEFUN_OUTPUT_DIR (else /output)
```

`main.py` (Python 3.13, standard library only, no network) reads a case
file — the game's trophies and every valid read of Exophase's trophy page
before the checkpoint — and writes one forecast file per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Method

An Exophase earn rate is earners / players-tracked, and both sides move
after launch, so the program models both. Players join the tracked
population over time (fitted as `log N` quadratic in log hours since the
launch anchor); a player of clock-age `a` has earned trophy `i` with
probability `A_i · Φ((ln a − ln μ_i) / σ)`, a lognormal CDF in clock age.
The rate the page shows is the mixture of that earn curve over the tracked
population's age distribution. `(A_i, μ_i)` are fitted per trophy against
the case history, with priors in `trophies.json` steadying trophies whose
history barely moves; the spread combines the fit residual, a base
uncertainty, and a random-walk term growing with the horizon. Full detail
in [MODEL.md](MODEL.md), including synthetic validation: the program beats
both house baselines in every tested scenario (matched family, wrong
family, and player growth 3× faster than forecast after the checkpoint).

---

<sub>Entered on [make.fun](https://make.fun/?ref=3zwarjv0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
