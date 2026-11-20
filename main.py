"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Model (cohort mixture, standard library only, no network, deterministic)
----------------------------------------------------------------------------
An Exophase earn rate is earners / players-tracked.  Both move after launch:

  * players join the tracked population over time, N(t); a player who joined
    at time s has age (t - s) at time t, and new joiners start at zero
    progress, which dilutes every rate;
  * a player of age a has earned trophy i with probability
    F_i(a) = A_i * Phi((ln a - mu_i) / sigma): a lognormal CDF in clock age.
    A_i is the eventual share of players who ever earn the trophy, mu_i the
    median clock-age at which an eventual earner has earned it, sigma shared.

The rate the page shows at time t is therefore the mixture

    r_i(t) = (1 / N(t)) * integral N'(s) * F_i(t - s) ds,   s in [t0, t],

with t0 the launch anchor (New Zealand midnight, 2026-11-18T11:00:00Z, the
first players).  The program fits N(t) as log N = a + b ln h + c (ln h)^2
(h = hours since t0), extrapolates it to the target, fits (A_i, mu_i) per
trophy by a grid search on the weighted logit error of the mixture against
the case history (weights ~ binomial precision, priors from trophies.json
steady trophies whose history barely moves), and reads the mixture at the
target for the median.  The spread combines the fit residual, a base
uncertainty and a random-walk term growing with the horizon.

Why this beats a trend fit: a straight line on logit(rate) against log time
cannot tell "players are progressing" from "new players are diluting the
rate".  Launch night shows the dilution plainly: the page sat unchanged at
638 players tracked from 23:48 to 04:00, then one refresh took it to 3,868
and every rate but the 100% one fell.  The mixture separates the two forces
and recombines them with the extrapolated player growth, so an early-story
trophy whose old cohort is finishing chapter 1 can still fall if the
tracked population is growing faster than its earners, and a late-story
trophy can rise steeply once its cohort reaches it.

Calibration (from the case files and the public Exophase page, 2026-11-20):
launch night chapter 1 sat at 1.25% with 638 tracked; by 2026-11-20T13:07Z
it was 11.10%, chapter 2 1.44%, chapters 3-4 and the final mission 0.02-0.03%
(2-4 players: the pre-launch cohort) with ~10k tracked.  The story is long
- the platinum needs ~300 hours of play - so chapter medians sit at tens to
hundreds of clock-hours and the platinum / 100% trophies are effectively
unreachable in week one.  Those facts live in trophies.json as priors; the
fit moves every trophy off its prior as the case history allows.

Fallbacks keep every case valid: flat player counts (the practice cases,
whose history never moves) fall back to the house baseline, because a curve
fitted to a flat window is not identified; any unexpected failure falls back
to the house baseline (last rate, random-walk spread).

Run: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input) and writing
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
HERE = Path(__file__).resolve().parent

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch anchor: the first players (New Zealand midnight) could start at
# 2026-11-18T11:00:00Z; the house's first read, 23:50Z that day, already
# shows 638 players tracked and nonzero rates.
T0 = datetime(2026, 11, 18, 11, 0, 0, tzinfo=timezone.utc)

# Grid for the per-trophy fit.  A is the eventual earn share, mu the median
# clock-age in hours at which an eventual earner has earned the trophy.
A_GRID = [0.01, 0.015, 0.022, 0.033, 0.05, 0.075, 0.11, 0.16, 0.24, 0.35,
          0.5, 0.7, 1.0]
MU_GRID = [2.0, 3.5, 6.0, 10.0, 17.0, 30.0, 50.0, 85.0, 140.0, 240.0,
           400.0, 650.0, 1000.0, 1500.0]
SIGMA_GRID = [0.7, 1.0, 1.4]

# Prior strength: a 2x move off the prior in A or mu costs LAMBDA_PRIOR
# times (ln 2 / 0.8)^2 in mean-squared-logit units - worth a few reads.
LAMBDA_PRIOR = 0.05
PRIOR_LOG_WIDTH = 0.8

# Spread: base uncertainty, fit residual, and a random-walk term in horizon.
SPREAD_BASE = 0.40
SPREAD_HORIZON = 0.30
SPREAD_MIN = 0.55
SPREAD_MAX = 1.6

MAX_READS = 48
MAX_AGE_HOURS = 1500.0
AGE_POINTS = 36

# Fallback priors by trophy class, used when trophies.json cannot be read.
CLASS_PRIORS = {
    "opening": (0.99, 1.0), "chapter1": (0.92, 80.0), "chapter2": (0.85, 200.0),
    "chapter3": (0.75, 400.0), "chapter4": (0.65, 600.0), "final": (0.5, 700.0),
    "story_early": (0.6, 120.0), "story_mid": (0.55, 300.0),
    "stranger": (0.35, 400.0), "strands": (0.2, 500.0), "easy": (0.45, 120.0),
    "misc": (0.25, 350.0), "economy": (0.2, 400.0), "grind": (0.15, 400.0),
    "grind_hard": (0.08, 550.0), "completion": (0.03, 1200.0),
    "platinum": (0.02, 900.0),
}


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as the starter does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_t0(t):
    return max(0.0, (t - T0).total_seconds() / 3600.0)


def load_priors():
    """Per-trophy (A0, mu0) priors from trophies.json, by trophy id."""
    try:
        data = json.loads((HERE / "trophies.json").read_text(encoding="utf-8"))
        out = {}
        for row in data["trophies"]:
            out[row["id"]] = (float(row["A0"]), float(row["mu0"]))
        if out:
            return out
    except Exception:
        pass
    return {}


def classify_prior(trophy):
    """Fallback prior from the trophy's tier, name and description."""
    name = trophy.get("name", "")
    desc = trophy.get("description", "")
    text = (name + " " + desc).lower()
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return CLASS_PRIORS["platinum"]
    if "100% completion" in text:
        return CLASS_PRIORS["completion"]
    if "chapter 1" in text:
        return CLASS_PRIORS["chapter1"]
    if "chapter 2" in text:
        return CLASS_PRIORS["chapter2"]
    if "chapter 3" in text:
        return CLASS_PRIORS["chapter3"]
    if "chapter 4" in text:
        return CLASS_PRIORS["chapter4"]
    if "final mission" in text and "for " not in text:
        return CLASS_PRIORS["final"]
    if "gone fishing" in text:
        return CLASS_PRIORS["opening"]
    if "final mission for" in text:
        return CLASS_PRIORS["stranger"]
    if "mission strands" in text:
        return CLASS_PRIORS["strands"]
    if "complete " in text and "mission" in text:
        return CLASS_PRIORS["story_mid"]
    if any(k in text for k in ("250 stunts", "gold medal in all", "100 locations",
                               "card sets", "50 challenges")):
        return CLASS_PRIORS["grind_hard"]
    if any(k in text for k in ("stunts", "races", "vehicles", "bank vaults")):
        return CLASS_PRIORS["grind"]
    if any(k in text for k in ("laundering", "spend $", "purchase all", "register")):
        return CLASS_PRIORS["economy"]
    if any(k in text for k in ("arcade game", "activity once", "fishing challenge",
                               "bowling", "meditate")):
        return CLASS_PRIORS["easy"]
    return CLASS_PRIORS["misc"]


def solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination; None if singular."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        s = m[i][3] - sum(m[i][j] * x[j] for j in range(i + 1, 3))
        x[i] = s / m[i][i]
    return x


def fit_player_growth(reads):
    """Fit log N = a + b ln h + c (ln h)^2 over the reads.

    Returns (a, b, c) or None when the counts are flat or the fit fails.
    """
    pts = []
    for r in reads:
        n = r["n"]
        h = r["h"]
        if n > 0 and h > 0:
            pts.append((math.log(h), math.log(n)))
    if len(pts) < 4:
        return None
    ns = [p[1] for p in pts]
    if max(ns) - min(ns) < math.log(1.05):
        return None
    # Normal equations for the quadratic least squares fit.
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for x, y in pts:
        v = [1.0, x, x * x]
        for i in range(3):
            atb[i] += v[i] * y
            for j in range(3):
                ata[i][j] += v[i] * v[j]
    coef = solve3(ata, atb)
    if coef is None:
        return None
    return coef


def growth_rate(h, coef):
    """d/dt log N at h hours since t0, per hour (h > 0)."""
    a, b, c = coef
    return (b + 2.0 * c * math.log(h)) / h


def log_n(h, coef):
    a, b, c = coef
    x = math.log(h)
    return a + b * x + c * x * x


def age_grid(max_age):
    """Log-spaced ages in hours, with the trapezoid width of each cell."""
    lo = math.log(0.25)
    hi = math.log(max_age)
    step = (hi - lo) / (AGE_POINTS - 1)
    ages = []
    widths = []
    for i in range(AGE_POINTS):
        a = math.exp(lo + step * i)
        ages.append(a)
        widths.append(a * step)
    return ages, widths


def age_weights(t_hours, coef, ages, widths):
    """Mixture weights over ages for a read at t_hours since t0.

    Weight of age a is N'(t - a) * da, normalised; ages whose join time
    predates the launch anchor get no weight.
    """
    w = []
    total = 0.0
    for a, da in zip(ages, widths):
        s = t_hours - a
        if s <= 0.05:
            w.append(0.0)
            continue
        rate = growth_rate(s, coef)
        if rate <= 0.0:
            w.append(0.0)
            continue
        val = math.exp(log_n(s, coef)) * rate * da
        w.append(val)
        total += val
    if total <= 0.0:
        # No usable growth curve: fall back to a uniform age distribution.
        return [1.0 / len(ages)] * len(ages)
    return [v / total for v in w]


def subsample(reads):
    """Evenly spaced reads, at most MAX_READS, always keeping the last."""
    if len(reads) <= MAX_READS:
        return reads
    step = (len(reads) - 1) / (MAX_READS - 1)
    idx = sorted({min(len(reads) - 1, int(round(i * step))) for i in range(MAX_READS)})
    return [reads[i] for i in idx]


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the scored five")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("no history")
    target = parse_time(case["target"])
    t_target = hours_since_t0(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    reads = []
    for r in history:
        t = parse_time(r["at"])
        reads.append({"h": hours_since_t0(t), "n": float(r["players_tracked"]),
                      "rates": r["rates"]})
    reads = subsample(reads)
    last_rates = reads[-1]["rates"]

    coef = fit_player_growth(reads)
    if coef is None:
        # The tracked population never moved (the practice cases, whose
        # history is one unchanged page): there is no signal of any move,
        # so the honest forecast is the last rate with a random-walk
        # spread - the house baseline.  A curve fitted to a flat window
        # is not identified: any slope passes through it and extrapolates
        # wildly, so do not let one speculate here.
        return baseline_case(case)

    max_age = max(MAX_AGE_HOURS, t_target * 1.2)
    ages, widths = age_grid(max_age)
    log_ages = [math.log(a) for a in ages]

    # Mixture value v(t; mu, sigma) = sum_j w_j(t) * Phi((ln a_j - mu)/sigma)
    # for every read and the target, for each sigma and mu.  Trophy-free.
    weights = [age_weights(r["h"], coef, ages, widths) for r in reads]
    w_target = age_weights(t_target, coef, ages, widths)

    def mixture_value(weights_row, mu, sigma):
        mu_log = math.log(mu)
        total = 0.0
        for w, la in zip(weights_row, log_ages):
            if w:
                total += w * phi((la - mu_log) / sigma)
        return total

    priors = load_priors()

    # Per trophy: observed rates, binomial-precision weights, prior.
    series = []
    for i, trophy in enumerate(trophies):
        obs = []
        wts = []
        for r in reads:
            rate = r["rates"][i]
            if rate is None:
                continue
            p = clamp(rate, 0.005, 99.995) / 100.0
            obs.append((r["h"], p))
            wts.append(min(r["n"] * p * (1.0 - p), 5000.0))
        a0, mu0 = priors.get(trophy["id"], classify_prior(trophy))
        series.append({"obs": obs, "wts": wts, "a0": a0, "mu0": mu0,
                       "last": last_rates[i]})

    # Pooled sigma: the value whose best per-trophy fits lose least overall.
    def eval_with_vals(obs, wts, a0, mu0, a, mu, vals):
        """Weighted mean squared logit error + prior penalty for one (A, mu),
        given the model's per-read values (which depend on mu, not A)."""
        err = 0.0
        wsum = 0.0
        for (h, p), wt, v in zip(obs, wts, vals):
            pred = clamp(a * v * 100.0, 0.005, 99.995) / 100.0
            d = math.log(p / (1.0 - p)) - math.log(pred / (1.0 - pred))
            err += wt * d * d
            wsum += wt
        if wsum <= 0:
            return None
        resid = err / wsum
        loss = resid + LAMBDA_PRIOR * (
            (math.log(a / a0) / PRIOR_LOG_WIDTH) ** 2
            + (math.log(mu / mu0) / PRIOR_LOG_WIDTH) ** 2)
        return loss, resid

    def model_vals(mu, sigma, wrows):
        return [mixture_value(w, mu, sigma) for w in wrows]

    def trophy_loss(obs, wts, a0, mu0, sigma, wrows, precomputed=None):
        best = None
        for mi, mu in enumerate(MU_GRID):
            vals = precomputed[mi] if precomputed is not None else \
                model_vals(mu, sigma, wrows)
            for a in A_GRID:
                r = eval_with_vals(obs, wts, a0, mu0, a, mu, vals)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], a, mu, r[1])
        return best

    # The mixture values depend on (sigma, mu), not on the trophy:
    # precompute them once per sigma for every read.
    precomputed = {}
    for sigma in SIGMA_GRID:
        precomputed[sigma] = [model_vals(mu, sigma, weights) for mu in MU_GRID]

    sigma_totals = {}
    for sigma in SIGMA_GRID:
        total = 0.0
        for s in series:
            if not s["obs"]:
                continue
            b = trophy_loss(s["obs"], s["wts"], s["a0"], s["mu0"], sigma,
                            weights, precomputed=precomputed[sigma])
            if b is not None:
                total += b[0]
        sigma_totals[sigma] = total
    sigma = min(SIGMA_GRID, key=lambda s: sigma_totals[s])

    rows = []
    for i, trophy in enumerate(trophies):
        s = series[i]
        last = s["last"]
        if last is not None and last >= 99.5:
            # Saturated: everyone tracked has it (the opening mission).
            median, spread = 99.9, 0.3
        elif not s["obs"]:
            median, spread = clamp(last if last is not None else 0.5), 1.2
        else:
            b = trophy_loss(s["obs"], s["wts"], s["a0"], s["mu0"], sigma,
                            weights, precomputed=precomputed[sigma])
            if b is None:
                median, spread = clamp(last if last is not None else 0.5), 1.2
            else:
                _, a_best, mu_best, resid = b
                # Local refinement around the coarse best.
                for dmu in (0.85, 1.0, 1.18):
                    mu2 = min(2000.0, max(0.5, mu_best * dmu))
                    vals2 = model_vals(mu2, sigma, weights)
                    for da in (0.85, 1.0, 1.18):
                        a2 = min(1.0, max(0.005, a_best * da))
                        r2 = eval_with_vals(s["obs"], s["wts"], s["a0"], s["mu0"],
                                            a2, mu2, vals2)
                        if r2 is not None and r2[0] < b[0]:
                            b = (r2[0], a2, mu2, r2[1])
                _, a_best, mu_best, resid = b
                median = a_best * mixture_value(w_target, mu_best, sigma) * 100.0
                median = clamp(median, 0.01, 99.99)
                s_fit = min(0.8, math.sqrt(max(resid, 0.0)))
                spread = math.sqrt(SPREAD_BASE ** 2 + s_fit ** 2
                                   + (SPREAD_HORIZON * math.sqrt(horizon_days)) ** 2)
                spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        lg = logit(median)
        q = [round4(inv_logit(lg + z * spread)) for z in Z]
        q = [clamp(v) for v in q]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def baseline_case(case):
    """House baseline: last rate, random-walk spread.  Always valid."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        q = [round4(inv_logit(lg + z * spread)) for z in Z]
        q = [clamp(v) for v in q]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = baseline_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
