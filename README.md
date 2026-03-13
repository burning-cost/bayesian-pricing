# bayesian-pricing

[![Tests](https://github.com/burning-cost/bayesian-pricing/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/bayesian-pricing/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/bayesian-pricing)](https://pypi.org/project/bayesian-pricing/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Hierarchical Bayesian models for insurance pricing thin-data segments.

## The problem

UK personal lines rating operates on multi-dimensional grids. A typical motor model has driver age × NCD × vehicle group × postcode area × occupation. That is potentially 4.5 million rating cells. With 1 million policies, most cells are either empty or contain fewer than 30 observations.

Standard approaches all fail at thin cells:

- **Saturated GLM**: one coefficient per cell. Overfits noise. A cell with 3 claims gets a relativity of 3/expected, which is meaningless.
- **Main-effects GLM**: forces multiplicativity. A young driver in a sports car has a rate exactly equal to young-driver-relativity × sports-car-relativity. Reality is super-multiplicative and the model cannot detect it.
- **Ridge/LASSO GLM**: uniform regularisation regardless of exposure. A cell with 5,000 policy-years gets the same shrinkage as one with 20 policy-years. Wrong.
- **GBM with min_data_in_leaf**: refuses to split on thin cells. Cannot borrow strength from related cells. No calibrated uncertainty.

The correct answer is **partial pooling**: thin segments borrow strength from related segments via a shared population distribution. The degree of borrowing is data-driven — determined by the ratio of within-segment sampling noise to between-segment signal variance. This is the Bayesian posterior.

Under Normal-Normal conjugacy, partial pooling is exactly Bühlmann-Straub credibility. This library generalises it to Poisson (frequency) and Gamma (severity) likelihoods, with multiple crossed random effects.

## Install

```bash
pip install "bayesian-pricing[pymc]"
# or with uv:
uv add "bayesian-pricing[pymc]"
```

PyMC 5.x is an optional dependency — it is not pulled in by default because it has C++ compiler requirements on some platforms. The `[pymc]` extra handles this. For GPU-accelerated inference on large portfolios:

```bash
pip install "bayesian-pricing[numpyro]"
# or with uv:
uv add "bayesian-pricing[numpyro]"
```

## Usage

Input is segment-level sufficient statistics — one row per rating cell, with exposure and claim count. This is the practical production design: aggregate your book to rating cells first, then run the model. A book with 500k policies typically has 5,000–20,000 non-empty rating cells. The model operates on those cells, making NUTS feasible on a standard machine.

```python
import polars as pl
from bayesian_pricing import HierarchicalFrequency, BayesianRelativities
from bayesian_pricing.frequency import SamplerConfig

# One row per rating cell
df = pl.DataFrame({
    "veh_group":  ["Supermini", "Supermini", "Sports", "Sports", "Saloon"],
    "age_band":   ["17-21", "31-40", "17-21", "31-40", "31-40"],
    "claims":     [8, 120, 3, 45, 200],
    "exposure":   [60.0, 900.0, 25.0, 350.0, 2000.0],
})

# Fit hierarchical Poisson model
model = HierarchicalFrequency(
    group_cols=["veh_group", "age_band"],
    prior_mean_rate=0.09,       # portfolio mean claim rate
    variance_prior_sigma=0.3,   # prior belief on between-segment variation
)

config = SamplerConfig(
    method="nuts",    # use "pathfinder" for fast iteration during model development
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=42,
)

model.fit(df, claim_count_col="claims", exposure_col="exposure", sampler_config=config)

# Posterior predictive means for each segment
preds = model.predict()
print(preds)
#   veh_group age_band      mean      p5       p50      p95  credibility_factor
#   Supermini    17-21    0.1234  0.0812   0.1201   0.1731           0.38
#   Sports       17-21    0.1891  0.1102   0.1845   0.2881           0.21  ← thin
# ...

# Variance components: how much does each factor drive frequency?
print(model.variance_components())
```

## Relativities

```python
rel = BayesianRelativities(model, hdi_prob=0.9)

# Full table for all factors
tables = rel.relativities()

# Single factor in rate-table format
veh_table = rel.relativities(factor="veh_group")
print(veh_table.table)
#   level      relativity  lower_90pct  upper_90pct  credibility_factor  interval_width
#   Sports          1.524        1.234        1.891           0.71               0.657
#   Saloon          1.000        0.921        1.082           0.94               0.161
#   Supermini       0.819        0.764        0.881           0.89               0.117

# Identify thin segments that need manual review
thin = rel.thin_segments(credibility_threshold=0.3)
print(thin)
# factor    level    credibility_factor    relativity
# veh_group Sports-17-21          0.18         1.84   ← sparse cell, wide CI

# Export for Excel / rate system import
summary_df = rel.summary()  # long format: factor, level, relativity, CI, credibility
summary_df.write_csv("bayesian_relativities.csv")
```

## Severity model

The severity model has the same API but uses a Gamma likelihood. It expects one row per rating cell with a mean claim cost and a claim count weight.

```python
import numpy as np
import polars as pl
from bayesian_pricing import HierarchicalSeverity
from bayesian_pricing.frequency import SamplerConfig

rng = np.random.default_rng(42)

# Segment-level severity data: one row per vehicle group
veh_groups = ["Supermini", "Supermini", "Sports", "Sports", "Saloon", "Saloon", "Estate"]
claim_counts = [42, 89, 15, 31, 120, 95, 67]
# Mean severity varies by vehicle group; sports cars cost more to repair
base_sev = {"Supermini": 1400, "Sports": 2800, "Saloon": 1700, "Estate": 1900}
avg_costs = [
    base_sev[g] * rng.uniform(0.85, 1.15)
    for g in veh_groups
]

sev_df = pl.DataFrame({
    "veh_group":     veh_groups,
    "avg_claim_cost": avg_costs,
    "claim_count":   claim_counts,
})

sev_model = HierarchicalSeverity(
    group_cols=["veh_group"],          # severity varies by vehicle, not driver age
    prior_mean_severity=1800.0,        # portfolio mean attritional claim cost
    variance_prior_sigma=0.2,          # severity has less between-segment variation than frequency
)

sev_model.fit(
    sev_df,
    severity_col="avg_claim_cost",
    weight_col="claim_count",          # segments with more claims get more influence
    sampler_config=SamplerConfig(method="nuts", draws=1000, tune=1000, chains=4),
)

sev_preds = sev_model.predict()
```

## Convergence diagnostics

MCMC results are only valid if the sampler converged. Check before using output:

```python
from bayesian_pricing.diagnostics import convergence_summary, posterior_predictive_check

# R-hat, ESS, divergence counts
diag = convergence_summary(model)
# Prints warnings if R-hat > 1.01 or ESS < 400

# Check model describes the data
ppc = posterior_predictive_check(model, claim_count_col="claims")
# Returns: mean, variance, p90, p95 checks
```

## Inference options

| Method | When to use | Speed | Accuracy |
|---|---|---|---|
| `SamplerConfig(method="pathfinder")` | Model development, prior sensitivity | Minutes | Good approximation |
| `SamplerConfig(method="nuts")` | Final production estimates | 20–60 min | Exact (asymptotically) |
| `SamplerConfig(nuts_sampler="numpyro")` | Large portfolios, GPU available | Fast on GPU | Exact |

For portfolios with more than 50k rating cells, consider the two-stage approach: fit a GBM on the full book, extract segment-level residuals, then run the Bayesian model on the residuals. The GBM captures dense-cell signal; the Bayesian model handles thin-cell pooling.

## Relationship to Bühlmann-Straub credibility

The Bühlmann-Straub credibility premium is the exact posterior mean of a hierarchical model under Normal-Normal conjugacy. This library generalises that result:

| Feature | Bühlmann-Straub | bayesian-pricing |
|---|---|---|
| Likelihood | Normal (symmetric loss) | Poisson, Gamma, NB |
| Number of grouping factors | One | Multiple crossed |
| Credible intervals | No (point estimates) | Yes (full posterior) |
| Hyperparameter uncertainty | Plugged in | Integrated out |
| Groups | > 20 needed for stable K | Works with 5+ |

For single-factor pricing with many groups (e.g., scheme pricing), Bühlmann-Straub is computationally trivial and entirely adequate. Use this library when you need multiple crossed random effects, non-Normal likelihoods, or full posterior uncertainty.

## Design decisions

**Non-centered parameterisation throughout.** The centered version (`u_i ~ Normal(0, sigma)`) creates funnel geometry in the posterior when sigma is small — which is exactly the case for well-regularised insurance models. HMC cannot traverse the funnel efficiently. The non-centered version decouples the raw offsets from the scale and eliminates this problem. See Twiecki (2017) for the clearest exposition.

**Segment-level input, not policy-level.** This is the practical production design. NUTS does not scale linearly with observation count. A model with 10,000 rating cells runs in minutes; a model with 1 million policy rows takes hours. Aggregate first.

**HalfNormal variance hyperpriors, not HalfCauchy.** HalfCauchy has heavy tails that allow unrealistically large random effects for thin cells — the opposite of the regularisation we want. HalfNormal (Gelman et al., 2013) produces appropriate shrinkage for insurance factors.

**Frequency-severity split, not Tweedie.** The split allows different pooling structures for frequency and severity. Young drivers have high frequency but similar severity to older drivers. A Tweedie cannot capture this. The Gamma likelihood handles attritional severity; model large claims separately with Pareto or log-normal.

**PyMC optional.** The library parses and validates data without PyMC. Tests for the data layer run in CI without it. This makes the library usable in environments where PyMC is hard to install.

## Read more

[Partial Pooling for Thin Rating Cells](https://burning-cost.github.io/2026/03/06/bayesian-hierarchical-models-for-thin-data-pricing.html) — why every other approach fails thin segments and how hierarchical Bayesian models solve it.

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [credibility](https://github.com/burning-cost/credibility) | Bühlmann-Straub credibility weighting — the closed-form special case of this library under Normal-Normal conjugacy |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated detection of missing GLM interactions — the complementary question to thin-cell regularisation |
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Synthetic UK motor and home datasets with known DGPs — useful for validating that the model recovers true parameters |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI and A/E drift detection for tracking when the deployed model needs a refit |

[All Burning Cost libraries →](https://burning-cost.github.io)

## References

1. Bühlmann, H. (1967). Experience rating and credibility. *ASTIN Bulletin*, 4(3), 199–207.
2. Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed. Chapter 5.
3. Ohlsson, E. (2008). Combining generalised linear models and credibility models. *Scandinavian Actuarial Journal*.
4. Krapu et al. (2023). Flexible hierarchical risk modeling for large insurance data via NumPyro. *arXiv:2312.07432*.
5. Twiecki, T. (2017). Why hierarchical models are awesome, tricky, and Bayesian. twiecki.io.
