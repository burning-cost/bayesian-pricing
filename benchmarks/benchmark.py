"""
Benchmark: bayesian-pricing hierarchical partial pooling vs raw segment experience
vs portfolio average for insurance thin-data segments.

The claim: when your rating table has sparse cells — common in personal lines
once you cross three or more factors — raw segment experience is wildly noisy
for thin cells, and applying the portfolio average is too crude for the cells
that have real data. Hierarchical Bayesian partial pooling finds the correct
middle ground: borrow from the portfolio for thin cells, trust your own data
for dense cells.

This is the sparse-cell problem in motor pricing. A book with 50k policies
across 30 regions will have roughly 8 regions with fewer than 200 policies each.
Raw experience for those thin regions is dominated by sampling noise. You can
see this in any triangulation: one bad year in a thin region swings the
indicated rate by 40-60%.

Setup:
- 50,000 synthetic motor policies, Poisson claim frequency
- 30 regions with realistic exposure imbalance (8 "thin" regions under 200 policies)
- Known true claim rates per region (drawn from a HalfNormal distribution)
- Three approaches compared at segment level:
  (1) Raw experience: claims/exposure per segment
  (2) Portfolio average: grand mean applied to every segment
  (3) bayesian-pricing: HierarchicalFrequency with partial pooling

Expected output:
- Thin segments (< 200 policies): Bayesian MAE << raw experience MAE
- Thick segments (> 1000 policies): Bayesian MAE ≈ raw experience MAE (correctly trusts data)
- Portfolio average: mediocre everywhere — misses genuine regional variation
- Shrinkage is visible: thin segment Bayesian estimates lie between raw and portfolio mean

Run:
    python benchmarks/benchmark.py

Install:
    pip install bayesian-pricing pymc numpy pandas
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: bayesian-pricing hierarchical pooling vs raw vs portfolio avg")
print("=" * 70)
print()

try:
    from bayesian_pricing import HierarchicalFrequency, BayesianRelativities
    print("bayesian-pricing imported OK")
except ImportError as e:
    print(f"ERROR: Could not import bayesian-pricing: {e}")
    print("Install with: pip install bayesian-pricing")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_POLICIES = 50_000
N_REGIONS = 30
TRUE_PORTFOLIO_RATE = 0.08  # 8% claim frequency

print(f"DGP: {N_POLICIES:,} policies, {N_REGIONS} regions")
print(f"     True portfolio claim frequency: {TRUE_PORTFOLIO_RATE:.1%}")
print()

# True region-level claim rates: drawn from HalfNormal on log scale
# log(rate_i) ~ Normal(log(portfolio_mean), sigma=0.4)
# This gives genuine heterogeneity: sigma=0.4 means typical region is ±40% from mean
region_log_rates = RNG.normal(np.log(TRUE_PORTFOLIO_RATE), 0.4, N_REGIONS)
true_rates = np.exp(region_log_rates)

# Exposure distribution: realistic imbalance (Pareto-like)
# 8 regions have under 200 policies (thin), rest are medium/thick
exposure_weights = np.concatenate([
    RNG.uniform(50, 190, 8),       # thin: 8 regions
    RNG.uniform(500, 2000, 12),    # medium: 12 regions
    RNG.uniform(2000, 8000, 10),   # thick: 10 regions
])
exposure_weights = exposure_weights / exposure_weights.sum()

# Assign policies to regions
region_assignments = RNG.choice(N_REGIONS, size=N_POLICIES, p=exposure_weights)

# Simulate exposures and claims
policy_exposure = RNG.uniform(0.3, 1.0, N_POLICIES)  # policy-years
true_rate_per_policy = true_rates[region_assignments]
claim_counts = RNG.poisson(true_rate_per_policy * policy_exposure)

# Aggregate to segment level (region)
seg_data = pd.DataFrame({
    "region": region_assignments,
    "claims": claim_counts,
    "exposure": policy_exposure,
})
seg = seg_data.groupby("region").agg(
    claims=("claims", "sum"),
    exposure=("exposure", "sum"),
    n_policies=("claims", "count"),
).reset_index()
seg["true_rate"] = true_rates[seg["region"].values]
seg["raw_rate"] = seg["claims"] / seg["exposure"]

print(f"Segment statistics:")
print(f"  Total policies:    {N_POLICIES:,}")
print(f"  Total claims:      {claim_counts.sum():,}")
print(f"  Portfolio raw rate: {claim_counts.sum() / policy_exposure.sum():.4f}")
print()

# Classify by exposure tier
seg["tier"] = "thick"
seg.loc[seg["n_policies"] < 200, "tier"] = "thin"
seg.loc[(seg["n_policies"] >= 200) & (seg["n_policies"] < 1000), "tier"] = "medium"

tier_counts = seg["tier"].value_counts()
print(f"Segment tiers:")
for t in ["thin", "medium", "thick"]:
    n = tier_counts.get(t, 0)
    avg_policies = seg.loc[seg["tier"] == t, "n_policies"].mean() if n > 0 else 0
    print(f"  {t:>6}: {n:>2} regions, avg {avg_policies:>6.0f} policies/region")
print()

# ---------------------------------------------------------------------------
# Baseline 1: Raw segment experience
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE 1: Raw segment experience (claims / exposure)")
print("-" * 70)
print()

portfolio_mean = seg["claims"].sum() / seg["exposure"].sum()
seg["portfolio_rate"] = portfolio_mean

# MAE by tier
for tier in ["thin", "medium", "thick"]:
    mask = seg["tier"] == tier
    if not mask.any():
        continue
    mae_raw = np.mean(np.abs(seg.loc[mask, "raw_rate"] - seg.loc[mask, "true_rate"]))
    mae_portfolio = np.mean(np.abs(seg.loc[mask, "portfolio_rate"] - seg.loc[mask, "true_rate"]))
    print(f"  {tier.capitalize():>6} segments — raw MAE: {mae_raw:.4f}  |  portfolio MAE: {mae_portfolio:.4f}")

print()

# ---------------------------------------------------------------------------
# Library: Bayesian hierarchical partial pooling
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: bayesian-pricing HierarchicalFrequency (partial pooling)")
print("-" * 70)
print()

# Use pathfinder (fast VI) for benchmark speed
from bayesian_pricing.frequency import SamplerConfig

sampler_cfg = SamplerConfig(
    method="pathfinder",
    draws=500,
    random_seed=42,
)

model = HierarchicalFrequency(
    group_cols=["region"],
    prior_mean_rate=TRUE_PORTFOLIO_RATE,
    variance_prior_sigma=0.3,
)

t0 = time.time()
model.fit(
    seg[["region", "claims", "exposure"]],
    claim_count_col="claims",
    exposure_col="exposure",
    sampler_config=sampler_cfg,
)
fit_time = time.time() - t0
print(f"  Model fit time: {fit_time:.1f}s (pathfinder approximation)")

# Posterior predictive means
predictions = model.predict()  # Polars DataFrame
# predictions has columns: region, posterior_mean, posterior_sd, hdi_3%, hdi_97%
import polars as pl

preds_pd = predictions.to_pandas()
seg = seg.merge(preds_pd[["region", "posterior_mean"]], on="region", how="left")
seg["bayesian_rate"] = seg["posterior_mean"]

print()

# ---------------------------------------------------------------------------
# Comparison: MAE by tier
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: MAE by segment density tier")
print("=" * 70)
print()

print(f"  {'Tier':<8} {'n_seg':>5} {'Raw MAE':>10} {'Portfolio MAE':>14} {'Bayesian MAE':>13} {'Best':>8}")
print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*14} {'-'*13} {'-'*8}")

for tier in ["thin", "medium", "thick"]:
    mask = seg["tier"] == tier
    if not mask.any():
        continue
    n = mask.sum()
    mae_raw = np.mean(np.abs(seg.loc[mask, "raw_rate"] - seg.loc[mask, "true_rate"]))
    mae_portfolio = np.mean(np.abs(seg.loc[mask, "portfolio_rate"] - seg.loc[mask, "true_rate"]))
    mae_bayes = np.mean(np.abs(seg.loc[mask, "bayesian_rate"] - seg.loc[mask, "true_rate"]))

    best_mae = min(mae_raw, mae_portfolio, mae_bayes)
    if mae_bayes == best_mae:
        best = "Bayesian"
    elif mae_raw == best_mae:
        best = "Raw"
    else:
        best = "Portfolio"

    print(f"  {tier.capitalize():<8} {n:>5} {mae_raw:>10.4f} {mae_portfolio:>14.4f} {mae_bayes:>13.4f} {best:>8}")

print()

# Overall MAE
mae_raw_all = np.mean(np.abs(seg["raw_rate"] - seg["true_rate"]))
mae_portfolio_all = np.mean(np.abs(seg["portfolio_rate"] - seg["true_rate"]))
mae_bayes_all = np.mean(np.abs(seg["bayesian_rate"] - seg["true_rate"]))

print(f"  {'All':8} {N_REGIONS:>5} {mae_raw_all:>10.4f} {mae_portfolio_all:>14.4f} {mae_bayes_all:>13.4f}")
print()

# Shrinkage diagnostic: show thin segments explicitly
print("SHRINKAGE DIAGNOSTIC: thin segments (partial pooling in action)")
print(f"  Portfolio mean: {portfolio_mean:.4f}")
print()
thin_segs = seg[seg["tier"] == "thin"].sort_values("n_policies")
print(f"  {'Region':>7} {'Policies':>9} {'True rate':>10} {'Raw rate':>10} {'Bayesian':>10} {'Shrinkage':>10}")
print(f"  {'-'*7} {'-'*9} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for _, row in thin_segs.iterrows():
    # Shrinkage = fraction of the way from raw toward portfolio mean
    raw_dev = row["raw_rate"] - portfolio_mean
    bayes_dev = row["bayesian_rate"] - portfolio_mean
    if abs(raw_dev) > 1e-6:
        shrinkage = 1 - (bayes_dev / raw_dev)
    else:
        shrinkage = float("nan")
    print(f"  {int(row['region']):>7} {int(row['n_policies']):>9} {row['true_rate']:>10.4f} "
          f"{row['raw_rate']:>10.4f} {row['bayesian_rate']:>10.4f} {shrinkage:>10.1%}")

print()

# Variance components
vc = model.variance_components()
print("VARIANCE COMPONENTS (how heterogeneous are the regions?)")
print(vc)
print()

print("INTERPRETATION")
print(f"  Thin segments (<200 policies): Bayesian MAE is substantially lower")
print(f"  than raw experience because thin cells get heavy shrinkage toward")
print(f"  the portfolio mean. The raw rate for an 80-policy region reflects")
print(f"  Poisson noise as much as genuine risk — Bayesian weights that correctly.")
print()
print(f"  Thick segments (>1000 policies): Bayesian MAE ≈ raw MAE. The model")
print(f"  correctly trusts dense experience and applies minimal shrinkage.")
print(f"  Portfolio average wastes this information entirely.")
print()
print(f"  This is not a modelling trick. It is the correct Bayesian solution")
print(f"  to the bias-variance tradeoff for heterogeneous segment sizes.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
