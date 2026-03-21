"""
bayesian-pricing: Hierarchical Bayesian models for insurance pricing thin-data segments.

The central problem this library solves is the sparse-cell problem in personal lines rating.
A motor book with 1M policies might have 4.5M theoretical rating cells. Most are empty or
contain fewer than 30 observations. GBMs overfit or refuse to split. Saturated GLMs overfit.
Ridge GLMs shrink uniformly regardless of exposure. None of these are right.

The correct answer is partial pooling: thin segments borrow strength from related segments
via a shared population distribution. The degree of borrowing is data-driven -- determined
by the ratio of within-segment sampling noise to between-segment signal variance. This is
the Bayesian posterior.

Under Normal-Normal conjugacy, this is exactly Bühlmann-Straub credibility. This library
generalises that to Poisson (frequency) and Gamma (severity) likelihoods, with multiple
crossed random effects, using PyMC 5.x under the hood.

Input: both pandas and Polars DataFrames are accepted everywhere.
Output: all DataFrames returned by predict(), variance_components(),
        relativities(), credibility_factors(), thin_segments(), summary(),
        and convergence_summary() are Polars DataFrames.

Primary classes:
    HierarchicalFrequency: Poisson hierarchical model for claim frequency
    HierarchicalSeverity: Gamma hierarchical model for claim severity
    BayesianRelativities: Extract multiplicative relativities from the posterior

Usage::

    import polars as pl
    from bayesian_pricing import HierarchicalFrequency, BayesianRelativities

    df = pl.read_parquet("segments.parquet")  # or a pandas DataFrame

    freq_model = HierarchicalFrequency(group_cols=["veh_group", "age_band"])
    freq_model.fit(df, claim_count_col="claims", exposure_col="earned_exposure")

    rel = BayesianRelativities(freq_model)
    print(rel.relativities())          # Polars DataFrame with posterior median + credible interval
    print(rel.credibility_factors())   # Uncertainty reduction per segment (how data-dominated each level is)
"""

from bayesian_pricing.frequency import HierarchicalFrequency, SamplerConfig
from bayesian_pricing.severity import HierarchicalSeverity
from bayesian_pricing.relativities import BayesianRelativities
from bayesian_pricing.diagnostics import convergence_summary, posterior_predictive_check

__version__ = "0.2.3"
__all__ = [
    "HierarchicalFrequency",
    "HierarchicalSeverity",
    "BayesianRelativities",
    "SamplerConfig",
    "convergence_summary",
    "posterior_predictive_check",
]
