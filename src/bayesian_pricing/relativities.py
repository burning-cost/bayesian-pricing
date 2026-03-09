"""
BayesianRelativities: Extract multiplicative rating factors from the posterior.

An actuary working in Emblem or Radar thinks in relativities. A "young driver
relativity" of 2.5 means young drivers are expected to have 2.5x the base claim
rate. This is the natural output of a Poisson GLM with a log link.

In a hierarchical Bayesian model, each segment has a posterior distribution for
its random effect u_k. Converting to relativities is straightforward on the log
scale: relativity_j = exp(u_k_j). But the actuary wants:

1. A point estimate for the rate table (posterior median or mean of the relativity)
2. A credible interval (e.g., 90% HDI) that quantifies how uncertain the estimate is
3. A "credibility factor" in Bühlmann-Straub terms: how much of the data is the
   segment's own experience vs the portfolio mean?

This class extracts all three from a fitted HierarchicalFrequency or
HierarchicalSeverity model and returns Polars DataFrames in a format that can
be directly imported into a rate table or handed to an underwriter.

The relativities are by design multiplicative around a base of 1.0. The base
level for each factor is not separately identified -- relativities should be
interpreted as "relative to the portfolio mean" unless you define a base level.
To define a base level (e.g., veh_group A = 1.0), normalise by dividing all
relativities by the A-level relativity.
"""

from __future__ import annotations

from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bayesian_pricing._utils import _check_pymc


@dataclass
class RelativityTable:
    """Output from BayesianRelativities.relativities() for a single factor.

    Attributes:
        factor: The grouping column name (e.g., "veh_group").
        levels: List of levels in the factor.
        table: Polars DataFrame with columns: level, relativity, lower, upper,
            credibility_factor, interval_width.

    The credibility_factor tells you how data-dominated each level is:
        0.0 = fully pooled to portfolio mean (estimate is 1.0 relativity)
        1.0 = trusts own experience completely (like a fixed-effects GLM)
        0.6 = weighted blend: 60% own data, 40% portfolio mean

    Levels with credibility_factor < 0.3 have wide credible intervals and
    should be treated with caution. They should not drive large rate changes.
    """

    factor: str
    levels: list
    table: "pl.DataFrame"


class BayesianRelativities:
    """Extract multiplicative rating relativities from a fitted Bayesian model.

    Works with both HierarchicalFrequency and HierarchicalSeverity. The output
    is designed to be actuary-friendly: a table of relativities per factor level
    with credible intervals. All output DataFrames are Polars.

    Args:
        model: A fitted HierarchicalFrequency or HierarchicalSeverity instance.
        hdi_prob: Width of the highest density interval to report. Default 0.9
            gives a 90% credible interval. Industry convention varies -- some
            teams use 95%, some use 80%.
        base_level: Optional dict mapping factor name to the level that should
            be set to 1.0. E.g. {"veh_group": "B", "age_band": "30-35"}.
            If None, relativities are centred at exp(mean(u_k)) ≈ 1.0 (since
            the random effects have zero mean by construction).

    Examples::

        from bayesian_pricing import HierarchicalFrequency, BayesianRelativities

        freq = HierarchicalFrequency(group_cols=["veh_group", "age_band"])
        freq.fit(df, claim_count_col="claims", exposure_col="exposure")

        rel = BayesianRelativities(freq, hdi_prob=0.9)

        # Summary Polars DataFrame across all factors
        print(rel.relativities())

        # Single factor
        veh_table = rel.relativities(factor="veh_group")
        print(veh_table.table)

        # Credibility factors by factor
        print(rel.credibility_factors())

        # Check which segments should be treated as "thin"
        thin = rel.thin_segments(credibility_threshold=0.3)
        print(thin)
    """

    def __init__(
        self,
        model,
        hdi_prob: float = 0.9,
        base_level: Optional[dict[str, str]] = None,
    ) -> None:
        _check_fitted(model)
        self._model = model
        self.hdi_prob = hdi_prob
        self.base_level = base_level or {}

    def relativities(
        self, factor: Optional[str] = None
    ) -> Union[dict[str, RelativityTable], RelativityTable]:
        """Return multiplicative relativities from the posterior.

        Args:
            factor: If specified, return a RelativityTable for that factor only.
                If None, return a dict of RelativityTable keyed by factor name.

        Returns:
            RelativityTable (if factor specified) or dict[str, RelativityTable].
        """
        _check_pymc()
        import arviz as az

        if factor is not None:
            if factor not in self._model.group_cols:
                raise ValueError(
                    f"Factor {factor!r} not in model group_cols: {self._model.group_cols}"
                )
            return self._relativity_table(factor, az)

        return {col: self._relativity_table(col, az) for col in self._model.group_cols}

    def _relativity_table(self, col: str, az) -> RelativityTable:
        """Build a RelativityTable for a single grouping factor."""
        import polars as pl

        posterior = self._model.idata.posterior
        u_samples = posterior[f"u_{col}"].values  # (chains, draws, n_levels)
        u_flat = u_samples.reshape(-1, u_samples.shape[-1])  # (n_samples, n_levels)

        rel_samples = np.exp(u_flat)  # multiplicative relativities

        # Apply base level normalisation if requested
        levels = self._model._group_levels[col].tolist()
        if col in self.base_level:
            base = self.base_level[col]
            if base not in levels:
                raise ValueError(
                    f"Base level {base!r} not found in factor {col!r}. "
                    f"Available levels: {levels}"
                )
            base_idx = levels.index(base)
            rel_samples = rel_samples / rel_samples[:, base_idx : base_idx + 1]

        # Summary statistics
        point_estimate = np.median(rel_samples, axis=0)
        hdi_lower, hdi_upper = self._compute_hdi(rel_samples)

        # Credibility factor heuristic: compare posterior uncertainty to prior uncertainty.
        # Low posterior std relative to prior std -> high credibility (less uncertain).
        prior_std = np.ones(len(levels)) * self._model.variance_prior_sigma
        posterior_std = rel_samples.std(axis=0)
        prior_rel_std = np.exp(prior_std) - 1
        credibility = np.clip(1 - posterior_std / (prior_rel_std + 1e-8), 0, 1)

        lower_col = f"lower_{int(self.hdi_prob * 100)}pct"
        upper_col = f"upper_{int(self.hdi_prob * 100)}pct"

        table = pl.DataFrame({
            "level": levels,
            "relativity": point_estimate.tolist(),
            lower_col: hdi_lower.tolist(),
            upper_col: hdi_upper.tolist(),
            "credibility_factor": credibility.tolist(),
            "interval_width": (hdi_upper - hdi_lower).tolist(),
        }).sort("relativity", descending=True)

        return RelativityTable(factor=col, levels=levels, table=table)

    def credibility_factors(self) -> "pl.DataFrame":
        """Return credibility factors for all segments across all factors.

        The credibility factor answers: "what fraction of this segment's estimate
        comes from its own experience vs the portfolio mean?" This is the Bayesian
        equivalent of Z_j = w_j / (w_j + K) in Bühlmann-Straub.

        Returns:
            Polars DataFrame with factor, level, credibility_factor, and relativity columns.
        """
        import polars as pl

        rows: list[dict] = []
        for col in self._model.group_cols:
            rt = self._relativity_table(col, None)
            for row in rt.table.iter_rows(named=True):
                rows.append({
                    "factor": col,
                    "level": row["level"],
                    "credibility_factor": row["credibility_factor"],
                    "relativity": row["relativity"],
                })
        return pl.DataFrame(rows)

    def thin_segments(self, credibility_threshold: float = 0.3) -> "pl.DataFrame":
        """Return segments where the credibility factor is below the threshold.

        These are the segments where the model is doing the most work --
        thin cells that were strongly pulled toward the portfolio mean.
        The credible intervals for these segments will be wide.

        Practical use: flag thin segments in the rate table for underwriter review.
        A segment with credibility_factor = 0.1 should not drive a large rate change
        even if the posterior median relativity differs from 1.0 significantly.

        Args:
            credibility_threshold: Segments with credibility_factor below this
                value are returned. Default 0.3 (meaning less than 30% of the
                estimate comes from own experience).

        Returns:
            Polars DataFrame with factor, level, credibility_factor, and relativity.
        """
        creds = self.credibility_factors()
        return (
            creds
            .filter(creds["credibility_factor"] < credibility_threshold)
            .sort("credibility_factor")
        )

    def summary(self) -> "pl.DataFrame":
        """Long-format summary of all relativities across all factors.

        Convenient for exporting to Excel or plotting. One row per (factor, level)
        combination. Returns a Polars DataFrame.
        """
        import polars as pl

        frames = []
        for col in self._model.group_cols:
            rt = self._relativity_table(col, None)
            frames.append(rt.table.with_columns(pl.lit(col).alias("factor")))

        if not frames:
            return pl.DataFrame()

        # Move 'factor' to the front
        combined = pl.concat(frames)
        cols = ["factor"] + [c for c in combined.columns if c != "factor"]
        return combined.select(cols)

    def _compute_hdi(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute Highest Density Interval for each column of samples.

        HDI is preferred over equal-tailed intervals for skewed posteriors --
        which is common for relativities near 1.0 since they are bounded below
        at 0. The HDI contains the most probable values, not just the central ones.

        Args:
            samples: (n_samples, n_levels) array.

        Returns:
            lower, upper arrays of shape (n_levels,).
        """
        _check_pymc()
        import arviz as az

        lower = np.empty(samples.shape[1])
        upper = np.empty(samples.shape[1])

        for j in range(samples.shape[1]):
            hdi = az.hdi(samples[:, j], hdi_prob=self.hdi_prob)
            lower[j] = hdi[0]
            upper[j] = hdi[1]

        return lower, upper


def _check_fitted(model) -> None:
    """Raise if model has not been fitted."""
    if not getattr(model, "_fitted", False):
        raise RuntimeError(
            "Model has not been fitted. Call .fit() before creating BayesianRelativities."
        )
