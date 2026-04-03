"""
HierarchicalSeverity: Gamma hierarchical model for claim severity.

Model structure:

    severity_i  ~  Gamma(alpha_gamma, alpha_gamma / mu_i)
    log(mu_i) = alpha + sum_k u_k[group_k_i]

    alpha  ~  Normal(log(portfolio_mean_severity), 0.5)

    For each grouping factor k:
        sigma_k  ~  HalfNormal(prior_sigma)
        z_k_j    ~  Normal(0, 1)
        u_k_j     = z_k_j * sigma_k

    alpha_gamma  ~  HalfNormal(2.0)   [Gamma shape parameter, controls severity dispersion]

Important: severity is conditional on a claim occurring. The input data should
contain one row per segment with the *average* claim cost and the *claim count*
(used as the effective weight). Do not include zero-claim observations -- the
severity model is defined only for claim counts > 0.

The Gamma distribution is parameterised in terms of mean and concentration:
    Gamma(alpha=concentration, beta=concentration/mean)
    E[Y] = mean, Var[Y] = mean^2 / concentration

The shape parameter alpha_gamma (concentration) is estimated from data. It captures
the within-segment coefficient of variation of individual claim amounts. UK attritional
motor severity typically has CV around 1.5-2.5, implying alpha_gamma ~ 0.16-0.44.
Large claims (bodily injury, theft totals) should be modelled separately -- this
model is intended for attritional severity only.

Why Gamma and not log-normal?
    - Gamma is the conjugate prior for Poisson rates, so the compound Poisson-Gamma
      pure premium has nice properties.
    - Log-normal gives a heavier right tail. In practice, for attritional claims,
      the Gamma and log-normal are hard to distinguish.
    - The Gamma is additive: the sum of independent Gamma random variables is Gamma.
      This is convenient for aggregate loss calculations.
    - If large claims are in the data, both distributions give poor fits. Better to
      remove large claims and model them separately (e.g., Pareto above a threshold).

For large claims (BI, total losses), consider parameterising a Pareto or log-normal
severity model manually using PyMC directly.

Both pandas and Polars DataFrames are accepted as input. All output DataFrames are
returned as Polars.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from bayesian_pricing._utils import (
    _check_pymc,
    _validate_positive,
    _validate_columns_present,
    _validate_no_nulls_in_group_cols,
    _segment_index,
    _to_pandas,
    DataFrameLike,
)
from bayesian_pricing.frequency import SamplerConfig


class HierarchicalSeverity:
    """Gamma hierarchical model for claim severity with partial pooling.

    Severity behaves differently from frequency. The between-segment heterogeneity
    is typically lower for severity than frequency (vehicle group matters more for
    frequency than for average repair cost). This means sigma_k for severity models
    is usually smaller than for frequency models, and more shrinkage occurs.

    The practical implication: a sparse segment (5 claims) with an unusually high
    average severity will be pulled strongly toward the portfolio mean severity.
    This is correct behaviour -- 5 claims is not enough to identify a genuinely
    different severity level.

    Args:
        group_cols: Column names for partial pooling. Usually a subset of the
            frequency model's group_cols -- severity may only be pooled on
            vehicle group, not driver age, if age doesn't affect repair cost.
        prior_mean_severity: Portfolio mean claim cost. Used to centre the
            intercept prior. Should be the average attritional claim amount,
            excluding large claims. Required for sensible priors.
        prior_sigma_severity: Width of intercept prior on log scale. Default 0.5.
        variance_prior_sigma: HalfNormal sigma for random effect variance
            hyperpriors. Typically smaller than for frequency (0.2 vs 0.3)
            because severity has less between-segment variation in practice.
        interaction_pairs: Two-way interactions to include. Same structure as
            HierarchicalFrequency. Rare to need these for severity.

    Examples::

        import polars as pl
        from bayesian_pricing import HierarchicalSeverity

        # One row per segment, claims > 0 only (Polars or pandas accepted)
        df = pl.DataFrame({
            "veh_group":        ["A", "A", "B", "B", "C"],
            "age_band":         ["17-21", "22-35", "17-21", "22-35", "22-35"],
            "claim_count":      [12, 45, 8, 120, 3],
            "avg_claim_cost":   [2100.0, 1850.0, 2400.0, 1950.0, 1700.0],
        })

        model = HierarchicalSeverity(
            group_cols=["veh_group"],
            prior_mean_severity=1950.0,
        )
        model.fit(
            df,
            severity_col="avg_claim_cost",
            weight_col="claim_count",
        )

        print(model.predict())  # returns a Polars DataFrame
    """

    def __init__(
        self,
        group_cols: list[str],
        prior_mean_severity: Optional[float] = None,
        prior_sigma_severity: float = 0.5,
        variance_prior_sigma: float = 0.2,
        interaction_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        if not group_cols:
            raise ValueError("group_cols must contain at least one column name")
        self.group_cols = list(group_cols)
        self.prior_mean_severity = prior_mean_severity
        self.prior_sigma_severity = prior_sigma_severity
        self.variance_prior_sigma = variance_prior_sigma
        self.interaction_pairs = interaction_pairs or []

        self._idata = None
        self._model = None
        self._coords: dict = {}
        self._segment_data: Optional[pd.DataFrame] = None
        self._group_indices: dict = {}
        self._group_levels: dict = {}
        self._fitted = False
        self._severity_col: str = ""
        self._weight_col: Optional[str] = None

    def fit(
        self,
        data: DataFrameLike,
        severity_col: str = "avg_claim_cost",
        weight_col: Optional[str] = None,
        sampler_config: Optional[SamplerConfig] = None,
    ) -> "HierarchicalSeverity":
        """Fit the hierarchical Gamma model.

        Args:
            data: DataFrame with one row per segment, claims > 0. Accepts both
                pandas and Polars DataFrames. Must contain group columns, the
                severity column, and optionally a weight column (claim counts).
            severity_col: Column containing average claim cost per segment.
                Must be strictly positive.
            weight_col: Optional column containing claim counts. If provided,
                higher-weight segments have more influence on the posterior.
                In practice this means segments with more claims shrink less.
                Strongly recommended -- without weights, a segment with 1 claim
                is treated as equally informative as one with 1,000 claims.
            sampler_config: Sampling configuration. See SamplerConfig.

        Returns:
            self
        """
        # Accept pandas or Polars; work internally with pandas
        # Validation runs before _check_pymc() so API tests don't require PyMC.
        df = _to_pandas(data).copy().reset_index(drop=True)

        required_cols = self.group_cols + [severity_col]
        if weight_col:
            required_cols.append(weight_col)
        _validate_columns_present(df, required_cols)
        _validate_positive(df[severity_col], severity_col)

        if weight_col and weight_col in df.columns:
            weights_check = df[weight_col].to_numpy(dtype=float)
            if np.any(weights_check <= 0):
                n_bad = int(np.sum(weights_check <= 0))
                raise ValueError(
                    f"weight_col '{weight_col}' contains {n_bad} zero or negative "
                    f"value(s). Weights must be strictly positive claim counts. "
                    f"Remove or exclude zero-claim segments before fitting."
                )

        _validate_no_nulls_in_group_cols(df, self.group_cols)

        _check_pymc()
        import pymc as pm

        if sampler_config is None:
            sampler_config = SamplerConfig()

        self._segment_data = df
        self._severity_col = severity_col
        self._weight_col = weight_col

        # Estimate portfolio mean severity if not provided
        if self.prior_mean_severity is None:
            if weight_col and weight_col in df.columns:
                weights = df[weight_col].values.astype(float)
                self.prior_mean_severity = float(
                    np.average(df[severity_col].values, weights=weights)
                )
            else:
                self.prior_mean_severity = float(df[severity_col].mean())

        observed_severity = df[severity_col].values.astype(float)

        # Effective weights: if claim counts provided, use as replication factor.
        # We model each segment as the average of claim_count observations, so the
        # effective information is proportional to claim_count.
        if weight_col and weight_col in df.columns:
            weights = df[weight_col].values.astype(float)
        else:
            weights = np.ones(len(df))

        # Segment indices
        for col in self.group_cols:
            idx, levels = _segment_index(df[col])
            self._group_indices[col] = idx
            self._group_levels[col] = levels

        coords = {}
        for col in self.group_cols:
            coords[col] = self._group_levels[col].tolist()
        self._coords = coords

        with pm.Model(coords=coords) as model:
            # Intercept (log-scale)
            alpha = pm.Normal(
                "alpha",
                mu=np.log(self.prior_mean_severity),
                sigma=self.prior_sigma_severity,
            )

            mu_log = alpha

            # Random effects (non-centered)
            for col in self.group_cols:
                sigma = pm.HalfNormal(
                    f"sigma_{col}",
                    sigma=self.variance_prior_sigma,
                )
                z_raw = pm.Normal(
                    f"z_{col}",
                    mu=0,
                    sigma=1,
                    dims=col,
                )
                u = pm.Deterministic(
                    f"u_{col}",
                    z_raw * sigma,
                    dims=col,
                )
                mu_log = mu_log + u[self._group_indices[col]]

            # Interaction random effects
            for col_a, col_b in self.interaction_pairs:
                n_a = len(self._group_levels[col_a])
                n_b = len(self._group_levels[col_b])
                sigma_ix = pm.HalfNormal(
                    f"sigma_{col_a}_x_{col_b}",
                    sigma=self.variance_prior_sigma * 0.5,
                )
                z_ix = pm.Normal(
                    f"z_{col_a}_x_{col_b}",
                    mu=0,
                    sigma=1,
                    shape=(n_a, n_b),
                )
                u_ix = pm.Deterministic(
                    f"u_{col_a}_x_{col_b}",
                    z_ix * sigma_ix,
                )
                ix_a = self._group_indices[col_a]
                ix_b = self._group_indices[col_b]
                mu_log = mu_log + u_ix[ix_a, ix_b]

            mu_severity = pm.math.exp(mu_log)

            # Gamma shape: controls within-segment CV.
            # UK attritional motor severity has CV ~1.5-2.5, implying shape ~0.16-0.44.
            # HalfNormal(sigma=0.5) has median ~0.34, placing most prior mass in the
            # attritional motor range. The previous default of sigma=2 put ~83% of
            # mass above shape=0.44, systematically underestimating dispersion.
            gamma_shape = pm.HalfNormal("gamma_shape", sigma=0.5)

            # Gamma parameterisation: mean = mu, concentration = gamma_shape
            # Each segment observation is a sample average over `weights` claims.
            # The variance of a sample mean of n Gamma(shape, rate) variables is
            # mu^2 / (shape * n). We model this by scaling the effective shape.
            effective_shape = gamma_shape * weights

            # PyMC 5 Gamma: use alpha/beta parameterization
            # alpha = concentration (shape), beta = concentration/mean (rate)
            _ = pm.Gamma(
                "severity",
                alpha=effective_shape,
                beta=effective_shape / mu_severity,
                observed=observed_severity,
            )

        self._model = model

        with model:
            if sampler_config.method == "nuts":
                idata = pm.sample(
                    draws=sampler_config.draws,
                    tune=sampler_config.tune,
                    chains=sampler_config.chains,
                    target_accept=sampler_config.target_accept,
                    nuts_sampler=sampler_config.nuts_sampler,
                    random_seed=sampler_config.random_seed,
                    progressbar=True,
                    return_inferencedata=True,
                )
            else:  # pathfinder (or advi fallback)
                try:
                    idata = pm.fit(
                        method="pathfinder",
                        draws=sampler_config.draws,
                        random_seed=sampler_config.random_seed,
                        progressbar=True,
                    )
                except (KeyError, ValueError):
                    import warnings
                    warnings.warn(
                        "pathfinder not available in this PyMC installation; "
                        "falling back to ADVI. Upgrade to PyMC>=5.3 for pathfinder.",
                        stacklevel=3,
                    )
                    approx = pm.fit(
                        n=20_000,
                        method="advi",
                        random_seed=sampler_config.random_seed,
                        progressbar=False,
                    )
                    idata = approx.sample(sampler_config.draws)

            idata = pm.sample_posterior_predictive(
                idata,
                extend_inferencedata=True,
            )

        self._idata = idata
        self._fitted = True
        return self

    def predict(self, quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)) -> "pl.DataFrame":
        """Return posterior mean severity and credible intervals for each segment.

        The posterior mean is the Bayesian credibility-weighted severity estimate
        for each segment. Thin segments (few claims) will be pulled toward the
        portfolio mean severity more strongly than well-populated segments.

        Returns:
            Polars DataFrame with segment identifiers plus 'mean' and quantile columns.
        """
        import polars as pl

        self._check_fitted()

        posterior = self._idata.posterior
        alpha_samples = posterior["alpha"].values.reshape(-1)

        log_sev_samples = np.broadcast_to(
            alpha_samples[:, np.newaxis],
            (len(alpha_samples), len(self._segment_data)),
        ).copy()

        for col in self.group_cols:
            u_samples = posterior[f"u_{col}"].values.reshape(-1, len(self._group_levels[col]))
            log_sev_samples += u_samples[:, self._group_indices[col]]

        for col_a, col_b in self.interaction_pairs:
            u_ix = posterior[f"u_{col_a}_x_{col_b}"].values
            u_ix = u_ix.reshape(-1, u_ix.shape[-2], u_ix.shape[-1])
            log_sev_samples += u_ix[:, self._group_indices[col_a], self._group_indices[col_b]]

        sev_samples = np.exp(log_sev_samples)

        result_dict: dict = {}
        for col in self.group_cols:
            result_dict[col] = self._segment_data[col].tolist()

        result_dict["mean"] = sev_samples.mean(axis=0).tolist()
        for q in quantiles:
            result_dict[f"p{int(q * 100)}"] = np.quantile(sev_samples, q, axis=0).tolist()

        return pl.DataFrame(result_dict)

    def variance_components(self) -> "pl.DataFrame":
        """Posterior summary of variance hyperparameters by grouping factor.

        Returns:
            Polars DataFrame with one row per grouping factor.
        """
        import polars as pl

        self._check_fitted()
        import arviz as az

        var_names = [f"sigma_{col}" for col in self.group_cols]
        var_names += [f"sigma_{a}_x_{b}" for a, b in self.interaction_pairs]

        summary_pd = az.summary(self._idata, var_names=var_names, hdi_prob=0.94)
        return pl.from_pandas(summary_pd.reset_index().rename(columns={"index": "parameter"}))

    @property
    def idata(self):
        """ArviZ InferenceData."""
        self._check_fitted()
        return self._idata

    @property
    def model(self):
        """The underlying PyMC model."""
        self._check_fitted()
        return self._model

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
