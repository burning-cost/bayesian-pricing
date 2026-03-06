"""
HierarchicalFrequency: Poisson hierarchical model for claim frequency.

The model structure is:

    claims_i  ~  Poisson(lambda_i * exposure_i)
    log(lambda_i) = alpha + sum_k u_k[group_k_i]

    alpha  ~  Normal(log(portfolio_mean_rate), 0.5)   [global intercept]

    For each grouping factor k:
        sigma_k    ~  HalfNormal(0.3)                 [variance hyperprior]
        z_k_j      ~  Normal(0, 1)                    [non-centered raw offsets]
        u_k_j       = z_k_j * sigma_k                 [actual group offsets]

Non-centered parameterization is mandatory. The centered version (u ~ Normal(0, sigma))
creates a funnel geometry in the posterior when sigma is small -- which is exactly the
case for well-regularised insurance models. NUTS cannot traverse the funnel efficiently:
step size is either too large for the narrow end or too small for the wide end. The
non-centered version decouples z from sigma so neither has geometry issues.

Reference: Twiecki (2017), "Why hierarchical models are awesome, tricky, and Bayesian."
           Gelman et al. (2013), BDA3 Chapter 13.

Input contract:
    The model accepts a DataFrame of segment-level sufficient statistics, not
    policy-level data. This is the practical production design: you aggregate
    your book to (segment, exposure, claims) and run the model on segments.
    For a book with 1M policies and 5,000 rating cells, the model operates
    on 5,000 rows -- making NUTS feasible on a standard machine.

    This also makes the model composable with a first-stage GBM: aggregate
    residuals from the GBM at segment level, then pool those residuals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from bayesian_pricing._utils import (
    _check_pymc,
    _validate_positive,
    _validate_columns_present,
    _portfolio_mean_rate,
    _segment_index,
)


@dataclass
class SamplerConfig:
    """Controls how PyMC draws from the posterior.

    For model development and prior sensitivity checks, use method="pathfinder".
    It runs in seconds rather than minutes. For final inference -- the numbers
    that go into the rate table -- use method="nuts".

    Args:
        method: "nuts" (exact MCMC) or "pathfinder" (fast VI approximation).
            Pathfinder is one to two orders of magnitude faster than NUTS.
            The trade-off: no R-hat convergence diagnostics, and the posterior
            may be slightly miscalibrated in the tails. Fine for development;
            use NUTS for production.
        draws: Posterior samples to keep after warmup.
        tune: Warmup/adaptation steps. For NUTS: 1,000 is usually sufficient
            for well-specified models. Increase to 2,000 if you see divergences.
        chains: Number of independent MCMC chains. 4 is standard. Reduces to
            check convergence (R-hat requires multiple chains).
        target_accept: NUTS dual averaging target. Default 0.8. If you see
            divergences after non-centering, increase to 0.9 or 0.95.
        nuts_sampler: "pymc" (default) or "numpyro". The numpyro backend uses
            JAX and runs on GPU if available. For large models (>10k parameters)
            or large segment tables (>50k rows), numpyro is materially faster.
        random_seed: Reproducibility. Set this in production.
    """

    method: str = "nuts"
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.8
    nuts_sampler: str = "pymc"
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.method not in ("nuts", "pathfinder"):
            raise ValueError(f"method must be 'nuts' or 'pathfinder', got {self.method!r}")
        if self.nuts_sampler not in ("pymc", "numpyro"):
            raise ValueError(
                f"nuts_sampler must be 'pymc' or 'numpyro', got {self.nuts_sampler!r}"
            )


class HierarchicalFrequency:
    """Poisson hierarchical model for claim frequency with partial pooling.

    This is a Bayesian generalisation of Bühlmann-Straub credibility to the
    Poisson case with multiple crossed random effects. The key property: thin
    segments (few claims, low exposure) shrink toward the portfolio mean;
    dense segments trust their own data. The degree of shrinkage is estimated
    from data, not hand-set.

    Compared to a fixed-effects Poisson GLM:
    - No overfitting in thin cells (they get credibility-weighted estimates)
    - Full posterior distributions, not just point estimates
    - Data-driven shrinkage, not uniform regularisation

    Compared to Bühlmann-Straub credibility:
    - Works with multiple interacting grouping factors
    - Correct likelihood for count data (Poisson, not Normal)
    - Full posterior uncertainty on the variance components themselves

    Args:
        group_cols: Column names that define the segments for partial pooling.
            Each column gets its own random effect. Two-way interactions are
            supported via ``interaction_pairs``.
            Example: ["veh_group", "age_band", "area_code"]

        prior_mean_rate: Informative prior on the portfolio mean claim rate.
            Used to centre the intercept prior. If None, estimated from data.
            Expressed as a rate (not log): 0.07 means 7% expected claim rate.
            Set this from your portfolio statistics, not from the thin-cell data
            you are modelling -- otherwise the prior defeats its own purpose.

        prior_sigma_rate: Width of the intercept prior on the log scale.
            Default 0.5 allows the intercept to span roughly ±60% from the
            prior mean on the multiplicative scale. Rarely needs tuning.

        variance_prior_sigma: HalfNormal sigma for each random effect's variance
            hyperprior. Interpreted on the log scale: 0.3 means the typical
            segment random effect is within ±30% of the portfolio mean.
            Increase to 0.5 if your book has strong segment-level heterogeneity
            (e.g., telematics segments vs standard rating). Decrease to 0.15 to
            impose stronger pooling when you believe segments are similar.

        interaction_pairs: Optional list of (col_a, col_b) tuples. Each pair
            gets a matrix of interaction random effects, partially pooled toward
            zero via a shared variance hyperprior. Only include actuarially
            motivated interactions -- the parameter count is n_levels_a * n_levels_b.
            Example: [("veh_group", "age_band")]

        overdispersion: If True, use Negative Binomial instead of Poisson.
            Recommended when count data shows overdispersion (variance > mean),
            which is the norm in insurance. The NB adds a dispersion parameter
            estimated from data. Computationally more expensive than Poisson.

    Examples::

        import pandas as pd
        from bayesian_pricing import HierarchicalFrequency

        # Segment-level data: one row per rating cell
        df = pd.DataFrame({
            "veh_group": ["A", "A", "B", "B", "C"],
            "age_band":  ["17-21", "22-35", "17-21", "22-35", "22-35"],
            "claims":    [12, 45, 8, 120, 3],
            "exposure":  [80, 400, 60, 900, 25],
        })

        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            prior_mean_rate=0.12,   # portfolio mean ~12% frequency
        )
        model.fit(df, claim_count_col="claims", exposure_col="exposure")

        # Posterior predictive means for each segment
        print(model.predict())

        # Variance components: how much each factor explains
        print(model.variance_components())
    """

    def __init__(
        self,
        group_cols: list[str],
        prior_mean_rate: Optional[float] = None,
        prior_sigma_rate: float = 0.5,
        variance_prior_sigma: float = 0.3,
        interaction_pairs: Optional[list[tuple[str, str]]] = None,
        overdispersion: bool = False,
    ) -> None:
        if not group_cols:
            raise ValueError("group_cols must contain at least one column name")
        self.group_cols = list(group_cols)
        self.prior_mean_rate = prior_mean_rate
        self.prior_sigma_rate = prior_sigma_rate
        self.variance_prior_sigma = variance_prior_sigma
        self.interaction_pairs = interaction_pairs or []
        self.overdispersion = overdispersion

        # Set after fit()
        self._trace = None
        self._idata = None
        self._model = None
        self._coords: dict = {}
        self._segment_data: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        claim_count_col: str = "claims",
        exposure_col: str = "exposure",
        sampler_config: Optional[SamplerConfig] = None,
    ) -> "HierarchicalFrequency":
        """Fit the hierarchical Poisson model to segment-level data.

        Args:
            data: DataFrame with one row per rating segment. Must contain the
                group columns specified in ``group_cols``, the claim count column,
                and the exposure column. Exposure should be in policy-years.
            claim_count_col: Column containing observed claim counts (integer).
            exposure_col: Column containing earned exposure in policy-years.
            sampler_config: Sampling settings. Default uses NUTS with 4 chains,
                1,000 warmup + 1,000 draw steps.

        Returns:
            self (for method chaining)
        """
        _check_pymc()
        import pymc as pm

        _validate_columns_present(data, self.group_cols + [claim_count_col, exposure_col])
        _validate_positive(data[exposure_col], exposure_col)

        if sampler_config is None:
            sampler_config = SamplerConfig()

        df = data.copy().reset_index(drop=True)

        # Estimate portfolio mean rate from data if not provided
        if self.prior_mean_rate is None:
            self.prior_mean_rate = _portfolio_mean_rate(
                df[claim_count_col], df[exposure_col]
            )

        observed_claims = df[claim_count_col].values.astype(int)
        exposure = df[exposure_col].values.astype(float)

        # Build segment indices for each grouping factor
        group_indices: dict[str, np.ndarray] = {}
        group_levels: dict[str, np.ndarray] = {}
        for col in self.group_cols:
            idx, levels = _segment_index(df[col])
            group_indices[col] = idx
            group_levels[col] = levels

        # Build interaction indices
        interaction_indices: dict[tuple[str, str], np.ndarray] = {}
        for col_a, col_b in self.interaction_pairs:
            if col_a not in self.group_cols or col_b not in self.group_cols:
                raise ValueError(
                    f"Interaction pair ({col_a!r}, {col_b!r}) references a column "
                    f"not in group_cols: {self.group_cols}"
                )

        # Coords for PyMC named dimensions
        coords = {}
        for col in self.group_cols:
            coords[col] = group_levels[col].tolist()
        for col_a, col_b in self.interaction_pairs:
            coords[f"{col_a}_x_{col_b}_a"] = group_levels[col_a].tolist()
            coords[f"{col_a}_x_{col_b}_b"] = group_levels[col_b].tolist()

        self._coords = coords

        with pm.Model(coords=coords) as model:
            # --- Global intercept ---
            # Informed by portfolio mean rate. The prior is on the log scale,
            # so Normal(log(mu), 0.5) puts 95% mass between mu/e and mu*e.
            alpha = pm.Normal(
                "alpha",
                mu=np.log(self.prior_mean_rate),
                sigma=self.prior_sigma_rate,
            )

            # --- Main random effects (one per grouping factor) ---
            # Non-centered parameterization throughout. See module docstring.
            mu_log = alpha

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
                mu_log = mu_log + u[group_indices[col]]

            # --- Interaction random effects ---
            # Each two-way interaction is a matrix of offsets, partially pooled
            # toward zero via a shared variance. Think of it as: "how much does
            # the veh_group effect change across age bands?" If sigma_interaction
            # is near zero, the model is effectively additive (like a main-effects GLM).
            for col_a, col_b in self.interaction_pairs:
                n_a = len(group_levels[col_a])
                n_b = len(group_levels[col_b])
                sigma_ix = pm.HalfNormal(
                    f"sigma_{col_a}_x_{col_b}",
                    sigma=self.variance_prior_sigma * 0.5,  # interactions smaller than main effects
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
                ix_a = group_indices[col_a]
                ix_b = group_indices[col_b]
                mu_log = mu_log + u_ix[ix_a, ix_b]

            # --- Likelihood ---
            lambda_ = pm.math.exp(mu_log) * exposure

            if self.overdispersion:
                # Negative Binomial: marginalises over Gamma-distributed Poisson rate.
                # Handles overdispersion that arises from unobserved heterogeneity.
                # alpha_nb controls dispersion: as alpha_nb -> inf, NB -> Poisson.
                alpha_nb = pm.HalfNormal("alpha_nb", sigma=2.0)
                _ = pm.NegativeBinomial(
                    "claims",
                    mu=lambda_,
                    alpha=alpha_nb,
                    observed=observed_claims,
                )
            else:
                _ = pm.Poisson(
                    "claims",
                    mu=lambda_,
                    observed=observed_claims,
                )

        self._model = model
        self._segment_data = df
        self._group_indices = group_indices
        self._group_levels = group_levels

        # Sample
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
            else:  # pathfinder
                idata = pm.fit(
                    method="pathfinder",
                    draws=sampler_config.draws,
                    random_seed=sampler_config.random_seed,
                    progressbar=True,
                )

            # Posterior predictive for diagnostics and calibration
            idata = pm.sample_posterior_predictive(
                idata,
                extend_inferencedata=True,
            )

        self._idata = idata
        self._fitted = True
        return self

    def predict(self, quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Return posterior predictive claim rates for each segment.

        The "claim rate" here is claims per policy-year -- the lambda parameter
        in the Poisson model, not the raw claim count.

        Returns a DataFrame indexed the same as the training data, with columns:
            - mean: posterior mean claim rate (use this as your point estimate)
            - The quantiles you specified (default: p5, p50, p95)
            - shrinkage: how much this segment was pulled toward the portfolio mean
              (0 = fully pooled to mean, 1 = trusts own data completely)

        Args:
            quantiles: Posterior quantiles to return. Default gives 90% credible
                interval + median.
        """
        self._check_fitted()
        import arviz as az

        # Extract posterior for alpha and all u_ terms
        posterior = self._idata.posterior

        # Build the log-rate posterior: alpha + sum of u_k[group_k]
        alpha_samples = posterior["alpha"].values.reshape(-1)  # (chains*draws,)

        log_rate_samples = np.broadcast_to(
            alpha_samples[:, np.newaxis],
            (len(alpha_samples), len(self._segment_data)),
        ).copy()

        for col in self.group_cols:
            u_samples = posterior[f"u_{col}"].values  # (chains, draws, n_levels)
            u_samples = u_samples.reshape(-1, u_samples.shape[-1])  # (chains*draws, n_levels)
            idx = self._group_indices[col]
            log_rate_samples += u_samples[:, idx]

        for col_a, col_b in self.interaction_pairs:
            u_ix = posterior[f"u_{col_a}_x_{col_b}"].values
            u_ix = u_ix.reshape(-1, u_ix.shape[-2], u_ix.shape[-1])
            ix_a = self._group_indices[col_a]
            ix_b = self._group_indices[col_b]
            log_rate_samples += u_ix[:, ix_a, ix_b]

        rate_samples = np.exp(log_rate_samples)  # (n_samples, n_segments)

        result = self._segment_data[self.group_cols].copy()
        result["mean"] = rate_samples.mean(axis=0)
        for q in quantiles:
            result[f"p{int(q * 100)}"] = np.quantile(rate_samples, q, axis=0)

        # Shrinkage: ratio of posterior variance to prior variance.
        # High shrinkage -> segment is data-dominated.
        # Low shrinkage -> segment was pooled toward the mean.
        prior_var = np.exp(self.prior_sigma_rate**2) - 1  # log-normal variance approximation
        posterior_var = rate_samples.var(axis=0)
        # Simple credibility-style shrinkage: compare segment rate to portfolio mean
        portfolio_mean = np.exp(alpha_samples.mean())
        observed_rates = (
            self._segment_data.iloc[:, -2].values  # claim count col
            / self._segment_data.iloc[:, -1].values  # exposure col
            if False  # disabled: use posterior instead
            else rate_samples.mean(axis=0)
        )

        # Credibility factor: Z = (posterior mean - portfolio mean) / (observed rate - portfolio mean)
        # Undefined when observed rate == portfolio mean; clamp to [0, 1].
        obs = self._get_observed_rates()
        denom = obs - portfolio_mean
        num = result["mean"].values - portfolio_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(np.abs(denom) < 1e-10, 0.5, np.clip(num / denom, 0, 1))
        result["credibility_factor"] = z

        return result

    def variance_components(self) -> pd.DataFrame:
        """Return posterior summary of variance hyperparameters.

        These are the sigma parameters that control how much each grouping factor
        is allowed to vary. Large sigma means the factor has a strong effect and
        segments genuinely differ. Small sigma means segments are similar and
        get pulled strongly toward the portfolio mean.

        On the log scale, sigma = 0.3 corresponds to a typical between-segment
        spread of about ±35% from the portfolio mean.

        Returns a DataFrame with one row per grouping factor (and one per
        interaction pair), with columns: mean, sd, hdi_3%, hdi_97%.
        """
        self._check_fitted()
        import arviz as az

        var_names = [f"sigma_{col}" for col in self.group_cols]
        var_names += [f"sigma_{a}_x_{b}" for a, b in self.interaction_pairs]

        summary = az.summary(
            self._idata,
            var_names=var_names,
            stat_funcs=None,
            hdi_prob=0.94,
        )

        # Add human interpretation column
        summary["typical_relativity_spread"] = np.exp(summary["mean"]) - 1

        return summary

    @property
    def idata(self):
        """ArviZ InferenceData object. Use for diagnostics via arviz directly."""
        self._check_fitted()
        return self._idata

    @property
    def model(self):
        """The underlying PyMC model object."""
        self._check_fitted()
        return self._model

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )

    def _get_observed_rates(self) -> np.ndarray:
        """Raw observed claim rates per segment (noisy, uncredibility-weighted)."""
        df = self._segment_data
        # The claim and exposure cols are the last two numeric columns added during fit
        # We preserve references to them via the stored DataFrame
        claim_col = [c for c in df.columns if c not in self.group_cols][0]
        exp_col = [c for c in df.columns if c not in self.group_cols][1]
        return df[claim_col].values / df[exp_col].values
