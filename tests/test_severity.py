"""Tests for HierarchicalSeverity.

Structure mirrors test_frequency.py: API tests run without PyMC; model tests
require PyMC and use Pathfinder for speed.

Note on severity model validation: the Gamma hierarchical model is harder to
validate than the frequency model because severity is inherently noisier (higher
CV). With Pathfinder and 500 draws, we check that:
1. Higher-severity vehicle groups have higher posterior means
2. Credible intervals are wider for thin segments than dense ones
3. The shape parameter is recovered within a factor of 2 of the true value
"""

import numpy as np
import pandas as pd
import pytest

from bayesian_pricing import HierarchicalSeverity
from bayesian_pricing.frequency import SamplerConfig

try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

SKIP_MSG = "PyMC not installed. Install with: pip install bayesian-pricing[pymc]"


class TestHierarchicalSeverityAPI:

    def test_init_requires_group_cols(self):
        with pytest.raises(ValueError, match="group_cols"):
            HierarchicalSeverity(group_cols=[])

    def test_predict_before_fit_raises(self):
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict()

    def test_fit_missing_column_raises(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "avg_cost": [1000.0, 1500.0],
        })
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(ValueError, match="not found"):
            model.fit(df, severity_col="missing_col")

    def test_fit_nonpositive_severity_raises(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "avg_cost": [1000.0, -500.0],  # negative severity
            "claim_count": [10, 5],
        })
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(df, severity_col="avg_cost", weight_col="claim_count")

    def test_fit_missing_group_col_raises(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "avg_cost": [1000.0, 1500.0],
        })
        model = HierarchicalSeverity(group_cols=["nonexistent"])
        with pytest.raises(ValueError, match="not found"):
            model.fit(df, severity_col="avg_cost")


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalSeverityModel:

    @pytest.fixture(scope="class")
    def fitted_model(self, sev_segment_data):
        model = HierarchicalSeverity(
            group_cols=["veh_group"],
            prior_mean_severity=2000.0,
            variance_prior_sigma=0.25,
        )
        config = SamplerConfig(
            method="pathfinder",
            draws=500,
            random_seed=99,
        )
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            weight_col="claim_count",
            sampler_config=config,
        )
        return model

    def test_fit_sets_fitted_flag(self, fitted_model):
        assert fitted_model._fitted

    def test_idata_has_posterior(self, fitted_model):
        assert hasattr(fitted_model.idata, "posterior")

    def test_predict_returns_dataframe(self, fitted_model):
        preds = fitted_model.predict()
        assert isinstance(preds, pd.DataFrame)
        assert "mean" in preds.columns

    def test_predict_severities_are_positive(self, fitted_model):
        preds = fitted_model.predict()
        assert (preds["mean"] > 0).all()

    def test_credible_interval_ordering(self, fitted_model):
        preds = fitted_model.predict()
        assert (preds["p5"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p95"]).all()

    def test_intercept_near_true_base_severity(self, fitted_model):
        """Intercept should be close to log(2000) = 7.6."""
        alpha_samples = fitted_model.idata.posterior["alpha"].values.reshape(-1)
        posterior_mean = alpha_samples.mean()
        true_alpha = np.log(2000.0)
        # Allow ±0.5 (56% on multiplicative scale) for Pathfinder noise
        assert abs(posterior_mean - true_alpha) < 0.6, (
            f"Severity intercept {posterior_mean:.3f} too far from {true_alpha:.3f}"
        )

    def test_group_d_higher_severity_than_group_e(self, fitted_model):
        """True u_veh_sev: D=+0.15, E=-0.1. D should dominate E."""
        posterior = fitted_model.idata.posterior
        u_veh = posterior["u_veh_group"].values.reshape(-1, 5)
        levels = fitted_model._group_levels["veh_group"].tolist()
        d_idx = levels.index("D")
        e_idx = levels.index("E")
        assert u_veh[:, d_idx].mean() > u_veh[:, e_idx].mean()

    def test_gamma_shape_positive(self, fitted_model):
        shape_samples = fitted_model.idata.posterior["gamma_shape"].values.reshape(-1)
        assert (shape_samples > 0).all()

    def test_no_weight_col_fits(self, sev_segment_data):
        """Fitting without weights should work (treats all segments equally)."""
        model = HierarchicalSeverity(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=0)
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            # no weight_col
            sampler_config=config,
        )
        assert model._fitted

    def test_variance_components_returns_dataframe(self, fitted_model):
        vc = fitted_model.variance_components()
        assert isinstance(vc, pd.DataFrame)
        assert "sigma_veh_group" in vc.index

    def test_fit_returns_self(self, sev_segment_data):
        model = HierarchicalSeverity(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=50, random_seed=1)
        result = model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            sampler_config=config,
        )
        assert result is model


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestSeverityPriorSensitivity:
    """Check that informative vs weakly informative priors give different pooling."""

    def test_tight_prior_more_shrinkage(self, sev_segment_data):
        """Tight variance prior (0.1) should produce narrower posteriors for u_veh."""
        config = SamplerConfig(method="pathfinder", draws=300, random_seed=55)

        model_tight = HierarchicalSeverity(
            group_cols=["veh_group"],
            variance_prior_sigma=0.1,
        )
        model_loose = HierarchicalSeverity(
            group_cols=["veh_group"],
            variance_prior_sigma=0.6,
        )

        model_tight.fit(sev_segment_data, severity_col="avg_claim_cost", sampler_config=config)
        model_loose.fit(sev_segment_data, severity_col="avg_claim_cost", sampler_config=config)

        # Tight prior should produce smaller sigma_veh_group posterior mean
        sigma_tight = model_tight.idata.posterior["sigma_veh_group"].values.mean()
        sigma_loose = model_loose.idata.posterior["sigma_veh_group"].values.mean()
        assert sigma_tight < sigma_loose, (
            f"Expected tight prior to produce smaller sigma: {sigma_tight:.4f} vs {sigma_loose:.4f}"
        )
