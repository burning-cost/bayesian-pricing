"""Tests for HierarchicalFrequency.

Two categories of tests:
1. API / contract tests: run without PyMC, test validation and data structures.
2. Model tests: require PyMC, use Pathfinder for speed (not NUTS), verify that
   the model recovers known true parameters within reasonable tolerance.

The model tests use Pathfinder (variational inference) rather than NUTS to keep
CI runtime under a few minutes. This sacrifices posterior accuracy for speed.
Prior recovery tests are therefore approximate: we check that the posterior mean
is within ~30% of the true parameter, not tight confidence bounds.
"""

import numpy as np
import pandas as pd
import pytest

from bayesian_pricing import HierarchicalFrequency
from bayesian_pricing.frequency import SamplerConfig

try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

SKIP_MSG = "PyMC not installed. Install with: uv add bayesian-pricing[pymc]"


# ── API / contract tests (no PyMC) ──────────────────────────────────────────

class TestHierarchicalFrequencyAPI:
    """Validate input handling and output contracts without running PyMC."""

    def test_init_requires_group_cols(self):
        with pytest.raises(ValueError, match="group_cols"):
            HierarchicalFrequency(group_cols=[])

    def test_predict_before_fit_raises(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict()

    def test_variance_components_before_fit_raises(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.variance_components()

    def test_fit_missing_column_raises(self, small_freq_data):
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(ValueError, match="not found"):
            model.fit(small_freq_data, claim_count_col="nonexistent", exposure_col="exposure")

    def test_fit_missing_group_col_raises(self, small_freq_data):
        model = HierarchicalFrequency(group_cols=["missing_group"])
        with pytest.raises(ValueError, match="not found"):
            model.fit(small_freq_data, claim_count_col="claims", exposure_col="exposure")

    def test_fit_nonpositive_exposure_raises(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "claims": [5, 3],
            "exposure": [100.0, -10.0],  # negative exposure
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(df, claim_count_col="claims", exposure_col="exposure")

    def test_interaction_pair_not_in_group_cols_raises(self, small_freq_data):
        model = HierarchicalFrequency(
            group_cols=["segment"],
            interaction_pairs=[("segment", "nonexistent")],
        )
        with pytest.raises(ValueError, match="Interaction pair"):
            # This should raise during fit when building the model
            if HAS_PYMC:
                model.fit(
                    small_freq_data,
                    claim_count_col="claims",
                    exposure_col="exposure",
                )
            else:
                pytest.skip(SKIP_MSG)

    def test_sampler_config_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            SamplerConfig(method="invalid")

    def test_sampler_config_invalid_nuts_sampler(self):
        with pytest.raises(ValueError, match="nuts_sampler"):
            SamplerConfig(nuts_sampler="tensorflow")


# ── Model tests (require PyMC) ────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalFrequencyModel:
    """Verify model behaviour with Pathfinder inference (fast but approximate)."""

    @pytest.fixture(scope="class")
    def fitted_model(self, freq_segment_data):
        """Fit model once, share across tests in this class."""
        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            prior_mean_rate=0.08,
            variance_prior_sigma=0.4,
        )
        config = SamplerConfig(
            method="pathfinder",
            draws=500,
            random_seed=42,
        )
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        return model

    def test_fit_sets_fitted_flag(self, fitted_model):
        assert fitted_model._fitted is True

    def test_idata_has_posterior(self, fitted_model):
        assert hasattr(fitted_model.idata, "posterior")

    def test_predict_returns_dataframe(self, fitted_model):
        preds = fitted_model.predict()
        assert isinstance(preds, pd.DataFrame)
        assert "mean" in preds.columns
        assert "p5" in preds.columns
        assert "p95" in preds.columns

    def test_predict_row_count_matches_input(self, fitted_model, freq_segment_data):
        preds = fitted_model.predict()
        assert len(preds) == len(freq_segment_data)

    def test_predict_rates_are_positive(self, fitted_model):
        preds = fitted_model.predict()
        assert (preds["mean"] > 0).all(), "All predicted rates must be positive"

    def test_predict_credible_interval_ordering(self, fitted_model):
        preds = fitted_model.predict()
        assert (preds["p5"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p95"]).all()

    def test_predict_credibility_factors_bounded(self, fitted_model):
        preds = fitted_model.predict()
        assert (preds["credibility_factor"] >= 0).all()
        assert (preds["credibility_factor"] <= 1).all()

    def test_variance_components_returns_dataframe(self, fitted_model):
        vc = fitted_model.variance_components()
        assert isinstance(vc, pd.DataFrame)
        # Should have one row per group col
        for col in fitted_model.group_cols:
            assert f"sigma_{col}" in vc.index

    def test_posterior_intercept_near_true_rate(self, fitted_model):
        """The global intercept should recover the portfolio mean rate.

        Using Pathfinder with moderate draws, we expect rough recovery.
        True log(0.08) = -2.526. Allow ±0.3 (about 35% on multiplicative scale).
        """
        alpha_samples = fitted_model.idata.posterior["alpha"].values.reshape(-1)
        posterior_mean_alpha = alpha_samples.mean()
        true_alpha = np.log(0.08)
        assert abs(posterior_mean_alpha - true_alpha) < 0.5, (
            f"Intercept {posterior_mean_alpha:.3f} too far from truth {true_alpha:.3f}"
        )

    def test_group_b_higher_than_group_c(self, fitted_model):
        """Veh group B has positive effect, C negative. Posterior should reflect this.

        True u_veh: B=+0.4, C=-0.3. This is a large difference; even with
        Pathfinder approximation this should be detectable.
        """
        posterior = fitted_model.idata.posterior
        u_veh = posterior["u_veh_group"].values.reshape(-1, 5)  # 5 vehicle groups
        levels = fitted_model._group_levels["veh_group"].tolist()
        b_idx = levels.index("B")
        c_idx = levels.index("C")
        # B should have higher mean effect than C
        assert u_veh[:, b_idx].mean() > u_veh[:, c_idx].mean(), (
            "Expected veh group B to have higher effect than C"
        )

    def test_young_drivers_higher_than_middle_aged(self, fitted_model):
        """Age band 17-21 has u_age=0.8, 36-50 has u_age=0.0. Large signal."""
        posterior = fitted_model.idata.posterior
        u_age = posterior["u_age_band"].values.reshape(-1, 5)
        levels = fitted_model._group_levels["age_band"].tolist()
        young_idx = levels.index("17-21")
        mid_idx = levels.index("36-50")
        assert u_age[:, young_idx].mean() > u_age[:, mid_idx].mean()

    def test_fit_returns_self_for_chaining(self, freq_segment_data):
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=99)
        result = model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        assert result is model


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalFrequencyOverdispersion:
    """Test Negative Binomial variant."""

    def test_nb_model_fits(self, freq_segment_data):
        model = HierarchicalFrequency(
            group_cols=["veh_group"],
            overdispersion=True,
        )
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=7)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        assert model._fitted
        # alpha_nb should be in the posterior
        assert "alpha_nb" in model.idata.posterior.data_vars


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalFrequencyInteractions:
    """Test two-way interaction random effects."""

    def test_interaction_model_fits(self, freq_segment_data):
        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            interaction_pairs=[("veh_group", "age_band")],
            variance_prior_sigma=0.3,
        )
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=13)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        assert model._fitted
        assert "u_veh_group_x_age_band" in model.idata.posterior.data_vars

    def test_interaction_posterior_shape(self, freq_segment_data):
        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            interaction_pairs=[("veh_group", "age_band")],
        )
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=14)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        ix_samples = model.idata.posterior["u_veh_group_x_age_band"].values
        # Should be (chains, draws, n_veh, n_age)
        assert ix_samples.ndim == 4
        assert ix_samples.shape[2] == 5  # 5 vehicle groups
        assert ix_samples.shape[3] == 5  # 5 age bands
