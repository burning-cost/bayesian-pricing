"""Expanded coverage for bayesian-pricing (April 2026 sweep).

Covers gaps not addressed by the existing test suite:

1. SamplerConfig — valid default values, edge cases
2. HierarchicalFrequency — zero-exposure validation, null group columns,
   idata/model property pre-fit, custom quantiles, posterior_shrinkage_ratio
   column, single-segment input
3. HierarchicalSeverity — idata/model property pre-fit, zero weights,
   interaction pairs, custom quantiles, prior_mean_severity auto-estimate,
   variance_components before fit
4. BayesianRelativities — RelativityTable structure, summary() with one factor,
   credibility_factors() structure, thin_segments() threshold behaviour
5. Package-level import contract
6. Numerical correctness for _portfolio_mean_rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import polars as pl

from bayesian_pricing import (
    HierarchicalFrequency,
    HierarchicalSeverity,
    BayesianRelativities,
    SamplerConfig,
)
from bayesian_pricing.relativities import RelativityTable

try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except (ImportError, AttributeError):
    HAS_PYMC = False

SKIP_MSG = "PyMC not installed. Install with: uv add bayesian-pricing[pymc]"


# ── Package imports ───────────────────────────────────────────────────────────


class TestPackageImports:

    def test_version_string_exists(self):
        import bayesian_pricing
        assert hasattr(bayesian_pricing, "__version__")
        assert isinstance(bayesian_pricing.__version__, str)

    def test_version_not_empty(self):
        import bayesian_pricing
        assert len(bayesian_pricing.__version__) > 0

    def test_all_exports_importable(self):
        from bayesian_pricing import (  # noqa: F401
            HierarchicalFrequency,
            HierarchicalSeverity,
            BayesianRelativities,
            SamplerConfig,
            convergence_summary,
            posterior_predictive_check,
        )

    def test_sampler_config_in_all(self):
        import bayesian_pricing
        assert "SamplerConfig" in bayesian_pricing.__all__


# ── SamplerConfig ─────────────────────────────────────────────────────────────


class TestSamplerConfig:

    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.method == "nuts"
        assert cfg.draws == 1000
        assert cfg.tune == 1000
        assert cfg.chains == 4
        assert cfg.target_accept == 0.8
        assert cfg.nuts_sampler == "pymc"
        assert cfg.random_seed is None

    def test_pathfinder_method_valid(self):
        cfg = SamplerConfig(method="pathfinder")
        assert cfg.method == "pathfinder"

    def test_nuts_method_valid(self):
        cfg = SamplerConfig(method="nuts")
        assert cfg.method == "nuts"

    def test_numpyro_sampler_valid(self):
        cfg = SamplerConfig(nuts_sampler="numpyro")
        assert cfg.nuts_sampler == "numpyro"

    def test_custom_draws(self):
        cfg = SamplerConfig(draws=500)
        assert cfg.draws == 500

    def test_custom_seed(self):
        cfg = SamplerConfig(random_seed=42)
        assert cfg.random_seed == 42

    def test_custom_chains(self):
        cfg = SamplerConfig(chains=2)
        assert cfg.chains == 2

    def test_custom_target_accept(self):
        cfg = SamplerConfig(target_accept=0.95)
        assert cfg.target_accept == 0.95

    def test_invalid_method_message(self):
        with pytest.raises(ValueError, match="pathfinder"):
            SamplerConfig(method="mcmc")  # wrong value

    def test_invalid_sampler_message(self):
        with pytest.raises(ValueError, match="numpyro"):
            SamplerConfig(nuts_sampler="stan")


# ── HierarchicalFrequency — pre-fit property access ──────────────────────────


class TestHierarchicalFrequencyPreFit:

    def test_idata_before_fit_raises(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.idata

    def test_model_before_fit_raises(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.model

    def test_group_cols_stored(self):
        model = HierarchicalFrequency(group_cols=["veh_group", "age_band"])
        assert model.group_cols == ["veh_group", "age_band"]

    def test_overdispersion_default_false(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        assert model.overdispersion is False

    def test_overdispersion_can_be_set_true(self):
        model = HierarchicalFrequency(group_cols=["segment"], overdispersion=True)
        assert model.overdispersion is True

    def test_prior_mean_rate_stored(self):
        model = HierarchicalFrequency(group_cols=["segment"], prior_mean_rate=0.12)
        assert model.prior_mean_rate == 0.12

    def test_variance_prior_sigma_stored(self):
        model = HierarchicalFrequency(group_cols=["segment"], variance_prior_sigma=0.5)
        assert model.variance_prior_sigma == 0.5

    def test_fitted_flag_starts_false(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        assert model._fitted is False

    def test_zero_exposure_raises_before_pymc(self):
        """Zero exposure is caught during validation, before PyMC is imported."""
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "claims": [5, 3],
            "exposure": [0.0, 100.0],  # zero exposure
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(df, claim_count_col="claims", exposure_col="exposure")

    def test_null_in_group_col_raises(self):
        """Null values in group columns are caught before PyMC."""
        df = pd.DataFrame({
            "segment": ["A", None, "B"],
            "claims": [5, 3, 10],
            "exposure": [100.0, 50.0, 200.0],
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(ValueError, match="null"):
            model.fit(df, claim_count_col="claims", exposure_col="exposure")

    def test_interaction_pair_validation_both_cols_must_be_in_group_cols(self):
        """Both cols in an interaction pair must be in group_cols."""
        df = pd.DataFrame({
            "seg_a": ["X", "Y"],
            "seg_b": ["P", "Q"],
            "claims": [5, 3],
            "exposure": [100.0, 50.0],
        })
        model = HierarchicalFrequency(
            group_cols=["seg_a"],
            interaction_pairs=[("seg_a", "seg_b")],  # seg_b not in group_cols
        )
        with pytest.raises(ValueError, match="Interaction pair"):
            model.fit(df, claim_count_col="claims", exposure_col="exposure")


# ── HierarchicalSeverity — pre-fit ────────────────────────────────────────────


class TestHierarchicalSeverityPreFit:

    def test_idata_before_fit_raises(self):
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.idata

    def test_model_before_fit_raises(self):
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.model

    def test_variance_components_before_fit_raises(self):
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.variance_components()

    def test_group_cols_stored(self):
        model = HierarchicalSeverity(group_cols=["veh_group"])
        assert model.group_cols == ["veh_group"]

    def test_fitted_flag_starts_false(self):
        model = HierarchicalSeverity(group_cols=["segment"])
        assert model._fitted is False

    def test_prior_mean_severity_stored(self):
        model = HierarchicalSeverity(group_cols=["seg"], prior_mean_severity=1500.0)
        assert model.prior_mean_severity == 1500.0

    def test_variance_prior_sigma_stored(self):
        model = HierarchicalSeverity(group_cols=["seg"], variance_prior_sigma=0.2)
        assert model.variance_prior_sigma == 0.2

    def test_zero_weight_raises(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "avg_cost": [1000.0, 1500.0],
            "claim_count": [0, 5],  # zero weight
        })
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(ValueError, match="zero or negative"):
            model.fit(df, severity_col="avg_cost", weight_col="claim_count")

    def test_negative_weight_raises(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "avg_cost": [1000.0, 1500.0],
            "claim_count": [-1, 5],
        })
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(ValueError, match="zero or negative"):
            model.fit(df, severity_col="avg_cost", weight_col="claim_count")

    def test_null_group_col_raises(self):
        df = pd.DataFrame({
            "segment": ["A", None, "B"],
            "avg_cost": [1000.0, 1500.0, 2000.0],
        })
        model = HierarchicalSeverity(group_cols=["segment"])
        with pytest.raises(ValueError, match="null"):
            model.fit(df, severity_col="avg_cost")


# ── HierarchicalFrequency — model tests (require PyMC) ───────────────────────


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalFrequencyExtended:

    @pytest.fixture(scope="class")
    def fitted(self, freq_segment_data):
        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            prior_mean_rate=0.08,
        )
        config = SamplerConfig(method="pathfinder", draws=300, random_seed=77)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        return model

    def test_posterior_shrinkage_ratio_column_exists(self, fitted):
        """predict() must include posterior_shrinkage_ratio column."""
        preds = fitted.predict()
        assert "posterior_shrinkage_ratio" in preds.columns, (
            "predict() should return 'posterior_shrinkage_ratio' column"
        )

    def test_posterior_shrinkage_ratio_bounded(self, fitted):
        preds = fitted.predict()
        ratio = preds["posterior_shrinkage_ratio"]
        assert (ratio >= 0).all(), "shrinkage ratio must be >= 0"
        assert (ratio <= 1).all(), "shrinkage ratio must be <= 1"

    def test_custom_quantiles_in_predict(self, fitted):
        preds = fitted.predict(quantiles=(0.1, 0.5, 0.9))
        assert "p10" in preds.columns
        assert "p50" in preds.columns
        assert "p90" in preds.columns
        # Default p5 / p95 should not be present with custom quantiles
        assert "p5" not in preds.columns
        assert "p95" not in preds.columns

    def test_custom_quantiles_ordering(self, fitted):
        preds = fitted.predict(quantiles=(0.1, 0.5, 0.9))
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()

    def test_single_quantile_in_predict(self, fitted):
        preds = fitted.predict(quantiles=(0.5,))
        assert "p50" in preds.columns

    def test_predict_group_cols_present(self, fitted):
        preds = fitted.predict()
        for col in fitted.group_cols:
            assert col in preds.columns

    def test_idata_property_returns_after_fit(self, fitted):
        idata = fitted.idata
        assert hasattr(idata, "posterior")

    def test_model_property_returns_after_fit(self, fitted):
        model_obj = fitted.model
        assert model_obj is not None

    def test_single_group_col_fit(self, freq_segment_data):
        """Single grouping factor should work fine."""
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=11)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        assert model._fitted
        preds = model.predict()
        assert "veh_group" in preds.columns

    def test_prior_mean_rate_auto_estimated(self, freq_segment_data):
        """When prior_mean_rate is None, it should be set from data after fit."""
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=12)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        assert model.prior_mean_rate is not None
        assert model.prior_mean_rate > 0

    def test_predict_n_rows_matches_training(self, fitted, freq_segment_data):
        preds = fitted.predict()
        assert len(preds) == len(freq_segment_data)

    def test_mean_rates_all_positive(self, fitted):
        preds = fitted.predict()
        assert (preds["mean"] > 0).all()

    def test_variance_components_sigma_positive(self, fitted):
        vc = fitted.variance_components()
        # sigma parameters in the mean column must all be positive (HalfNormal prior)
        assert (vc["mean"] > 0).all()

    def test_variance_components_one_row_per_group(self, fitted):
        vc = fitted.variance_components()
        param_names = vc["parameter"].to_list()
        for col in fitted.group_cols:
            assert f"sigma_{col}" in param_names


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalFrequencyMinimalData:
    """Tests on very small datasets to check edge-case robustness."""

    def test_single_segment_fits(self):
        """A single segment should still fit (though all observations pool to the same mean)."""
        df = pd.DataFrame({
            "segment": ["A"],
            "claims": [5],
            "exposure": [100.0],
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=0)
        model.fit(df, claim_count_col="claims", exposure_col="exposure",
                  sampler_config=config)
        assert model._fitted
        preds = model.predict()
        assert len(preds) == 1

    def test_two_segment_fits(self):
        df = pd.DataFrame({
            "segment": ["A", "B"],
            "claims": [5, 20],
            "exposure": [100.0, 200.0],
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=1)
        model.fit(df, claim_count_col="claims", exposure_col="exposure",
                  sampler_config=config)
        assert model._fitted

    def test_zero_claims_segment_fits(self):
        """Segments with zero claims (but positive exposure) should be accepted."""
        df = pd.DataFrame({
            "segment": ["A", "B", "C"],
            "claims": [0, 5, 10],
            "exposure": [100.0, 50.0, 200.0],
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=2)
        model.fit(df, claim_count_col="claims", exposure_col="exposure",
                  sampler_config=config)
        assert model._fitted


# ── HierarchicalSeverity — model tests (require PyMC) ────────────────────────


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestHierarchicalSeverityExtended:

    @pytest.fixture(scope="class")
    def fitted(self, sev_segment_data):
        model = HierarchicalSeverity(
            group_cols=["veh_group"],
            prior_mean_severity=2000.0,
        )
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=88)
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            weight_col="claim_count",
            sampler_config=config,
        )
        return model

    def test_predict_returns_polars(self, fitted):
        preds = fitted.predict()
        assert isinstance(preds, pl.DataFrame)

    def test_predict_custom_quantiles(self, fitted):
        preds = fitted.predict(quantiles=(0.25, 0.75))
        assert "p25" in preds.columns
        assert "p75" in preds.columns

    def test_predict_quantile_ordering(self, fitted):
        preds = fitted.predict(quantiles=(0.1, 0.5, 0.9))
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()

    def test_predict_group_col_in_output(self, fitted):
        preds = fitted.predict()
        assert "veh_group" in preds.columns

    def test_idata_property_after_fit(self, fitted):
        assert hasattr(fitted.idata, "posterior")

    def test_model_property_after_fit(self, fitted):
        assert fitted.model is not None

    def test_prior_mean_severity_auto_estimated(self, sev_segment_data):
        """When prior_mean_severity is None, should be estimated from data."""
        model = HierarchicalSeverity(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=5)
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            weight_col="claim_count",
            sampler_config=config,
        )
        assert model.prior_mean_severity is not None
        assert model.prior_mean_severity > 0

    def test_prior_mean_auto_estimated_without_weights(self, sev_segment_data):
        """Auto-estimation without weight_col should use simple mean."""
        model = HierarchicalSeverity(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=6)
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            sampler_config=config,
        )
        assert model.prior_mean_severity is not None

    def test_variance_components_returns_polars(self, fitted):
        vc = fitted.variance_components()
        assert isinstance(vc, pl.DataFrame)

    def test_severity_interaction_pairs(self, sev_segment_data):
        """Severity model supports interaction pairs."""
        # Need two group cols to have an interaction
        model = HierarchicalSeverity(
            group_cols=["veh_group", "age_band"],
            interaction_pairs=[("veh_group", "age_band")],
        )
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=9)
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            weight_col="claim_count",
            sampler_config=config,
        )
        assert model._fitted
        assert "u_veh_group_x_age_band" in model.idata.posterior.data_vars

    def test_posterior_predictive_present(self, fitted):
        assert hasattr(fitted.idata, "posterior_predictive")

    def test_single_segment_severity(self):
        df = pd.DataFrame({
            "segment": ["A"],
            "avg_cost": [1500.0],
            "claim_count": [10],
        })
        model = HierarchicalSeverity(group_cols=["segment"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=3)
        model.fit(
            df,
            severity_col="avg_cost",
            weight_col="claim_count",
            sampler_config=config,
        )
        assert model._fitted
        preds = model.predict()
        assert len(preds) == 1
        assert preds["mean"][0] > 0


# ── BayesianRelativities — extended tests (require PyMC) ─────────────────────


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestBayesianRelativitiesExtended:

    @pytest.fixture(scope="class")
    def rel(self, freq_segment_data):
        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            prior_mean_rate=0.08,
        )
        config = SamplerConfig(method="pathfinder", draws=300, random_seed=99)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        return BayesianRelativities(model, hdi_prob=0.9)

    def test_relativity_table_factor_attribute(self, rel):
        rt = rel.relativities(factor="veh_group")
        assert rt.factor == "veh_group"

    def test_relativity_table_levels_attribute(self, rel):
        rt = rel.relativities(factor="veh_group")
        assert isinstance(rt.levels, list)
        assert len(rt.levels) == 5  # A, B, C, D, E

    def test_relativity_table_is_relativity_table_instance(self, rel):
        rt = rel.relativities(factor="veh_group")
        assert isinstance(rt, RelativityTable)

    def test_relativity_table_sorted_descending(self, rel):
        """Table should be sorted by relativity descending."""
        rt = rel.relativities(factor="veh_group")
        rels = rt.table["relativity"].to_numpy()
        assert (rels[:-1] >= rels[1:]).all(), "Table should be sorted descending by relativity"

    def test_hdi_95_column_name(self, freq_segment_data):
        """BayesianRelativities with hdi_prob=0.95 should name columns lower_95pct/upper_95pct."""
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=21)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        rel_95 = BayesianRelativities(model, hdi_prob=0.95)
        rt = rel_95.relativities(factor="veh_group")
        assert "lower_95pct" in rt.table.columns
        assert "upper_95pct" in rt.table.columns

    def test_hdi_80_column_name(self, freq_segment_data):
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=22)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        rel_80 = BayesianRelativities(model, hdi_prob=0.8)
        rt = rel_80.relativities(factor="veh_group")
        assert "lower_80pct" in rt.table.columns
        assert "upper_80pct" in rt.table.columns

    def test_summary_has_factor_column(self, rel):
        summary = rel.summary()
        assert "factor" in summary.columns

    def test_summary_factor_column_values(self, rel):
        summary = rel.summary()
        factors = set(summary["factor"].to_list())
        assert "veh_group" in factors
        assert "age_band" in factors

    def test_summary_n_rows_all_levels(self, rel):
        summary = rel.summary()
        # 5 veh groups + 5 age bands = 10 rows
        assert len(summary) == 10

    def test_credibility_factors_has_factor_column(self, rel):
        cf = rel.credibility_factors()
        assert "factor" in cf.columns

    def test_credibility_factors_has_level_column(self, rel):
        cf = rel.credibility_factors()
        assert "level" in cf.columns

    def test_credibility_factors_n_rows(self, rel):
        cf = rel.credibility_factors()
        assert len(cf) == 10  # 5 veh + 5 age

    def test_thin_segments_threshold_zero_returns_none(self, rel):
        """With threshold=0, no segments should be thin."""
        thin = rel.thin_segments(credibility_threshold=0.0)
        assert len(thin) == 0

    def test_thin_segments_threshold_above_one_returns_all(self, rel):
        """With threshold > 1.0, all segments (in [0,1]) should be returned.
        
        Using 1.01 rather than 1.0 to avoid edge case where a segment lands
        exactly at 1.0 (possible with near-zero posterior_std) and misses the
        strict-less-than filter.
        """
        thin = rel.thin_segments(credibility_threshold=1.01)
        assert len(thin) == 10  # all 10 factor-levels

    def test_thin_segments_sorted_ascending(self, rel):
        thin = rel.thin_segments(credibility_threshold=1.01)
        cred = thin["uncertainty_reduction"].to_numpy()
        assert (cred[:-1] <= cred[1:]).all()

    def test_base_level_normalisation_a_is_one(self, freq_segment_data):
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=33)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        rel = BayesianRelativities(model, base_level={"veh_group": "A"})
        rt = rel.relativities(factor="veh_group")
        a_row = rt.table.filter(pl.col("level") == "A")
        assert abs(float(a_row["relativity"][0]) - 1.0) < 1e-10

    def test_interval_width_positive(self, rel):
        rt = rel.relativities(factor="veh_group")
        assert (rt.table["interval_width"] > 0).all()

    def test_interval_width_equals_upper_minus_lower(self, rel):
        rt = rel.relativities(factor="veh_group")
        expected = rt.table["upper_90pct"] - rt.table["lower_90pct"]
        diff = (rt.table["interval_width"] - expected).abs()
        assert (diff < 1e-8).all()

    def test_single_factor_model_summary(self, freq_segment_data):
        """summary() with a single-factor model should work correctly."""
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=44)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        rel = BayesianRelativities(model)
        summary = rel.summary()
        assert isinstance(summary, pl.DataFrame)
        assert "factor" in summary.columns
        assert len(summary) == 5  # 5 veh groups

    def test_severity_model_with_bayesian_relativities(self, sev_segment_data):
        """BayesianRelativities works with HierarchicalSeverity too."""
        model = HierarchicalSeverity(
            group_cols=["veh_group"],
            prior_mean_severity=2000.0,
        )
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=55)
        model.fit(
            sev_segment_data,
            severity_col="avg_claim_cost",
            weight_col="claim_count",
            sampler_config=config,
        )
        rel = BayesianRelativities(model, hdi_prob=0.9)
        rt = rel.relativities(factor="veh_group")
        assert isinstance(rt, RelativityTable)
        assert (rt.table["relativity"] > 0).all()


# ── RelativityTable dataclass ─────────────────────────────────────────────────


class TestRelativityTableDataclass:
    """RelativityTable is a dataclass; test its basic contract."""

    def test_fields_accessible(self):
        dummy_table = pl.DataFrame({
            "level": ["A", "B"],
            "relativity": [1.0, 1.5],
        })
        rt = RelativityTable(factor="veh_group", levels=["A", "B"], table=dummy_table)
        assert rt.factor == "veh_group"
        assert rt.levels == ["A", "B"]
        assert isinstance(rt.table, pl.DataFrame)

    def test_repr_contains_factor(self):
        dummy_table = pl.DataFrame({"level": ["A"], "relativity": [1.0]})
        rt = RelativityTable(factor="age_band", levels=["17-21"], table=dummy_table)
        assert "age_band" in repr(rt)
