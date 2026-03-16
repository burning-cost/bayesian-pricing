"""Tests for BayesianRelativities.

Validates the posterior-to-relativity pipeline:
1. Relativities are positive (they are multiplicative factors, bounded below at 0)
2. Credible intervals have correct ordering
3. Base level normalisation works correctly
4. thin_segments() correctly identifies low-credibility levels
5. Output is in a format an actuary can use directly

All output DataFrames are Polars. API tests are partially PyMC-free (they test
error handling on unfitted models). All substantive tests require a fitted model.
"""

import numpy as np
import pandas as pd
import pytest

import polars as pl

from bayesian_pricing import HierarchicalFrequency, BayesianRelativities
from bayesian_pricing.frequency import SamplerConfig
from bayesian_pricing.relativities import RelativityTable

try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except (ImportError, AttributeError):
    HAS_PYMC = False

SKIP_MSG = "PyMC not installed. Install with: uv add bayesian-pricing[pymc]"


class TestBayesianRelativitiesAPI:

    def test_unfitted_model_raises(self):
        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            BayesianRelativities(model)

    def test_invalid_factor_raises(self):
        """After fit, requesting a non-existent factor should raise."""
        if not HAS_PYMC:
            pytest.skip(SKIP_MSG)

        df = pd.DataFrame({
            "segment": ["A", "B", "C"],
            "claims": [10, 5, 20],
            "exposure": [100.0, 50.0, 200.0],
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=0)
        model.fit(df, claim_count_col="claims", exposure_col="exposure", sampler_config=config)
        rel = BayesianRelativities(model)
        with pytest.raises(ValueError, match="not in model group_cols"):
            rel.relativities(factor="nonexistent")

    def test_invalid_base_level_raises(self):
        """Requesting a base level that doesn't exist should raise."""
        if not HAS_PYMC:
            pytest.skip(SKIP_MSG)

        df = pd.DataFrame({
            "segment": ["A", "B", "C"],
            "claims": [10, 5, 20],
            "exposure": [100.0, 50.0, 200.0],
        })
        model = HierarchicalFrequency(group_cols=["segment"])
        config = SamplerConfig(method="pathfinder", draws=100, random_seed=0)
        model.fit(df, claim_count_col="claims", exposure_col="exposure", sampler_config=config)
        rel = BayesianRelativities(model, base_level={"segment": "Z"})  # Z doesn't exist
        with pytest.raises(ValueError, match="Base level"):
            rel.relativities(factor="segment")


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestBayesianRelativitiesOutput:

    @pytest.fixture(scope="class")
    def fitted_rel(self, freq_segment_data):
        """Fitted model and BayesianRelativities object, shared across tests."""
        model = HierarchicalFrequency(
            group_cols=["veh_group", "age_band"],
            prior_mean_rate=0.08,
        )
        config = SamplerConfig(method="pathfinder", draws=500, random_seed=42)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        return BayesianRelativities(model, hdi_prob=0.9)

    def test_relativities_returns_dict(self, fitted_rel):
        result = fitted_rel.relativities()
        assert isinstance(result, dict)
        assert "veh_group" in result
        assert "age_band" in result

    def test_each_factor_is_relativity_table(self, fitted_rel):
        result = fitted_rel.relativities()
        for factor, rt in result.items():
            assert isinstance(rt, RelativityTable)
            assert rt.factor == factor

    def test_relativity_table_has_required_columns(self, fitted_rel):
        rt = fitted_rel.relativities(factor="veh_group")
        required_cols = {"level", "relativity", "lower_90pct", "upper_90pct",
                         "uncertainty_reduction", "interval_width"}
        assert required_cols.issubset(set(rt.table.columns))

    def test_relativity_table_is_polars(self, fitted_rel):
        rt = fitted_rel.relativities(factor="veh_group")
        assert isinstance(rt.table, pl.DataFrame)

    def test_relativities_are_positive(self, fitted_rel):
        result = fitted_rel.relativities()
        for factor, rt in result.items():
            assert (rt.table["relativity"] > 0).all(), (
                f"Non-positive relativity in factor {factor}"
            )

    def test_credible_intervals_ordered(self, fitted_rel):
        result = fitted_rel.relativities()
        for factor, rt in result.items():
            assert (rt.table["lower_90pct"] <= rt.table["upper_90pct"]).all()

    def test_credibility_factors_bounded(self, fitted_rel):
        cred = fitted_rel.credibility_factors()
        assert isinstance(cred, pl.DataFrame)
        assert (cred["uncertainty_reduction"] >= 0).all()
        assert (cred["uncertainty_reduction"] <= 1).all()

    def test_base_level_normalisation(self, freq_segment_data):
        """With base_level={"veh_group": "A"}, group A should have relativity 1.0."""
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=300, random_seed=7)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        rel = BayesianRelativities(model, base_level={"veh_group": "A"})
        rt = rel.relativities(factor="veh_group")
        a_row = rt.table.filter(rt.table["level"] == "A")
        assert len(a_row) == 1
        assert abs(float(a_row["relativity"][0]) - 1.0) < 1e-10, (
            "Base level 'A' should have relativity exactly 1.0"
        )

    def test_veh_group_d_higher_relativity_than_e(self, fitted_rel):
        """True u_veh: D=0.7, E=-0.5. D should have clearly higher relativity."""
        rt = fitted_rel.relativities(factor="veh_group")
        d_rel = float(rt.table.filter(rt.table["level"] == "D")["relativity"][0])
        e_rel = float(rt.table.filter(rt.table["level"] == "E")["relativity"][0])
        assert d_rel > e_rel, (
            f"Expected D ({d_rel:.3f}) > E ({e_rel:.3f}) in veh_group relativities"
        )

    def test_young_drivers_higher_relativity(self, fitted_rel):
        """True u_age: 17-21=0.8, 36-50=0.0. Young should dominate."""
        rt = fitted_rel.relativities(factor="age_band")
        young_rel = float(rt.table.filter(rt.table["level"] == "17-21")["relativity"][0])
        mid_rel = float(rt.table.filter(rt.table["level"] == "36-50")["relativity"][0])
        assert young_rel > mid_rel

    def test_summary_is_polars_long_format(self, fitted_rel):
        summary = fitted_rel.summary()
        assert isinstance(summary, pl.DataFrame)
        assert "factor" in summary.columns
        assert "level" in summary.columns
        assert "relativity" in summary.columns
        # Should have rows for all levels of all factors
        assert len(summary) == 5 + 5  # 5 veh groups + 5 age bands

    def test_thin_segments_returns_low_credibility(self, fitted_rel):
        thin = fitted_rel.thin_segments(credibility_threshold=0.5)
        assert isinstance(thin, pl.DataFrame)
        if len(thin) > 0:
            assert (thin["uncertainty_reduction"] < 0.5).all()

    def test_thin_segments_sorted_ascending(self, fitted_rel):
        thin = fitted_rel.thin_segments(credibility_threshold=1.0)  # all segments
        if len(thin) > 1:
            cred = thin["uncertainty_reduction"].to_numpy()
            assert (cred[1:] >= cred[:-1]).all(), "thin_segments should be sorted by uncertainty_reduction"

    def test_single_factor_returns_relativity_table_not_dict(self, fitted_rel):
        result = fitted_rel.relativities(factor="veh_group")
        assert isinstance(result, RelativityTable)

    def test_hdi_prob_affects_interval_width(self, freq_segment_data):
        """Wider HDI should produce wider credible intervals."""
        model = HierarchicalFrequency(group_cols=["veh_group"])
        config = SamplerConfig(method="pathfinder", draws=300, random_seed=8)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )

        rel_90 = BayesianRelativities(model, hdi_prob=0.9)
        rel_50 = BayesianRelativities(model, hdi_prob=0.5)

        rt_90 = rel_90.relativities(factor="veh_group")
        rt_50 = rel_50.relativities(factor="veh_group")

        # Merge on level to compare interval widths
        merged = rt_90.table.select(["level", "interval_width"]).join(
            rt_50.table.select(["level", "interval_width"]),
            on="level",
            suffix="_50",
        ).rename({"interval_width": "interval_width_90"})

        assert (merged["interval_width_90"] >= merged["interval_width_50"]).all(), (
            "90% HDI should be wider than 50% HDI for all levels"
        )
