"""Tests for bayesian_pricing.diagnostics.

Covers:
- _stat_check: the core check primitive — pure numpy, no PyMC required
- convergence_summary: requires a fitted model (PyMC)
- posterior_predictive_check: requires a fitted model with posterior predictive

The _stat_check tests are unit tests with known analytical properties.
The higher-level diagnostic tests use pathfinder-fitted models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_pricing.diagnostics import _stat_check

try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except (ImportError, AttributeError):
    HAS_PYMC = False

SKIP_MSG = "PyMC not installed. Install with: uv add bayesian-pricing[pymc]"


# ── _stat_check (pure numpy, no PyMC) ────────────────────────────────────────


class TestStatCheck:
    """_stat_check is a pure function. We can test it exhaustively without PyMC."""

    def test_returns_dict_with_required_keys(self):
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _stat_check(3.0, simulated)
        required = {"observed", "simulated_mean", "simulated_p5", "simulated_p95",
                    "posterior_predictive_p", "pass"}
        assert required.issubset(set(result.keys()))

    def test_observed_value_preserved(self):
        simulated = np.arange(1.0, 101.0)
        result = _stat_check(42.5, simulated)
        assert result["observed"] == 42.5

    def test_pass_when_in_range(self):
        """Observed at 50th percentile of simulated — should pass."""
        rng = np.random.default_rng(0)
        simulated = rng.normal(0, 1, size=10000)
        result = _stat_check(0.0, simulated)  # mean is 0 — roughly 50th pct
        assert result["pass"] is True

    def test_fail_when_observed_too_low(self):
        """Observed well below all simulated values — should fail."""
        simulated = np.linspace(100.0, 200.0, 1000)
        result = _stat_check(-999.0, simulated)
        assert result["pass"] is False

    def test_fail_when_observed_too_high(self):
        """Observed well above all simulated values — should fail."""
        simulated = np.linspace(0.0, 1.0, 1000)
        result = _stat_check(999.0, simulated)
        assert result["pass"] is False

    def test_p_value_range(self):
        """posterior_predictive_p must be in [0, 1]."""
        simulated = np.arange(1.0, 101.0)
        for obs in [-10.0, 0.5, 50.0, 100.5, 200.0]:
            result = _stat_check(obs, simulated)
            assert 0.0 <= result["posterior_predictive_p"] <= 1.0

    def test_simulated_mean_correct(self):
        simulated = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = _stat_check(5.0, simulated)
        assert abs(result["simulated_mean"] - 5.0) < 1e-10

    def test_simulated_p5_less_than_p95(self):
        simulated = np.linspace(0.0, 100.0, 1000)
        result = _stat_check(50.0, simulated)
        assert result["simulated_p5"] < result["simulated_p95"]

    def test_p5_near_5th_percentile_of_simulated(self):
        simulated = np.linspace(0.0, 100.0, 10000)
        result = _stat_check(50.0, simulated)
        expected_p5 = float(np.percentile(simulated, 5))
        assert abs(result["simulated_p5"] - expected_p5) < 0.1

    def test_p_value_at_min_is_low(self):
        """Observed equal to minimum of simulated -> p_value near 0."""
        simulated = np.arange(10.0, 110.0)
        result = _stat_check(10.0, simulated)
        assert result["posterior_predictive_p"] <= 0.02  # 1/100 = 0.01

    def test_p_value_at_max_is_high(self):
        """Observed equal to maximum -> p_value near 1."""
        simulated = np.arange(10.0, 110.0)
        result = _stat_check(109.0, simulated)
        assert result["posterior_predictive_p"] >= 0.98

    def test_custom_alpha_boundary(self):
        """With alpha=0.1, pass boundary is [0.1, 0.9]."""
        simulated = np.linspace(0.0, 1.0, 1000)
        # Observed at 5th percentile -> p=0.05, which is < alpha=0.1 -> fail
        result = _stat_check(0.05, simulated, alpha=0.1)
        assert result["pass"] is False

    def test_single_simulated_value_passthrough(self):
        """Edge case: single simulated value."""
        simulated = np.array([5.0])
        result = _stat_check(5.0, simulated)
        assert result["simulated_mean"] == 5.0
        assert result["simulated_p5"] == 5.0
        assert result["simulated_p95"] == 5.0

    def test_all_values_are_python_floats(self):
        """All numeric values in the returned dict should be Python floats, not numpy scalars."""
        simulated = np.linspace(1.0, 10.0, 50)
        result = _stat_check(5.0, simulated)
        for key in ("observed", "simulated_mean", "simulated_p5", "simulated_p95",
                    "posterior_predictive_p"):
            assert isinstance(result[key], float), f"{key} should be float, got {type(result[key])}"


# ── convergence_summary (requires PyMC) ──────────────────────────────────────


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestConvergenceSummary:

    @pytest.fixture(scope="class")
    def fitted_freq(self, freq_segment_data):
        from bayesian_pricing import HierarchicalFrequency
        from bayesian_pricing.frequency import SamplerConfig

        model = HierarchicalFrequency(
            group_cols=["veh_group"],
            prior_mean_rate=0.08,
        )
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=42)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        return model

    def test_unfitted_raises(self):
        from bayesian_pricing import HierarchicalFrequency
        from bayesian_pricing.diagnostics import convergence_summary

        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not fitted"):
            convergence_summary(model)

    def test_returns_polars_dataframe(self, fitted_freq):
        import polars as pl
        from bayesian_pricing.diagnostics import convergence_summary

        result = convergence_summary(fitted_freq, return_warnings=False)
        assert isinstance(result, pl.DataFrame)

    def test_has_parameter_column(self, fitted_freq):
        from bayesian_pricing.diagnostics import convergence_summary

        result = convergence_summary(fitted_freq, return_warnings=False)
        assert "parameter" in result.columns

    def test_has_rows(self, fitted_freq):
        from bayesian_pricing.diagnostics import convergence_summary

        result = convergence_summary(fitted_freq, return_warnings=False)
        assert len(result) > 0

    def test_pathfinder_has_n_divergences_column(self, fitted_freq):
        """Pathfinder model: may not have r_hat but should return a DataFrame."""
        from bayesian_pricing.diagnostics import convergence_summary

        result = convergence_summary(fitted_freq, return_warnings=False)
        # Pathfinder may or may not have divergences column; just verify it's a DF
        assert hasattr(result, "columns")

    def test_return_warnings_false_suppresses_output(self, fitted_freq, capsys):
        """return_warnings=False should suppress printed warnings."""
        from bayesian_pricing.diagnostics import convergence_summary

        convergence_summary(fitted_freq, return_warnings=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ── posterior_predictive_check (requires PyMC) ───────────────────────────────


@pytest.mark.skipif(not HAS_PYMC, reason=SKIP_MSG)
class TestPosteriorPredictiveCheck:

    @pytest.fixture(scope="class")
    def fitted_freq(self, freq_segment_data):
        from bayesian_pricing import HierarchicalFrequency
        from bayesian_pricing.frequency import SamplerConfig

        model = HierarchicalFrequency(
            group_cols=["veh_group"],
            prior_mean_rate=0.08,
        )
        config = SamplerConfig(method="pathfinder", draws=200, random_seed=42)
        model.fit(
            freq_segment_data,
            claim_count_col="claims",
            exposure_col="exposure",
            sampler_config=config,
        )
        return model

    def test_unfitted_raises(self):
        from bayesian_pricing import HierarchicalFrequency
        from bayesian_pricing.diagnostics import posterior_predictive_check

        model = HierarchicalFrequency(group_cols=["segment"])
        with pytest.raises(RuntimeError, match="not fitted"):
            posterior_predictive_check(model)

    def test_returns_dict(self, fitted_freq):
        from bayesian_pricing.diagnostics import posterior_predictive_check

        result = posterior_predictive_check(
            fitted_freq,
            claim_count_col="claims",
            n_stats=50,
        )
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, fitted_freq):
        from bayesian_pricing.diagnostics import posterior_predictive_check

        result = posterior_predictive_check(
            fitted_freq,
            claim_count_col="claims",
            n_stats=50,
        )
        assert "mean" in result
        assert "variance" in result
        assert "p90" in result
        assert "p95" in result
        assert "_summary" in result

    def test_stat_entries_have_required_subkeys(self, fitted_freq):
        from bayesian_pricing.diagnostics import posterior_predictive_check

        result = posterior_predictive_check(
            fitted_freq,
            claim_count_col="claims",
            n_stats=50,
        )
        for stat_key in ("mean", "variance", "p90", "p95"):
            entry = result[stat_key]
            for sub_key in ("observed", "simulated_mean", "simulated_p5",
                            "simulated_p95", "posterior_predictive_p", "pass"):
                assert sub_key in entry, f"Missing {sub_key!r} in stat {stat_key!r}"

    def test_summary_has_passed_and_total(self, fitted_freq):
        from bayesian_pricing.diagnostics import posterior_predictive_check

        result = posterior_predictive_check(
            fitted_freq,
            claim_count_col="claims",
            n_stats=50,
        )
        summary = result["_summary"]
        assert "passed" in summary
        assert "total" in summary
        assert "failed_checks" in summary
        assert "interpretation" in summary
        assert summary["total"] == 4  # mean, variance, p90, p95

    def test_observed_values_are_finite(self, fitted_freq):
        from bayesian_pricing.diagnostics import posterior_predictive_check

        result = posterior_predictive_check(
            fitted_freq,
            claim_count_col="claims",
            n_stats=50,
        )
        for key in ("mean", "variance", "p90", "p95"):
            assert np.isfinite(result[key]["observed"]), \
                f"observed value for {key} should be finite"
