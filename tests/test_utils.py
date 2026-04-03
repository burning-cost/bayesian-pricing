"""Tests for bayesian_pricing._utils internal utilities.

These are all pure Python / numpy / pandas tests — no PyMC required.
The utils module is the foundation everything else builds on, yet was
previously entirely untested. Any regression here would silently corrupt
model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import polars as pl

from bayesian_pricing._utils import (
    _to_pandas,
    _validate_columns_present,
    _validate_positive,
    _portfolio_mean_rate,
    _segment_index,
    _check_numpy_for_pymc,
)


# ── _to_pandas ────────────────────────────────────────────────────────────────


class TestToPandas:

    def test_pandas_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_polars_converted(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = _to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert list(result["a"]) == [1, 2, 3]

    def test_polars_values_preserved(self):
        df = pl.DataFrame({"x": ["foo", "bar"], "y": [1.5, 2.5]})
        result = _to_pandas(df)
        assert result["x"].tolist() == ["foo", "bar"]
        assert result["y"].tolist() == [1.5, 2.5]

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas or Polars DataFrame"):
            _to_pandas([1, 2, 3])

    def test_invalid_type_mentions_type_name(self):
        with pytest.raises(TypeError, match="list"):
            _to_pandas([1, 2, 3])

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_pandas({"a": [1, 2]})

    def test_numpy_array_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_pandas(np.array([[1, 2], [3, 4]]))

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_pandas(None)

    def test_polars_empty_dataframe(self):
        df = pl.DataFrame({"a": [], "b": []})
        result = _to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]

    def test_pandas_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = _to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ── _validate_columns_present ─────────────────────────────────────────────────


class TestValidateColumnsPresent:

    def test_all_present_no_error(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        _validate_columns_present(df, ["a", "b"])

    def test_empty_list_no_error(self):
        df = pd.DataFrame({"a": [1]})
        _validate_columns_present(df, [])

    def test_single_missing_raises(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="not found"):
            _validate_columns_present(df, ["a", "missing"])

    def test_multiple_missing_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            _validate_columns_present(df, ["x", "y", "z"])

    def test_error_mentions_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="foo"):
            _validate_columns_present(df, ["foo"])

    def test_error_mentions_available_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="a"):
            _validate_columns_present(df, ["missing"])

    def test_exact_columns_present(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        _validate_columns_present(df, ["x", "y"])  # No error expected

    def test_superset_ok(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        _validate_columns_present(df, ["a", "c"])  # extra cols fine


# ── _validate_positive ────────────────────────────────────────────────────────


class TestValidatePositive:

    def test_all_positive_no_error(self):
        s = pd.Series([1.0, 2.0, 100.0], name="exposure")
        _validate_positive(s, "exposure")  # No error

    def test_single_element_positive(self):
        s = pd.Series([0.001], name="x")
        _validate_positive(s, "x")  # No error

    def test_zero_raises(self):
        s = pd.Series([1.0, 0.0, 2.0], name="exposure")
        with pytest.raises(ValueError, match="strictly positive"):
            _validate_positive(s, "exposure")

    def test_negative_raises(self):
        s = pd.Series([1.0, -0.001, 3.0], name="cost")
        with pytest.raises(ValueError, match="strictly positive"):
            _validate_positive(s, "cost")

    def test_all_negative_raises(self):
        s = pd.Series([-1.0, -2.0], name="x")
        with pytest.raises(ValueError, match="strictly positive"):
            _validate_positive(s, "x")

    def test_error_mentions_column_name(self):
        s = pd.Series([1.0, 0.0], name="my_col")
        with pytest.raises(ValueError, match="my_col"):
            _validate_positive(s, "my_col")

    def test_error_mentions_count_of_bad_values(self):
        s = pd.Series([0.0, -1.0, 1.0], name="x")
        with pytest.raises(ValueError, match="2"):
            _validate_positive(s, "x")

    def test_error_mentions_min_value(self):
        s = pd.Series([1.0, -5.0, 2.0], name="x")
        with pytest.raises(ValueError, match="-5"):
            _validate_positive(s, "x")

    def test_very_small_positive_ok(self):
        s = pd.Series([1e-10, 1e-15], name="tiny")
        _validate_positive(s, "tiny")  # Should not raise

    def test_integer_series_ok(self):
        s = pd.Series([1, 2, 3], name="counts")
        _validate_positive(s, "counts")  # integers, all positive

    def test_integer_series_with_zero_raises(self):
        s = pd.Series([1, 0, 3], name="counts")
        with pytest.raises(ValueError, match="strictly positive"):
            _validate_positive(s, "counts")


# ── _portfolio_mean_rate ──────────────────────────────────────────────────────


class TestPortfolioMeanRate:

    def test_basic_calculation(self):
        claims = pd.Series([10.0, 20.0, 30.0])
        exposure = pd.Series([100.0, 200.0, 300.0])
        rate = _portfolio_mean_rate(claims, exposure)
        # (10+20+30) / (100+200+300) = 60/600 = 0.1
        assert abs(rate - 0.1) < 1e-10

    def test_zero_exposure_raises(self):
        claims = pd.Series([10.0])
        exposure = pd.Series([0.0])
        with pytest.raises(ValueError, match="zero"):
            _portfolio_mean_rate(claims, exposure)

    def test_zero_claims_returns_small_value(self):
        """Zero claims must not produce log(0) later; returns small positive."""
        claims = pd.Series([0.0, 0.0])
        exposure = pd.Series([100.0, 200.0])
        rate = _portfolio_mean_rate(claims, exposure)
        assert rate > 0
        assert rate < 0.01  # Should be small

    def test_single_segment(self):
        claims = pd.Series([5.0])
        exposure = pd.Series([100.0])
        rate = _portfolio_mean_rate(claims, exposure)
        assert abs(rate - 0.05) < 1e-10

    def test_high_exposure_segment_dominates(self):
        # Segment 1: rate = 0.05 (1/20), exposure 10
        # Segment 2: rate = 0.50 (100/200), exposure 200
        # Portfolio: (1 + 100) / (20 + 200) = 101/220 ≈ 0.459
        claims = pd.Series([1.0, 100.0])
        exposure = pd.Series([20.0, 200.0])
        rate = _portfolio_mean_rate(claims, exposure)
        assert abs(rate - 101 / 220) < 1e-10

    def test_returns_float(self):
        claims = pd.Series([5])
        exposure = pd.Series([100])
        result = _portfolio_mean_rate(claims, exposure)
        assert isinstance(result, float)


# ── _segment_index ────────────────────────────────────────────────────────────


class TestSegmentIndex:

    def test_basic_categorical_encoding(self):
        s = pd.Series(["A", "B", "C", "A", "B"], name="group")
        indices, levels = _segment_index(s)
        # Levels should be sorted (Categorical default)
        assert list(levels) == ["A", "B", "C"]
        assert list(indices) == [0, 1, 2, 0, 1]

    def test_indices_are_numpy_array(self):
        s = pd.Series(["X", "Y"], name="g")
        indices, levels = _segment_index(s)
        assert isinstance(indices, np.ndarray)

    def test_levels_are_numpy_array(self):
        s = pd.Series(["X", "Y"], name="g")
        indices, levels = _segment_index(s)
        assert isinstance(levels, np.ndarray)

    def test_single_level(self):
        s = pd.Series(["A", "A", "A"], name="seg")
        indices, levels = _segment_index(s)
        assert list(levels) == ["A"]
        assert list(indices) == [0, 0, 0]

    def test_levels_are_sorted(self):
        # Pandas Categorical sorts alphabetically by default
        s = pd.Series(["C", "A", "B"], name="g")
        _, levels = _segment_index(s)
        assert list(levels) == ["A", "B", "C"]

    def test_indices_are_non_negative(self):
        s = pd.Series(["X", "Y", "Z"], name="g")
        indices, _ = _segment_index(s)
        assert (indices >= 0).all()

    def test_null_values_raise(self):
        s = pd.Series(["A", None, "B"], name="g")
        with pytest.raises(ValueError, match="null"):
            _segment_index(s)

    def test_null_error_mentions_count(self):
        s = pd.Series(["A", None, None, "B"], name="g")
        with pytest.raises(ValueError, match="2"):
            _segment_index(s)

    def test_null_error_mentions_column_name(self):
        s = pd.Series(["A", None], name="veh_group")
        with pytest.raises(ValueError, match="veh_group"):
            _segment_index(s)

    def test_numeric_categories_work(self):
        s = pd.Series([3, 1, 2, 1, 3], name="num_group")
        indices, levels = _segment_index(s)
        # Levels should be [1, 2, 3]
        assert list(levels) == [1, 2, 3]
        assert indices[0] == 2   # 3 -> index 2
        assert indices[1] == 0   # 1 -> index 0

    def test_indices_cover_all_levels(self):
        s = pd.Series(["A", "B", "C", "D", "E"], name="g")
        indices, levels = _segment_index(s)
        assert set(indices) == set(range(len(levels)))

    def test_many_repetitions(self):
        """Large repetitive series should encode correctly."""
        rng = np.random.default_rng(1)
        choices = rng.choice(["X", "Y", "Z"], size=1000)
        s = pd.Series(choices, name="g")
        indices, levels = _segment_index(s)
        assert set(levels) == {"X", "Y", "Z"}
        assert len(indices) == 1000
        # Reconstruct from indices and verify
        reconstructed = np.array(levels)[indices]
        assert (reconstructed == choices).all()


# ── _check_numpy_for_pymc ─────────────────────────────────────────────────────


class TestCheckNumpyForPymc:

    def test_current_numpy_ok(self):
        """This environment must have a numpy version that passes the check,
        OR the check raises a RuntimeError with a helpful message."""
        import numpy as np
        numpy_version = tuple(int(x) for x in np.__version__.split(".")[:2])
        if numpy_version >= (2, 0):
            # Should not raise
            _check_numpy_for_pymc()
        else:
            # Should raise with clear message
            with pytest.raises(RuntimeError, match="numpy>=2.0"):
                _check_numpy_for_pymc()

    def test_error_message_mentions_pymc(self):
        """If numpy is old, error must mention PyMC version requirement."""
        import numpy as np
        numpy_version = tuple(int(x) for x in np.__version__.split(".")[:2])
        if numpy_version < (2, 0):
            with pytest.raises(RuntimeError, match="PyMC"):
                _check_numpy_for_pymc()
        else:
            pytest.skip("numpy is >=2.0 in this environment")
