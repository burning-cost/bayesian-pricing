"""Internal utilities for bayesian-pricing."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


# Type alias for DataFrame inputs we accept
DataFrameLike = Union[pd.DataFrame, "pl.DataFrame"]


def _to_pandas(data: DataFrameLike) -> pd.DataFrame:
    """Accept a pandas or Polars DataFrame and return a pandas DataFrame.

    This is the single point of conversion. All public API methods call this
    at the boundary so the internal machinery always works with pandas (which
    PyMC and ArviZ require). Callers never need to know what type was passed in.
    """
    # Import lazily so polars is not a hard dependency
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
    except ImportError:
        pass  # polars not installed; anything passed in must already be pandas

    if isinstance(data, pd.DataFrame):
        return data

    raise TypeError(
        f"Expected a pandas or Polars DataFrame, got {type(data).__name__}. "
        "Install polars with: uv add polars"
    )


def _check_numpy_for_pymc() -> None:
    """Raise a clear RuntimeError if numpy is too old for PyMC 5.8+.

    PyMC 5.8 and later depend on pytensor 2.18+ which requires numpy>=2.0.
    Environments with pinned numpy (e.g. Databricks serverless, some conda
    setups) will install bayesian-pricing[pymc] successfully but fail at import
    time with an opaque AttributeError. This check surfaces the issue clearly.
    """
    numpy_version = tuple(int(x) for x in np.__version__.split(".")[:2])
    if numpy_version < (2, 0):
        raise RuntimeError(
            f"PyMC 5.8+ requires numpy>=2.0, but numpy {np.__version__} is installed.\n\n"
            "On most systems, upgrading numpy is straightforward:\n\n"
            "    pip install 'numpy>=2.0'\n\n"
            "On Databricks serverless, numpy is locked at the system level and cannot\n"
            "be upgraded. To use bayesian-pricing on Databricks serverless, use a\n"
            "Databricks ML Runtime cluster (14.3+) which ships with numpy 2.x, or\n"
            "install the NumPyro backend instead:\n\n"
            "    pip install 'bayesian-pricing[numpyro]'\n\n"
            "See: https://github.com/burning-cost/bayesian-pricing#installation"
        )


def _check_pymc() -> None:
    """Raise a helpful error if PyMC is not installed or incompatible."""
    # Check numpy version first — the PyMC import itself will fail with a
    # confusing AttributeError if numpy < 2.0, so we pre-empt it here.
    _check_numpy_for_pymc()

    try:
        import pymc  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyMC is required for fitting Bayesian models. Install it with:\n\n"
            "    uv add pymc\n\n"
            "Or install this package with the pymc extras:\n\n"
            "    uv add 'bayesian-pricing[pymc]'\n\n"
            "PyMC requires numpy>=2.0. See the installation notes above if\n"
            "you are in an environment with a locked numpy version.\n\n"
            "For GPU acceleration (large portfolios), install with NumPyro backend:\n"
            "    uv add 'bayesian-pricing[numpyro]'"
        )


def _validate_columns_present(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any column is missing from the DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def _validate_no_nulls_in_group_cols(df: pd.DataFrame, group_cols: list[str]) -> None:
    """Raise ValueError if any group column contains null values.

    This check runs before _check_pymc() so callers without PyMC installed
    still get a clear ValueError (not an ImportError) for null group inputs.
    """
    for col in group_cols:
        if col not in df.columns:
            continue  # _validate_columns_present handles missing columns
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"Group column '{col}' contains {null_count} null value(s). "
                "Fill or drop null rows before fitting. "
                "Null group labels cannot be assigned to a partial-pooling segment."
            )


def _validate_positive(series: pd.Series, name: str) -> None:
    """Raise ValueError if any value is non-positive."""
    if (series <= 0).any():
        n_bad = (series <= 0).sum()
        raise ValueError(
            f"Column '{name}' must be strictly positive. "
            f"Found {n_bad} non-positive values. "
            f"Min value: {series.min()}"
        )


def _portfolio_mean_rate(
    claims: pd.Series, exposure: pd.Series
) -> float:
    """Compute exposure-weighted portfolio mean claim rate.

    This is the maximum likelihood estimate of the overall Poisson rate --
    total claims divided by total exposure. Used as the prior mean for the
    intercept when the user does not provide one.

    The prior should ideally come from a long-run average, not the training data,
    to avoid the prior adapting to the same data the likelihood uses. But in
    practice, for a weakly informative prior (sigma=0.5 on log scale), this
    makes little difference.
    """
    total_claims = float(claims.sum())
    total_exposure = float(exposure.sum())
    if total_exposure == 0:
        raise ValueError("Total exposure is zero. Cannot compute portfolio mean rate.")
    if total_claims == 0:
        # Avoid log(0) in prior; use a small positive value
        return 0.001
    return total_claims / total_exposure


def _segment_index(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Convert a categorical series to integer indices and unique levels.

    Returns:
        (indices, levels): indices maps each row to a position in levels.
    """
    # Use pandas Categorical for consistent ordering
    cat = pd.Categorical(series)
    indices = cat.codes.copy()
    levels = np.array(cat.categories)

    if (indices < 0).any():
        n_null = (indices < 0).sum()
        raise ValueError(
            f"Column '{series.name}' contains {n_null} null/NaN values. "
            "Fill or drop these before fitting."
        )

    return indices, levels
