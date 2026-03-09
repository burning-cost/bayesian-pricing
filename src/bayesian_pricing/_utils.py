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


def _check_pymc() -> None:
    """Raise a helpful ImportError if PyMC is not installed."""
    try:
        import pymc  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyMC is required for fitting Bayesian models. Install it with:\n\n"
            "    uv add pymc\n\n"
            "Or install this package with the pymc extras:\n\n"
            "    uv add bayesian-pricing[pymc]\n\n"
            "PyMC requires C++ compiler tools on some platforms. See:\n"
            "    https://www.pymc.io/projects/docs/en/stable/installation.html\n\n"
            "For GPU acceleration (large portfolios), install with NumPyro backend:\n"
            "    uv add bayesian-pricing[numpyro]"
        )


def _validate_columns_present(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any column is missing from the DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {df.columns.tolist()}"
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
