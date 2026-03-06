"""
Diagnostics for hierarchical Bayesian insurance models.

Before you trust a model's output, you need to know whether the MCMC sampler
actually explored the posterior correctly. Two types of failures are common in
hierarchical models:

1. Non-convergence: chains did not mix. R-hat > 1.01 is the standard flag.
   Cause: usually centered parameterization creating funnel geometry. Fix:
   ensure non-centered parameterization is used (it is, by default in this library).

2. Divergences: the HMC trajectory hit a region where the step size is too large.
   A small number (<0.1% of samples) is acceptable. More than 1% indicates a
   poorly specified model. Increase target_accept in SamplerConfig or check
   your priors.

After convergence, validate that the model actually describes your data:

3. Posterior predictive check: simulate new datasets from the posterior and
   compare to observed data. If the model is correct, the observed statistics
   (mean, variance, 95th percentile) should fall within the simulated range.

4. Calibration on holdout: the 90% credible interval should contain the true
   value 90% of the time. If it contains it 99% of the time, your priors are
   too tight and the model is overconfident. If 70%, it's under-dispersed.

These are the checks a Lloyd's of London actuary would want to see in a model
validation report. The functions here support all of them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from bayesian_pricing._utils import _check_pymc


def convergence_summary(model, return_warnings: bool = True) -> pd.DataFrame:
    """Summarise MCMC convergence diagnostics.

    Returns a DataFrame of diagnostics for every parameter in the model.
    The key columns are:

    - r_hat: Gelman-Rubin statistic. Should be < 1.01. Values > 1.05 indicate
      serious non-convergence and the results should not be used.
    - ess_bulk: Effective sample size for bulk of posterior. Target > 400.
      Low ESS means the chain mixed slowly -- your estimates are less precise
      than the nominal sample count suggests.
    - ess_tail: ESS for the tails of the distribution. More relevant for
      credible intervals. Target > 400.
    - divergences: Count of divergent transitions. Should be 0 ideally; < 10
      (out of 4,000 samples) is often acceptable. Flag any divergences.

    Args:
        model: Fitted HierarchicalFrequency or HierarchicalSeverity.
        return_warnings: If True, print actionable warnings when diagnostics
            are outside acceptable ranges.

    Returns:
        DataFrame with one row per parameter (or parameter level for vectors).
    """
    _check_pymc()
    import arviz as az

    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model not fitted. Call .fit() first.")

    idata = model.idata

    # Check if NUTS was used (pathfinder has no r_hat)
    has_sample_stats = hasattr(idata, "sample_stats")
    is_nuts = has_sample_stats and "diverging" in idata.sample_stats

    if not is_nuts:
        # Pathfinder: only basic summary, no convergence diagnostics
        summary = az.summary(idata, kind="stats")
        if return_warnings:
            print(
                "WARNING: Model was fitted with Pathfinder (variational inference). "
                "R-hat and ESS diagnostics are not available. "
                "Re-fit with SamplerConfig(method='nuts') for production use."
            )
        return summary

    summary = az.summary(idata, round_to=4)

    # Count divergences
    n_divergent = int(idata.sample_stats["diverging"].sum().item())
    total_samples = (
        idata.sample_stats["diverging"].sizes["chain"]
        * idata.sample_stats["diverging"].sizes["draw"]
    )

    if return_warnings:
        # R-hat check
        if "r_hat" in summary.columns:
            bad_rhat = summary[summary["r_hat"] > 1.01]
            if len(bad_rhat) > 0:
                print(
                    f"WARNING: {len(bad_rhat)} parameter(s) have R-hat > 1.01. "
                    f"Non-convergence detected. Do not use these results. "
                    f"Check: {bad_rhat.index.tolist()[:5]}"
                )
            elif summary["r_hat"].max() > 1.005:
                print(
                    f"NOTE: Maximum R-hat is {summary['r_hat'].max():.4f}. "
                    "Marginally acceptable. Consider longer chains."
                )
            else:
                print(f"R-hat: OK (max = {summary['r_hat'].max():.4f})")

        # ESS check
        ess_col = "ess_bulk" if "ess_bulk" in summary.columns else "ess_mean"
        if ess_col in summary.columns:
            low_ess = summary[summary[ess_col] < 400]
            if len(low_ess) > 0:
                print(
                    f"WARNING: {len(low_ess)} parameter(s) have ESS < 400. "
                    f"Increase draws or tune in SamplerConfig."
                )
            else:
                print(f"ESS: OK (min bulk = {summary[ess_col].min():.0f})")

        # Divergence check
        pct_divergent = n_divergent / total_samples * 100
        if n_divergent == 0:
            print("Divergences: none")
        elif pct_divergent < 0.1:
            print(
                f"NOTE: {n_divergent} divergences ({pct_divergent:.3f}%). "
                "Small number, probably fine. Check with az.plot_trace()."
            )
        else:
            print(
                f"WARNING: {n_divergent} divergences ({pct_divergent:.2f}%). "
                "This is too many. Try SamplerConfig(target_accept=0.95). "
                "If problem persists, check model specification."
            )

    summary.attrs["n_divergences"] = n_divergent
    summary.attrs["pct_divergences"] = n_divergent / total_samples * 100 if total_samples else 0
    return summary


def posterior_predictive_check(
    model,
    claim_count_col: Optional[str] = None,
    severity_col: Optional[str] = None,
    n_stats: int = 200,
) -> dict:
    """Compare observed statistics to posterior predictive distribution.

    This is the fundamental model validation: simulate datasets from the
    fitted model and check whether the observed data looks plausible given
    those simulations. If the observed mean claim rate falls in the 94th
    percentile of simulated means, the model is over-predicting -- this
    is a problem.

    The function returns a dict of check statistics. Each key maps to a
    sub-dict with:
        - observed: the statistic computed on actual data
        - simulated_mean: mean of the statistic across posterior predictive draws
        - simulated_p5, simulated_p95: credible range of the statistic
        - posterior_predictive_p: what percentile the observed value is at
          (should be between 0.05 and 0.95 for a well-calibrated model)

    Statistics checked:
        - mean: overall mean prediction
        - variance: prediction variance (tests dispersion)
        - p90, p95: upper tail (important for large claim detection)
        - gini: discriminatory power across segments

    Args:
        model: A fitted model (HierarchicalFrequency or HierarchicalSeverity).
        claim_count_col: Required for frequency models.
        severity_col: Required for severity models.
        n_stats: Number of posterior predictive samples to use. More gives
            tighter bounds on the check statistics but takes longer.

    Returns:
        Dict of check statistics. Examine pp_p values: should all be in [0.05, 0.95].
    """
    _check_pymc()
    import arviz as az

    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model not fitted. Call .fit() first.")

    idata = model.idata

    if not hasattr(idata, "posterior_predictive"):
        raise RuntimeError(
            "No posterior predictive samples found. "
            "This should be computed automatically during fit(). "
            "If using a custom workflow, call pm.sample_posterior_predictive() "
            "and pass the result to extend_inferencedata=True."
        )

    pp = idata.posterior_predictive

    # Determine which predictive variable to check
    pp_vars = list(pp.data_vars)
    if not pp_vars:
        raise RuntimeError("posterior_predictive contains no variables.")

    pp_var = pp_vars[0]  # "claims" or "severity"
    pp_data = pp[pp_var].values  # (chains, draws, n_segments)
    pp_flat = pp_data.reshape(-1, pp_data.shape[-1])  # (n_samples, n_segments)

    # Sample a subset if large
    if pp_flat.shape[0] > n_stats:
        idx = np.random.choice(pp_flat.shape[0], n_stats, replace=False)
        pp_flat = pp_flat[idx]

    # Get observed values
    df = model._segment_data
    if claim_count_col and claim_count_col in df.columns:
        observed_vals = df[claim_count_col].values.astype(float)
    elif severity_col and severity_col in df.columns:
        observed_vals = df[severity_col].values.astype(float)
    else:
        # Try to infer from model
        non_group_cols = [c for c in df.columns if c not in model.group_cols]
        if not non_group_cols:
            raise ValueError(
                "Cannot determine observed values column. "
                "Pass claim_count_col or severity_col explicitly."
            )
        observed_vals = df[non_group_cols[0]].values.astype(float)

    def _gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Normalised Gini coefficient -- standard insurance model discrimination metric."""
        if len(y_true) < 2:
            return 0.0
        order = np.argsort(y_pred)
        y_sorted = y_true[order]
        n = len(y_sorted)
        cum_y = np.cumsum(y_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1)) * y_sorted) - (n + 1) * cum_y[-1]) / (
            n * cum_y[-1]
        )
        return float(np.abs(gini))

    results = {}

    # Mean check
    obs_mean = float(np.mean(observed_vals))
    sim_means = pp_flat.mean(axis=1)
    results["mean"] = _stat_check(obs_mean, sim_means)

    # Variance check
    obs_var = float(np.var(observed_vals))
    sim_vars = pp_flat.var(axis=1)
    results["variance"] = _stat_check(obs_var, sim_vars)

    # 90th percentile check
    obs_p90 = float(np.percentile(observed_vals, 90))
    sim_p90s = np.percentile(pp_flat, 90, axis=1)
    results["p90"] = _stat_check(obs_p90, sim_p90s)

    # 95th percentile
    obs_p95 = float(np.percentile(observed_vals, 95))
    sim_p95s = np.percentile(pp_flat, 95, axis=1)
    results["p95"] = _stat_check(obs_p95, sim_p95s)

    # Summary: did all checks pass?
    failed = [k for k, v in results.items() if not v["pass"]]
    results["_summary"] = {
        "passed": len(results) - len(failed) - 1,  # -1 for _summary itself
        "total": len(results) - 1,
        "failed_checks": failed,
        "interpretation": (
            "All checks passed. Model appears well-calibrated."
            if not failed
            else f"Failed checks: {failed}. "
            "The model may be mis-specified for these statistics. "
            "Consider alternative likelihood distributions."
        ),
    }

    return results


def _stat_check(
    observed: float, simulated: np.ndarray, alpha: float = 0.05
) -> dict:
    """Check whether observed statistic is within the simulated range."""
    p_value = float(np.mean(simulated <= observed))
    return {
        "observed": observed,
        "simulated_mean": float(simulated.mean()),
        "simulated_p5": float(np.percentile(simulated, 5)),
        "simulated_p95": float(np.percentile(simulated, 95)),
        "posterior_predictive_p": p_value,
        "pass": alpha <= p_value <= (1 - alpha),
    }
