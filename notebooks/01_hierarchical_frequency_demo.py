# Databricks notebook source
# MAGIC %md
# MAGIC # Hierarchical Bayesian Frequency Model - UK Motor Insurance Demo
# MAGIC
# MAGIC This notebook demonstrates `bayesian-pricing` on a synthetic UK motor portfolio.
# MAGIC
# MAGIC **The problem**: You have a motor book with 500k policies across vehicle groups,
# MAGIC age bands, and areas. Most rating cells are sparse - fewer than 30 observations.
# MAGIC Standard GLMs either overfit (saturated model) or pool everything indiscriminately
# MAGIC (main-effects only). Neither gives you calibrated estimates for thin cells.
# MAGIC
# MAGIC **The solution**: Hierarchical Bayesian model with partial pooling. Thin segments
# MAGIC borrow strength from related segments via a shared population distribution.
# MAGIC The degree of borrowing is data-driven - set by the posterior, not hand-tuned.
# MAGIC
# MAGIC **What you will see**:
# MAGIC 1. Data generation: realistic UK motor segments with known true relativities
# MAGIC 2. Model fitting with Pathfinder (fast) and NUTS (accurate)
# MAGIC 3. Posterior diagnostics: R-hat, ESS, divergences
# MAGIC 4. Extracting multiplicative relativities in rate-table format
# MAGIC 5. Identifying thin segments that need manual review
# MAGIC 6. Pure premium calculation: frequency × severity

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

# MAGIC %pip install bayesian-pricing[pymc] pymc arviz matplotlib

# COMMAND ----------

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from bayesian_pricing import HierarchicalFrequency, HierarchicalSeverity, BayesianRelativities
from bayesian_pricing.frequency import SamplerConfig
from bayesian_pricing.diagnostics import convergence_summary, posterior_predictive_check

np.random.seed(42)
rng = np.random.default_rng(42)

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK Motor Portfolio
# MAGIC
# MAGIC We generate a realistic segment-level dataset. The true model:
# MAGIC
# MAGIC ```
# MAGIC log(lambda_ij) = log(0.09) + u_veh[j] + u_age[i]
# MAGIC ```
# MAGIC
# MAGIC True relativities are embedded so we can check posterior recovery.
# MAGIC
# MAGIC Exposure is deliberately skewed: common combinations (mid-age, mid-group) get
# MAGIC thousands of policy-years; rare combinations (young drivers, sports cars) get tens.
# MAGIC This is the realistic UK pattern.

# COMMAND ----------

# True model parameters
BASE_FREQ = 0.09  # 9% annual claim frequency (UK motor attritional)

# Vehicle groups (analogous to ABI groups, simplified)
VEH_GROUPS = {
    "Supermini":     -0.20,  # VW Polo, Ford Fiesta - low frequency
    "Hatchback":     -0.05,  # Golf, Focus
    "Saloon":         0.00,  # Base category
    "SUV":            0.10,  # Higher theft, parking claims
    "Sports":         0.35,  # High frequency, young driver interaction
    "Van/4x4":        0.15,  # Higher claim rate
    "Prestige":       0.20,  # Expensive repairs drive BI frequency up
}

# Driver age bands
AGE_BANDS = {
    "17-21":  0.85,   # 2.3x base - super-additive with sports cars
    "22-25":  0.45,   # 1.6x base
    "26-30":  0.20,   # 1.2x base
    "31-40":  0.00,   # Base category
    "41-50": -0.05,
    "51-60": -0.10,
    "61-70": -0.05,
    "71+":    0.15,   # Higher frequency again - cognitive decline
}


def generate_segment_data(
    veh_groups: dict,
    age_bands: dict,
    base_freq: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate segment-level motor insurance data with realistic exposure distribution."""
    rng = np.random.default_rng(seed)
    rows = []

    for veh, u_veh in veh_groups.items():
        for age, u_age in age_bands.items():

            # Realistic exposure: young x sports = thin; mid x standard = dense
            is_young = age in ("17-21", "22-25")
            is_sports = veh in ("Sports", "Prestige", "Prestige")
            is_common = veh in ("Hatchback", "Saloon") and age in ("31-40", "41-50")

            if is_young and is_sports:
                exposure = rng.integers(15, 50)    # Thin cell: few young-driver sports policies
            elif is_common:
                exposure = rng.integers(2000, 5000)  # Dense: most common combination
            elif is_young:
                exposure = rng.integers(100, 400)
            else:
                exposure = rng.integers(300, 1500)

            true_rate = base_freq * np.exp(u_veh + u_age)
            claims = int(rng.poisson(true_rate * exposure))

            rows.append({
                "veh_group": veh,
                "age_band": age,
                "exposure": float(exposure),
                "claims": claims,
                "true_rate": true_rate,
                "true_relativity": np.exp(u_veh + u_age),
            })

    df = pd.DataFrame(rows)
    df["observed_rate"] = df["claims"] / df["exposure"]
    return df


df = generate_segment_data(VEH_GROUPS, AGE_BANDS, BASE_FREQ)
print(f"Segments: {len(df)}")
print(f"Total claims: {df['claims'].sum():,}")
print(f"Total exposure: {df['exposure'].sum():,.0f} policy-years")
print(f"Portfolio mean frequency: {df['claims'].sum() / df['exposure'].sum():.4f}")
print()
print("Exposure distribution:")
print(df["exposure"].describe().round(0))

# COMMAND ----------

# MAGIC %md ### Visualise the sparse-cell problem

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: exposure distribution (log scale)
ax = axes[0]
ax.hist(df["exposure"], bins=30, edgecolor="white", color="#2E86AB")
ax.set_xlabel("Segment exposure (policy-years)")
ax.set_ylabel("Count")
ax.set_title("Exposure distribution across segments\n(log scale)")
ax.set_xscale("log")
ax.axvline(30, color="red", linestyle="--", label="n=30 threshold")
ax.legend()
n_thin = (df["exposure"] < 30).sum()
ax.text(0.05, 0.9, f"{n_thin} thin segments\n(exposure < 30)", transform=ax.transAxes,
        color="red", fontsize=10)

# Right: observed rate vs true rate (GLM would overfit thin cells)
ax = axes[1]
scatter = ax.scatter(
    df["true_rate"],
    df["observed_rate"],
    c=np.log(df["exposure"]),
    cmap="RdYlGn",
    alpha=0.7,
    s=50,
)
ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.5, label="Perfect fit")
ax.set_xlabel("True claim rate")
ax.set_ylabel("Observed rate (claims/exposure)")
ax.set_title("Observed vs true rate\nColour = log(exposure)")
plt.colorbar(scatter, ax=ax, label="log(exposure)")
ax.legend()

plt.tight_layout()
plt.savefig("/tmp/sparse_cell_problem.png", dpi=150, bbox_inches="tight")
plt.show()
print("Note: thin segments (red dots) have noisy observed rates far from the true rate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit the Hierarchical Frequency Model
# MAGIC
# MAGIC We start with Pathfinder for a quick look at the posterior, then use NUTS
# MAGIC for the production-quality estimate.

# COMMAND ----------

# MAGIC %md ### 2a. Quick exploration with Pathfinder

# COMMAND ----------

model_fast = HierarchicalFrequency(
    group_cols=["veh_group", "age_band"],
    prior_mean_rate=0.09,          # We know the portfolio mean; inform the prior
    variance_prior_sigma=0.4,       # Allows ±50% between-segment variation on log scale
)

config_fast = SamplerConfig(
    method="pathfinder",
    draws=500,
    random_seed=42,
)

model_fast.fit(df, claim_count_col="claims", exposure_col="exposure", sampler_config=config_fast)
print("Pathfinder fit complete.")

preds_fast = model_fast.predict()
print("\nFirst 5 segment predictions:")
display(preds_fast.head(10))

# COMMAND ----------

# MAGIC %md ### 2b. NUTS for production-quality posterior

# COMMAND ----------

model = HierarchicalFrequency(
    group_cols=["veh_group", "age_band"],
    prior_mean_rate=0.09,
    variance_prior_sigma=0.4,
)

config_nuts = SamplerConfig(
    method="nuts",
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.85,
    random_seed=42,
)

model.fit(df, claim_count_col="claims", exposure_col="exposure", sampler_config=config_nuts)
print("NUTS fit complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Convergence Diagnostics
# MAGIC
# MAGIC Before interpreting results, verify the sampler converged. This is not optional.

# COMMAND ----------

diag = convergence_summary(model)
print("\nParameter summary (first 20 rows):")
display(diag.head(20))

# COMMAND ----------

# Check for any non-convergence
if "r_hat" in diag.columns:
    bad_rhat = diag[diag["r_hat"] > 1.01]
    if len(bad_rhat) > 0:
        print(f"WARNING: {len(bad_rhat)} parameters with R-hat > 1.01:")
        display(bad_rhat)
    else:
        print(f"All R-hat values below 1.01. Max = {diag['r_hat'].max():.4f}")

# COMMAND ----------

# Trace plot for key parameters
az.plot_trace(
    model.idata,
    var_names=["alpha", "sigma_veh_group", "sigma_age_band"],
    figsize=(14, 8),
)
plt.tight_layout()
plt.savefig("/tmp/trace_plot.png", dpi=120, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Posterior Predictive Check
# MAGIC
# MAGIC Does the model actually describe the data? If the mean of the posterior predictive
# MAGIC distribution doesn't match the observed mean, the model is misspecified.

# COMMAND ----------

ppc = posterior_predictive_check(model, claim_count_col="claims")
for stat_name, stat in ppc.items():
    if stat_name.startswith("_"):
        continue
    status = "PASS" if stat["pass"] else "FAIL"
    print(f"[{status}] {stat_name}: observed={stat['observed']:.4f}, "
          f"sim 90% CI=[{stat['simulated_p5']:.4f}, {stat['simulated_p95']:.4f}], "
          f"pp_p={stat['posterior_predictive_p']:.3f}")

print(f"\nSummary: {ppc['_summary']['interpretation']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Relativities - Rate Table Format
# MAGIC
# MAGIC This is what the pricing actuary actually needs: a table of multiplicative
# MAGIC factors per rating variable, with credible intervals.

# COMMAND ----------

rel = BayesianRelativities(model, hdi_prob=0.9)

# All factors
all_relativities = rel.relativities()

for factor, rt in all_relativities.items():
    print(f"\n{'='*60}")
    print(f"Factor: {factor}")
    print(f"{'='*60}")
    display(rt.table)

# COMMAND ----------

# Full summary in long format (suitable for Excel export)
summary_df = rel.summary()
print("Long-format summary (all factors x levels):")
display(summary_df)

# COMMAND ----------

# MAGIC %md ### Compare posterior relativities to true values

# COMMAND ----------

# Vehicle group comparison
rt_veh = rel.relativities(factor="veh_group")
veh_table = rt_veh.table.copy()

# True relativities by vehicle group (exp of u_veh, normalised so Saloon=1)
true_veh_rel = {
    v: np.exp(u) / np.exp(VEH_GROUPS["Saloon"])
    for v, u in VEH_GROUPS.items()
}
veh_table["true_relativity"] = veh_table["level"].map(true_veh_rel)
veh_table["recovery_error_pct"] = (
    (veh_table["relativity"] - veh_table["true_relativity"]) / veh_table["true_relativity"] * 100
)
print("Vehicle group relativity recovery:")
display(veh_table[["level", "true_relativity", "relativity", "lower_90pct", "upper_90pct",
                    "recovery_error_pct", "uncertainty_reduction"]].round(3))

# COMMAND ----------

# MAGIC %md ### Plot relativities with credible intervals

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, factor in zip(axes, ["veh_group", "age_band"]):
    rt = all_relativities[factor]
    t = rt.table.sort_values("level").reset_index(drop=True)

    y_pos = range(len(t))
    ax.errorbar(
        t["relativity"],
        y_pos,
        xerr=[t["relativity"] - t["lower_90pct"], t["upper_90pct"] - t["relativity"]],
        fmt="o",
        color="#2E86AB",
        capsize=5,
        linewidth=2,
        markersize=8,
        label="Posterior median + 90% HDI",
    )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="No effect")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(t["level"])
    ax.set_xlabel("Multiplicative relativity")
    ax.set_title(f"{factor}")
    ax.legend()

plt.suptitle("Posterior relativities with 90% credible intervals", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("/tmp/relativities_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Identifying Thin Segments
# MAGIC
# MAGIC The credibility factor tells you how much weight each segment puts on its
# MAGIC own data vs the portfolio mean. Low credibility = thin segment = treat with caution.

# COMMAND ----------

thin = rel.thin_segments(credibility_threshold=0.4)
print(f"Thin segments (credibility factor < 0.4): {len(thin)}")
display(thin)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Variance Components
# MAGIC
# MAGIC How much does each rating factor actually explain?
# MAGIC
# MAGIC The sigma parameters tell you the magnitude of between-segment variation.
# MAGIC A large sigma for "veh_group" means vehicle type drives substantial frequency
# MAGIC differences. A small sigma means segments cluster tightly around the mean.

# COMMAND ----------

vc = model.variance_components()
print("Variance components (log-scale sigma):")
display(vc)

print("\nInterpretation on multiplicative scale:")
for col in model.group_cols:
    sigma_mean = float(vc.loc[f"sigma_{col}", "mean"])
    factor_range = np.exp(2 * sigma_mean)  # ±2 sigma on log scale
    print(f"  {col}: sigma={sigma_mean:.3f} → 2σ spread ≈ {factor_range:.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Shrinkage Visualisation
# MAGIC
# MAGIC The partial pooling in action: how much does the model shrink thin segments
# MAGIC toward the portfolio mean, compared to a naive observed-rate estimate?

# COMMAND ----------

preds = model.predict()
preds = preds.merge(
    df[["veh_group", "age_band", "true_rate", "observed_rate", "exposure", "claims"]],
    on=["veh_group", "age_band"],
)
preds["portfolio_mean"] = df["claims"].sum() / df["exposure"].sum()

fig, ax = plt.subplots(figsize=(10, 7))

# Scatter: x = observed rate, y = posterior mean
sc = ax.scatter(
    preds["observed_rate"],
    preds["mean"],
    c=np.log(preds["exposure"]),
    cmap="RdYlGn",
    alpha=0.8,
    s=60,
    zorder=3,
)

# Perfect fit line
rate_range = [0, preds[["observed_rate", "mean"]].max().max() * 1.1]
ax.plot(rate_range, rate_range, "k--", alpha=0.4, label="No shrinkage (observed = posterior)")

# Portfolio mean reference
pm_val = preds["portfolio_mean"].iloc[0]
ax.axhline(pm_val, color="blue", alpha=0.3, linestyle=":", label=f"Portfolio mean ({pm_val:.3f})")
ax.axvline(pm_val, color="blue", alpha=0.3, linestyle=":")

plt.colorbar(sc, ax=ax, label="log(exposure)")
ax.set_xlabel("Observed claim rate (noisy)")
ax.set_ylabel("Posterior mean claim rate (credibility-weighted)")
ax.set_title("Partial pooling in action\nRed = sparse segments, green = dense")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/shrinkage_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("Segments below the diagonal: shrunk TOWARD portfolio mean (correct for thin segments)")
print("Segments near the diagonal: trusted own data (correct for dense segments)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Severity Model
# MAGIC
# MAGIC Fit a Gamma hierarchical model for claim severity, then combine with
# MAGIC frequency to get pure premium relativities.

# COMMAND ----------

# Generate severity data consistent with our frequency data
from tests.conftest import sev_segment_data  # for demo purposes we re-generate here

# Severity: varies by vehicle group only (not age band - repair cost is similar across ages)
SEV_VEH = {
    "Supermini": -0.05,
    "Hatchback":  0.00,
    "Saloon":     0.00,
    "SUV":        0.10,
    "Sports":     0.25,  # Sports cars: expensive parts, specialist repair
    "Van/4x4":    0.05,
    "Prestige":   0.35,  # Prestige: very expensive parts
}
BASE_SEV = 1800.0  # £1,800 average attritional claim

sev_rows = []
for _, row in df.iterrows():
    if row["claims"] == 0:
        continue
    veh = row["veh_group"]
    true_sev = BASE_SEV * np.exp(SEV_VEH[veh])
    n_claims = int(row["claims"])
    # Simulate actual claim amounts, then aggregate to segment average
    amounts = rng.gamma(shape=1.5, scale=true_sev / 1.5, size=n_claims)
    sev_rows.append({
        "veh_group": veh,
        "age_band": row["age_band"],
        "claim_count": n_claims,
        "avg_claim_cost": float(amounts.mean()),
        "true_severity": true_sev,
    })

sev_df = pd.DataFrame(sev_rows)
print(f"Severity segments: {len(sev_df)}")
print(f"Mean observed severity: £{sev_df['avg_claim_cost'].mean():,.0f}")

# COMMAND ----------

sev_model = HierarchicalSeverity(
    group_cols=["veh_group"],      # severity by veh group only
    prior_mean_severity=BASE_SEV,
    variance_prior_sigma=0.25,
)

sev_config = SamplerConfig(method="nuts", draws=1000, tune=1000, chains=4, random_seed=42)
sev_model.fit(sev_df, severity_col="avg_claim_cost", weight_col="claim_count",
              sampler_config=sev_config)
print("Severity model fit complete.")

sev_rel = BayesianRelativities(sev_model, hdi_prob=0.9)
sev_rt = sev_rel.relativities(factor="veh_group")
display(sev_rt.table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Pure Premium Relativities
# MAGIC
# MAGIC Combine frequency and severity relativities to get pure premium relativities.
# MAGIC This is what feeds the rate table.

# COMMAND ----------

freq_rt = rel.relativities(factor="veh_group")

# Merge frequency and severity relativities
pp_table = freq_rt.table[["level", "relativity", "uncertainty_reduction"]].rename({"uncertainty_reduction": "uncertainty_reduction_freq"}).merge(
    sev_rt.table[["level", "relativity", "uncertainty_reduction"]].rename({"uncertainty_reduction": "uncertainty_reduction_sev"}),
    on="level",
    suffixes=("_freq", "_sev"),
)
pp_table["pure_premium_relativity"] = pp_table["relativity_freq"] * pp_table["relativity_sev"]

# True pure premium relativity for comparison
pp_table["true_pp_rel"] = pp_table["level"].map(
    {v: np.exp(VEH_GROUPS[v]) * np.exp(SEV_VEH[v]) for v in VEH_GROUPS}
)
pp_table["true_pp_rel"] = pp_table["true_pp_rel"] / pp_table["true_pp_rel"].mean()

pp_table = pp_table.sort_values("pure_premium_relativity", ascending=False)
print("Pure premium relativities by vehicle group:")
display(pp_table[["level", "relativity_freq", "relativity_sev",
                   "pure_premium_relativity", "true_pp_rel"]].round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What the hierarchical Bayesian model gives you that a standard GLM doesn't:**
# MAGIC
# MAGIC 1. **Calibrated estimates for thin cells**: instead of overfitting or refusing to
# MAGIC    estimate, the model gives you a posterior distribution that reflects genuine
# MAGIC    uncertainty. Thin cells get wide credible intervals; dense cells get tight ones.
# MAGIC
# MAGIC 2. **Data-driven pooling**: the degree of shrinkage is estimated from data.
# MAGIC    If vehicle group effects are large (sigma_veh = 0.4), the model allows large
# MAGIC    differences. If they are small, it pools aggressively. You do not set this.
# MAGIC
# MAGIC 3. **Credibility factors in Bühlmann-Straub terms**: actuaries already understand
# MAGIC    credibility. The model gives you Z_j for each segment directly.
# MAGIC
# MAGIC 4. **Uncertainty on relativities**: the 90% credible interval tells you whether
# MAGIC    a 1.3x relativity is well-determined (CI: 1.2–1.4) or uncertain (CI: 0.9–1.8).
# MAGIC    The latter should not drive a rate change.
# MAGIC
# MAGIC **Practical limitations to document in the validation report:**
# MAGIC - Model is fitted on segment-level sufficient statistics. Within-segment heterogeneity
# MAGIC   is not modelled.
# MAGIC - Convergence must be checked (R-hat, ESS, divergences) before using results.
# MAGIC - Pathfinder is faster but approximate - use NUTS for final production estimates.
# MAGIC - The Gamma likelihood assumes attritional claims only. Large claims (BI, total loss)
# MAGIC   need separate treatment.

# COMMAND ----------

# Export rate table for use in Radar/Emblem
output_path = "/tmp/bayesian_relativities.csv"
pp_table.to_csv(output_path, index=False)
print(f"Rate table exported to {output_path}")
print("\nDone. The bayesian_relativities.csv file is ready for import into your rating system.")
