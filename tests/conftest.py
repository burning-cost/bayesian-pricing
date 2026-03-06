"""
Shared fixtures for bayesian-pricing tests.

The fixtures generate synthetic insurance segment data that reproduces realistic
characteristics of UK personal lines:

- Poisson-distributed claim counts with Gamma-distributed severity
- Mixed segment exposure (some well-populated, some sparse)
- Multiple grouping factors (vehicle group, age band)
- True relativities embedded so we can check posterior recovery
"""

import numpy as np
import pandas as pd
import pytest

try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def freq_segment_data(rng):
    """Segment-level frequency data with known true relativities.

    True model:
        log(lambda) = log(0.08) + u_veh + u_age
        u_veh: A=0.0, B=0.4, C=-0.3, D=0.7, E=-0.5
        u_age: 17-21=0.8, 22-35=0.2, 36-50=0.0, 51-65=-0.1, 66+=0.1
    """
    veh_groups = ["A", "B", "C", "D", "E"]
    age_bands  = ["17-21", "22-35", "36-50", "51-65", "66+"]

    # True random effects (log scale)
    u_veh = {"A": 0.0, "B": 0.4, "C": -0.3, "D": 0.7, "E": -0.5}
    u_age = {"17-21": 0.8, "22-35": 0.2, "36-50": 0.0, "51-65": -0.1, "66+": 0.1}

    base_rate = 0.08

    rows = []
    for veh in veh_groups:
        for age in age_bands:
            # Exposure: sparse for dangerous combinations, dense for common ones
            if veh in ("D", "E") and age == "17-21":
                exposure = rng.integers(20, 60)  # thin segment
            elif veh == "A" and age == "36-50":
                exposure = rng.integers(800, 1200)  # dense
            else:
                exposure = rng.integers(100, 500)

            true_rate = base_rate * np.exp(u_veh[veh] + u_age[age])
            claims = rng.poisson(true_rate * exposure)

            rows.append({
                "veh_group": veh,
                "age_band": age,
                "exposure": float(exposure),
                "claims": int(claims),
                "true_rate": true_rate,
            })

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sev_segment_data(rng, freq_segment_data):
    """Segment-level severity data consistent with freq_segment_data.

    True model:
        log(mu_sev) = log(2000) + u_veh_sev
        u_veh_sev: A=0.0, B=0.1, C=-0.05, D=0.15, E=-0.1
    Shape parameter (concentration): 1.5 (moderate dispersion)
    """
    u_veh_sev = {"A": 0.0, "B": 0.1, "C": -0.05, "D": 0.15, "E": -0.1}
    base_severity = 2000.0

    rows = []
    for _, row in freq_segment_data.iterrows():
        if row["claims"] == 0:
            continue  # severity model only applies to segments with claims

        veh = row["veh_group"]
        true_sev = base_severity * np.exp(u_veh_sev[veh])
        shape = 1.5

        # Simulate individual claim amounts and take mean
        n_claims = max(int(row["claims"]), 1)
        claim_amounts = rng.gamma(
            shape=shape,
            scale=true_sev / shape,
            size=n_claims,
        )
        avg_cost = float(claim_amounts.mean())

        rows.append({
            "veh_group": row["veh_group"],
            "age_band": row["age_band"],
            "claim_count": n_claims,
            "avg_claim_cost": avg_cost,
            "true_severity": true_sev,
        })

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def small_freq_data(rng):
    """Minimal frequency dataset for fast API tests (no PyMC needed)."""
    return pd.DataFrame({
        "segment": ["A", "B", "C", "D"],
        "claims":  [10, 5, 20, 1],
        "exposure": [100.0, 50.0, 200.0, 10.0],
    })
