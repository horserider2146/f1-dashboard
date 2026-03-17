"""
Driver Consistency Index (DCI) — novel metric.

DCI = 1 / Var(lap_time_s), normalised 0–1 across all drivers.
Higher DCI → more consistent driver.
"""
import pandas as pd
import numpy as np


def compute_dci(laps: list[dict]) -> list[dict]:
    """
    Compute DCI for every driver in the provided lap list.

    Returns a list of dicts sorted by DCI descending:
        [{driver_id, mean_lap_s, std_lap_s, variance, dci, dci_normalised, n_laps}]
    """
    df = pd.DataFrame(laps)
    if df.empty or "lap_time_s" not in df.columns:
        return []

    df = df.dropna(subset=["lap_time_s"])
    df = df[df["lap_time_s"] > 0]

    # Exclude outlier laps (pit in/out, safety car) using 3-sigma filter per driver
    def _clean(grp):
        mu, sigma = grp["lap_time_s"].mean(), grp["lap_time_s"].std()
        return grp[abs(grp["lap_time_s"] - mu) <= 3 * sigma] if sigma > 0 else grp

    df = df.groupby("driver_id", group_keys=False).apply(_clean)

    stats = (
        df.groupby("driver_id")["lap_time_s"]
        .agg(n_laps="count", mean_lap_s="mean", std_lap_s="std", variance="var")
        .reset_index()
    )
    stats = stats[stats["n_laps"] >= 3]  # need at least 3 laps for meaningful variance
    stats = stats[stats["variance"] > 0]

    # DCI = 1 / variance  (higher = more consistent)
    stats["dci"] = 1.0 / stats["variance"]

    # Normalise to 0–1
    dci_min, dci_max = stats["dci"].min(), stats["dci"].max()
    if dci_max > dci_min:
        stats["dci_normalised"] = (stats["dci"] - dci_min) / (dci_max - dci_min)
    else:
        stats["dci_normalised"] = 1.0

    stats = stats.sort_values("dci_normalised", ascending=False).reset_index(drop=True)
    stats["rank"] = range(1, len(stats) + 1)

    return stats.round(6).to_dict(orient="records")


def dci_championship_correlation(dci_results: list[dict],
                                  standings: list[dict]) -> dict:
    """
    Pearson correlation between DCI (normalised) and championship points.

    standings: [{driver_id, points}]
    Returns: {r, p_value, interpretation}
    """
    from scipy import stats as sp

    if not dci_results or not standings:
        return {}

    dci_df = pd.DataFrame(dci_results)[["driver_id", "dci_normalised"]]
    pts_df = pd.DataFrame(standings)[["driver_id", "points"]]
    merged = dci_df.merge(pts_df, on="driver_id").dropna()

    if len(merged) < 3:
        return {}

    r, p = sp.pearsonr(merged["dci_normalised"], merged["points"])

    if abs(r) >= 0.7:
        interp = "Strong"
    elif abs(r) >= 0.4:
        interp = "Moderate"
    else:
        interp = "Weak"
    direction = "positive" if r > 0 else "negative"

    return {
        "r": round(r, 4),
        "p_value": round(p, 4),
        "n": len(merged),
        "interpretation": f"{interp} {direction} correlation (r={r:.3f}, p={p:.4f})",
        "significant": bool(p < 0.05),
    }
