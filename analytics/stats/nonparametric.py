"""
Unit 7 — Nonparametric tests.
Wilcoxon signed-rank, Mann-Whitney U, Friedman test.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp


def _clean(series: pd.Series) -> pd.Series:
    s = series.dropna()
    s = s[s > 0]
    mu, sigma = s.mean(), s.std()
    return s[abs(s - mu) <= 3 * sigma] if sigma > 0 else s


# ── Wilcoxon signed-rank: driver before vs after safety car ──────────────────

def wilcoxon_before_after_sc(laps: list[dict],
                               sc_laps: list[int],
                               driver_id: str) -> dict:
    """
    Wilcoxon signed-rank test: driver's lap times before vs after safety car.
    Requires paired laps (one lap before SC, one lap after).
    """
    df = pd.DataFrame(laps)
    if df.empty or not sc_laps:
        return {"error": "No lap data or no safety car laps provided."}

    drv = df[df["driver_id"] == driver_id][["lap_number", "lap_time_s"]].dropna()
    if drv.empty:
        return {"error": f"No laps found for {driver_id}."}

    drv = drv.sort_values("lap_number")

    # Classify laps: before SC = laps within 3 before first SC lap
    # After SC = laps within 3 after last SC lap
    sc_start = min(sc_laps)
    sc_end = max(sc_laps)
    window = 5

    before = _clean(drv[drv["lap_number"].between(sc_start - window, sc_start - 1)]["lap_time_s"])
    after = _clean(drv[drv["lap_number"].between(sc_end + 1, sc_end + window)]["lap_time_s"])

    min_n = min(len(before), len(after))
    if min_n < 2:
        return {"error": "Not enough laps before/after safety car for this driver."}

    # Paired: use first min_n from each
    b = before.iloc[:min_n].values
    a = after.iloc[:min_n].values

    stat, p = sp.wilcoxon(b, a)

    faster_after = a.mean() < b.mean()

    return {
        "test": "Wilcoxon Signed-Rank",
        "driver_id": driver_id,
        "n_pairs": min_n,
        "mean_before_sc": round(float(b.mean()), 4),
        "mean_after_sc": round(float(a.mean()), 4),
        "w_statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "verdict": (
            f"{'Significant' if p < 0.05 else 'No significant'} change in lap times "
            f"around safety car (W={stat:.1f}, p={p:.4f}). "
            f"{driver_id} was {'faster' if faster_after else 'slower'} after SC restart."
        ),
        "laps_before": list(b),
        "laps_after": list(a),
    }


# ── Mann-Whitney U: compare two teams ────────────────────────────────────────

def mann_whitney_teams(laps: list[dict],
                        team_a: str,
                        team_b: str,
                        group_col: str = "team") -> dict:
    """
    Mann-Whitney U test: lap times of team A vs team B.
    No normality assumption required.
    """
    df = pd.DataFrame(laps)
    if df.empty or group_col not in df.columns:
        return {"error": f"Missing '{group_col}' column."}

    a = _clean(df[df[group_col] == team_a]["lap_time_s"])
    b = _clean(df[df[group_col] == team_b]["lap_time_s"])

    if len(a) < 3 or len(b) < 3:
        return {"error": "Not enough laps for one or both groups."}

    stat, p = sp.mannwhitneyu(a, b, alternative="two-sided")

    # Effect size: rank-biserial correlation
    n1, n2 = len(a), len(b)
    r_effect = 1 - (2 * stat) / (n1 * n2)

    faster = team_a if a.median() < b.median() else team_b

    return {
        "test": "Mann-Whitney U",
        "group_a": team_a,
        "group_b": team_b,
        "n_a": n1,
        "n_b": n2,
        "median_a": round(float(a.median()), 4),
        "median_b": round(float(b.median()), 4),
        "u_statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "rank_biserial_r": round(float(r_effect), 4),
        "significant": bool(p < 0.05),
        "verdict": (
            f"{'Significant' if p < 0.05 else 'No significant'} difference "
            f"(U={stat:.1f}, p={p:.4f}). "
            f"{faster} has faster lap times (median)."
        ),
        "laps_a": a.tolist(),
        "laps_b": b.tolist(),
    }


# ── Friedman test: rank multiple drivers across multiple races ────────────────

def friedman_test(multi_race_laps: list[dict]) -> dict:
    """
    Friedman test: are lap time rankings consistent across multiple drivers?
    multi_race_laps: list of laps with 'driver_id', 'lap_number', 'lap_time_s'.

    Uses median lap time per driver per lap as the block structure:
    rows = lap numbers (blocks), columns = drivers (treatments).
    """
    df = pd.DataFrame(multi_race_laps)
    if df.empty:
        return {"error": "No data provided."}

    pivot = (df.groupby(["lap_number", "driver_id"])["lap_time_s"]
             .median().unstack("driver_id"))
    pivot = pivot.dropna()

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {"error": "Need at least 3 laps × 3 drivers for Friedman test."}

    stat, p = sp.friedmanchisquare(*[pivot[col].values for col in pivot.columns])

    # Kendall's W (effect size)
    k = pivot.shape[1]
    n = pivot.shape[0]
    w = stat / (k * (n - 1))

    return {
        "test": "Friedman Test",
        "n_blocks": n,
        "n_treatments": k,
        "drivers": list(pivot.columns),
        "chi2_statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "kendalls_w": round(float(w), 4),
        "significant": bool(p < 0.05),
        "verdict": (
            f"{'Significant' if p < 0.05 else 'No significant'} difference in lap time "
            f"rankings across drivers (χ²={stat:.3f}, p={p:.4f}, W={w:.3f})."
        ),
        "driver_medians": {
            drv: round(float(pivot[drv].median()), 4)
            for drv in pivot.columns
        },
    }
