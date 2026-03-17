"""
Unit 3 — ANOVA
One-way ANOVA (team lap times), Two-way ANOVA (team × compound), Tukey HSD post-hoc.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp
from itertools import combinations


def _clean_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lap_time_s"])
    df = df[df["lap_time_s"] > 0].copy()
    mu, sigma = df["lap_time_s"].mean(), df["lap_time_s"].std()
    return df[abs(df["lap_time_s"] - mu) <= 3 * sigma]


# ── One-way ANOVA: team lap times ─────────────────────────────────────────────

def one_way_anova(laps: list[dict], group_col: str = "team") -> dict:
    """
    One-way ANOVA: does mean lap time differ significantly across teams?
    If significant, runs Tukey HSD post-hoc.
    """
    df = _clean_lap_times(pd.DataFrame(laps))

    if group_col not in df.columns:
        # fall back to driver_id if team not present
        group_col = "driver_id"

    groups = {g: grp["lap_time_s"].values
              for g, grp in df.groupby(group_col)
              if len(grp) >= 3}

    if len(groups) < 2:
        return {"error": "Need at least 2 groups with ≥3 laps each."}

    f_stat, p_value = sp.f_oneway(*groups.values())

    # Effect size: eta-squared
    grand_mean = df["lap_time_s"].mean()
    ss_between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in groups.values())
    ss_total = sum((v - grand_mean) ** 2 for v in groups.values()).sum() if False else \
               df["lap_time_s"].apply(lambda x: (x - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    result = {
        "test": "One-way ANOVA",
        "group_col": group_col,
        "n_groups": len(groups),
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "eta_squared": round(float(eta_sq), 4),
        "significant": bool(p_value < 0.05),
        "verdict": (
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference "
            f"across {group_col}s (F={f_stat:.3f}, p={p_value:.4f})"
        ),
        "group_stats": [
            {
                "group": g,
                "n": len(v),
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std()), 4),
            }
            for g, v in groups.items()
        ],
    }

    # Tukey HSD post-hoc if significant
    if p_value < 0.05:
        result["tukey_hsd"] = _tukey_hsd(groups)

    return result


def _tukey_hsd(groups: dict) -> list[dict]:
    """Tukey HSD pairwise comparisons."""
    group_names = list(groups.keys())
    grand_n = sum(len(v) for v in groups.values())
    k = len(groups)

    # MSE (within-group mean square error)
    ss_within = sum(np.sum((v - v.mean()) ** 2) for v in groups.values())
    df_within = grand_n - k
    ms_within = ss_within / df_within if df_within > 0 else 1e-9

    results = []
    for g1, g2 in combinations(group_names, 2):
        v1, v2 = groups[g1], groups[g2]
        n_harm = 2 / (1 / len(v1) + 1 / len(v2))  # harmonic mean n
        q = abs(v1.mean() - v2.mean()) / np.sqrt(ms_within / n_harm)
        # p-value from Studentized range distribution (approximation)
        p_approx = float(sp.t.sf(q / np.sqrt(2), df=df_within) * 2)
        results.append({
            "group_a": g1,
            "group_b": g2,
            "mean_diff": round(float(v1.mean() - v2.mean()), 4),
            "q_statistic": round(float(q), 4),
            "p_approx": round(min(p_approx, 1.0), 4),
            "significant": bool(p_approx < 0.05),
        })

    return sorted(results, key=lambda x: x["p_approx"])


# ── Two-way ANOVA: team × compound ────────────────────────────────────────────

def two_way_anova(laps: list[dict]) -> dict:
    """
    Two-way ANOVA: team (A) × compound (B) on lap time.
    Tests main effects of team and compound + their interaction.
    """
    df = _clean_lap_times(pd.DataFrame(laps))

    if "team" not in df.columns or "compound" not in df.columns:
        return {"error": "Need 'team' and 'compound' columns."}

    df = df[["team", "compound", "lap_time_s"]].dropna()
    df = df[df.groupby(["team", "compound"])["lap_time_s"].transform("count") >= 2]

    if df.empty:
        return {"error": "Insufficient data for two-way ANOVA."}

    try:
        import statsmodels.formula.api as smf
        model = smf.ols("lap_time_s ~ C(team) + C(compound) + C(team):C(compound)",
                        data=df).fit()
        from statsmodels.stats.anova import anova_lm
        table = anova_lm(model, typ=2)

        rows = []
        for idx, row in table.iterrows():
            rows.append({
                "source": str(idx),
                "sum_sq": round(float(row["sum_sq"]), 4),
                "df": round(float(row["df"]), 2),
                "F": round(float(row["F"]), 4) if not np.isnan(row["F"]) else None,
                "p_value": round(float(row["PR(>F)"]), 6) if not np.isnan(row["PR(>F)"]) else None,
                "significant": bool(row["PR(>F)"] < 0.05) if not np.isnan(row["PR(>F)"]) else False,
            })

        # Interaction plot data: mean lap time per (team, compound)
        interaction = (df.groupby(["team", "compound"])["lap_time_s"]
                       .mean().reset_index()
                       .rename(columns={"lap_time_s": "mean_lap_s"})
                       .round(4).to_dict(orient="records"))

        return {
            "test": "Two-way ANOVA",
            "anova_table": rows,
            "interaction_plot_data": interaction,
            "n_observations": len(df),
        }

    except ImportError:
        return {"error": "statsmodels not installed — two-way ANOVA unavailable."}
    except Exception as e:
        return {"error": str(e)}
