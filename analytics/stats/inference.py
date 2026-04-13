"""
Unit 1 — MLE + Confidence Intervals + Bayesian inference
Unit 2 — Hypothesis testing (t-test, z-test)
"""
import numpy as np
import pandas as pd
from scipy import stats as sp


# ── Unit 1: MLE lap-time distribution ─────────────────────────────────────────

def mle_lap_distribution(laps: list[dict], driver_id: str) -> dict:
    """
    Fit a Normal distribution to one driver's lap times via MLE.
    Returns histogram bins + fitted curve + MLE estimates + 95% CI.
    """
    df = pd.DataFrame(laps)
    if df.empty:
        return {}

    drv = df[df["driver_id"] == driver_id]["lap_time_s"].dropna()
    drv = drv[drv > 0]

    # Remove obvious outlier laps (pit in/out)
    mu0, s0 = drv.mean(), drv.std()
    drv = drv[abs(drv - mu0) <= 3 * s0]

    if len(drv) < 5:
        return {}

    n = len(drv)
    mu_hat = float(drv.mean())        # MLE mean
    sigma_hat = float(drv.std(ddof=0))  # MLE std (biased, as per MLE)
    sigma_se = sigma_hat / np.sqrt(2 * n)  # std error of sigma

    # 95% CI for mean using t-distribution
    ci_low, ci_high = sp.t.interval(0.95, df=n - 1,
                                     loc=mu_hat,
                                     scale=sp.sem(drv))

    # Histogram for plotting
    counts, bin_edges = np.histogram(drv, bins=20, density=True)
    bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    # Fitted PDF values
    x_fit = np.linspace(drv.min(), drv.max(), 200)
    pdf_fit = sp.norm.pdf(x_fit, loc=mu_hat, scale=sigma_hat)

    return {
        "driver_id": driver_id,
        "n_laps": n,
        "mu_hat": round(mu_hat, 4),
        "sigma_hat": round(sigma_hat, 4),
        "sigma_se": round(sigma_se, 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "histogram": {"counts": counts.tolist(), "bin_centres": bin_centres},
        "fit_curve": {"x": x_fit.tolist(), "y": pdf_fit.tolist()},
        "interpretation": (
            f"MLE: μ̂ = {mu_hat:.3f}s, σ̂ = {sigma_hat:.3f}s. "
            f"95% CI for mean: [{ci_low:.3f}s, {ci_high:.3f}s]"
        ),
    }


# ── Unit 1: Bayesian win probability ──────────────────────────────────────────

def bayesian_win_probability(laps: list[dict],
                              results: list[dict] | None = None) -> list[dict]:
    """
    P(Finish position | grid_position) via Bayes' theorem.
    Prior = uniform Dirichlet over grid slots.
    Likelihood = empirical finish counts per grid position from results.

    Returns list of per-grid probabilities including separate posterior means for
    finishing 1st, 2nd, and 3rd.
    """
    if not results:
        # Use lap data to approximate: driver with best mean lap time as proxy
        return []

    res_df = pd.DataFrame(results)
    if "position" not in res_df.columns and "final_position" in res_df.columns:
        res_df = res_df.rename(columns={"final_position": "position"})
    if "grid_position" not in res_df.columns or "position" not in res_df.columns:
        return []

    res_df = res_df.dropna(subset=["grid_position", "position"])
    res_df["grid_position"] = res_df["grid_position"].astype(int)
    res_df["position"] = pd.to_numeric(res_df["position"], errors="coerce")
    res_df = res_df.dropna(subset=["position"])
    res_df["position"] = res_df["position"].astype(int)

    res_df["p1"] = (res_df["position"] == 1).astype(int)
    res_df["p2"] = (res_df["position"] == 2).astype(int)
    res_df["p3"] = (res_df["position"] == 3).astype(int)

    grid_max = int(res_df["grid_position"].max())
    output = []
    for g in range(1, min(grid_max + 1, 21)):
        subset = res_df[res_df["grid_position"] == g]
        n_races = len(subset)
        n_p1 = int(subset["p1"].sum())
        n_p2 = int(subset["p2"].sum())
        n_p3 = int(subset["p3"].sum())

        # Beta-Binomial with Beta(1,1) prior for each finish-position event.
        p1_posterior = (1 + n_p1) / (2 + n_races)
        p2_posterior = (1 + n_p2) / (2 + n_races)
        p3_posterior = (1 + n_p3) / (2 + n_races)

        output.append({
            "grid_pos": g,
            "n_races": n_races,
            "n_wins": n_p1,
            "n_p1": n_p1,
            "n_p2": n_p2,
            "n_p3": n_p3,
            "prior": round(1 / grid_max, 4),
            "posterior": round(p1_posterior, 4),
            "posterior_p1": round(p1_posterior, 4),
            "posterior_p2": round(p2_posterior, 4),
            "posterior_p3": round(p3_posterior, 4),
        })

    return output


# ── Unit 2: Two-sample t-test ──────────────────────────────────────────────────

def two_sample_ttest(laps: list[dict],
                     driver_a: str,
                     driver_b: str,
                     alternative: str = "two-sided",
                     alpha: float = 0.05) -> dict:
    """
    Independent Welch two-sample t-test on clean lap times for two drivers.
    Alternative is defined relative to driver_a mean lap time.
    """
    df = pd.DataFrame(laps)
    if df.empty:
        return {}

    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        return {"error": f"Unsupported alternative '{alternative}'."}
    if not 0 < alpha < 1:
        return {"error": "Alpha must be between 0 and 1."}

    def _clean_laps(drv_id):
        x = df[df["driver_id"] == drv_id]["lap_time_s"].dropna()
        x = x[x > 0]
        mu, s = x.mean(), x.std()
        return x[abs(x - mu) <= 3 * s] if s > 0 else x

    a = _clean_laps(driver_a)
    b = _clean_laps(driver_b)

    if len(a) < 3 or len(b) < 3:
        return {"error": "Not enough laps for one or both drivers."}

    mean_a = float(a.mean())
    mean_b = float(b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    se_sq_a = var_a / len(a)
    se_sq_b = var_b / len(b)
    standard_error = np.sqrt(se_sq_a + se_sq_b)
    if standard_error == 0:
        return {"error": "Lap-time variation is zero; t-test is undefined."}

    t_stat = (mean_a - mean_b) / standard_error
    welch_df_num = (se_sq_a + se_sq_b) ** 2
    welch_df_den = 0.0
    if len(a) > 1:
        welch_df_den += (se_sq_a ** 2) / (len(a) - 1)
    if len(b) > 1:
        welch_df_den += (se_sq_b ** 2) / (len(b) - 1)
    welch_df = welch_df_num / welch_df_den if welch_df_den > 0 else max(1, len(a) + len(b) - 2)

    if alternative == "two-sided":
        p_value = 2 * sp.t.sf(abs(t_stat), df=welch_df)
        null_hypothesis = f"{driver_a} and {driver_b} have equal mean lap time."
        alternative_hypothesis = f"{driver_a} and {driver_b} have different mean lap times."
    elif alternative == "less":
        p_value = sp.t.cdf(t_stat, df=welch_df)
        null_hypothesis = f"{driver_a} is not faster than {driver_b} (mean_A >= mean_B)."
        alternative_hypothesis = f"{driver_a} is faster than {driver_b} (mean_A < mean_B)."
    else:
        p_value = sp.t.sf(t_stat, df=welch_df)
        null_hypothesis = f"{driver_a} is not slower than {driver_b} (mean_A <= mean_B)."
        alternative_hypothesis = f"{driver_a} is slower than {driver_b} (mean_A > mean_B)."
    p_value = float(p_value)

    # Cohen's d
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    # Plain-English effect size
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    faster = driver_a if mean_a < mean_b else driver_b
    significant = p_value < alpha

    if alternative == "two-sided":
        outcome_text = "different mean lap times"
    elif alternative == "less":
        outcome_text = f"{driver_a} has a lower mean lap time than {driver_b}"
    else:
        outcome_text = f"{driver_a} has a higher mean lap time than {driver_b}"

    verdict = (
        f"{'Significant' if significant else 'No significant'} evidence that {outcome_text} "
        f"(p={'< 0.001' if p_value < 0.001 else f'{p_value:.4f}'}, α={alpha:.2f}). "
        f"Effect size: {effect} (d={cohens_d:.3f}). "
        f"Observed averages: {driver_a}={mean_a:.3f}s, {driver_b}={mean_b:.3f}s; "
        f"{faster} is faster on average."
    )

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "alpha": round(float(alpha), 4),
        "alternative": alternative,
        "null_hypothesis": null_hypothesis,
        "alternative_hypothesis": alternative_hypothesis,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "std_a": round(float(a.std()), 4),
        "std_b": round(float(b.std()), 4),
        "t_statistic": round(float(t_stat), 4),
        "degrees_of_freedom": round(float(welch_df), 4),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(float(cohens_d), 4),
        "effect_size": effect,
        "significant": bool(significant),
        "verdict": verdict,
        "laps_a": a.tolist(),
        "laps_b": b.tolist(),
    }


def z_test_pit_stop_time(pit_stops: list[dict],
                          driver_id: str,
                          population_mean: float | None = None,
                          alternative: str = "two-sided",
                          alpha: float = 0.05) -> dict:
    """
    One-sample z-test comparing a driver's mean pit stop time to the field.
    """
    df = pd.DataFrame(pit_stops)
    if df.empty or "pit_duration" not in df.columns:
        return {}
    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        return {"error": f"Unsupported alternative '{alternative}'."}
    if not 0 < alpha < 1:
        return {"error": "Alpha must be between 0 and 1."}

    all_times = df["pit_duration"].dropna()
    drv_times = df[df["driver_id"] == driver_id]["pit_duration"].dropna()

    if len(drv_times) < 2:
        return {"error": "Not enough pit stops."}

    pop_mean = population_mean or float(all_times.mean())
    pop_std = float(all_times.std())
    if pop_std <= 0 or np.isnan(pop_std):
        return {"error": "Field pit-stop variation is zero; z-test is undefined."}

    driver_mean = float(drv_times.mean())

    z = (driver_mean - pop_mean) / (pop_std / np.sqrt(len(drv_times)))
    if alternative == "two-sided":
        p = 2 * sp.norm.sf(abs(z))
        null_hypothesis = f"{driver_id} has the same mean pit-stop time as the field."
        alternative_hypothesis = f"{driver_id} has a different mean pit-stop time from the field."
    elif alternative == "less":
        p = sp.norm.cdf(z)
        null_hypothesis = f"{driver_id} is not quicker than the field in pit stops (mean_driver >= mean_field)."
        alternative_hypothesis = f"{driver_id} is quicker than the field in pit stops (mean_driver < mean_field)."
    else:
        p = sp.norm.sf(z)
        null_hypothesis = f"{driver_id} is not slower than the field in pit stops (mean_driver <= mean_field)."
        alternative_hypothesis = f"{driver_id} is slower than the field in pit stops (mean_driver > mean_field)."

    significant = bool(p < alpha)
    if alternative == "two-sided":
        outcome_text = "a different mean pit-stop time from the field"
    elif alternative == "less":
        outcome_text = "a lower mean pit-stop time than the field"
    else:
        outcome_text = "a higher mean pit-stop time than the field"

    return {
        "driver_id": driver_id,
        "alpha": round(float(alpha), 4),
        "alternative": alternative,
        "null_hypothesis": null_hypothesis,
        "alternative_hypothesis": alternative_hypothesis,
        "n_stops": len(drv_times),
        "driver_mean": round(driver_mean, 3),
        "field_mean": round(pop_mean, 3),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p), 6),
        "significant": significant,
        "verdict": (
            f"{'Significant' if significant else 'No significant'} evidence that {driver_id} has {outcome_text} "
            f"(z={z:.3f}, p={'< 0.001' if p < 0.001 else f'{p:.4f}'}, α={alpha:.2f})"
        ),
    }
