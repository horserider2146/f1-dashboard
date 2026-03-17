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
    P(Win | grid_position) via Bayes' theorem.
    Prior = uniform Dirichlet over grid slots.
    Likelihood = empirical win counts per grid position from results.

    Returns list of {grid_pos, prior, likelihood, posterior} sorted by grid_pos.
    """
    if not results:
        # Use lap data to approximate: driver with best mean lap time as proxy
        return []

    res_df = pd.DataFrame(results)
    if "grid_position" not in res_df.columns or "position" not in res_df.columns:
        return []

    res_df = res_df.dropna(subset=["grid_position", "position"])
    res_df["grid_position"] = res_df["grid_position"].astype(int)
    res_df["won"] = (res_df["position"] == 1).astype(int)

    grid_max = int(res_df["grid_position"].max())
    output = []
    for g in range(1, min(grid_max + 1, 21)):
        subset = res_df[res_df["grid_position"] == g]
        n_races = len(subset)
        n_wins = int(subset["won"].sum())
        # Beta-Binomial: prior Beta(1,1) (uniform), posterior Beta(1+wins, 1+losses)
        alpha = 1 + n_wins
        beta_param = 1 + (n_races - n_wins)
        posterior_mean = alpha / (alpha + beta_param)
        output.append({
            "grid_pos": g,
            "n_races": n_races,
            "n_wins": n_wins,
            "prior": round(1 / grid_max, 4),
            "posterior": round(posterior_mean, 4),
        })

    return output


# ── Unit 2: Two-sample t-test ──────────────────────────────────────────────────

def two_sample_ttest(laps: list[dict],
                     driver_a: str,
                     driver_b: str) -> dict:
    """
    Independent two-sample t-test on clean lap times for two drivers.
    Returns t-stat, p-value, Cohen's d, and plain-English verdict.
    """
    df = pd.DataFrame(laps)
    if df.empty:
        return {}

    def _clean_laps(drv_id):
        x = df[df["driver_id"] == drv_id]["lap_time_s"].dropna()
        x = x[x > 0]
        mu, s = x.mean(), x.std()
        return x[abs(x - mu) <= 3 * s] if s > 0 else x

    a = _clean_laps(driver_a)
    b = _clean_laps(driver_b)

    if len(a) < 3 or len(b) < 3:
        return {"error": "Not enough laps for one or both drivers."}

    t_stat, p_value = sp.ttest_ind(a, b, equal_var=False)  # Welch's t-test

    # Cohen's d
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

    # Plain-English effect size
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    faster = driver_a if a.mean() < b.mean() else driver_b
    significant = p_value < 0.05

    verdict = (
        f"{'Significant' if significant else 'No significant'} difference "
        f"(p={'< 0.001' if p_value < 0.001 else f'{p_value:.4f}'}, α=0.05). "
        f"Effect size: {effect} (d={cohens_d:.3f}). "
        f"{faster} is faster on average."
    )

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "std_a": round(float(a.std()), 4),
        "std_b": round(float(b.std()), 4),
        "t_statistic": round(float(t_stat), 4),
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
                          population_mean: float | None = None) -> dict:
    """
    One-sample z-test: is this driver's mean pit stop time different from
    the field average?
    """
    df = pd.DataFrame(pit_stops)
    if df.empty or "pit_duration" not in df.columns:
        return {}

    all_times = df["pit_duration"].dropna()
    drv_times = df[df["driver_id"] == driver_id]["pit_duration"].dropna()

    if len(drv_times) < 2:
        return {"error": "Not enough pit stops."}

    pop_mean = population_mean or float(all_times.mean())
    pop_std = float(all_times.std())

    z = (drv_times.mean() - pop_mean) / (pop_std / np.sqrt(len(drv_times)))
    p = 2 * sp.norm.sf(abs(z))

    return {
        "driver_id": driver_id,
        "n_stops": len(drv_times),
        "driver_mean": round(float(drv_times.mean()), 3),
        "field_mean": round(pop_mean, 3),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "verdict": (
            f"{'Significant' if p < 0.05 else 'No significant'} difference from "
            f"field average (z={z:.3f}, p={p:.4f})"
        ),
    }
