"""
Page: Applied Statistical Workflow.

This page keeps all statistical content in one continuous analysis pipeline that
shows how methods are applied to race decisions, instead of splitting everything
into disconnected topic chunks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sp
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from dashboard import api_client as api


@st.cache_data(show_spinner=False)
def _load_laps_df(year: int, gp: str) -> pd.DataFrame:
    try:
        return pd.DataFrame(api.get_laps(year, gp))
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_drivers_df(year: int, gp: str) -> pd.DataFrame:
    try:
        return pd.DataFrame(api.get_drivers(year, gp))
    except Exception:
        return pd.DataFrame()


def _clean_lap_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return s
    mu = float(s.mean())
    sigma = float(s.std())
    if sigma <= 0 or np.isnan(sigma):
        return s
    return s[np.abs(s - mu) <= 3 * sigma]


def _sampling_distribution(series: pd.Series,
                           sample_size: int,
                           draws: int = 2500) -> dict | None:
    clean = _clean_lap_series(series)
    if len(clean) < sample_size:
        return None
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(clean.values, size=sample_size, replace=True).mean()
        for _ in range(draws)
    ])
    return {
        "means": means,
        "mean_of_means": float(means.mean()),
        "std_error": float(means.std(ddof=1)),
        "n": len(clean),
    }


def _chi_square_independence(df: pd.DataFrame,
                             row_col: str,
                             col_col: str) -> dict | None:
    if row_col not in df.columns or col_col not in df.columns:
        return None
    work = df[[row_col, col_col]].dropna()
    if work.empty:
        return None

    contingency = pd.crosstab(work[row_col], work[col_col])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None

    chi2, p_value, dof, expected = sp.chi2_contingency(contingency)
    expected_df = pd.DataFrame(expected,
                               index=contingency.index,
                               columns=contingency.columns)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "contingency": contingency,
        "expected": expected_df,
        "significant": bool(p_value < 0.05),
    }


def _chi_square_goodness_of_fit(series: pd.Series) -> dict | None:
    s = series.dropna()
    if s.nunique() < 2:
        return None

    observed = s.value_counts().sort_index()
    expected = np.repeat(observed.sum() / len(observed), len(observed))
    chi2, p_value = sp.chisquare(observed.values, f_exp=expected)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "observed": observed,
        "expected": expected,
        "significant": bool(p_value < 0.05),
    }


def _one_sample_ttest(laps_df: pd.DataFrame, driver_id: str) -> dict | None:
    if laps_df.empty:
        return None
    if "driver_id" not in laps_df.columns or "lap_time_s" not in laps_df.columns:
        return None

    all_laps = _clean_lap_series(laps_df["lap_time_s"])
    drv_laps = _clean_lap_series(
        laps_df.loc[laps_df["driver_id"] == driver_id, "lap_time_s"]
    )

    if len(all_laps) < 5 or len(drv_laps) < 3:
        return None

    pop_mean = float(all_laps.mean())
    t_stat, p_value = sp.ttest_1samp(drv_laps, popmean=pop_mean)

    return {
        "driver_mean": float(drv_laps.mean()),
        "overall_mean": pop_mean,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": int(len(drv_laps)),
        "significant": bool(p_value < 0.05),
    }


def _sign_test(laps_df: pd.DataFrame, driver_a: str, driver_b: str) -> dict | None:
    needed = {"driver_id", "lap_number", "lap_time_s"}
    if laps_df.empty or not needed.issubset(set(laps_df.columns)):
        return None

    a_df = (laps_df[laps_df["driver_id"] == driver_a][["lap_number", "lap_time_s"]]
            .dropna())
    b_df = (laps_df[laps_df["driver_id"] == driver_b][["lap_number", "lap_time_s"]]
            .dropna())

    merged = a_df.merge(b_df, on="lap_number", suffixes=("_a", "_b"))
    if len(merged) < 6:
        return None

    diff = merged["lap_time_s_a"] - merged["lap_time_s_b"]
    non_zero = diff[diff != 0]
    if len(non_zero) < 5:
        return None

    faster_a = int((non_zero < 0).sum())
    faster_b = int((non_zero > 0).sum())

    # Two-sided binomial sign test under H0: P(A faster) = 0.5
    p_value = sp.binomtest(faster_a, n=len(non_zero), p=0.5, alternative="two-sided").pvalue

    return {
        "paired_laps": int(len(non_zero)),
        "faster_a": faster_a,
        "faster_b": faster_b,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def _functional_form_check(raw_df: pd.DataFrame) -> dict | None:
    needed = {"grid_position", "position"}
    if raw_df.empty or not needed.issubset(set(raw_df.columns)):
        return None

    clean = raw_df[["grid_position", "position"]].dropna().astype(float)
    if len(clean) < 5:
        return None

    x = clean[["grid_position"]].values
    y = clean["position"].values

    linear = LinearRegression().fit(x, y)
    y_linear = linear.predict(x)
    r2_linear = r2_score(y, y_linear)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x)
    quad = LinearRegression().fit(x_poly, y)
    y_quad = quad.predict(x_poly)
    r2_quad = r2_score(y, y_quad)

    xs = np.linspace(x.min(), x.max(), 60).reshape(-1, 1)
    ys_linear = linear.predict(xs)
    ys_quad = quad.predict(poly.transform(xs))

    return {
        "r2_linear": float(r2_linear),
        "r2_quadratic": float(r2_quad),
        "x_plot": xs.flatten(),
        "y_linear": ys_linear,
        "y_quad": ys_quad,
        "x_raw": x.flatten(),
        "y_raw": y,
    }


def _transformation_check(raw_df: pd.DataFrame) -> dict | None:
    if raw_df.empty or "avg_pit_time" not in raw_df.columns:
        return None

    s = pd.to_numeric(raw_df["avg_pit_time"], errors="coerce").dropna()
    if len(s) < 4:
        return None

    original_skew = float(sp.skew(s, bias=False))
    shifted = s - s.min() + 1e-6
    transformed = np.log1p(shifted)
    transformed_skew = float(sp.skew(transformed, bias=False))

    return {
        "original_skew": original_skew,
        "log_skew": transformed_skew,
        "improves": abs(transformed_skew) < abs(original_skew),
    }


def _canonical_correlation(raw_df: pd.DataFrame) -> dict | None:
    x_cols = ["grid_position", "num_stops", "avg_pit_time"]
    y_cols = ["position", "avg_lap_time"]
    needed = set(x_cols + y_cols)

    if raw_df.empty or not needed.issubset(set(raw_df.columns)):
        return None

    clean = raw_df[x_cols + y_cols].dropna().astype(float)
    if len(clean) < 6:
        return None

    x = clean[x_cols]
    y = clean[y_cols]

    x_std = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    y_std = (y - y.mean()) / y.std(ddof=0).replace(0, 1)

    cca = CCA(n_components=1)
    x_c, y_c = cca.fit_transform(x_std, y_std)
    corr = float(np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1])

    return {
        "canonical_corr": corr,
        "n_obs": int(len(clean)),
        "x_weights": dict(zip(x_cols, cca.x_weights_[:, 0].round(4))),
        "y_weights": dict(zip(y_cols, cca.y_weights_[:, 0].round(4))),
    }


def _poisson_count_model(raw_df: pd.DataFrame) -> dict | None:
    needed = {"num_stops", "grid_position", "avg_lap_time"}
    if raw_df.empty or not needed.issubset(set(raw_df.columns)):
        return None

    try:
        import statsmodels.api as sm
    except Exception:
        return {"error": "statsmodels not installed for Poisson count modeling."}

    clean = raw_df[["num_stops", "grid_position", "avg_lap_time"]].dropna().copy()
    if len(clean) < 6:
        return None

    clean["num_stops"] = clean["num_stops"].clip(lower=0).round().astype(int)
    if clean["num_stops"].nunique() < 2:
        return None

    y = clean["num_stops"]
    x = sm.add_constant(clean[["grid_position", "avg_lap_time"]].astype(float))

    model = sm.GLM(y, x, family=sm.families.Poisson()).fit()

    coef_rows = []
    for name in model.params.index:
        coef_rows.append({
            "term": name,
            "coef": round(float(model.params[name]), 4),
            "exp_coef": round(float(np.exp(model.params[name])), 4),
            "p_value": round(float(model.pvalues[name]), 6),
        })

    return {
        "aic": round(float(model.aic), 3),
        "deviance": round(float(model.deviance), 3),
        "coefficients": coef_rows,
    }


def _nonlinear_tyre_model(laps_df: pd.DataFrame, driver_id: str) -> dict | None:
    needed = {"driver_id", "tyre_age", "lap_time_s"}
    if laps_df.empty or not needed.issubset(set(laps_df.columns)):
        return None

    clean = laps_df.loc[laps_df["driver_id"] == driver_id,
                        ["tyre_age", "lap_time_s"]].dropna().copy()
    clean = clean[clean["lap_time_s"] > 0]
    if len(clean) < 8:
        return None

    x = clean[["tyre_age"]].astype(float).values
    y = clean["lap_time_s"].astype(float).values

    linear = LinearRegression().fit(x, y)
    y_linear = linear.predict(x)
    r2_linear = r2_score(y, y_linear)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x)
    quad = LinearRegression().fit(x_poly, y)
    y_quad = quad.predict(x_poly)
    r2_quad = r2_score(y, y_quad)

    xs = np.linspace(x.min(), x.max(), 70).reshape(-1, 1)
    y_line_plot = linear.predict(xs)
    y_quad_plot = quad.predict(poly.transform(xs))

    return {
        "r2_linear": float(r2_linear),
        "r2_quadratic": float(r2_quad),
        "x_raw": x.flatten(),
        "y_raw": y,
        "x_plot": xs.flatten(),
        "y_line_plot": y_line_plot,
        "y_quad_plot": y_quad_plot,
    }


def _team_dummy_regression(dci_df: pd.DataFrame,
                           drivers_df: pd.DataFrame) -> dict | None:
    needed_dci = {"driver_id", "mean_lap_s"}
    needed_driver = {"driver_id", "team"}
    if dci_df.empty or drivers_df.empty:
        return None
    if not needed_dci.issubset(set(dci_df.columns)):
        return None
    if not needed_driver.issubset(set(drivers_df.columns)):
        return None

    try:
        import statsmodels.api as sm
    except Exception:
        return {"error": "statsmodels not installed for dummy regression."}

    merged = dci_df[["driver_id", "mean_lap_s"]].merge(
        drivers_df[["driver_id", "team"]], on="driver_id", how="left"
    ).dropna()

    counts = merged["team"].value_counts()
    keep = counts[counts >= 2].index
    merged = merged[merged["team"].isin(keep)]

    if merged["team"].nunique() < 2:
        return None

    x = pd.get_dummies(merged["team"], prefix="team", drop_first=True, dtype=float)
    x = sm.add_constant(x)
    y = merged["mean_lap_s"].astype(float)

    model = sm.OLS(y, x).fit()

    rows = []
    for term in model.params.index:
        rows.append({
            "term": term,
            "coef": round(float(model.params[term]), 4),
            "p_value": round(float(model.pvalues[term]), 6),
        })

    return {
        "r_squared": round(float(model.rsquared), 4),
        "f_stat": round(float(model.fvalue), 4) if model.fvalue is not None else None,
        "f_p_value": round(float(model.f_pvalue), 6) if model.f_pvalue is not None else None,
        "coefficients": rows,
    }


def _run_manova_from_dci(dci_df: pd.DataFrame,
                         drivers_df: pd.DataFrame) -> dict | None:
    needed_dci = {"driver_id", "mean_lap_s", "std_lap_s"}
    needed_driver = {"driver_id", "team"}
    if dci_df.empty or drivers_df.empty:
        return None
    if not needed_dci.issubset(set(dci_df.columns)):
        return None
    if not needed_driver.issubset(set(drivers_df.columns)):
        return None

    try:
        from statsmodels.multivariate.manova import MANOVA
    except Exception:
        return {"error": "statsmodels MANOVA is not available."}

    merged = dci_df[["driver_id", "mean_lap_s", "std_lap_s"]].merge(
        drivers_df[["driver_id", "team"]], on="driver_id", how="left"
    ).dropna()

    counts = merged["team"].value_counts()
    keep = counts[counts >= 2].index
    merged = merged[merged["team"].isin(keep)]

    if merged["team"].nunique() < 2:
        return None

    model = MANOVA.from_formula("mean_lap_s + std_lap_s ~ team", data=merged)
    mv = model.mv_test()

    try:
        stat_df = mv.results["team"]["stat"].copy()
        pillai = stat_df.loc["Pillai's trace"]
        return {
            "table": stat_df.reset_index().rename(columns={"index": "stat"}),
            "pillai_f": float(pillai["F Value"]),
            "pillai_p": float(pillai["Pr > F"]),
        }
    except Exception:
        return {"summary": str(mv)}


def render(year: int, gp: str):
    st.header("Applied Statistical Workflow")
    st.caption(
        f"Race: {gp} {year}. This page follows one continuous analysis path: "
        "baseline -> inference -> group effects -> modeling -> decision validation."
    )

    try:
        api._get(f"/stats/{year}/{gp}/dci")
    except Exception as e:
        msg = str(e)
        if "404" in msg or "not available" in msg.lower() or "not cached" in msg.lower():
            st.warning(
                f"Race data not available for {gp} {year}. "
                "Only cached races work offline: 2025 Australian, Chinese, Bahrain, Miami."
            )
        else:
            st.error(f"Could not load race data: {e}")
        return

    st.markdown(
        """
        <style>
        .flow-card {
            border: 1px solid #2a2a2a;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0;
            background: linear-gradient(135deg, #111111 0%, #171717 100%);
        }
        .flow-title {
            font-weight: 700;
            letter-spacing: 0.2px;
            margin-bottom: 0.25rem;
            color: #f5f5f5;
        }
        .flow-sub {
            color: #b8b8b8;
            font-size: 0.9rem;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-card">
            <p class="flow-title">How to read this page</p>
            <p class="flow-sub">
                Start at step 1 and move down. Each step answers a practical race question,
                then applies the right statistical tools to that question.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    focus = st.selectbox(
        "Primary analysis objective",
        options=[
            "Explain who was fastest and why",
            "Test if observed differences are statistically real",
            "Build prediction and decision models",
            "Validate results with assumption-robust methods",
        ],
        key="flow_focus",
    )
    st.caption(f"Current objective: {focus}")

    laps_df = _load_laps_df(year, gp)
    drivers_df = _load_drivers_df(year, gp)

    if laps_df.empty:
        st.error("No lap data available for this race.")
        return

    if "team" not in laps_df.columns and not drivers_df.empty and {"driver_id", "team"}.issubset(drivers_df.columns):
        laps_df = laps_df.merge(drivers_df[["driver_id", "team"]], on="driver_id", how="left")

    driver_ids = sorted(laps_df["driver_id"].dropna().unique().tolist()) if "driver_id" in laps_df.columns else []
    if not driver_ids:
        st.error("No drivers found in race data.")
        return

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------
    st.subheader("1) Establish baseline pace and uncertainty")
    st.caption("Applied topics: statistical inference framework, likelihood, MLE, confidence intervals, frequentist and Bayesian setup.")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("Question: Which drivers are consistently quick lap after lap?")
        if st.button("Compute consistency baseline", key="flow_dci_btn"):
            with st.spinner("Computing consistency metrics..."):
                try:
                    dci_data = api._get(f"/stats/{year}/{gp}/dci")
                    dci_df = pd.DataFrame(dci_data)
                    fig = px.bar(
                        dci_df,
                        x="driver_id",
                        y="dci_normalised",
                        color="dci_normalised",
                        color_continuous_scale="RdYlGn",
                        title="Driver consistency baseline",
                        labels={"driver_id": "Driver", "dci_normalised": "Normalised DCI"},
                        template="plotly_dark",
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width="stretch")

                    show_cols = [
                        "rank",
                        "driver_id",
                        "dci_normalised",
                        "mean_lap_s",
                        "std_lap_s",
                        "n_laps",
                    ]
                    st.dataframe(dci_df[show_cols], width="stretch", hide_index=True)
                except Exception as e:
                    st.error(f"Consistency baseline failed: {e}")

        if st.button("Run DCI vs points correlation", key="flow_dci_corr_btn"):
            with st.spinner("Computing DCI-points correlation..."):
                try:
                    corr_result = api._get(f"/stats/{year}/{gp}/dci/correlation")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Pearson r", corr_result.get("r", "-"))
                    m2.metric("p-value", corr_result.get("p_value", "-"))
                    m3.metric("Significant", "Yes" if corr_result.get("significant") else "No")
                    st.info(corr_result.get("interpretation", ""))
                except Exception as e:
                    st.error(f"Correlation failed: {e}")

    with col_r:
        mle_driver = st.selectbox("Driver for likelihood and MLE", driver_ids, key="flow_mle_driver")
        if st.button("Fit likelihood and confidence interval", key="flow_mle_btn"):
            with st.spinner("Fitting distribution..."):
                try:
                    mle = api._get(f"/stats/{year}/{gp}/mle/{mle_driver}")

                    fig = go.Figure()
                    hist = mle["histogram"]
                    fit_curve = mle["fit_curve"]

                    fig.add_trace(go.Bar(
                        x=hist["bin_centres"],
                        y=hist["counts"],
                        name="Observed density",
                        marker_color="#5ea0ff",
                        opacity=0.7,
                    ))
                    fig.add_trace(go.Scatter(
                        x=fit_curve["x"],
                        y=fit_curve["y"],
                        name="Likelihood-based normal fit",
                        line=dict(color="#ff4f4f", width=2.4),
                    ))
                    fig.add_vline(
                        x=mle["ci_95_low"],
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="CI low",
                    )
                    fig.add_vline(
                        x=mle["ci_95_high"],
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="CI high",
                    )
                    fig.update_layout(
                        title=f"Lap-time likelihood profile: {mle_driver}",
                        xaxis_title="Lap time (s)",
                        yaxis_title="Density",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, width="stretch")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("mu_hat", f"{mle['mu_hat']:.3f}s")
                    m2.metric("sigma_hat", f"{mle['sigma_hat']:.3f}s")
                    m3.metric("95% CI low", f"{mle['ci_95_low']:.3f}s")
                    m4.metric("95% CI high", f"{mle['ci_95_high']:.3f}s")
                except Exception as e:
                    st.error(f"MLE analysis failed: {e}")

    sampling_driver = st.selectbox("Driver for sampling distribution", driver_ids,
                                   key="flow_sampling_driver")
    sample_size = st.slider("Sample size for mean estimation", 5, 25, 12,
                            key="flow_sampling_size")
    if st.button("Simulate sampling distribution", key="flow_sampling_btn"):
        sim = _sampling_distribution(
            laps_df.loc[laps_df["driver_id"] == sampling_driver, "lap_time_s"],
            sample_size=sample_size,
        )
        if not sim:
            st.warning("Not enough laps to simulate this sample size.")
        else:
            means_df = pd.DataFrame({"sample_mean": sim["means"]})
            fig = px.histogram(
                means_df,
                x="sample_mean",
                nbins=35,
                title=f"Sampling distribution of mean lap time ({sampling_driver})",
                template="plotly_dark",
            )
            fig.add_vline(
                x=sim["mean_of_means"],
                line_dash="dash",
                line_color="#ff4f4f",
                annotation_text="Mean of sample means",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Estimated sampling SE = {sim['std_error']:.4f}s from {len(sim['means'])} resamples."
            )

    if st.button("Update prior to posterior (Bayesian)", key="flow_bayes_btn"):
        with st.spinner("Computing posterior probabilities..."):
            try:
                bayes_data = api._get(f"/stats/{year}/{gp}/bayes-win")
                if bayes_data:
                    bayes_df = pd.DataFrame(bayes_data)
                    fig = px.bar(
                        bayes_df,
                        x="grid_pos",
                        y="posterior",
                        title="Posterior P(win | grid position)",
                        labels={"grid_pos": "Grid", "posterior": "Posterior probability"},
                        color="posterior",
                        color_continuous_scale="Reds",
                        template="plotly_dark",
                    )
                    fig.add_scatter(
                        x=bayes_df["grid_pos"],
                        y=bayes_df["prior"],
                        mode="lines",
                        name="Prior",
                        line=dict(color="white", dash="dash"),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No historical race results available for Bayesian update.")
            except Exception as e:
                st.error(f"Bayesian analysis failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------
    st.subheader("2) Test claims under uncertainty")
    st.caption("Applied topics: population mean testing, one-sample and two-sample t-tests, large-sample z-test, rejection and non-rejection regions, Type I and Type II errors.")

    alpha = st.slider("Significance level alpha", 0.01, 0.10, 0.05, 0.01,
                      key="flow_alpha")

    c1, c2 = st.columns(2)
    with c1:
        driver_a = st.selectbox("Driver A", driver_ids, key="flow_t_a")
    with c2:
        candidates_b = [d for d in driver_ids if d != driver_a]
        driver_b = st.selectbox("Driver B", candidates_b, key="flow_t_b")

    if st.button("Run two-sample t-test", key="flow_ttest_btn"):
        with st.spinner("Running two-sample test..."):
            try:
                t_res = api._get(
                    f"/stats/{year}/{gp}/ttest",
                    params={"driver_a": driver_a, "driver_b": driver_b},
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{driver_a} mean", f"{t_res['mean_a']:.3f}s")
                m2.metric(f"{driver_b} mean", f"{t_res['mean_b']:.3f}s")
                m3.metric("t-statistic", f"{t_res['t_statistic']:.3f}")
                m4.metric("p-value", f"{t_res['p_value']:.5f}")

                m5, m6 = st.columns(2)
                m5.metric("Cohen's d", f"{t_res['cohens_d']:.3f}")
                m6.metric("Effect", t_res["effect_size"])

                st.success(t_res["verdict"]) if t_res["significant"] else st.info(t_res["verdict"])

                # Rejection region visualization (frequentist decision boundary)
                dfree = max(1, t_res["n_a"] + t_res["n_b"] - 2)
                x_vals = np.linspace(-4.5, 4.5, 500)
                y_vals = sp.t.pdf(x_vals, df=dfree)
                crit = sp.t.ppf(1 - alpha / 2, df=dfree)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="t distribution"))
                fig.add_vline(x=crit, line_dash="dash", line_color="orange", annotation_text="+critical")
                fig.add_vline(x=-crit, line_dash="dash", line_color="orange", annotation_text="-critical")
                fig.add_vline(
                    x=t_res["t_statistic"],
                    line_dash="solid",
                    line_color="#ff4f4f",
                    annotation_text="observed t",
                )
                fig.update_layout(
                    title=f"Rejection region at alpha={alpha:.2f}",
                    xaxis_title="t value",
                    yaxis_title="Density",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, width="stretch")

                # Approximate Type II error using normal approximation to power
                n_eff = (t_res["n_a"] * t_res["n_b"]) / (t_res["n_a"] + t_res["n_b"])
                z_alpha = sp.norm.ppf(1 - alpha / 2)
                effect = abs(t_res["cohens_d"])
                approx_power = float(np.clip(sp.norm.cdf(effect * np.sqrt(n_eff) - z_alpha), 0, 1))
                beta = 1 - approx_power
                st.caption(
                    f"Approximate Type I error = {alpha:.2f}, Type II error (beta) approx = {beta:.2f}, "
                    f"power approx = {approx_power:.2f}."
                )
            except Exception as e:
                st.error(f"Two-sample t-test failed: {e}")

    one_sample_driver = st.selectbox("Driver for one-sample mean test",
                                     driver_ids,
                                     key="flow_one_sample_driver")
    if st.button("Run one-sample t-test vs field mean", key="flow_one_sample_btn"):
        one_sample = _one_sample_ttest(laps_df, one_sample_driver)
        if not one_sample:
            st.warning("Not enough data for one-sample test.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Driver mean", f"{one_sample['driver_mean']:.3f}s")
            m2.metric("Field mean", f"{one_sample['overall_mean']:.3f}s")
            m3.metric("t-statistic", f"{one_sample['t_stat']:.3f}")
            m4.metric("p-value", f"{one_sample['p_value']:.5f}")
            if one_sample["significant"]:
                st.success("Driver mean is significantly different from overall mean.")
            else:
                st.info("No significant difference from overall mean.")

    z_driver = st.selectbox("Driver for large-sample z-test", driver_ids,
                            key="flow_z_driver")
    if st.button("Run pit-stop z-test", key="flow_z_btn"):
        with st.spinner("Running z-test..."):
            try:
                z_res = api._get(f"/stats/{year}/{gp}/ztest/{z_driver}")
                if not z_res.get("available", True):
                    st.info(z_res.get("message", "Pit-stop duration data unavailable."))
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Driver mean pit", f"{z_res['driver_mean']:.3f}s")
                    m2.metric("Field mean pit", f"{z_res['field_mean']:.3f}s")
                    m3.metric("z-statistic", f"{z_res['z_statistic']:.3f}")
                    m4.metric("p-value", f"{z_res['p_value']:.5f}")
                    st.success(z_res["verdict"]) if z_res["significant"] else st.info(z_res["verdict"])
            except Exception as e:
                st.error(f"Z-test failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------
    st.subheader("3) Evaluate group effects and distribution fit")
    st.caption("Applied topics: chi-square independence and goodness-of-fit, distribution appropriateness, one-way/two-way ANOVA, F-test interpretation, and MANOVA.")

    chi_col_1, chi_col_2 = st.columns(2)

    with chi_col_1:
        st.markdown("Chi-square test of independence: Team vs compound usage")
        chi_ind = _chi_square_independence(laps_df, "team", "compound")
        if not chi_ind:
            st.info("Insufficient team/compound data for chi-square independence.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("chi2", f"{chi_ind['chi2']:.3f}")
            m2.metric("p-value", f"{chi_ind['p_value']:.5f}")
            m3.metric("DoF", chi_ind["dof"])

            heat = px.imshow(
                chi_ind["contingency"],
                title="Observed counts: team x compound",
                text_auto=True,
                template="plotly_dark",
            )
            st.plotly_chart(heat, width="stretch")

            st.success("Dependent relationship detected.") if chi_ind["significant"] else st.info("No strong dependency detected.")

    with chi_col_2:
        st.markdown("Chi-square goodness-of-fit: Compound mix vs equal allocation")
        if "compound" in laps_df.columns:
            chi_gof = _chi_square_goodness_of_fit(laps_df["compound"])
        else:
            chi_gof = None

        if not chi_gof:
            st.info("Insufficient compound data for goodness-of-fit.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("chi2", f"{chi_gof['chi2']:.3f}")
            m2.metric("p-value", f"{chi_gof['p_value']:.5f}")
            obs_df = pd.DataFrame({
                "compound": chi_gof["observed"].index,
                "observed": chi_gof["observed"].values,
                "expected_equal": chi_gof["expected"],
            })
            fig = go.Figure()
            fig.add_bar(x=obs_df["compound"], y=obs_df["observed"], name="Observed")
            fig.add_bar(x=obs_df["compound"], y=obs_df["expected_equal"], name="Expected")
            fig.update_layout(
                barmode="group",
                title="Observed vs expected distribution",
                template="plotly_dark",
            )
            st.plotly_chart(fig, width="stretch")
            st.success("Distribution differs from equal allocation.") if chi_gof["significant"] else st.info("No clear deviation from equal allocation.")

    anova_group = st.radio("One-way ANOVA grouping", ["team", "driver_id"],
                           horizontal=True, key="flow_anova_group")
    if st.button("Run one-way ANOVA", key="flow_anova_btn"):
        with st.spinner("Running one-way ANOVA..."):
            try:
                anova = api._get(
                    f"/stats/{year}/{gp}/anova/one-way",
                    params={"group": anova_group},
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("F-statistic", f"{anova['f_statistic']:.4f}")
                m2.metric("p-value", f"{anova['p_value']:.5f}")
                m3.metric("eta^2", f"{anova['eta_squared']:.4f}")
                st.success(anova["verdict"]) if anova["significant"] else st.info(anova["verdict"])

                if anova.get("tukey_hsd"):
                    st.caption("Post-hoc (Tukey HSD)")
                    st.dataframe(pd.DataFrame(anova["tukey_hsd"]),
                                 width="stretch",
                                 hide_index=True)
            except Exception as e:
                st.error(f"One-way ANOVA failed: {e}")

    if st.button("Run two-way ANOVA (team x compound)", key="flow_anova2_btn"):
        with st.spinner("Running two-way ANOVA..."):
            try:
                anova2 = api._get(f"/stats/{year}/{gp}/anova/two-way")
                if anova2.get("anova_table"):
                    anova_table = pd.DataFrame(anova2["anova_table"])
                    st.dataframe(anova_table, width="stretch", hide_index=True)
                    if anova2.get("interaction_plot_data"):
                        idf = pd.DataFrame(anova2["interaction_plot_data"])
                        fig = px.line(
                            idf,
                            x="team",
                            y="mean_lap_s",
                            color="compound",
                            title="Interaction profile: team x compound",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info(anova2.get("error", "Two-way ANOVA output unavailable."))
            except Exception as e:
                st.error(f"Two-way ANOVA failed: {e}")

    if st.button("Run MANOVA (mean and variance by team)", key="flow_manova_btn"):
        with st.spinner("Running MANOVA..."):
            try:
                dci_df = pd.DataFrame(api._get(f"/stats/{year}/{gp}/dci"))
                manova = _run_manova_from_dci(dci_df, drivers_df)
                if not manova:
                    st.info("Insufficient data for MANOVA.")
                elif manova.get("error"):
                    st.info(manova["error"])
                elif manova.get("table") is not None:
                    st.dataframe(manova["table"], width="stretch", hide_index=True)
                    st.caption(
                        f"Pillai's trace test: F={manova['pillai_f']:.3f}, p={manova['pillai_p']:.5f}"
                    )
                else:
                    st.text(manova.get("summary", "MANOVA completed."))
            except Exception as e:
                st.error(f"MANOVA failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------
    st.subheader("4) Build regression explanations and diagnose assumptions")
    st.caption("Applied topics: simple/multiple/multivariate regression framing, least squares, residual analysis, model assumptions, dummy-variable regression, autocorrelation, heteroskedasticity, specification diagnostics, and transformations.")

    if st.button("Run OLS model with diagnostics", key="flow_ols_btn"):
        with st.spinner("Fitting OLS model..."):
            try:
                ols = api._get(f"/stats/{year}/{gp}/regression/ols")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R2", ols["r_squared"])
                m2.metric("Adj R2", ols["adj_r_squared"])
                m3.metric("F-stat", ols["f_statistic"])
                m4.metric("Durbin-Watson", ols["durbin_watson"])

                if ols.get("breusch_pagan"):
                    bp = ols["breusch_pagan"]
                    st.caption(
                        f"Breusch-Pagan: statistic={bp['statistic']:.3f}, p={bp['p_value']:.5f}, "
                        f"heteroskedastic={bp['heteroskedastic']}"
                    )

                st.dataframe(pd.DataFrame(ols["coefficients"]),
                             width="stretch",
                             hide_index=True)

                if ols.get("vif"):
                    st.caption("Multicollinearity check (VIF)")
                    st.dataframe(pd.DataFrame(ols["vif"]),
                                 width="stretch",
                                 hide_index=True)

                c1, c2 = st.columns(2)
                with c1:
                    fig_res = go.Figure(go.Scatter(
                        x=ols["fitted"],
                        y=ols["residuals"],
                        mode="markers",
                        marker=dict(color="#5ea0ff", size=8),
                    ))
                    fig_res.add_hline(y=0, line_dash="dash", line_color="white")
                    fig_res.update_layout(
                        title="Residuals vs fitted",
                        xaxis_title="Fitted values",
                        yaxis_title="Residual",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_res, width="stretch")

                with c2:
                    qq = ols["qq_plot"]
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=qq["theoretical"],
                        y=qq["sample"],
                        mode="markers",
                        name="Residual quantiles",
                    ))
                    min_q = min(qq["theoretical"])
                    max_q = max(qq["theoretical"])
                    fig_qq.add_trace(go.Scatter(
                        x=[min_q, max_q],
                        y=[min_q, max_q],
                        mode="lines",
                        line=dict(color="#ff4f4f", dash="dash"),
                        name="Normal line",
                    ))
                    fig_qq.update_layout(
                        title="Q-Q plot",
                        xaxis_title="Theoretical quantiles",
                        yaxis_title="Observed quantiles",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_qq, width="stretch")
            except Exception as e:
                st.error(f"OLS regression failed: {e}")

    if st.button("Run model specification checks", key="flow_spec_btn"):
        with st.spinner("Running functional form and transformation checks..."):
            try:
                corr_payload = api._get(f"/stats/{year}/{gp}/correlation-matrix")
                raw_df = pd.DataFrame(corr_payload.get("raw_data", []))

                form_check = _functional_form_check(raw_df)
                transform_check = _transformation_check(raw_df)

                if form_check:
                    m1, m2 = st.columns(2)
                    m1.metric("Linear R2", f"{form_check['r2_linear']:.4f}")
                    m2.metric("Quadratic R2", f"{form_check['r2_quadratic']:.4f}")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=form_check["x_raw"],
                        y=form_check["y_raw"],
                        mode="markers",
                        name="Observed",
                    ))
                    fig.add_trace(go.Scatter(
                        x=form_check["x_plot"],
                        y=form_check["y_linear"],
                        mode="lines",
                        name="Linear fit",
                    ))
                    fig.add_trace(go.Scatter(
                        x=form_check["x_plot"],
                        y=form_check["y_quad"],
                        mode="lines",
                        name="Quadratic fit",
                    ))
                    fig.update_layout(
                        title="Functional form check: position vs grid",
                        xaxis_title="Grid position",
                        yaxis_title="Finish position",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, width="stretch")

                if transform_check:
                    m1, m2 = st.columns(2)
                    m1.metric("Skew before transform", f"{transform_check['original_skew']:.3f}")
                    m2.metric("Skew after log transform", f"{transform_check['log_skew']:.3f}")
                    if transform_check["improves"]:
                        st.info("Log transformation improves distribution symmetry for avg_pit_time.")
                    else:
                        st.info("Log transformation does not improve symmetry for this race sample.")

                if not form_check and not transform_check:
                    st.info("Insufficient data for specification checks.")
            except Exception as e:
                st.error(f"Specification checks failed: {e}")

    if st.button("Run team dummy-variable regression", key="flow_dummy_btn"):
        with st.spinner("Fitting dummy-variable model..."):
            try:
                dci_df = pd.DataFrame(api._get(f"/stats/{year}/{gp}/dci"))
                dummy = _team_dummy_regression(dci_df, drivers_df)
                if not dummy:
                    st.info("Insufficient data for team dummy regression.")
                elif dummy.get("error"):
                    st.info(dummy["error"])
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("R2", dummy["r_squared"])
                    m2.metric("F-stat", dummy["f_stat"])
                    m3.metric("Model p-value", dummy["f_p_value"])
                    st.dataframe(pd.DataFrame(dummy["coefficients"]),
                                 width="stretch",
                                 hide_index=True)
            except Exception as e:
                st.error(f"Dummy regression failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Step 5
    # ------------------------------------------------------------------
    st.subheader("5) Relate variables and stabilize model behavior")
    st.caption("Applied topics: correlation and covariance analysis, canonical correlation, ridge/lasso regularization, and nonlinear regression.")

    if st.button("Compute correlation, covariance, and canonical correlation", key="flow_corr_btn"):
        with st.spinner("Computing dependency structure..."):
            try:
                corr_payload = api._get(f"/stats/{year}/{gp}/correlation-matrix")
                cols = corr_payload.get("columns", [])
                corr_df = pd.DataFrame(corr_payload.get("correlation_matrix", {}))
                raw_df = pd.DataFrame(corr_payload.get("raw_data", []))

                if not corr_df.empty:
                    fig = px.imshow(
                        corr_df,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        title="Pearson correlation matrix",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, width="stretch")

                if not raw_df.empty and cols:
                    cov_df = raw_df[cols].cov()
                    st.caption("Covariance matrix")
                    st.dataframe(cov_df.round(4), width="stretch")

                    if {"grid_position", "position"}.issubset(set(raw_df.columns)):
                        fig_sc = px.scatter(
                            raw_df,
                            x="grid_position",
                            y="position",
                            trendline="ols",
                            title="Grid position vs finish position",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig_sc, width="stretch")

                cca_res = _canonical_correlation(raw_df)
                if cca_res:
                    m1, m2 = st.columns(2)
                    m1.metric("Canonical correlation", f"{cca_res['canonical_corr']:.4f}")
                    m2.metric("Observations", cca_res["n_obs"])
                    st.caption("X-set weights (strategy variables)")
                    st.dataframe(pd.DataFrame([
                        {"variable": k, "weight": v} for k, v in cca_res["x_weights"].items()
                    ]), width="stretch", hide_index=True)
                    st.caption("Y-set weights (outcome variables)")
                    st.dataframe(pd.DataFrame([
                        {"variable": k, "weight": v} for k, v in cca_res["y_weights"].items()
                    ]), width="stretch", hide_index=True)
                else:
                    st.info("Canonical correlation not available for this race sample.")
            except Exception as e:
                st.error(f"Correlation analysis failed: {e}")

    if st.button("Run ridge and lasso", key="flow_ridge_lasso_btn"):
        with st.spinner("Fitting regularized regressions..."):
            try:
                rl = api._get(f"/stats/{year}/{gp}/regression/regularised")
                m1, m2 = st.columns(2)
                m1.metric("Ridge R2", rl["ridge_r2"])
                m2.metric("Lasso R2", rl["lasso_r2"])
                st.caption(
                    "Lasso selected features: "
                    f"{', '.join(rl['lasso_selected_features']) if rl['lasso_selected_features'] else 'none'}"
                )

                coef_df = pd.DataFrame(rl["coefficients"])
                fig = go.Figure()
                fig.add_bar(x=coef_df["feature"], y=coef_df["ridge_coef"], name="Ridge")
                fig.add_bar(x=coef_df["feature"], y=coef_df["lasso_coef"], name="Lasso")
                fig.update_layout(
                    barmode="group",
                    title="Regularization comparison",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Ridge/Lasso failed: {e}")

    nonlinear_driver = st.selectbox("Driver for nonlinear tyre-age model",
                                    driver_ids,
                                    key="flow_nonlinear_driver")
    if st.button("Fit nonlinear tyre-age regression", key="flow_nonlinear_btn"):
        nonlinear = _nonlinear_tyre_model(laps_df, nonlinear_driver)
        if not nonlinear:
            st.info("Insufficient tyre-age observations for nonlinear model.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Linear R2", f"{nonlinear['r2_linear']:.4f}")
            m2.metric("Quadratic R2", f"{nonlinear['r2_quadratic']:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nonlinear["x_raw"],
                y=nonlinear["y_raw"],
                mode="markers",
                name="Observed",
            ))
            fig.add_trace(go.Scatter(
                x=nonlinear["x_plot"],
                y=nonlinear["y_line_plot"],
                mode="lines",
                name="Linear",
            ))
            fig.add_trace(go.Scatter(
                x=nonlinear["x_plot"],
                y=nonlinear["y_quad_plot"],
                mode="lines",
                name="Quadratic",
            ))
            fig.update_layout(
                title=f"Nonlinear model check: {nonlinear_driver}",
                xaxis_title="Tyre age (laps)",
                yaxis_title="Lap time (s)",
                template="plotly_dark",
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # ------------------------------------------------------------------
    # Step 6
    # ------------------------------------------------------------------
    st.subheader("6) Predict outcomes and evaluate decision risk")
    st.caption("Applied topics: logistic regression, binary outcomes, multiple logistic features, model comparison, and count outcomes via Poisson regression.")

    if st.button("Run logistic regression (podium probability)", key="flow_logistic_btn"):
        with st.spinner("Fitting logistic model..."):
            try:
                logit = api._get(f"/stats/{year}/{gp}/logistic")
                m1, m2, m3 = st.columns(3)
                m1.metric("AUC", logit["auc"])
                cv_auc_val = logit.get("cv_auc")
                m2.metric("CV AUC", round(cv_auc_val, 4) if cv_auc_val is not None else "n/a")
                m3.metric("Podiums in data", logit["n_podiums"])

                coef_df = pd.DataFrame(logit["coefficients"])
                fig_or = px.bar(
                    coef_df,
                    x="feature",
                    y="odds_ratio",
                    title="Odds ratio interpretation",
                    template="plotly_dark",
                    labels={"odds_ratio": "Odds ratio"},
                )
                fig_or.add_hline(y=1, line_dash="dash", line_color="white")
                st.plotly_chart(fig_or, width="stretch")

                roc = logit["roc_curve"]
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=roc["fpr"],
                    y=roc["tpr"],
                    mode="lines",
                    name="Logistic ROC",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random baseline",
                    line=dict(dash="dash"),
                ))
                fig_roc.update_layout(
                    title="ROC curve",
                    xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_roc, width="stretch")

                for row in logit["coefficients"]:
                    st.caption(row["interpretation"])
            except Exception as e:
                st.error(f"Logistic regression failed: {e}")

    if st.button("Compare classical vs ML models", key="flow_compare_btn"):
        with st.spinner("Comparing models..."):
            try:
                cmp_res = api._get(f"/stats/{year}/{gp}/model-comparison")
                rows = [cmp_res["logistic_regression"]]
                if cmp_res.get("xgboost"):
                    rows.append(cmp_res["xgboost"])
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                if cmp_res.get("driver_predictions"):
                    pred_df = pd.DataFrame(cmp_res["driver_predictions"])
                    pred_df["podium"] = pred_df["actual_podium"].map({1: "Podium", 0: "No podium"})
                    fig = px.bar(
                        pred_df,
                        x="driver_id",
                        y="lr_prob",
                        color="podium",
                        color_discrete_map={"Podium": "#00d4ff", "No podium": "#444"},
                        title="Driver-level podium probabilities",
                        template="plotly_dark",
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="white")
                    st.plotly_chart(fig, width="stretch")

                if not cmp_res.get("comparison_available"):
                    st.info("XGBoost comparison not available for this race sample.")
            except Exception as e:
                st.error(f"Model comparison failed: {e}")

    if st.button("Run count-outcome model (num_stops)", key="flow_count_btn"):
        with st.spinner("Fitting Poisson count model..."):
            try:
                corr_payload = api._get(f"/stats/{year}/{gp}/correlation-matrix")
                raw_df = pd.DataFrame(corr_payload.get("raw_data", []))
                count_model = _poisson_count_model(raw_df)

                if not count_model:
                    st.info("Insufficient data for Poisson model.")
                elif count_model.get("error"):
                    st.info(count_model["error"])
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("AIC", count_model["aic"])
                    m2.metric("Deviance", count_model["deviance"])
                    st.dataframe(pd.DataFrame(count_model["coefficients"]),
                                 width="stretch",
                                 hide_index=True)
            except Exception as e:
                st.error(f"Count-outcome modeling failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Step 7
    # ------------------------------------------------------------------
    st.subheader("7) Validate with assumption-robust nonparametric checks")
    st.caption("Applied topics: sign test, rank-sum style comparisons, and nonparametric consistency checks.")

    c1, c2 = st.columns(2)
    with c1:
        sign_a = st.selectbox("Sign test - Driver A", driver_ids, key="flow_sign_a")
    with c2:
        sign_b = st.selectbox("Sign test - Driver B",
                              [d for d in driver_ids if d != sign_a],
                              key="flow_sign_b")

    if st.button("Run sign test", key="flow_sign_btn"):
        sign_res = _sign_test(laps_df, sign_a, sign_b)
        if not sign_res:
            st.info("Insufficient paired laps for sign test.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Paired laps", sign_res["paired_laps"])
            m2.metric(f"{sign_a} faster", sign_res["faster_a"])
            m3.metric(f"{sign_b} faster", sign_res["faster_b"])
            m4.metric("p-value", f"{sign_res['p_value']:.5f}")
            st.success("Directional difference is significant.") if sign_res["significant"] else st.info("No significant directional edge.")

    mw_group = st.radio("Grouping for Mann-Whitney", ["driver_id", "team"],
                        horizontal=True, key="flow_mw_group")

    if mw_group == "team":
        if "team" in laps_df.columns:
            mw_values = sorted(laps_df["team"].dropna().astype(str).unique().tolist())
        elif not drivers_df.empty and "team" in drivers_df.columns:
            mw_values = sorted(drivers_df["team"].dropna().astype(str).unique().tolist())
        else:
            mw_values = []
    else:
        mw_values = driver_ids

    mw_a, mw_b = None, None
    if len(mw_values) < 2:
        st.info("Need at least two groups for Mann-Whitney U.")
    else:
        mw_col1, mw_col2 = st.columns(2)
        with mw_col1:
            mw_a = st.selectbox("Mann-Whitney group A", mw_values, key="flow_mw_a")
        with mw_col2:
            mw_b_choices = [d for d in mw_values if d != mw_a]
            mw_b = st.selectbox("Mann-Whitney group B", mw_b_choices, key="flow_mw_b")

    if st.button("Run Mann-Whitney U", key="flow_mw_btn"):
        if not mw_a or not mw_b:
            st.warning("Select two valid groups for Mann-Whitney U.")
        else:
            with st.spinner("Running Mann-Whitney U..."):
                try:
                    mw = api._get(
                        f"/stats/{year}/{gp}/mann-whitney",
                        params={"group_a": mw_a, "group_b": mw_b, "group_col": mw_group},
                    )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"{mw_a} median", f"{mw['median_a']:.3f}s")
                    m2.metric(f"{mw_b} median", f"{mw['median_b']:.3f}s")
                    m3.metric("U", f"{mw['u_statistic']:.3f}")
                    m4.metric("p-value", f"{mw['p_value']:.5f}")
                    st.success(mw["verdict"]) if mw["significant"] else st.info(mw["verdict"])
                except Exception as e:
                    st.error(f"Mann-Whitney failed: {e}")

    wil_driver = st.selectbox("Wilcoxon driver", driver_ids, key="flow_wil_driver")
    if st.button("Run Wilcoxon before/after SC", key="flow_wil_btn"):
        with st.spinner("Running Wilcoxon test..."):
            try:
                wil = api._get(f"/stats/{year}/{gp}/wilcoxon/{wil_driver}")
                if not wil.get("available", True):
                    st.info(wil.get("message", "No safety car window for Wilcoxon test."))
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mean before SC", f"{wil['mean_before_sc']:.3f}s")
                    m2.metric("Mean after SC", f"{wil['mean_after_sc']:.3f}s")
                    m3.metric("p-value", f"{wil['p_value']:.5f}")
                    st.success(wil["verdict"]) if wil["significant"] else st.info(wil["verdict"])
            except Exception as e:
                st.error(f"Wilcoxon failed: {e}")

    fried_default = driver_ids[:5] if len(driver_ids) >= 5 else driver_ids
    fried_drivers = st.multiselect("Drivers for Friedman test (3+)",
                                   driver_ids,
                                   default=fried_default,
                                   key="flow_friedman_drivers")
    if st.button("Run Friedman test", key="flow_friedman_btn"):
        if len(fried_drivers) < 3:
            st.warning("Select at least 3 drivers.")
        else:
            with st.spinner("Running Friedman test..."):
                try:
                    fried = api._get(
                        f"/stats/{year}/{gp}/friedman",
                        params={"drivers": ",".join(fried_drivers)},
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("chi2", f"{fried['chi2_statistic']:.3f}")
                    m2.metric("p-value", f"{fried['p_value']:.5f}")
                    m3.metric("Kendall W", f"{fried['kendalls_w']:.3f}")
                    st.success(fried["verdict"]) if fried["significant"] else st.info(fried["verdict"])
                except Exception as e:
                    st.error(f"Friedman test failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Step 8
    # ------------------------------------------------------------------
    st.subheader("8) Consolidate findings into an application summary")

    if st.button("Load synthesis summary", key="flow_summary_btn"):
        with st.spinner("Loading summary..."):
            try:
                summary = api._get(f"/stats/{year}/{gp}/summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Drivers analyzed", summary["n_drivers"])
                m2.metric("ANOVA significant", "Yes" if summary["anova_significant"] else "No")
                m3.metric(
                    "ANOVA p-value",
                    f"{summary['anova_p_value']:.5f}" if summary["anova_p_value"] else "-",
                )
                st.caption("Top consistency performers")
                st.dataframe(pd.DataFrame(summary["dci_top5"]),
                             width="stretch",
                             hide_index=True)
            except Exception as e:
                st.error(f"Summary failed: {e}")

    coverage_rows = [
        ("Statistical inference framework", "Step 1 baseline + Step 2 decisions"),
        ("Likelihood function", "Step 1 likelihood fit plot"),
        ("Maximum Likelihood Estimation (MLE)", "Step 1 driver-level distribution fitting"),
        ("Computing and comparing MLE estimates", "Step 1 metric panel by driver"),
        ("Confidence intervals", "Step 1 MLE confidence bounds"),
        ("Plotting likelihood functions", "Step 1 observed vs fitted density"),
        ("Frequentist inference", "Step 2 t and z hypothesis testing"),
        ("Bayesian theorem", "Step 1 prior-to-posterior update"),
        ("Posterior distributions", "Step 1 posterior win probability chart"),
        ("Population mean and difference testing", "Step 2 one-sample and two-sample tests"),
        ("Estimating population mean", "Step 1 sampling distribution and CI"),
        ("Large sample tests", "Step 2 z-test"),
        ("Sampling distributions", "Step 1 bootstrap simulation"),
        ("Student's t-distribution", "Step 2 rejection-region plot"),
        ("One-sample and two-sample t-tests", "Step 2 dedicated tests"),
        ("Rejection and non-rejection regions", "Step 2 critical-value visualization"),
        ("Type I and Type II errors", "Step 2 alpha/beta/power panel"),
        ("Hypothesis testing using Z-test", "Step 2 pit-stop z-test"),
        ("F-test basics", "Step 3 and Step 4 ANOVA/OLS F-statistics"),
        ("Chi-square test of independence", "Step 3 team vs compound contingency"),
        ("Chi-square goodness-of-fit", "Step 3 compound distribution check"),
        ("Appropriateness of distributions", "Step 3 goodness-of-fit + Step 4 transformations"),
        ("One-way ANOVA", "Step 3 one-way ANOVA"),
        ("Two-way ANOVA", "Step 3 two-way ANOVA"),
        ("MANOVA", "Step 3 multivariate team effect check"),
        ("Simple linear regression", "Step 4 functional form linear fit"),
        ("Least squares estimation", "Step 4 OLS model"),
        ("Residual analysis", "Step 4 residual and Q-Q plots"),
        ("Model assumptions", "Step 4 diagnostics and tests"),
        ("Multiple regression", "Step 4 OLS with multiple predictors"),
        ("Multivariate regression", "Step 4 combined outcome framing"),
        ("Dummy variable regression models", "Step 4 team dummy regression"),
        ("Autocorrelation testing", "Step 4 Durbin-Watson"),
        ("Heteroscedasticity testing", "Step 4 Breusch-Pagan"),
        ("Model specification and diagnostics", "Step 4 specification checks"),
        ("Functional form testing", "Step 4 linear vs quadratic comparison"),
        ("Transformation of variables", "Step 4 skewness and log transform check"),
        ("Correlation analysis", "Step 5 Pearson matrix"),
        ("Covariance analysis", "Step 5 covariance matrix"),
        ("Canonical correlation analysis", "Step 5 canonical-correlation panel"),
        ("Ridge regression", "Step 5 regularization"),
        ("Lasso regression", "Step 5 regularization and feature selection"),
        ("Nonlinear regression models", "Step 5 tyre-age nonlinear fit"),
        ("Logistic regression model", "Step 6 logistic model"),
        ("Binary outcome modeling", "Step 6 podium probability"),
        ("Count outcomes", "Step 6 Poisson model for num_stops"),
        ("Multiple logistic regression", "Step 6 multi-feature logistic model"),
        ("Sign test", "Step 7 sign test"),
        ("Rank sum test", "Step 7 Mann-Whitney U"),
    ]

    st.caption("Topic-to-application mapping in the current workflow")
    st.dataframe(
        pd.DataFrame(coverage_rows, columns=["Topic", "Where it is applied"]),
        width="stretch",
        hide_index=True,
    )
