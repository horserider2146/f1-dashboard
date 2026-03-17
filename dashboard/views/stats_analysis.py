"""
Page: Statistical Analysis — 10 interactive panels covering all 7 syllabus units.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard import api_client as api


def render(year: int, gp: str):
    st.header("📊 Statistical Analysis")
    st.caption(f"Race: **{gp} {year}** | All tests use FastF1 telemetry data")

    # Quick availability check
    try:
        api._get(f"/stats/{year}/{gp}/dci")
    except Exception as e:
        msg = str(e)
        if "404" in msg or "not available" in msg.lower() or "not cached" in msg.lower():
            st.warning(
                f"Race data not available for **{gp} {year}**. "
                f"Only cached races work offline: **2025 Australian, Chinese, Bahrain, Miami**. "
                f"Select one of those from the sidebar."
            )
        else:
            st.error(f"Could not load race data: {e}")
        return

    tab_labels = [
        "🏅 DCI",
        "📈 MLE & Bayes",
        "🔬 t-test / z-test",
        "📊 ANOVA",
        "📉 Regression",
        "🔗 Correlation",
        "⚖️ Logistic Reg",
        "🤖 Model Compare",
        "🎲 Nonparametric",
        "📋 Summary",
    ]
    tabs = st.tabs(tab_labels)

    # ── Load drivers list (shared across panels) ──────────────────────────────
    try:
        drivers_data = api.get_drivers(year, gp)
        driver_ids = [d["driver_id"] for d in drivers_data] if drivers_data else []
    except Exception:
        driver_ids = []

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 0: Driver Consistency Index
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Driver Consistency Index (DCI)")
        st.markdown(
            "**DCI = 1 / Var(lap\\_time\\_s)**, normalised 0–1. "
            "Higher → more consistent. Outlier laps (pit in/out, SC) are removed."
        )
        if st.button("Compute DCI", key="dci_btn"):
            with st.spinner("Computing DCI…"):
                try:
                    data = api._get(f"/stats/{year}/{gp}/dci")
                    if data:
                        df = pd.DataFrame(data)
                        fig = px.bar(
                            df, x="driver_id", y="dci_normalised",
                            color="dci_normalised",
                            color_continuous_scale="RdYlGn",
                            title="Driver Consistency Index (normalised)",
                            labels={"driver_id": "Driver", "dci_normalised": "DCI (0–1)"},
                            template="plotly_dark",
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(
                            df[["rank", "driver_id", "dci_normalised", "mean_lap_s",
                                "std_lap_s", "n_laps"]].rename(columns={
                                "rank": "Rank", "driver_id": "Driver",
                                "dci_normalised": "DCI", "mean_lap_s": "Mean Lap (s)",
                                "std_lap_s": "Std Dev (s)", "n_laps": "Laps",
                            }),
                            use_container_width=True, hide_index=True,
                        )
                except Exception as e:
                    st.error(f"DCI failed: {e}")

        st.divider()
        st.subheader("DCI vs Championship Points Correlation")
        if st.button("Run Correlation", key="dci_corr_btn"):
            with st.spinner("Computing correlation…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/dci/correlation")
                    if result:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Pearson r", result.get("r", "—"))
                        col2.metric("p-value", result.get("p_value", "—"))
                        col3.metric("Significant", "✅ Yes" if result.get("significant") else "❌ No")
                        st.info(result.get("interpretation", ""))
                    else:
                        st.info("No standings data available for this season.")
                except Exception as e:
                    st.error(f"Correlation failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: MLE + Bayesian
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("MLE Lap-Time Distribution (Unit 1)")
        driver_sel = st.selectbox("Select driver", driver_ids, key="mle_driver")
        if st.button("Fit Distribution", key="mle_btn") and driver_sel:
            with st.spinner("Fitting Normal distribution via MLE…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/mle/{driver_sel}")
                    if result:
                        fig = go.Figure()
                        h = result["histogram"]
                        fig.add_trace(go.Bar(
                            x=h["bin_centres"], y=h["counts"],
                            name="Observed", marker_color="steelblue", opacity=0.7,
                        ))
                        fc = result["fit_curve"]
                        fig.add_trace(go.Scatter(
                            x=fc["x"], y=fc["y"],
                            name=f"Normal fit (μ={result['mu_hat']:.3f}s)",
                            line=dict(color="#E8002D", width=2.5),
                        ))
                        # CI bands
                        fig.add_vline(x=result["ci_95_low"], line_dash="dash",
                                      line_color="orange", annotation_text="95% CI low")
                        fig.add_vline(x=result["ci_95_high"], line_dash="dash",
                                      line_color="orange", annotation_text="95% CI high")
                        fig.update_layout(
                            title=f"MLE Lap-Time Distribution — {driver_sel}",
                            xaxis_title="Lap Time (s)", yaxis_title="Density",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("μ̂ (MLE mean)", f"{result['mu_hat']:.3f}s")
                        col2.metric("σ̂ (MLE std)", f"{result['sigma_hat']:.3f}s")
                        col3.metric("95% CI low", f"{result['ci_95_low']:.3f}s")
                        col4.metric("95% CI high", f"{result['ci_95_high']:.3f}s")
                        st.info(result.get("interpretation", ""))
                except Exception as e:
                    st.error(f"MLE failed: {e}")

        st.divider()
        st.subheader("Bayesian Win Probability by Grid Position (Unit 1)")
        if st.button("Compute P(Win|Grid)", key="bayes_btn"):
            with st.spinner("Computing posterior…"):
                try:
                    data = api._get(f"/stats/{year}/{gp}/bayes-win")
                    if data:
                        df = pd.DataFrame(data)
                        fig = px.bar(
                            df, x="grid_pos", y="posterior",
                            title="P(Win | Grid Position) — Bayesian Posterior",
                            labels={"grid_pos": "Grid Position", "posterior": "P(Win)"},
                            color="posterior", color_continuous_scale="Reds",
                            template="plotly_dark",
                        )
                        fig.add_scatter(x=df["grid_pos"], y=df["prior"],
                                        name="Prior (uniform)", mode="lines",
                                        line=dict(dash="dash", color="white"))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Historical results data not available for Bayesian update.")
                except Exception as e:
                    st.error(f"Bayesian analysis failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: t-test / z-test
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Head-to-Head t-Test (Unit 2)")
        col1, col2 = st.columns(2)
        with col1:
            drv_a = st.selectbox("Driver A", driver_ids, key="ttest_a")
        with col2:
            options_b = [d for d in driver_ids if d != drv_a]
            drv_b = st.selectbox("Driver B", options_b, key="ttest_b")

        if st.button("Run t-Test", key="ttest_btn"):
            with st.spinner("Running Welch's t-test…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/ttest",
                                      params={"driver_a": drv_a, "driver_b": drv_b})
                    if result:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(f"{drv_a} mean", f"{result['mean_a']:.3f}s")
                        col2.metric(f"{drv_b} mean", f"{result['mean_b']:.3f}s")
                        col3.metric("t-statistic", f"{result['t_statistic']:.3f}")
                        col4.metric("p-value", f"{result['p_value']:.4f}")

                        col5, col6 = st.columns(2)
                        col5.metric("Cohen's d", f"{result['cohens_d']:.3f}")
                        col6.metric("Effect size", result["effect_size"])

                        if result["significant"]:
                            st.success(result["verdict"])
                        else:
                            st.info(result["verdict"])

                        # Violin plot
                        plot_df = pd.DataFrame({
                            "Lap Time (s)": result["laps_a"] + result["laps_b"],
                            "Driver": [drv_a] * len(result["laps_a"]) + [drv_b] * len(result["laps_b"]),
                        })
                        fig = px.violin(plot_df, x="Driver", y="Lap Time (s)",
                                        box=True, points="all",
                                        title=f"Lap Time Distribution: {drv_a} vs {drv_b}",
                                        template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"t-test failed: {e}")

        st.divider()
        st.subheader("Pit Stop Z-Test (Unit 2)")
        drv_z = st.selectbox("Driver", driver_ids, key="ztest_drv")
        if st.button("Run Z-Test", key="ztest_btn"):
            with st.spinner("Running z-test…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/ztest/{drv_z}")
                    if not result.get("available", True):
                        st.info(result.get("message", "Pit duration data not available."))
                    elif result:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Driver mean pit", f"{result['driver_mean']:.3f}s")
                        col2.metric("Field mean pit", f"{result['field_mean']:.3f}s")
                        col3.metric("z-statistic", f"{result['z_statistic']:.3f}")
                        if result["significant"]:
                            st.success(result["verdict"])
                        else:
                            st.info(result["verdict"])
                except Exception as e:
                    st.error(f"Z-test failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: ANOVA
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("One-Way ANOVA — Team Lap Times (Unit 3)")
        group_by = st.radio("Group by", ["team", "driver_id"], horizontal=True, key="anova_group")
        if st.button("Run One-Way ANOVA", key="anova1_btn"):
            with st.spinner("Running ANOVA…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/anova/one-way",
                                      params={"group": group_by})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("F-statistic", f"{result['f_statistic']:.4f}")
                    col2.metric("p-value", f"{result['p_value']:.4f}")
                    col3.metric("η² (effect size)", f"{result['eta_squared']:.4f}")

                    if result["significant"]:
                        st.success(result["verdict"])
                    else:
                        st.info(result["verdict"])

                    # Box plot
                    laps_data = api.get_laps(year, gp)
                    if laps_data and group_by in pd.DataFrame(laps_data).columns:
                        df_plot = pd.DataFrame(laps_data).dropna(subset=["lap_time_s"])
                        fig = px.box(df_plot, x=group_by, y="lap_time_s",
                                     title=f"Lap Time Distribution by {group_by}",
                                     template="plotly_dark",
                                     labels={group_by: group_by.title(),
                                             "lap_time_s": "Lap Time (s)"})
                        st.plotly_chart(fig, use_container_width=True)

                    if "tukey_hsd" in result and result["tukey_hsd"]:
                        st.subheader("Tukey HSD Post-Hoc")
                        tukey_df = pd.DataFrame(result["tukey_hsd"])
                        st.dataframe(tukey_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"ANOVA failed: {e}")

        st.divider()
        st.subheader("Two-Way ANOVA — Team × Compound (Unit 3)")
        if st.button("Run Two-Way ANOVA", key="anova2_btn"):
            with st.spinner("Running two-way ANOVA…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/anova/two-way")
                    if "anova_table" in result:
                        st.dataframe(pd.DataFrame(result["anova_table"]),
                                     use_container_width=True, hide_index=True)
                        if result.get("interaction_plot_data"):
                            idf = pd.DataFrame(result["interaction_plot_data"])
                            fig = px.line(idf, x="team", y="mean_lap_s",
                                          color="compound",
                                          title="Interaction Plot: Team × Compound",
                                          template="plotly_dark",
                                          labels={"team": "Team",
                                                  "mean_lap_s": "Mean Lap Time (s)"})
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(result.get("error", "Two-way ANOVA failed."))
                except Exception as e:
                    st.error(f"Two-way ANOVA failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: OLS + Ridge/Lasso Regression
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("OLS Multiple Regression (Units 4–5)")
        st.caption("Predicts: `final_position ~ grid_pos + avg_pit_time + num_stops + avg_lap_time`")
        if st.button("Run OLS Regression", key="ols_btn"):
            with st.spinner("Fitting OLS model…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/regression/ols")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R²", result["r_squared"])
                    col2.metric("Adj R²", result["adj_r_squared"])
                    col3.metric("F-statistic", result["f_statistic"])
                    col4.metric("Durbin-Watson", result["durbin_watson"])

                    st.subheader("Coefficient Table")
                    coef_df = pd.DataFrame(result["coefficients"])
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

                    if result.get("vif"):
                        st.subheader("VIF (Variance Inflation Factors)")
                        st.dataframe(pd.DataFrame(result["vif"]),
                                     use_container_width=True, hide_index=True)

                    col_resid, col_qq = st.columns(2)
                    with col_resid:
                        fig_r = go.Figure(go.Scatter(
                            x=result["fitted"], y=result["residuals"],
                            mode="markers", marker=dict(color="steelblue", size=8),
                        ))
                        fig_r.add_hline(y=0, line_dash="dash", line_color="white")
                        fig_r.update_layout(title="Residuals vs Fitted",
                                            xaxis_title="Fitted", yaxis_title="Residual",
                                            template="plotly_dark")
                        st.plotly_chart(fig_r, use_container_width=True)

                    with col_qq:
                        qq = result["qq_plot"]
                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(x=qq["theoretical"], y=qq["sample"],
                                                     mode="markers", name="Sample quantiles"))
                        mn, mx = min(qq["theoretical"]), max(qq["theoretical"])
                        fig_qq.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                                     mode="lines", name="Normal line",
                                                     line=dict(dash="dash", color="red")))
                        fig_qq.update_layout(title="Q-Q Plot",
                                             xaxis_title="Theoretical Quantiles",
                                             yaxis_title="Sample Quantiles",
                                             template="plotly_dark")
                        st.plotly_chart(fig_qq, use_container_width=True)

                    if result.get("breusch_pagan"):
                        bp = result["breusch_pagan"]
                        st.caption(
                            f"Breusch-Pagan test: statistic={bp['statistic']:.3f}, "
                            f"p={bp['p_value']:.4f} — "
                            f"{'Heteroskedastic' if bp['heteroskedastic'] else 'Homoskedastic'}"
                        )
                except Exception as e:
                    st.error(f"OLS regression failed: {e}")

        st.divider()
        st.subheader("Ridge & Lasso Regression (Unit 5)")
        if st.button("Run Ridge/Lasso", key="rl_btn"):
            with st.spinner("Fitting Ridge and Lasso…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/regression/regularised")
                    col1, col2 = st.columns(2)
                    col1.metric("Ridge R²", result["ridge_r2"])
                    col2.metric("Lasso R²", result["lasso_r2"])
                    st.caption(f"Lasso selected features: **{', '.join(result['lasso_selected_features']) or 'none'}**")
                    coef_df = pd.DataFrame(result["coefficients"])
                    fig = go.Figure()
                    fig.add_bar(x=coef_df["feature"], y=coef_df["ridge_coef"],
                                name="Ridge", marker_color="steelblue")
                    fig.add_bar(x=coef_df["feature"], y=coef_df["lasso_coef"],
                                name="Lasso", marker_color="#E8002D")
                    fig.update_layout(barmode="group",
                                      title="Ridge vs Lasso Coefficients (standardised)",
                                      template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Ridge/Lasso failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Correlation Matrix
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Pearson Correlation Matrix (Unit 5)")
        if st.button("Compute Correlation Matrix", key="corr_btn"):
            with st.spinner("Computing correlations…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/correlation-matrix")
                    cols = result["columns"]
                    corr = pd.DataFrame(result["correlation_matrix"])
                    fig = px.imshow(
                        corr, text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title="Pearson Correlation Matrix",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Scatter Plot for Selected Pair")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X variable", cols, key="corr_x")
                    with col2:
                        y_var = st.selectbox("Y variable", [c for c in cols if c != x_var],
                                             key="corr_y")
                    if result.get("raw_data"):
                        raw_df = pd.DataFrame(result["raw_data"])
                        fig2 = px.scatter(raw_df, x=x_var, y=y_var,
                                          trendline="ols",
                                          title=f"{x_var} vs {y_var}",
                                          template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Correlation matrix failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 6: Logistic Regression
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Logistic Regression — P(Podium) (Unit 6)")
        if st.button("Run Logistic Regression", key="logit_btn"):
            with st.spinner("Fitting logistic regression…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/logistic")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("AUC", result["auc"])
                    col2.metric("CV AUC", result["cv_auc"])
                    col3.metric("Podiums in data", result["n_podiums"])

                    st.subheader("Coefficients & Odds Ratios")
                    coef_df = pd.DataFrame(result["coefficients"])
                    fig_or = px.bar(coef_df, x="feature", y="odds_ratio",
                                    title="Odds Ratios (>1 increases podium probability)",
                                    template="plotly_dark",
                                    labels={"odds_ratio": "Odds Ratio"})
                    fig_or.add_hline(y=1, line_dash="dash", line_color="white")
                    st.plotly_chart(fig_or, use_container_width=True)
                    for row in result["coefficients"]:
                        st.caption(row["interpretation"])

                    st.subheader("ROC Curve")
                    roc = result["roc_curve"]
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=roc["fpr"], y=roc["tpr"],
                        name=f"Logistic (AUC={result['auc']:.3f})",
                        line=dict(color="#E8002D", width=2),
                    ))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                  name="Random", line=dict(dash="dash")))
                    fig_roc.update_layout(
                        title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.error(f"Logistic regression failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 7: Model Comparison (Classical vs ML)
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Classical Statistics vs Machine Learning (Unit 6)")
        st.markdown(
            "Same data. Three modelling approaches: **OLS** (interpretable coefficients), "
            "**Logistic Regression** (probabilistic), **XGBoost** (non-linear). "
            "This comparison is the project's strongest academic differentiator."
        )
        if st.button("Run Model Comparison", key="compare_btn"):
            with st.spinner("Running both models…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/model-comparison")
                    lr = result["logistic_regression"]
                    xgb = result.get("xgboost")

                    metrics = [lr]
                    if xgb:
                        metrics.append(xgb)

                    mdf = pd.DataFrame(metrics)
                    st.dataframe(mdf, use_container_width=True, hide_index=True)

                    if result.get("driver_predictions"):
                        pred_df = pd.DataFrame(result["driver_predictions"])
                        fig = px.bar(pred_df, x="driver_id", y="lr_prob",
                                     color="actual_podium",
                                     title="Logistic Regression P(Podium) per Driver",
                                     labels={"driver_id": "Driver",
                                             "lr_prob": "P(Podium)",
                                             "actual_podium": "Actual Podium"},
                                     template="plotly_dark")
                        fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                                      annotation_text="Decision boundary")
                        st.plotly_chart(fig, use_container_width=True)

                    if not result.get("comparison_available"):
                        st.info("XGBoost comparison unavailable — model may need training data.")
                except Exception as e:
                    st.error(f"Model comparison failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 8: Nonparametric Tests
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("Nonparametric Tests (Unit 7)")

        st.markdown("#### Wilcoxon Signed-Rank — Before vs After Safety Car")
        drv_wil = st.selectbox("Driver", driver_ids, key="wil_drv")
        if st.button("Run Wilcoxon Test", key="wil_btn"):
            with st.spinner("Running Wilcoxon test…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/wilcoxon/{drv_wil}")
                    if not result.get("available", True):
                        st.info(result.get("message", "No safety car laps in this race."))
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mean before SC", f"{result['mean_before_sc']:.3f}s")
                        col2.metric("Mean after SC", f"{result['mean_after_sc']:.3f}s")
                        col3.metric("p-value", f"{result['p_value']:.4f}")
                        if result["significant"]:
                            st.success(result["verdict"])
                        else:
                            st.info(result["verdict"])

                        plot_df = pd.DataFrame({
                            "Lap Time (s)": result["laps_before"] + result["laps_after"],
                            "Period": (["Before SC"] * len(result["laps_before"]) +
                                       ["After SC"] * len(result["laps_after"])),
                        })
                        fig = px.box(plot_df, x="Period", y="Lap Time (s)",
                                     points="all", title=f"Wilcoxon: {drv_wil} Before/After SC",
                                     template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Wilcoxon test failed: {e}")

        st.divider()
        st.markdown("#### Mann-Whitney U — Compare Two Groups")
        col1, col2 = st.columns(2)
        with col1:
            mw_a = st.selectbox("Group A", driver_ids, key="mw_a")
        with col2:
            mw_b = st.selectbox("Group B", [d for d in driver_ids if d != mw_a], key="mw_b")
        mw_col = st.radio("Group by", ["driver_id", "team"], horizontal=True, key="mw_col")
        if st.button("Run Mann-Whitney U", key="mw_btn"):
            with st.spinner("Running Mann-Whitney U…"):
                try:
                    result = api._get(
                        f"/stats/{year}/{gp}/mann-whitney",
                        params={"group_a": mw_a, "group_b": mw_b, "group_col": mw_col},
                    )
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{mw_a} median", f"{result['median_a']:.3f}s")
                    col2.metric(f"{mw_b} median", f"{result['median_b']:.3f}s")
                    col3.metric("p-value", f"{result['p_value']:.4f}")
                    if result["significant"]:
                        st.success(result["verdict"])
                    else:
                        st.info(result["verdict"])
                    plot_df = pd.DataFrame({
                        "Lap Time (s)": result["laps_a"] + result["laps_b"],
                        "Group": ([mw_a] * len(result["laps_a"]) +
                                  [mw_b] * len(result["laps_b"])),
                    })
                    fig = px.violin(plot_df, x="Group", y="Lap Time (s)", box=True, points="all",
                                    template="plotly_dark",
                                    title=f"Mann-Whitney: {mw_a} vs {mw_b}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Mann-Whitney failed: {e}")

        st.divider()
        st.markdown("#### Friedman Test — Driver Ranking Consistency")
        friedman_drivers = st.multiselect(
            "Select 3+ drivers", driver_ids, default=driver_ids[:5], key="friedman_drvs"
        )
        if st.button("Run Friedman Test", key="friedman_btn"):
            with st.spinner("Running Friedman test…"):
                try:
                    result = api._get(
                        f"/stats/{year}/{gp}/friedman",
                        params={"drivers": ",".join(friedman_drivers)},
                    )
                    col1, col2, col3 = st.columns(3)
                    col1.metric("χ² statistic", f"{result['chi2_statistic']:.4f}")
                    col2.metric("p-value", f"{result['p_value']:.4f}")
                    col3.metric("Kendall's W", f"{result['kendalls_w']:.4f}")
                    if result["significant"]:
                        st.success(result["verdict"])
                    else:
                        st.info(result["verdict"])
                    medians = result.get("driver_medians", {})
                    if medians:
                        med_df = pd.DataFrame(list(medians.items()),
                                              columns=["Driver", "Median Lap (s)"])
                        fig = px.bar(med_df, x="Driver", y="Median Lap (s)",
                                     title="Driver Median Lap Times",
                                     template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Friedman test failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 9: Summary
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("Statistical Summary")
        if st.button("Load Summary", key="summary_btn"):
            with st.spinner("Loading summary…"):
                try:
                    result = api._get(f"/stats/{year}/{gp}/summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Drivers analysed", result["n_drivers"])
                    col2.metric("ANOVA significant",
                                "✅ Yes" if result["anova_significant"] else "❌ No")
                    col3.metric("ANOVA p-value",
                                f"{result['anova_p_value']:.4f}" if result["anova_p_value"] else "—")

                    st.subheader("Top 5 Most Consistent Drivers")
                    top5 = pd.DataFrame(result["dci_top5"])
                    st.dataframe(top5, use_container_width=True, hide_index=True)

                    st.markdown("""
                    ---
                    ### Syllabus Coverage
                    | Unit | Topic | Status |
                    |------|-------|--------|
                    | 1 | MLE + CI + Bayesian | ✅ |
                    | 2 | Hypothesis testing (t, z) | ✅ |
                    | 3 | ANOVA (one-way + two-way) | ✅ |
                    | 4–5 | Regression + correlation | ✅ |
                    | 6 | Logistic regression | ✅ |
                    | 7 | Nonparametric tests | ✅ |
                    | Novel | Driver Consistency Index | ✅ |
                    """)
                except Exception as e:
                    st.error(f"Summary failed: {e}")
