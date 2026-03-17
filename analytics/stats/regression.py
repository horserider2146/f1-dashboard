"""
Units 4–5 — Regression analysis.
OLS multiple regression, Ridge, Lasso, VIF, residual diagnostics,
Breusch-Pagan, Durbin-Watson, Pearson correlation matrix.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp


def _build_regression_df(laps: list[dict],
                          pit_stops: list[dict],
                          results: list[dict]) -> pd.DataFrame:
    """
    Merge lap summary, pit stop summary, and final results into
    one row per driver for regression modelling.
    """
    lap_df = pd.DataFrame(laps)
    if lap_df.empty:
        return pd.DataFrame()

    # Aggregate per driver
    lap_summary = (lap_df.groupby("driver_id")["lap_time_s"]
                   .agg(avg_lap_time="mean", std_lap_time="std")
                   .reset_index())

    if pit_stops:
        pit_df = pd.DataFrame(pit_stops)
        if "pit_duration" in pit_df.columns:
            pit_summary = (pit_df.groupby("driver_id")
                           .agg(num_stops=("pit_duration", "count"),
                                avg_pit_time=("pit_duration", "mean"))
                           .reset_index())
        else:
            pit_summary = (pit_df.groupby("driver_id")
                           .size().reset_index(name="num_stops"))
            pit_summary["avg_pit_time"] = np.nan
    else:
        pit_summary = pd.DataFrame(columns=["driver_id", "num_stops", "avg_pit_time"])

    df = lap_summary.merge(pit_summary, on="driver_id", how="left")

    if results:
        res_df = pd.DataFrame(results)
        keep = [c for c in ["driver_id", "position", "grid_position", "team", "compound"]
                if c in res_df.columns]
        df = df.merge(res_df[keep].drop_duplicates("driver_id"), on="driver_id", how="left")

    df = df.dropna(subset=["avg_lap_time"])
    return df


def ols_regression(laps: list[dict],
                   pit_stops: list[dict],
                   results: list[dict]) -> dict:
    """
    OLS: final_position ~ grid_pos + avg_pit_time + num_stops + avg_lap_time
    Returns coefficients, p-values, R², residuals, VIF, diagnostic tests.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.stattools import durbin_watson
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError:
        return {"error": "statsmodels required for OLS regression."}

    df = _build_regression_df(laps, pit_stops, results)
    if df.empty or "position" not in df.columns:
        return {"error": "Insufficient data for regression."}

    feature_cols = [c for c in ["grid_position", "avg_pit_time", "num_stops", "avg_lap_time"]
                    if c in df.columns]
    # Drop features with >50% missing to avoid shrinking the sample
    df_model = df[["position"] + feature_cols].copy()
    for col in feature_cols[:]:
        if df_model[col].isna().mean() > 0.5:
            feature_cols.remove(col)
    df_model = df_model[["position"] + feature_cols].dropna()

    if len(df_model) < max(3, len(feature_cols) + 1):
        return {"error": f"Not enough observations for regression (need {len(feature_cols)+1}, got {len(df_model)})."}

    X = sm.add_constant(df_model[feature_cols].astype(float))
    y = df_model["position"].astype(float)

    model = sm.OLS(y, X).fit()

    # Coefficients table
    coef_table = []
    for feat in X.columns:
        coef_table.append({
            "feature": feat,
            "coefficient": round(float(model.params[feat]), 4),
            "std_error": round(float(model.bse[feat]), 4),
            "t_stat": round(float(model.tvalues[feat]), 4),
            "p_value": round(float(model.pvalues[feat]), 6),
            "significant": bool(model.pvalues[feat] < 0.05),
        })

    # VIF
    vif_data = []
    if len(feature_cols) > 1:
        X_vif = df_model[feature_cols].astype(float)
        for i, col in enumerate(X_vif.columns):
            vif_data.append({
                "feature": col,
                "vif": round(float(variance_inflation_factor(X_vif.values, i)), 3),
            })

    residuals = model.resid.tolist()
    fitted = model.fittedvalues.tolist()

    # Durbin-Watson
    dw = float(durbin_watson(residuals))

    # Breusch-Pagan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        bp = {"statistic": round(float(bp_stat), 4), "p_value": round(float(bp_p), 4),
              "heteroskedastic": bool(bp_p < 0.05)}
    except Exception:
        bp = {}

    # Q-Q plot data
    qq = sp.probplot(residuals, dist="norm")
    qq_data = {"theoretical": list(qq[0][0]), "sample": list(qq[0][1])}

    return {
        "model": "OLS Multiple Regression",
        "n_obs": len(df_model),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic": round(float(model.fvalue), 4),
        "f_p_value": round(float(model.f_pvalue), 6),
        "aic": round(float(model.aic), 2),
        "coefficients": coef_table,
        "vif": vif_data,
        "residuals": [round(r, 4) for r in residuals],
        "fitted": [round(f, 4) for f in fitted],
        "durbin_watson": round(dw, 4),
        "breusch_pagan": bp,
        "qq_plot": qq_data,
    }


def lasso_ridge_regression(laps: list[dict],
                            pit_stops: list[dict],
                            results: list[dict]) -> dict:
    """
    Ridge and Lasso regression — same features as OLS.
    Returns selected features (Lasso) and coefficient comparison table.
    """
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import numpy as np

    df = _build_regression_df(laps, pit_stops, results)
    if df.empty or "position" not in df.columns:
        return {"error": "Insufficient data."}

    feature_cols = [c for c in ["grid_position", "avg_pit_time", "num_stops", "avg_lap_time"]
                    if c in df.columns]
    df_model = df[["position"] + feature_cols].copy()
    for col in feature_cols[:]:
        if df_model[col].isna().mean() > 0.5:
            feature_cols.remove(col)
    df_model = df_model[["position"] + feature_cols].dropna()
    if len(df_model) < max(3, len(feature_cols) + 1):
        return {"error": "Not enough data."}

    X = df_model[feature_cols].astype(float).values
    y = df_model["position"].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1, max_iter=10000)
    ridge.fit(X_scaled, y)
    lasso.fit(X_scaled, y)

    comparison = []
    for i, feat in enumerate(feature_cols):
        comparison.append({
            "feature": feat,
            "ridge_coef": round(float(ridge.coef_[i]), 4),
            "lasso_coef": round(float(lasso.coef_[i]), 4),
            "lasso_selected": lasso.coef_[i] != 0,
        })

    return {
        "model": "Ridge / Lasso Comparison",
        "n_obs": len(df_model),
        "ridge_r2": round(float(ridge.score(X_scaled, y)), 4),
        "lasso_r2": round(float(lasso.score(X_scaled, y)), 4),
        "lasso_selected_features": [c["feature"] for c in comparison if c["lasso_selected"]],
        "coefficients": comparison,
    }


def correlation_matrix(laps: list[dict],
                        pit_stops: list[dict],
                        results: list[dict]) -> dict:
    """
    Pearson correlation matrix across key race variables.
    Returns matrix values + scatter pairs for selected cell clicks.
    """
    df = _build_regression_df(laps, pit_stops, results)
    if df.empty:
        return {}

    numeric_cols = [c for c in
                    ["avg_lap_time", "std_lap_time", "avg_pit_time",
                     "num_stops", "grid_position", "position"]
                    if c in df.columns]
    # Drop columns with >50% missing before requiring complete rows
    df_num = df[numeric_cols].copy()
    numeric_cols = [c for c in numeric_cols if df_num[c].isna().mean() <= 0.5]
    df_num = df_num[numeric_cols].dropna()
    if len(df_num) < 3:
        return {"error": "Not enough data."}

    corr = df_num.corr(method="pearson")
    p_matrix = pd.DataFrame(np.ones((len(numeric_cols), len(numeric_cols))),
                             index=numeric_cols, columns=numeric_cols)
    for i, c1 in enumerate(numeric_cols):
        for j, c2 in enumerate(numeric_cols):
            if i != j:
                _, p = sp.pearsonr(df_num[c1], df_num[c2])
                p_matrix.loc[c1, c2] = p

    return {
        "columns": numeric_cols,
        "correlation_matrix": corr.round(4).to_dict(),
        "p_value_matrix": p_matrix.round(4).to_dict(),
        "raw_data": df_num.round(4).to_dict(orient="records"),
    }
