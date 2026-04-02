"""
Unit 6 — Logistic Regression.
P(podium) from grid position, pit strategy, team, compound.
Odds ratios, ROC/AUC curve. Compared against XGBoost predictor.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp


def _build_model_df(laps: list[dict],
                    pit_stops: list[dict],
                    results: list[dict]) -> pd.DataFrame:
    lap_df = pd.DataFrame(laps)
    if lap_df.empty:
        return pd.DataFrame()

    lap_summary = (lap_df.groupby("driver_id")["lap_time_s"]
                   .agg(avg_lap_time="mean").reset_index())

    if pit_stops:
        pit_df = pd.DataFrame(pit_stops)
        if "pit_duration" in pit_df.columns:
            pit_s = (pit_df.groupby("driver_id")
                     .agg(num_stops=("pit_duration", "count"),
                          avg_pit_time=("pit_duration", "mean"))
                     .reset_index())
        else:
            pit_s = pit_df.groupby("driver_id").size().reset_index(name="num_stops")
            pit_s["avg_pit_time"] = np.nan
        lap_summary = lap_summary.merge(pit_s, on="driver_id", how="left")

    if results:
        res_df = pd.DataFrame(results)
        keep = [c for c in ["driver_id", "position", "grid_position"] if c in res_df.columns]
        lap_summary = lap_summary.merge(
            res_df[keep].drop_duplicates("driver_id"), on="driver_id", how="left"
        )

    lap_summary = lap_summary.dropna(subset=["avg_lap_time"])
    if "position" in lap_summary.columns:
        lap_summary["podium"] = (lap_summary["position"] <= 3).astype(int)

    return lap_summary


def logistic_regression(laps: list[dict],
                         pit_stops: list[dict],
                         results: list[dict]) -> dict:
    """
    Logistic regression: P(podium) ~ grid_position + num_stops + avg_lap_time.
    Returns coefficients, odds ratios, ROC/AUC data.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.model_selection import cross_val_score

    df = _build_model_df(laps, pit_stops, results)
    if df.empty or "podium" not in df.columns:
        return {"error": "Insufficient data for logistic regression."}

    feature_cols = [c for c in ["grid_position", "num_stops", "avg_lap_time", "avg_pit_time"]
                    if c in df.columns]
    # Drop columns with >50% missing so we keep all drivers
    df_model = df[["driver_id", "podium"] + feature_cols].copy()
    for col in feature_cols[:]:
        if df_model[col].isna().mean() > 0.5:
            feature_cols.remove(col)
    df_model = df_model[["driver_id", "podium"] + feature_cols].dropna()

    if len(df_model) < 4 or df_model["podium"].nunique() < 2:
        return {"error": "Not enough data or only one class present."}

    X = df_model[feature_cols].astype(float).values
    y = df_model["podium"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_scaled, y)

    y_prob = clf.predict_proba(X_scaled)[:, 1]
    auc = float(roc_auc_score(y, y_prob))

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)

    # Odds ratios (exp of coefficients on original scale)
    # Unscale: coef_original = coef_scaled / std
    coef_original = clf.coef_[0] / scaler.scale_
    odds_ratios = np.exp(coef_original)

    coef_table = []
    for i, feat in enumerate(feature_cols):
        coef_table.append({
            "feature": feat,
            "coefficient": round(float(coef_original[i]), 4),
            "odds_ratio": round(float(odds_ratios[i]), 4),
            "interpretation": (
                f"OR={odds_ratios[i]:.3f}: each 1-unit increase in {feat} "
                f"{'increases' if odds_ratios[i] > 1 else 'decreases'} podium odds "
                f"by {abs(odds_ratios[i]-1)*100:.1f}%"
            ),
        })

    # CV score
    try:
        cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(df_model)),
                                     scoring="roc_auc")
        cv_auc = float(cv_scores.mean())
        if np.isnan(cv_auc):
            cv_auc = auc
    except Exception:
        cv_auc = auc

    return {
        "model": "Logistic Regression",
        "n_obs": len(df_model),
        "n_podiums": int(y.sum()),
        "auc": round(auc, 4),
        "cv_auc": round(cv_auc, 4),
        "coefficients": coef_table,
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
        },
        "predictions": [
            {"driver_id": row["driver_id"],
             "actual_podium": int(row["podium"]),
             "prob_podium": round(float(y_prob[i]), 4)}
            for i, (_, row) in enumerate(df_model.iterrows())
        ],
    }


def compare_models(laps: list[dict],
                   pit_stops: list[dict],
                   results: list[dict],
                   xgb_predictions: list[dict] | None = None) -> dict:
    """
    Side-by-side comparison: Logistic Regression vs XGBoost predictor.
    Returns accuracy, AUC, RMSE for both models on the same data.
    """
    from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = _build_model_df(laps, pit_stops, results)
    if df.empty or "podium" not in df.columns:
        return {"error": "Insufficient data for model comparison."}

    feature_cols = [c for c in ["grid_position", "num_stops", "avg_lap_time"]
                    if c in df.columns]
    df_model = df[["driver_id", "podium"] + feature_cols].dropna()

    if len(df_model) < 4 or df_model["podium"].nunique() < 2:
        return {"error": "Insufficient data."}

    X = df_model[feature_cols].astype(float).values
    y = df_model["podium"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_scaled, y)
    lr_prob = lr.predict_proba(X_scaled)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    lr_metrics = {
        "model": "Logistic Regression",
        "accuracy": round(float(accuracy_score(y, lr_pred)), 4),
        "auc": round(float(roc_auc_score(y, lr_prob)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, lr_prob))), 4),
    }

    # XGBoost comparison using provided predictions
    xgb_metrics = None
    if xgb_predictions:
        xgb_df = pd.DataFrame(xgb_predictions)
        merged = df_model.merge(xgb_df[["driver_id", "predicted_position"]],
                                on="driver_id", how="inner")
        if len(merged) >= 3:
            actual_pos = df[df["driver_id"].isin(merged["driver_id"])]["position"]
            xgb_pred_podium = (merged["predicted_position"] <= 3).astype(int)
            actual_podium = merged["podium"]
            xgb_metrics = {
                "model": "XGBoost",
                "accuracy": round(float(accuracy_score(actual_podium, xgb_pred_podium)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(
                    actual_podium, xgb_pred_podium.astype(float)))), 4),
            }

    return {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "comparison_available": xgb_metrics is not None,
        "driver_predictions": [
            {
                "driver_id": row["driver_id"],
                "actual_podium": int(row["podium"]),
                "lr_prob": round(float(lr_prob[i]), 4),
            }
            for i, (_, row) in enumerate(df_model.iterrows())
        ],
    }
