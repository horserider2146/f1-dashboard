"""
Race outcome predictor using Random Forest / XGBoost.

Features:
  - grid_position
  - avg_qualifying_pace (relative to pole)
  - team_avg_points (season form)
  - num_pit_stops
  - compound_diversity (strategy complexity)
  - lap_pace_percentile (first-stint pace)

Target: final_position
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

MODEL_DIR = Path("./cache/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _build_features(race_results_df: pd.DataFrame,
                    lap_df: pd.DataFrame,
                    pit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Assemble feature matrix from raw DataFrames.

    race_results_df: columns driver_id, grid_position, final_position, team, points
    lap_df: columns driver_id, lap_number, lap_time_s, compound
    pit_df: columns driver_id, lap (optional — inferred from lap_df if None)
    """
    features = race_results_df[["driver_id", "grid_position", "final_position"]].copy()

    # ── Lap pace in first stint (laps 5-15 to skip outliers) ──────────────────
    early_laps = lap_df[lap_df["lap_number"].between(5, 15)].copy()
    early_pace = (early_laps.groupby("driver_id")["lap_time_s"]
                  .mean()
                  .reset_index()
                  .rename(columns={"lap_time_s": "early_pace"}))
    features = features.merge(early_pace, on="driver_id", how="left")

    # Normalise pace relative to fastest car
    if "early_pace" in features.columns:
        min_pace = features["early_pace"].min()
        features["pace_gap_to_leader"] = features["early_pace"] - min_pace

    # ── Pit stop count ─────────────────────────────────────────────────────────
    if pit_df is not None and not pit_df.empty:
        stops = (pit_df.groupby("driver_id").size()
                 .reset_index(name="num_pit_stops"))
    else:
        pit_laps = lap_df[lap_df.get("is_pit_lap", pd.Series(False, index=lap_df.index))]
        stops = (pit_laps.groupby("driver_id").size()
                 .reset_index(name="num_pit_stops"))
    features = features.merge(stops, on="driver_id", how="left")
    features["num_pit_stops"] = features["num_pit_stops"].fillna(1).astype(int)

    # ── Compound diversity ─────────────────────────────────────────────────────
    compound_div = (lap_df.groupby("driver_id")["compound"]
                    .nunique()
                    .reset_index(name="compound_diversity"))
    features = features.merge(compound_div, on="driver_id", how="left")

    features = features.fillna(features.median(numeric_only=True))
    return features


class RaceOutcomePredictor:
    """
    Predicts final race position given pre-race and in-race features.
    Two models available: 'rf' (Random Forest) and 'xgb' (XGBoost).
    """

    def __init__(self, model_type: str = "xgb"):
        self.model_type = model_type
        self._model = None
        self._feature_cols = [
            "grid_position",
            "pace_gap_to_leader",
            "num_pit_stops",
            "compound_diversity",
        ]

    def _build_model(self):
        if self.model_type == "xgb":
            return XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
        )

    def fit(self, features_df: pd.DataFrame):
        """
        Train the predictor on historical race feature data.
        features_df must contain final_position + the feature columns.
        """
        cols = [c for c in self._feature_cols if c in features_df.columns]
        X = features_df[cols].values
        y = features_df["final_position"].values

        self._model = self._build_model()
        self._model.fit(X, y)
        self._feature_cols = cols
        return self

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Return predicted finishing positions (as floats, round for actual position)."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call .fit() first.")
        X = features_df[self._feature_cols].fillna(0).values
        preds = self._model.predict(X)
        return pd.Series(np.clip(np.round(preds), 1, 20).astype(int),
                         index=features_df.index)

    def cross_validate(self, features_df: pd.DataFrame) -> dict:
        """5-fold CV — returns mean MAE and R²."""
        cols = [c for c in self._feature_cols if c in features_df.columns]
        X = features_df[cols].fillna(0).values
        y = features_df["final_position"].values
        model = self._build_model()
        mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
        r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        return {"mae": round(mae, 3), "r2": round(r2, 3)}

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame()
        imp = getattr(self._model, "feature_importances_", None)
        if imp is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False)

    def save(self, path: str | None = None):
        path = path or str(MODEL_DIR / f"race_predictor_{self.model_type}.joblib")
        joblib.dump({"model": self._model, "features": self._feature_cols}, path)

    def load(self, path: str | None = None):
        path = path or str(MODEL_DIR / f"race_predictor_{self.model_type}.joblib")
        obj = joblib.load(path)
        self._model = obj["model"]
        self._feature_cols = obj["features"]
        return self


# ── Convenience function ───────────────────────────────────────────────────────

def build_and_train(race_results_df: pd.DataFrame,
                    lap_df: pd.DataFrame,
                    pit_df: pd.DataFrame | None = None,
                    model_type: str = "xgb") -> RaceOutcomePredictor:
    """One-shot: build features, train, return fitted predictor."""
    features = _build_features(race_results_df, lap_df, pit_df)
    predictor = RaceOutcomePredictor(model_type=model_type)
    predictor.fit(features)
    return predictor
