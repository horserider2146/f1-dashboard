"""
Tyre degradation model.

Models:
  1. Linear regression: lap_time = base_time + deg_rate * tyre_age
  2. Per-compound deg rates fitted on historical lap data
  3. Stint remaining life estimator
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

MODEL_DIR = Path("./cache/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# Typical baseline deg rates (seconds per lap per tyre-age lap)
# Used as fallback when there is not enough data to fit.
BASELINE_DEG_RATES = {
    "SOFT": 0.08,
    "MEDIUM": 0.05,
    "HARD": 0.03,
    "INTERMEDIATE": 0.04,
    "WET": 0.02,
}


class TyreDegradationModel:
    """
    Fits a linear degradation model per compound per driver.

        lap_time = base_time + deg_rate * tyre_age + fuel_offset * lap_number

    fuel_offset captures the lap-time improvement as fuel burns off (~0.03 s/lap).
    """

    def __init__(self):
        # Maps (driver_id, compound) -> (base_time, deg_rate, fuel_offset)
        self._params: dict[tuple, tuple] = {}

    def fit(self, lap_df: pd.DataFrame):
        """
        Fit the model on a lap DataFrame.

        Required columns: driver_id, compound, tyre_age, lap_number, lap_time_s
        """
        lap_df = lap_df.dropna(subset=["tyre_age", "lap_number", "lap_time_s", "compound"])
        # Remove outliers (pit laps, safety cars) — keep only "normal" laps
        lap_df = lap_df[lap_df["lap_time_s"] > 0]
        # Remove top/bottom 5% lap times per driver as outliers
        lap_df = (lap_df.groupby("driver_id", group_keys=False)
                  .apply(lambda g: g[
                      (g["lap_time_s"] >= g["lap_time_s"].quantile(0.05)) &
                      (g["lap_time_s"] <= g["lap_time_s"].quantile(0.95))
                  ]))

        for (driver, compound), group in lap_df.groupby(["driver_id", "compound"]):
            if len(group) < 4:
                continue
            X = group[["tyre_age", "lap_number"]].values.astype(float)
            y = group["lap_time_s"].values.astype(float)
            reg = LinearRegression()
            reg.fit(X, y)
            base_time = reg.intercept_
            deg_rate = reg.coef_[0]
            fuel_offset = reg.coef_[1]
            self._params[(driver, compound)] = (base_time, deg_rate, fuel_offset)

        return self

    def predict(self, driver_id: str, compound: str,
                tyre_age: int, lap_number: int) -> float:
        """Predict lap time (seconds) for given conditions."""
        if (driver_id, compound) in self._params:
            base, deg, fuel = self._params[(driver_id, compound)]
        else:
            # Fallback: use baseline deg rate + median base time
            deg = BASELINE_DEG_RATES.get(compound.upper(), 0.05)
            base = 90.0  # generic 90 s base
            fuel = -0.03
        return base + deg * tyre_age + fuel * lap_number

    def predict_stint(self, driver_id: str, compound: str,
                      stint_start_lap: int, stint_length: int) -> pd.DataFrame:
        """Predict lap times for an entire stint."""
        rows = []
        for age in range(1, stint_length + 1):
            lap = stint_start_lap + age - 1
            t = self.predict(driver_id, compound, age, lap)
            rows.append({"lap_number": lap, "tyre_age": age, "predicted_lap_time": t})
        return pd.DataFrame(rows)

    def get_deg_rate(self, driver_id: str, compound: str) -> float:
        """Return degradation rate (seconds per lap per tyre age unit)."""
        if (driver_id, compound) in self._params:
            return self._params[(driver_id, compound)][1]
        return BASELINE_DEG_RATES.get(compound.upper(), 0.05)

    def optimal_pit_window(self, driver_id: str, compound: str,
                           current_tyre_age: int, lap_number: int,
                           threshold_seconds: float = 1.5) -> int | None:
        """
        Estimate how many more laps until performance drops below threshold
        relative to a fresh tyre of the same compound.

        Returns laps remaining in optimal window, or None if already past it.
        """
        fresh_time = self.predict(driver_id, compound, 1, lap_number)
        for extra_laps in range(0, 40):
            age = current_tyre_age + extra_laps
            lap = lap_number + extra_laps
            predicted = self.predict(driver_id, compound, age, lap)
            if predicted - fresh_time >= threshold_seconds:
                return extra_laps
        return None  # still within window at 40 laps

    def save(self, path: str | None = None):
        path = path or str(MODEL_DIR / "tyre_model.joblib")
        joblib.dump(self._params, path)

    def load(self, path: str | None = None):
        path = path or str(MODEL_DIR / "tyre_model.joblib")
        self._params = joblib.load(path)
        return self


# ── Compound-level statistics ──────────────────────────────────────────────────

def compound_summary(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate statistics per compound for a race.
    Returns: compound, mean_lap_time, deg_rate (slope from linear fit), stint_count
    """
    lap_df = lap_df.dropna(subset=["compound", "tyre_age", "lap_time_s"])
    # Remove clear outliers
    lap_df = lap_df[lap_df["lap_time_s"] > 0]

    rows = []
    for compound, group in lap_df.groupby("compound"):
        if len(group) < 3:
            continue
        mean_lt = group["lap_time_s"].mean()
        # Linear fit tyre_age → lap_time
        x = group["tyre_age"].values.reshape(-1, 1).astype(float)
        y = group["lap_time_s"].values.astype(float)
        reg = LinearRegression().fit(x, y)
        rows.append({
            "compound": compound,
            "mean_lap_time_s": round(mean_lt, 3),
            "deg_rate_s_per_lap": round(float(reg.coef_[0]), 4),
            "stint_count": group["driver_id"].nunique() if "driver_id" in group else len(group),
        })
    return pd.DataFrame(rows).sort_values("mean_lap_time_s")
