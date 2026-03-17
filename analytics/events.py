"""
Race event detection:
  - Overtakes
  - DRS activations
  - Safety car / VSC laps (from lap time spikes)
  - Fastest laps
"""
import pandas as pd
import numpy as np


# ── Overtake detection ─────────────────────────────────────────────────────────

def detect_overtakes(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect position changes between consecutive laps.

    An overtake is recorded when driver A's position on lap N+1 is better
    (lower) than on lap N — excluding pit-lap changes.

    Returns: lap_number, driver_id, position_before, position_after, laps_gained
    """
    if "position" not in lap_df.columns:
        return pd.DataFrame()

    lap_df = (lap_df
              .sort_values(["driver_id", "lap_number"])
              .dropna(subset=["position"]))

    records = []
    for driver, group in lap_df.groupby("driver_id"):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev = group.iloc[i - 1]
            curr = group.iloc[i]
            # Skip pit laps — position change there isn't an overtake
            if prev.get("is_pit_lap", False) or curr.get("is_pit_lap", False):
                continue
            gained = int(prev["position"]) - int(curr["position"])
            if gained > 0:
                records.append({
                    "lap_number": int(curr["lap_number"]),
                    "driver_id": driver,
                    "position_before": int(prev["position"]),
                    "position_after": int(curr["position"]),
                    "positions_gained": gained,
                })

    return pd.DataFrame(records).sort_values("lap_number")


# ── DRS activation detection ───────────────────────────────────────────────────

def detect_drs_activations(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect DRS open/close events from telemetry.

    DRS column values:
      0  = closed (or detection point)
      8  = eligible (within 1 s of car ahead)
      10 = open
      14 = open (some cars)

    Returns: driver_id, lap_number, ts_open, ts_close, duration_s
    """
    if "drs" not in telemetry_df.columns:
        return pd.DataFrame()

    drs_open_vals = {10, 12, 14}
    tel = telemetry_df.copy()
    tel["drs_open"] = tel["drs"].isin(drs_open_vals)

    records = []
    for (driver, lap), group in tel.groupby(["driver_id", "lap_number"]):
        group = group.sort_values("ts")
        in_drs = False
        start_ts = None

        for _, row in group.iterrows():
            if row["drs_open"] and not in_drs:
                in_drs = True
                start_ts = row["ts"]
            elif not row["drs_open"] and in_drs:
                records.append({
                    "driver_id": driver,
                    "lap_number": lap,
                    "ts_open": start_ts,
                    "ts_close": row["ts"],
                    "duration_s": round(row["ts"] - start_ts, 3),
                })
                in_drs = False

    return pd.DataFrame(records)


# ── Safety car / VSC lap detection ────────────────────────────────────────────

def detect_safety_car_laps(lap_df: pd.DataFrame,
                            sc_threshold_multiplier: float = 1.35) -> pd.DataFrame:
    """
    Identify likely safety-car or VSC laps by detecting lap times significantly
    above the session median.

    sc_threshold_multiplier: lap_time > median * multiplier => SC lap candidate

    Returns: lap_number, median_lap_time, detected_lap_time, sc_candidate (bool)
    """
    lap_df = lap_df.dropna(subset=["lap_time_s"])
    median_time = lap_df["lap_time_s"].median()
    threshold = median_time * sc_threshold_multiplier

    # Aggregate per lap (average across drivers)
    per_lap = (lap_df.groupby("lap_number")["lap_time_s"]
               .mean()
               .reset_index()
               .rename(columns={"lap_time_s": "avg_lap_time"}))

    per_lap["sc_candidate"] = per_lap["avg_lap_time"] > threshold
    per_lap["median_lap_time"] = round(median_time, 3)
    per_lap["avg_lap_time"] = per_lap["avg_lap_time"].round(3)
    return per_lap.sort_values("lap_number")


# ── Fastest lap ────────────────────────────────────────────────────────────────

def fastest_laps(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the single fastest lap for each driver in the race.
    """
    lap_df = lap_df.dropna(subset=["lap_time_s"])
    idx = lap_df.groupby("driver_id")["lap_time_s"].idxmin()
    result = lap_df.loc[idx, ["driver_id", "lap_number", "lap_time_s", "compound"]].copy()
    return result.sort_values("lap_time_s").reset_index(drop=True)


# ── Lap time delta ─────────────────────────────────────────────────────────────

def lap_time_delta(lap_df: pd.DataFrame,
                   driver_a: str, driver_b: str) -> pd.DataFrame:
    """
    Compute the lap-by-lap time delta between two drivers.
    Positive = A is slower than B on that lap.
    """
    a = lap_df[lap_df["driver_id"] == driver_a][["lap_number", "lap_time_s"]].copy()
    b = lap_df[lap_df["driver_id"] == driver_b][["lap_number", "lap_time_s"]].copy()
    merged = a.merge(b, on="lap_number", suffixes=("_a", "_b"))
    merged["delta_s"] = (merged["lap_time_s_a"] - merged["lap_time_s_b"]).round(3)
    merged = merged.rename(columns={
        "lap_time_s_a": f"lap_time_{driver_a}",
        "lap_time_s_b": f"lap_time_{driver_b}",
    })
    return merged.sort_values("lap_number")
