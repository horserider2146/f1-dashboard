"""
Strategy analysis: undercut/overcut detection, stint builder, strategy comparison.
"""
import pandas as pd
import numpy as np


# ── Stint builder ──────────────────────────────────────────────────────────────

def build_stints(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tyre stint summary from lap data.

    Required columns: driver_id, lap_number, compound, tyre_age, is_pit_lap
    Returns: driver_id, stint_number, compound, start_lap, end_lap, laps_on_tyre
    """
    lap_df = lap_df.sort_values(["driver_id", "lap_number"])
    stints = []

    for driver, group in lap_df.groupby("driver_id"):
        group = group.reset_index(drop=True)
        stint_num = 1
        stint_start = int(group.loc[0, "lap_number"])
        compound = group.loc[0, "compound"] if "compound" in group.columns else "UNKNOWN"

        for i, row in group.iterrows():
            # New stint starts after a pit lap
            if i > 0 and group.loc[i - 1, "is_pit_lap"]:
                # close previous stint
                stints.append({
                    "driver_id": driver,
                    "stint_number": stint_num,
                    "compound": compound,
                    "start_lap": stint_start,
                    "end_lap": int(group.loc[i - 1, "lap_number"]),
                    "laps_on_tyre": int(group.loc[i - 1, "lap_number"]) - stint_start + 1,
                })
                stint_num += 1
                stint_start = int(row["lap_number"])
                compound = row.get("compound", "UNKNOWN")

        # Close final stint
        last = group.iloc[-1]
        stints.append({
            "driver_id": driver,
            "stint_number": stint_num,
            "compound": compound,
            "start_lap": stint_start,
            "end_lap": int(last["lap_number"]),
            "laps_on_tyre": int(last["lap_number"]) - stint_start + 1,
        })

    return pd.DataFrame(stints)


# ── Pit stop summary ───────────────────────────────────────────────────────────

def pit_stop_summary(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pit stop events from lap data.
    Returns: driver_id, lap, old_compound, new_compound, stop_number
    """
    lap_df = lap_df.sort_values(["driver_id", "lap_number"])
    stops = []
    for driver, group in lap_df.groupby("driver_id"):
        group = group.reset_index(drop=True)
        stop_num = 0
        for i, row in group.iterrows():
            if row.get("is_pit_lap") and i + 1 < len(group):
                stop_num += 1
                next_row = group.iloc[i + 1]
                stops.append({
                    "driver_id": driver,
                    "lap": int(row["lap_number"]),
                    "old_compound": row.get("compound", "UNKNOWN"),
                    "new_compound": next_row.get("compound", "UNKNOWN"),
                    "stop_number": stop_num,
                })
    return pd.DataFrame(stops)


# ── Undercut / Overcut detector ────────────────────────────────────────────────

def detect_undercuts(lap_df: pd.DataFrame,
                     window_laps: int = 3,
                     gain_threshold_s: float = 0.5) -> pd.DataFrame:
    """
    Detect undercut attempts between pairs of drivers.

    An undercut occurs when:
      - Driver A pits N laps before Driver B
      - After both are on fresh tyres, Driver A has a faster cumulative pace

    Returns a DataFrame of detected undercut events with success flag.
    """
    pit_df = pit_stop_summary(lap_df)
    if pit_df.empty:
        return pd.DataFrame()

    lap_df = lap_df.sort_values(["driver_id", "lap_number"]).copy()
    lap_times = lap_df.pivot(index="lap_number", columns="driver_id", values="lap_time_s")

    results = []
    drivers = pit_df["driver_id"].unique().tolist()

    for i, d_a in enumerate(drivers):
        for d_b in drivers[i + 1:]:
            stops_a = pit_df[pit_df["driver_id"] == d_a]["lap"].tolist()
            stops_b = pit_df[pit_df["driver_id"] == d_b]["lap"].tolist()

            for pit_a in stops_a:
                for pit_b in stops_b:
                    gap = pit_b - pit_a
                    if not (1 <= gap <= 5):
                        continue

                    # Compare lap times in the window after both pitted
                    compare_start = pit_b + 1
                    compare_end = compare_start + window_laps

                    try:
                        times_a = lap_times.loc[compare_start:compare_end, d_a].dropna()
                        times_b = lap_times.loc[compare_start:compare_end, d_b].dropna()
                    except KeyError:
                        continue

                    if times_a.empty or times_b.empty:
                        continue

                    avg_a = times_a.mean()
                    avg_b = times_b.mean()
                    gain = avg_b - avg_a  # positive = A is faster

                    results.append({
                        "undercut_driver": d_a,
                        "target_driver": d_b,
                        "pit_lap_undercut": pit_a,
                        "pit_lap_target": pit_b,
                        "pit_gap_laps": gap,
                        "avg_time_undercut": round(avg_a, 3),
                        "avg_time_target": round(avg_b, 3),
                        "time_gain_s": round(gain, 3),
                        "success": gain >= gain_threshold_s,
                    })

    return pd.DataFrame(results)


# ── Gap analysis ───────────────────────────────────────────────────────────────

def position_change_summary(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise net position changes for each driver over the race.
    Returns: driver_id, start_position, end_position, net_change
    """
    if "position" not in lap_df.columns:
        return pd.DataFrame()

    lap_df = lap_df.dropna(subset=["position"])
    rows = []
    for driver, group in lap_df.groupby("driver_id"):
        group = group.sort_values("lap_number")
        start = int(group.iloc[0]["position"])
        end = int(group.iloc[-1]["position"])
        rows.append({
            "driver_id": driver,
            "start_position": start,
            "end_position": end,
            "net_change": start - end,  # positive = gained places
        })
    return pd.DataFrame(rows).sort_values("end_position")


# ── Strategy comparison ────────────────────────────────────────────────────────

def compare_strategies(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a human-readable strategy string per driver.
    Example: "MEDIUM(18) → HARD(35)"
    """
    rows = []
    for driver, group in stints_df.groupby("driver_id"):
        group = group.sort_values("stint_number")
        strategy = " → ".join(
            f"{row['compound']}({row['laps_on_tyre']})"
            for _, row in group.iterrows()
        )
        rows.append({
            "driver_id": driver,
            "strategy": strategy,
            "num_stops": len(group) - 1,
        })
    return pd.DataFrame(rows)
