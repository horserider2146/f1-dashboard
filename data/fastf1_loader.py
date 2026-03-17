"""
FastF1 data loader — historical lap data and telemetry.
"""
import fastf1
import pandas as pd
from pathlib import Path
from config.settings import settings

# Enable FastF1 cache
fastf1.Cache.enable_cache(settings.fastf1_cache_dir)


def load_session(year: int, gp: str, session_type: str = "R"):
    """
    Load a FastF1 session.

    session_type: 'R' = Race, 'Q' = Qualifying, 'FP1'/'FP2'/'FP3' = Practice
    Returns a fastf1.core.Session object (fully loaded).
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    return session


def get_lap_data(session) -> pd.DataFrame:
    """
    Extract per-lap data for all drivers from a loaded session.
    Returns a clean DataFrame with columns aligned to the DB schema.
    """
    laps = session.laps.copy()

    # Keep only relevant columns
    cols = {
        "Driver": "driver_id",
        "LapNumber": "lap_number",
        "LapTime": "lap_time_s",
        "Sector1Time": "sector1_s",
        "Sector2Time": "sector2_s",
        "Sector3Time": "sector3_s",
        "Compound": "compound",
        "TyreLife": "tyre_age",
        "SpeedI1": "speed_avg",    # fastest speed trap as proxy
        "Position": "position",
        "PitOutTime": "_pit_out",
        "PitInTime": "_pit_in",
    }
    available = {k: v for k, v in cols.items() if k in laps.columns}
    df = laps[list(available.keys())].rename(columns=available)

    # Convert timedelta columns to seconds (float)
    for col in ["lap_time_s", "sector1_s", "sector2_s", "sector3_s"]:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

    # Pit lap flag — lap where driver pitted in
    if "_pit_in" in df.columns:
        df["is_pit_lap"] = df["_pit_in"].notna()
        df.drop(columns=["_pit_in"], inplace=True, errors="ignore")
    if "_pit_out" in df.columns:
        df.drop(columns=["_pit_out"], inplace=True, errors="ignore")

    df["lap_number"] = df["lap_number"].astype("Int64")
    df["position"] = df["position"].astype("Int64")
    df["tyre_age"] = df["tyre_age"].astype("Int64")

    return df.reset_index(drop=True)


def get_telemetry(session, driver: str, lap_number: int | None = None) -> pd.DataFrame:
    """
    Extract car telemetry for a driver (optionally a specific lap).
    Returns DataFrame with columns: lap_number, ts, speed, throttle,
    brake, gear, rpm, x, y, drs.
    """
    if lap_number is not None:
        laps = session.laps.pick_driver(driver).pick_lap(lap_number)
    else:
        laps = session.laps.pick_driver(driver)

    frames = []
    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue
        tel = tel.copy()
        tel["lap_number"] = int(lap["LapNumber"])
        tel["driver_id"] = driver
        frames.append(tel)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    rename = {
        "Time": "ts",
        "Speed": "speed",
        "Throttle": "throttle",
        "Brake": "brake",
        "nGear": "gear",
        "RPM": "rpm",
        "X": "x",
        "Y": "y",
        "DRS": "drs",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "ts" in df.columns:
        df["ts"] = pd.to_timedelta(df["ts"]).dt.total_seconds()

    keep = ["driver_id", "lap_number", "ts", "speed", "throttle", "brake",
            "gear", "rpm", "x", "y", "drs"]
    df = df[[c for c in keep if c in df.columns]]
    return df.reset_index(drop=True)


def get_driver_info(session) -> pd.DataFrame:
    """
    Return a DataFrame with driver_id, driver_name, team, team_color.
    """
    rows = []
    for abbr, info in session.results.iterrows():
        rows.append({
            "driver_id": info.get("Abbreviation", abbr),
            "driver_name": f"{info.get('FirstName', '')} {info.get('LastName', '')}".strip(),
            "team": info.get("TeamName", ""),
            "team_color": f"#{info.get('TeamColor', 'FFFFFF')}",
        })
    return pd.DataFrame(rows)


def get_race_metadata(session) -> dict:
    """Return dict with race metadata for the races table."""
    event = session.event
    return {
        "race_name": event.get("EventName", ""),
        "track": event.get("Location", ""),
        "season": int(event.get("EventDate").year),
        "date": str(event.get("EventDate").date()),
        "total_laps": int(session.total_laps) if hasattr(session, "total_laps") else None,
        "round": int(event.get("RoundNumber", 0)),
    }


def list_available_races(year: int) -> pd.DataFrame:
    """Return a DataFrame listing all races in a given season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule[["RoundNumber", "EventName", "Location", "EventDate"]].copy()
