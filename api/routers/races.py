"""
Races router — season schedule, race metadata, lap data, driver list.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import pandas as pd

from api.schemas import RaceEvent, RaceMetadata, DriverInfo, LapRecord, FastestLap
from data.fastf1_loader import (
    list_available_races,
    load_session,
    get_lap_data,
    get_driver_info,
    get_race_metadata,
)
from analytics.events import fastest_laps

router = APIRouter(prefix="/races", tags=["races"])

# In-memory session cache to avoid re-loading within a single server run.
_session_cache: dict = {}


def _cache_key(year: int, gp: str, session_type: str) -> str:
    return f"{year}|{gp}|{session_type}"


def _get_session(year: int, gp: str, session_type: str = "R"):
    key = _cache_key(year, gp, session_type)
    if key not in _session_cache:
        try:
            _session_cache[key] = load_session(year, gp, session_type)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Session not found: {e}")
    return _session_cache[key]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/schedule/{year}", response_model=List[RaceEvent])
def get_schedule(year: int):
    """Return the full race calendar for a given season."""
    try:
        df = list_available_races(year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    events = []
    for _, row in df.iterrows():
        events.append(RaceEvent(
            round=int(row["RoundNumber"]),
            event_name=str(row["EventName"]),
            location=str(row["Location"]),
            date=str(row["EventDate"])[:10],
        ))
    return events


@router.get("/{year}/{gp}/metadata", response_model=RaceMetadata)
def get_race_info(year: int, gp: str):
    """Return metadata for a specific race weekend."""
    session = _get_session(year, gp)
    meta = get_race_metadata(session)
    return RaceMetadata(**meta)


@router.get("/{year}/{gp}/drivers", response_model=List[DriverInfo])
def get_drivers(year: int, gp: str):
    """Return driver information for a race weekend."""
    session = _get_session(year, gp)
    df = get_driver_info(session)
    return [DriverInfo(**row) for _, row in df.iterrows()]


@router.get("/{year}/{gp}/laps", response_model=List[LapRecord])
def get_laps(year: int, gp: str, driver: str | None = None):
    """
    Return lap data for a race.  Optionally filter by driver abbreviation.
    """
    session = _get_session(year, gp)
    df = get_lap_data(session)

    if driver:
        df = df[df["driver_id"] == driver.upper()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Driver '{driver}' not found.")

    # Replace NaN with None so Pydantic serialises cleanly
    df = df.where(pd.notna(df), None)
    records = df.to_dict(orient="records")
    return [LapRecord(**r) for r in records]


@router.get("/{year}/{gp}/fastest-laps", response_model=List[FastestLap])
def get_fastest_laps(year: int, gp: str):
    """Return the single fastest lap per driver."""
    session = _get_session(year, gp)
    lap_df = get_lap_data(session)
    fl = fastest_laps(lap_df)
    fl = fl.where(pd.notna(fl), None)
    return [FastestLap(**r) for r in fl.to_dict(orient="records")]


@router.get("/{year}/{gp}/position-history")
def get_position_history(year: int, gp: str):
    """
    Return lap-by-lap positions for all drivers — used for position change graph.
    Returns: { driver_id: { lap_number: position, ... }, ... }
    """
    session = _get_session(year, gp)
    df = get_lap_data(session)

    if "position" not in df.columns:
        raise HTTPException(status_code=404, detail="Position data not available.")

    pivot = (df.dropna(subset=["position"])
               .pivot_table(index="lap_number", columns="driver_id",
                            values="position", aggfunc="first")
               .sort_index())

    # Convert to JSON-friendly dict: { driver: { lap: position } }
    result = {}
    for driver in pivot.columns:
        series = pivot[driver].dropna()
        result[driver] = {int(k): int(v) for k, v in series.items()}
    return result
