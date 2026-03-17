"""
Telemetry router — per-driver, per-lap car telemetry with track coordinates.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from api.schemas import TelemetryPoint
from data.fastf1_loader import load_session, get_telemetry

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

_session_cache: dict = {}


def _get_session(year: int, gp: str, session_type: str = "R"):
    key = f"{year}|{gp}|{session_type}"
    if key not in _session_cache:
        try:
            _session_cache[key] = load_session(year, gp, session_type)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Session not found: {e}")
    return _session_cache[key]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/{driver}", response_model=List[TelemetryPoint])
def get_driver_telemetry(
    year: int,
    gp: str,
    driver: str,
    lap: Optional[int] = Query(None, description="Specific lap number (omit for all laps)"),
    session_type: str = Query("R", description="R=Race, Q=Qualifying, FP1/FP2/FP3"),
):
    """
    Return car telemetry for a driver.
    Optionally filter to a specific lap.
    """
    session = _get_session(year, gp, session_type)
    df = get_telemetry(session, driver.upper(), lap_number=lap)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry found for driver '{driver}'" +
                   (f" on lap {lap}" if lap else ""),
        )

    df = df.where(pd.notna(df), None)
    return [TelemetryPoint(**r) for r in df.to_dict(orient="records")]


@router.get("/{year}/{gp}/{driver}/track-map")
def get_track_map(
    year: int,
    gp: str,
    driver: str,
    lap: int = Query(..., description="Lap number for track map coordinates"),
    session_type: str = Query("R"),
):
    """
    Return X/Y track coordinates for a specific lap — used to draw the circuit outline.
    Returns a list of {ts, x, y} objects.
    """
    session = _get_session(year, gp, session_type)
    df = get_telemetry(session, driver.upper(), lap_number=lap)

    if df.empty:
        raise HTTPException(status_code=404, detail="Telemetry not found.")

    if "x" not in df.columns or "y" not in df.columns:
        raise HTTPException(status_code=404, detail="Track coordinates not available.")

    cols = ["ts", "x", "y"]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df.to_dict(orient="records")


@router.get("/{year}/{gp}/multi-driver/track-animation")
def get_track_animation(
    year: int,
    gp: str,
    lap: int = Query(..., description="Start lap number"),
    drivers: str = Query(..., description="Comma-separated driver abbreviations, e.g. HAM,VER"),
    lap_end: Optional[int] = Query(None, description="End lap (inclusive). Omit for single lap. -1 = full race."),
    session_type: str = Query("R"),
    max_frames: int = Query(2000, description="Cap total animation frames per driver to avoid browser lag"),
):
    """
    Return X/Y positions for multiple drivers — single lap, lap range, or full race.
    Returns: { driver: [{ts, x, y}, ...], ... }
    """
    session = _get_session(year, gp, session_type)
    driver_list = [d.strip().upper() for d in drivers.split(",")]

    def _load_driver(drv: str):
        """Load and downsample telemetry for one driver. Returns (drv, records) or (drv, None)."""
        try:
            if lap_end is not None and lap_end == -1:
                df = get_telemetry(session, drv, lap_number=None)
            elif lap_end is not None and lap_end > lap:
                df_all = get_telemetry(session, drv, lap_number=None)
                if not df_all.empty and "lap_number" in df_all.columns:
                    df = df_all[df_all["lap_number"].between(lap, lap_end)].copy()
                else:
                    df = df_all
            else:
                df = get_telemetry(session, drv, lap_number=lap)

            if df.empty or "x" not in df.columns:
                return drv, None

            sub = df[["ts", "x", "y"]].dropna().reset_index(drop=True)
            if len(sub) > max_frames:
                step = max(1, len(sub) // max_frames)
                sub = sub.iloc[::step].reset_index(drop=True)
            return drv, sub.to_dict(orient="records")
        except Exception:
            return drv, None

    # Load all drivers in parallel to cut wall-clock time
    result = {}
    with ThreadPoolExecutor(max_workers=min(6, len(driver_list))) as pool:
        futures = {pool.submit(_load_driver, drv): drv for drv in driver_list}
        for future in as_completed(futures):
            drv, records = future.result()
            if records:
                result[drv] = records

    if not result:
        raise HTTPException(status_code=404, detail="No telemetry found for requested drivers.")

    return result


@router.get("/{year}/{gp}/{driver}/speed-trace")
def get_speed_trace(
    year: int,
    gp: str,
    driver: str,
    lap: int = Query(..., description="Lap number"),
    session_type: str = Query("R"),
):
    """
    Return speed, throttle, brake, and gear vs. distance for a driver's lap.
    The 'Distance' column comes from FastF1 telemetry.
    """
    session = _get_session(year, gp, session_type)
    drv = driver.upper()

    try:
        lap_obj = session.laps.pick_driver(drv).pick_lap(lap).iloc[0]
        tel = lap_obj.get_telemetry().add_distance()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Telemetry error: {e}")

    rename = {
        "Distance": "distance",
        "Speed": "speed",
        "Throttle": "throttle",
        "Brake": "brake",
        "nGear": "gear",
        "RPM": "rpm",
        "DRS": "drs",
    }
    tel = tel.rename(columns={k: v for k, v in rename.items() if k in tel.columns})
    keep = ["distance", "speed", "throttle", "brake", "gear", "rpm", "drs"]
    tel = tel[[c for c in keep if c in tel.columns]]
    tel = tel.where(pd.notna(tel), None)
    return tel.to_dict(orient="records")
