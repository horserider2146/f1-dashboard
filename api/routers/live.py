"""
Live data router — proxies OpenF1 API for real-time race information.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

import data.openf1_client as openf1

router = APIRouter(prefix="/live", tags=["live"])


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/session/latest")
def get_latest_session():
    """Return metadata for the most recent F1 session."""
    try:
        return openf1.get_latest_session() or {}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/positions/{session_key}")
def get_positions(session_key: int, driver_number: Optional[int] = Query(None)):
    """Return driver positions for a session."""
    try:
        df = openf1.get_live_positions(session_key)
        if driver_number:
            df = df[df["driver_number"] == driver_number]
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/laps/{session_key}")
def get_laps(session_key: int, driver_number: Optional[int] = Query(None)):
    """Return lap timing data for a session."""
    try:
        df = openf1.get_live_laps(session_key, driver_number=driver_number)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/pit-stops/{session_key}")
def get_pit_stops(session_key: int):
    """Return pit stop events for a session."""
    try:
        df = openf1.get_pit_stops(session_key)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/car-data/{session_key}/{driver_number}")
def get_car_data(session_key: int, driver_number: int):
    """Return live car telemetry for a driver."""
    try:
        df = openf1.get_car_data(session_key, driver_number)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/leaderboard/{session_key}")
def get_leaderboard(session_key: int):
    """
    Composite leaderboard: positions + laps + tyres + gaps.
    """
    try:
        df = openf1.build_leaderboard(session_key)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/track-status/{session_key}")
def get_track_status(session_key: int):
    """Return race control messages (SC, VSC, flags, DRS zones)."""
    try:
        df = openf1.get_race_control(session_key)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/stints/{session_key}")
def get_stints(session_key: int, driver_number: Optional[int] = Query(None)):
    """Return tyre stint data for a session."""
    try:
        df = openf1.get_stints(session_key, driver_number=driver_number)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")


@router.get("/track-map/{session_key}/{driver_number}")
def get_track_map_live(session_key: int, driver_number: int):
    """Return live GPS/track coordinates for a driver."""
    try:
        df = openf1.get_location(session_key, driver_number)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 error: {e}")
