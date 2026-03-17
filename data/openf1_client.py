"""
OpenF1 API client — live and recent race timing data.
Docs: https://openf1.org/
"""
import httpx
import pandas as pd
from typing import Optional
from config.settings import settings

BASE = settings.openf1_base_url


def _get(endpoint: str, params: dict | None = None) -> list[dict]:
    """Raw GET against the OpenF1 REST API. Returns list of dicts."""
    url = f"{BASE}/{endpoint}"
    with httpx.Client(timeout=15) as client:
        resp = client.get(url, params=params or {})
        resp.raise_for_status()
        return resp.json()


# ── Session helpers ────────────────────────────────────────────────────────────

def get_latest_session() -> dict | None:
    """Return metadata for the most recent session."""
    data = _get("sessions", {"order_by": "date_end", "limit": 1})
    return data[0] if data else None


def get_session(year: int, gp: str, session_type: str = "Race") -> dict | None:
    """Find a session by year, GP name, and type."""
    data = _get("sessions", {
        "year": year,
        "circuit_short_name": gp,
        "session_name": session_type,
    })
    return data[0] if data else None


def get_sessions_for_year(year: int) -> pd.DataFrame:
    data = _get("sessions", {"year": year})
    return pd.DataFrame(data)


# ── Live timing ────────────────────────────────────────────────────────────────

def get_live_positions(session_key: int) -> pd.DataFrame:
    """Current position of every driver in a session."""
    data = _get("position", {"session_key": session_key})
    return pd.DataFrame(data)


def get_live_laps(session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
    """All laps completed in a session (optionally filtered by driver)."""
    params = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    data = _get("laps", params)
    return pd.DataFrame(data)


def get_intervals(session_key: int) -> pd.DataFrame:
    """Gap to leader / gap to car ahead for each driver."""
    data = _get("intervals", {"session_key": session_key})
    return pd.DataFrame(data)


def get_stints(session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
    """Tyre stint data — compound, lap in/out."""
    params = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    data = _get("stints", params)
    return pd.DataFrame(data)


def get_pit_stops(session_key: int) -> pd.DataFrame:
    """Pit stop events with lap number and duration."""
    data = _get("pit", {"session_key": session_key})
    return pd.DataFrame(data)


def get_car_data(session_key: int, driver_number: int,
                 date_start: Optional[str] = None,
                 date_end: Optional[str] = None) -> pd.DataFrame:
    """
    High-frequency car telemetry (speed, throttle, brake, gear, rpm, drs).
    Optionally slice by ISO-8601 timestamps.
    """
    params = {"session_key": session_key, "driver_number": driver_number}
    if date_start:
        params["date>"] = date_start
    if date_end:
        params["date<"] = date_end
    data = _get("car_data", params)
    return pd.DataFrame(data)


def get_location(session_key: int, driver_number: int) -> pd.DataFrame:
    """
    GPS / track position data (x, y, z coordinates).
    Used for the animated track map.
    """
    data = _get("location", {"session_key": session_key, "driver_number": driver_number})
    return pd.DataFrame(data)


def get_drivers(session_key: int) -> pd.DataFrame:
    """Driver list for a session with name, team, and team colour."""
    data = _get("drivers", {"session_key": session_key})
    return pd.DataFrame(data)


def get_race_control(session_key: int) -> pd.DataFrame:
    """Race control messages: safety car, VSC, flags, DRS zones."""
    data = _get("race_control", {"session_key": session_key})
    return pd.DataFrame(data)


# ── Convenience: build live leaderboard ───────────────────────────────────────

def build_leaderboard(session_key: int) -> pd.DataFrame:
    """
    Combine position, lap, interval, and stint data into a single
    leaderboard DataFrame. Suitable for the dashboard live view.
    """
    positions = get_live_positions(session_key)
    laps = get_live_laps(session_key)
    stints = get_stints(session_key)
    intervals = get_intervals(session_key)

    if positions.empty:
        return pd.DataFrame()

    # Keep latest position per driver
    if "date" in positions.columns:
        positions["date"] = pd.to_datetime(positions["date"])
        positions = (positions.sort_values("date")
                     .groupby("driver_number").last().reset_index())

    # Latest completed lap per driver
    if not laps.empty and "lap_number" in laps.columns:
        latest_laps = (laps.sort_values("lap_number")
                       .groupby("driver_number").last().reset_index()
                       [["driver_number", "lap_number", "lap_duration"]])
    else:
        latest_laps = pd.DataFrame(columns=["driver_number"])

    # Current tyre compound (last stint)
    if not stints.empty and "compound" in stints.columns:
        current_tyre = (stints.sort_values("stint_number")
                        .groupby("driver_number").last().reset_index()
                        [["driver_number", "compound", "stint_number"]])
    else:
        current_tyre = pd.DataFrame(columns=["driver_number"])

    # Latest gap
    if not intervals.empty:
        latest_gap = (intervals.sort_values("date") if "date" in intervals.columns
                      else intervals)
        latest_gap = (latest_gap.groupby("driver_number").last().reset_index()
                      [["driver_number", "gap_to_leader"]]
                      if "gap_to_leader" in intervals.columns
                      else pd.DataFrame(columns=["driver_number"]))
    else:
        latest_gap = pd.DataFrame(columns=["driver_number"])

    board = positions[["driver_number", "position"]].copy()
    for extra in [latest_laps, current_tyre, latest_gap]:
        if not extra.empty:
            board = board.merge(extra, on="driver_number", how="left")

    board = board.sort_values("position")
    return board.reset_index(drop=True)
