"""
Thin HTTP client used by the Streamlit dashboard to call the FastAPI backend.
All methods return plain Python dicts / lists — no Pydantic on the dashboard side.
"""
import requests
from typing import Any, Optional

BASE_URL = "http://localhost:8000"

# Analytics endpoints require FastF1 to load from disk on first call — allow more time.
_DEFAULT_TIMEOUT = 90
_POST_TIMEOUT = 120


def _get(path: str, params: dict | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, json: dict | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, json=json, timeout=_POST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _resolve_session_key(session_key: Optional[str]) -> Optional[int]:
    """Return an integer session key; fetches the latest session if none given."""
    if session_key and str(session_key).strip():
        try:
            return int(str(session_key).strip())
        except ValueError:
            pass
    try:
        data = _get("/live/session/latest")
        return data.get("session_key") if data else None
    except Exception:
        return None


# ── Races ──────────────────────────────────────────────────────────────────────

def get_schedule(year: int) -> list:
    return _get(f"/races/schedule/{year}")


def get_race_metadata(year: int, gp: str) -> dict:
    return _get(f"/races/{year}/{gp}/metadata")


def get_drivers(year: int, gp: str) -> list:
    return _get(f"/races/{year}/{gp}/drivers")


def get_laps(year: int, gp: str, driver: Optional[str] = None) -> list:
    params = {"driver": driver} if driver else None
    return _get(f"/races/{year}/{gp}/laps", params=params)


def get_fastest_laps(year: int, gp: str) -> list:
    return _get(f"/races/{year}/{gp}/fastest-laps")


def get_position_history(year: int, gp: str) -> dict:
    return _get(f"/races/{year}/{gp}/position-history")


# ── Telemetry ──────────────────────────────────────────────────────────────────

def get_speed_trace(year: int, gp: str, driver: str, lap: int) -> list:
    return _get(f"/telemetry/{year}/{gp}/{driver}/speed-trace", params={"lap": lap})


def get_track_map(year: int, gp: str, driver: str, lap: int) -> list:
    return _get(f"/telemetry/{year}/{gp}/{driver}/track-map", params={"lap": lap})


def get_track_animation(year: int, gp: str, lap: int, drivers: str,
                        lap_end: Optional[int] = None) -> dict:
    params: dict = {"lap": lap, "drivers": drivers}
    if lap_end is not None:
        params["lap_end"] = lap_end
    url = f"{BASE_URL}/telemetry/{year}/{gp}/multi-driver/track-animation"
    # Parallel loading still takes time on first load — allow up to 5 minutes
    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    return resp.json()


# ── Live ───────────────────────────────────────────────────────────────────────

def get_latest_session() -> dict:
    return _get("/live/session/latest")


def get_live_leaderboard(session_key: Optional[str] = None) -> list:
    sk = _resolve_session_key(session_key)
    if sk is None:
        return []
    return _get(f"/live/leaderboard/{sk}")


def get_live_pit_stops(session_key: Optional[str] = None) -> list:
    sk = _resolve_session_key(session_key)
    if sk is None:
        return []
    return _get(f"/live/pit-stops/{sk}")


def get_track_status(session_key: Optional[str] = None) -> list:
    sk = _resolve_session_key(session_key)
    if sk is None:
        return []
    return _get(f"/live/track-status/{sk}")


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_stints(year: int, gp: str, driver: Optional[str] = None) -> list:
    params = {"driver": driver} if driver else None
    return _get(f"/analytics/{year}/{gp}/stints", params=params)


def get_pit_stops(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/pit-stops")


def get_strategy_comparison(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/strategy-comparison")


def get_undercuts(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/undercuts")


def get_overtakes(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/overtakes")


def get_safety_car_laps(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/safety-car-laps")


def get_compound_summary(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/tyre-deg/compound-summary")


def predict_tyre_deg(year: int, gp: str, driver: str,
                     compound: str, start_lap: int, length: int) -> list:
    return _get(
        f"/analytics/{year}/{gp}/tyre-deg/predict",
        params={
            "driver": driver,
            "compound": compound,
            "stint_start_lap": start_lap,
            "stint_length": length,
        },
    )


def get_lap_delta(year: int, gp: str, driver_a: str, driver_b: str) -> list:
    return _get(
        f"/analytics/{year}/{gp}/lap-delta",
        params={"driver_a": driver_a, "driver_b": driver_b},
    )


def train_predictor(year: int, gp: str) -> dict:
    return _post(f"/analytics/{year}/{gp}/train-predictor")


def predict_outcome(year: int, gp: str) -> list:
    return _get(f"/analytics/{year}/{gp}/predict-outcome")
