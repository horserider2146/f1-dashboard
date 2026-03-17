"""
Analytics router — tyre degradation, strategy, overtakes, safety car, predictor.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd

from api.schemas import (
    TyreStint, PitStopEvent, UndercutEvent, OvertakeEvent,
    SafetyCarLap, TyreDegPrediction, StrategyComparison,
    CompoundSummary, RacePrediction,
)
from data.fastf1_loader import load_session, get_lap_data
from analytics.tyre_model import TyreDegradationModel, compound_summary
from analytics.strategy import (
    build_stints, pit_stop_summary, detect_undercuts, compare_strategies,
    position_change_summary,
)
from analytics.events import (
    detect_overtakes, detect_safety_car_laps, fastest_laps,
    lap_time_delta,
)
from analytics.predictor import RaceOutcomePredictor, _build_features, MODEL_DIR
from analytics.strategy import pit_stop_summary

router = APIRouter(prefix="/analytics", tags=["analytics"])

_session_cache: dict = {}
_tyre_models: dict = {}
_predictors: dict = {}


def _get_session(year: int, gp: str, session_type: str = "R"):
    key = f"{year}|{gp}|{session_type}"
    if key not in _session_cache:
        try:
            session = load_session(year, gp, session_type)
            _ = session.laps  # verify timing data actually loaded
            _session_cache[key] = session
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Race data not available for {gp} {year}. "
                       f"Only cached races (2025: Australian, Chinese, Bahrain, Miami) work offline. Error: {e}"
            )
    return _session_cache[key]


def _get_lap_df(year: int, gp: str) -> pd.DataFrame:
    session = _get_session(year, gp)
    return get_lap_data(session)


def _get_tyre_model(year: int, gp: str) -> TyreDegradationModel:
    key = f"{year}|{gp}"
    if key not in _tyre_models:
        lap_df = _get_lap_df(year, gp)
        model = TyreDegradationModel()
        model.fit(lap_df)
        _tyre_models[key] = model
    return _tyre_models[key]


# ── Stints & strategy ──────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/stints", response_model=List[TyreStint])
def get_stints(year: int, gp: str, driver: Optional[str] = Query(None)):
    """Return tyre stint breakdown for all (or one) driver."""
    df = _get_lap_df(year, gp)
    if driver:
        df = df[df["driver_id"] == driver.upper()]
    stints = build_stints(df)
    if stints.empty:
        return []
    stints = stints.where(pd.notna(stints), None)
    return [TyreStint(**r) for r in stints.to_dict(orient="records")]


@router.get("/{year}/{gp}/pit-stops", response_model=List[PitStopEvent])
def get_pit_stops(year: int, gp: str, driver: Optional[str] = Query(None)):
    """Return pit stop events."""
    df = _get_lap_df(year, gp)
    if driver:
        df = df[df["driver_id"] == driver.upper()]
    stops = pit_stop_summary(df)
    if stops.empty:
        return []
    stops = stops.where(pd.notna(stops), None)
    return [PitStopEvent(**r) for r in stops.to_dict(orient="records")]


@router.get("/{year}/{gp}/strategy-comparison", response_model=List[StrategyComparison])
def get_strategy_comparison(year: int, gp: str):
    """Return a human-readable strategy string per driver."""
    df = _get_lap_df(year, gp)
    stints = build_stints(df)
    if stints.empty:
        return []
    result = compare_strategies(stints)
    return [StrategyComparison(**r) for r in result.to_dict(orient="records")]


@router.get("/{year}/{gp}/undercuts", response_model=List[UndercutEvent])
def get_undercuts(
    year: int,
    gp: str,
    window_laps: int = Query(3, description="Laps after pit stop to compare pace"),
    gain_threshold: float = Query(0.5, description="Minimum time gain to count as success"),
):
    """Detect undercut/overcut attempts between drivers."""
    df = _get_lap_df(year, gp)
    result = detect_undercuts(df, window_laps=window_laps,
                              gain_threshold_s=gain_threshold)
    if result.empty:
        return []
    result = result.where(pd.notna(result), None)
    return [UndercutEvent(**r) for r in result.to_dict(orient="records")]


# ── Tyre degradation ───────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/tyre-deg/predict", response_model=List[TyreDegPrediction])
def predict_tyre_deg(
    year: int,
    gp: str,
    driver: str = Query(..., description="Driver abbreviation e.g. HAM"),
    compound: str = Query(..., description="Tyre compound e.g. SOFT"),
    stint_start_lap: int = Query(1),
    stint_length: int = Query(20),
):
    """Predict lap times across a stint using the fitted degradation model."""
    model = _get_tyre_model(year, gp)
    pred_df = model.predict_stint(driver.upper(), compound.upper(),
                                  stint_start_lap, stint_length)
    rows = []
    for _, row in pred_df.iterrows():
        rows.append(TyreDegPrediction(
            driver_id=driver.upper(),
            compound=compound.upper(),
            lap_number=int(row["lap_number"]),
            tyre_age=int(row["tyre_age"]),
            predicted_lap_time=round(float(row["predicted_lap_time"]), 3),
        ))
    return rows


@router.get("/{year}/{gp}/tyre-deg/compound-summary", response_model=List[CompoundSummary])
def get_compound_summary(year: int, gp: str):
    """Return degradation stats per compound for the race."""
    df = _get_lap_df(year, gp)
    summary = compound_summary(df)
    if summary.empty:
        return []
    return [CompoundSummary(**r) for r in summary.to_dict(orient="records")]


@router.get("/{year}/{gp}/tyre-deg/optimal-pit-window")
def optimal_pit_window(
    year: int,
    gp: str,
    driver: str = Query(...),
    compound: str = Query(...),
    current_tyre_age: int = Query(...),
    lap_number: int = Query(...),
    threshold: float = Query(1.5, description="Seconds of deg before pit is recommended"),
):
    """Return estimated laps remaining in optimal window before degradation hits threshold."""
    model = _get_tyre_model(year, gp)
    laps_left = model.optimal_pit_window(
        driver.upper(), compound.upper(),
        current_tyre_age, lap_number,
        threshold_seconds=threshold,
    )
    return {
        "driver_id": driver.upper(),
        "compound": compound.upper(),
        "current_tyre_age": current_tyre_age,
        "laps_until_threshold": laps_left,
        "recommendation": "Pit soon" if (laps_left is not None and laps_left <= 3)
                          else "Still in window",
    }


# ── Event detection ────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/overtakes", response_model=List[OvertakeEvent])
def get_overtakes(year: int, gp: str, driver: Optional[str] = Query(None)):
    """Detect on-track overtakes (position changes on non-pit laps)."""
    df = _get_lap_df(year, gp)
    result = detect_overtakes(df)
    if result.empty:
        return []
    if driver:
        result = result[result["driver_id"] == driver.upper()]
    return [OvertakeEvent(**r) for r in result.to_dict(orient="records")]


@router.get("/{year}/{gp}/safety-car-laps", response_model=List[SafetyCarLap])
def get_safety_car_laps(
    year: int,
    gp: str,
    threshold: float = Query(1.35, description="Multiplier above median to flag SC lap"),
):
    """Detect likely safety car / VSC laps from lap time spikes."""
    df = _get_lap_df(year, gp)
    result = detect_safety_car_laps(df, sc_threshold_multiplier=threshold)
    return [SafetyCarLap(**r) for r in result.to_dict(orient="records")]


@router.get("/{year}/{gp}/lap-delta")
def get_lap_delta(
    year: int,
    gp: str,
    driver_a: str = Query(..., description="First driver abbreviation"),
    driver_b: str = Query(..., description="Second driver abbreviation"),
):
    """Return lap-by-lap time delta between two drivers. Positive = A is slower."""
    df = _get_lap_df(year, gp)
    result = lap_time_delta(df, driver_a.upper(), driver_b.upper())
    if result.empty:
        raise HTTPException(status_code=404, detail="Not enough data for both drivers.")
    return result.to_dict(orient="records")


@router.get("/{year}/{gp}/position-changes")
def get_position_changes(year: int, gp: str):
    """Return net position gain/loss per driver over the race."""
    df = _get_lap_df(year, gp)
    result = position_change_summary(df)
    if result.empty:
        raise HTTPException(status_code=404, detail="Position data not available.")
    return result.to_dict(orient="records")


# ── Race outcome predictor ─────────────────────────────────────────────────────

@router.post("/{year}/{gp}/train-predictor")
def train_predictor(year: int, gp: str):
    """
    Train the race outcome predictor on this race's lap data.
    Call this once after a race finishes.
    """
    import math
    key = f"{year}|{gp}"
    session = _get_session(year, gp)
    lap_df = _get_lap_df(year, gp)

    # Build race_results_df: driver_id, grid_position, final_position, team, points
    try:
        res = session.results.copy()

        def _int_or_none(val):
            try:
                f = float(val)
                return int(f) if not math.isnan(f) else None
            except Exception:
                return None

        race_results_df = pd.DataFrame({
            "driver_id": res["Abbreviation"].astype(str).str.strip(),
            "grid_position": res["GridPosition"].apply(_int_or_none),
            "final_position": res["Position"].apply(_int_or_none),
            "team": res.get("TeamName", pd.Series(["Unknown"] * len(res))).astype(str),
            "points": res.get("Points", pd.Series([0] * len(res))).apply(
                lambda v: float(v) if v == v else 0.0
            ),
        })
        race_results_df = race_results_df.dropna(subset=["driver_id", "final_position"])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not build race results: {e}")

    # Build pit stop DataFrame
    try:
        pit_df = pit_stop_summary(lap_df)
        if pit_df.empty:
            pit_df = None
    except Exception:
        pit_df = None

    # Build feature matrix and train
    try:
        features_df = _build_features(race_results_df, lap_df, pit_df)
        predictor = RaceOutcomePredictor()
        predictor.fit(features_df)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    _predictors[key] = predictor
    model_path = MODEL_DIR / f"predictor_{year}_{gp.replace(' ', '_')}.joblib"
    predictor.save(str(model_path))
    return {"status": "trained", "race": gp, "year": year, "n_drivers": len(features_df)}


@router.get("/{year}/{gp}/predict-outcome", response_model=List[RacePrediction])
def predict_outcome(year: int, gp: str):
    """
    Return predicted final positions using the trained predictor.
    Call /train-predictor first.
    """
    import math
    key = f"{year}|{gp}"
    if key not in _predictors:
        model_path = MODEL_DIR / f"predictor_{year}_{gp.replace(' ', '_')}.joblib"
        if model_path.exists():
            p = RaceOutcomePredictor()
            p.load(str(model_path))
            _predictors[key] = p
        else:
            raise HTTPException(
                status_code=400,
                detail="Predictor not trained yet. POST to /train-predictor first.",
            )

    session = _get_session(year, gp)
    lap_df = _get_lap_df(year, gp)

    # Build race_results_df for feature engineering
    try:
        res = session.results.copy()

        def _int_or_none(val):
            try:
                f = float(val)
                return int(f) if not math.isnan(f) else None
            except Exception:
                return None

        race_results_df = pd.DataFrame({
            "driver_id": res["Abbreviation"].astype(str).str.strip(),
            "grid_position": res["GridPosition"].apply(_int_or_none),
            "final_position": res["Position"].apply(_int_or_none),
            "team": res.get("TeamName", pd.Series(["Unknown"] * len(res))).astype(str),
            "points": res.get("Points", pd.Series([0] * len(res))).apply(
                lambda v: float(v) if v == v else 0.0
            ),
        })
        race_results_df = race_results_df.dropna(subset=["driver_id"])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not build race results: {e}")

    try:
        pit_df = pit_stop_summary(lap_df)
        if pit_df.empty:
            pit_df = None
    except Exception:
        pit_df = None

    features_df = _build_features(race_results_df, lap_df, pit_df)
    predictor = _predictors[key]
    preds = predictor.predict(features_df)

    return [
        RacePrediction(
            driver_id=str(features_df.iloc[i]["driver_id"]),
            predicted_position=int(preds.iloc[i]),
        )
        for i in range(len(features_df))
    ]
