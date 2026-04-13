"""
Statistical Analysis Router — 18 endpoints covering all 7 syllabus units.
All responses pass through _jsonify() which converts numpy types to native Python.
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json
import math
from pathlib import Path
import numpy as np

from config.settings import settings
from data.fastf1_loader import load_session, get_lap_data, get_driver_info

router = APIRouter(prefix="/stats", tags=["statistics"])


# ── Numpy-safe JSON serialiser ────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _j(data) -> JSONResponse:
    """Serialise any dict/list, replacing numpy types, NaN and Inf with JSON-safe values."""
    # First pass: convert numpy types; replace NaN/Inf tokens that json.dumps emits
    raw = json.dumps(data, cls=_NumpyEncoder, allow_nan=True)
    raw = raw.replace(": NaN", ": null").replace(": Infinity", ": null").replace(": -Infinity", ": null")
    return JSONResponse(content=json.loads(raw))


# ── Shared data helpers (return plain Python — NOT JSONResponse) ──────────────

_session_cache: dict = {}


def _get_session(year: int, gp: str, session_type: str = "R"):
    key = (year, gp, session_type)
    if key not in _session_cache:
        session = load_session(year, gp, session_type)
        try:
            _ = session.laps  # verify timing data actually loaded
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Lap data not available for {gp} {year}. "
                       f"This race is not cached — only 2025 races are available offline."
            )
        _session_cache[key] = session
    return _session_cache[key]


def _get_laps(year: int, gp: str) -> list[dict]:
    import pandas as pd
    session = _get_session(year, gp)
    laps = get_lap_data(session)
    if laps is None or laps.empty:
        return []

    if "team" not in laps.columns:
        try:
            res = session.results[["Abbreviation", "TeamName"]].copy()
            res.columns = ["driver_id", "team"]
            laps = laps.merge(res, on="driver_id", how="left")
        except Exception:
            laps["team"] = "Unknown"

    cols = [c for c in ["driver_id", "lap_number", "lap_time_s",
                         "compound", "team", "tyre_age"] if c in laps.columns]
    return laps[cols].dropna(subset=["lap_time_s"]).to_dict(orient="records")


def _get_pit_stops(year: int, gp: str) -> list[dict]:
    import pandas as pd
    try:
        from analytics.strategy import pit_stop_summary
        laps_df = get_lap_data(_get_session(year, gp))
        stops = pit_stop_summary(laps_df)
        if stops.empty:
            return []
        if "pit_duration" not in stops.columns:
            try:
                # Build lap_time lookup: (driver_id, lap_number) -> lap_time_s
                lap_lookup = {}
                for _, r in laps_df.iterrows():
                    try:
                        lap_num = r["lap_number"]
                        lt = r.get("lap_time_s")
                        if lt is not None and not (isinstance(lt, float) and math.isnan(lt)) \
                                and lap_num is not None and str(lap_num) != "<NA>":
                            lap_lookup[(str(r["driver_id"]), int(lap_num))] = float(lt)
                    except (TypeError, ValueError):
                        continue
                # Driver median lap time
                driver_median = {}
                for drv, grp in laps_df.groupby("driver_id"):
                    times = grp["lap_time_s"].dropna()
                    if len(times) > 0:
                        driver_median[str(drv)] = float(times.median())
                # Assign pit_duration as pit lap time minus driver median
                pit_durations = []
                for _, row in stops.iterrows():
                    key = (str(row["driver_id"]), int(row["lap"]))
                    lap_time = lap_lookup.get(key)
                    median = driver_median.get(str(row["driver_id"]))
                    if lap_time is not None and median is not None:
                        pit_durations.append(round(max(0.0, lap_time - median), 3))
                    else:
                        pit_durations.append(None)
                stops = stops.copy()
                stops["pit_duration"] = pit_durations
            except Exception:
                pass
        return stops.to_dict(orient="records")
    except Exception:
        return []


def _get_results(year: int, gp: str) -> list[dict]:
    """Race results from FastF1 session (position + grid_position)."""
    try:
        session = _get_session(year, gp)
        results = session.results
        if results is not None and not results.empty:
            def _int_or_none(val):
                try:
                    f = float(val)
                    return int(f) if not math.isnan(f) else None
                except Exception:
                    return None
            rows = []
            for _, row in results.iterrows():
                drv = str(row.get("Abbreviation", "")).strip()
                if not drv:
                    continue
                rows.append({
                    "driver_id": drv,
                    "position": _int_or_none(row.get("Position")),
                    "grid_position": _int_or_none(row.get("GridPosition")),
                    "team": str(row.get("TeamName", "")),
                })
            if rows:
                return rows
    except Exception:
        pass
    try:
        from data.ergast_client import get_race_results
        return get_race_results(year, gp) or []
    except Exception:
        return []


def _get_cached_gps(year: int) -> list[str]:
    """Return GP names available in the local FastF1 cache for a season."""
    year_dir = Path(settings.fastf1_cache_dir) / str(year)
    if not year_dir.exists():
        return []

    gps: list[str] = []
    for child in sorted(year_dir.iterdir()):
        if not child.is_dir():
            continue
        parts = child.name.split("_", 1)
        if len(parts) != 2:
            continue
        gps.append(parts[1].replace("_", " "))
    return gps


# ── Unit Novel: Driver Consistency Index ──────────────────────────────────────

@router.get("/{year}/{gp}/dci")
def dci(year: int, gp: str):
    from analytics.stats.dci import compute_dci
    laps = _get_laps(year, gp)
    if not laps:
        raise HTTPException(404, "No lap data.")
    return _j(compute_dci(laps))


@router.get("/{year}/{gp}/dci/correlation")
def dci_correlation(year: int, gp: str):
    from analytics.stats.dci import compute_dci, dci_championship_correlation
    laps = _get_laps(year, gp)
    if not laps:
        raise HTTPException(404, "No lap data.")
    dci_results = compute_dci(laps)
    try:
        from data.ergast_client import get_driver_standings
        standings = get_driver_standings(year) or []
    except Exception:
        standings = []
    return _j(dci_championship_correlation(dci_results, standings))


# ── Unit 1 ─────────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/mle/{driver_id}")
def mle_distribution(year: int, gp: str, driver_id: str):
    from analytics.stats.inference import mle_lap_distribution
    laps = _get_laps(year, gp)
    result = mle_lap_distribution(laps, driver_id.upper())
    if not result:
        raise HTTPException(404, f"Not enough lap data for {driver_id}.")
    return _j(result)


@router.get("/{year}/{gp}/bayes-win")
def bayes_win(year: int, gp: str):
    from analytics.stats.inference import bayesian_win_probability
    laps = _get_laps(year, gp)
    results = _get_results(year, gp)
    return _j(bayesian_win_probability(laps, results))


@router.get("/{year}/{gp}/bayes-win-season")
def bayes_win_season(year: int, gp: str):
    from analytics.stats.inference import bayesian_win_probability
    season_rows: list[dict] = []

    try:
        from data.ergast_client import get_all_results_for_season
        season_results = get_all_results_for_season(year)
        if season_results is not None and not season_results.empty:
            season_rows = season_results.to_dict(orient="records")
    except Exception:
        season_rows = []

    if not season_rows:
        cached_rows: list[dict] = []
        for cached_gp in _get_cached_gps(year):
            cached_rows.extend(_get_results(year, cached_gp))
        season_rows = cached_rows

    return _j(bayesian_win_probability([], season_rows))


# ── Unit 2 ─────────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/ttest")
def ttest(year: int, gp: str,
          driver_a: str = Query(...),
          driver_b: str = Query(...),
          alternative: str = Query("two-sided", regex="^(two-sided|less|greater)$"),
          alpha: float = Query(0.05, gt=0.0, lt=1.0)):
    from analytics.stats.inference import two_sample_ttest
    laps = _get_laps(year, gp)
    result = two_sample_ttest(
        laps,
        driver_a.upper(),
        driver_b.upper(),
        alternative=alternative,
        alpha=alpha,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


@router.get("/{year}/{gp}/ztest/{driver_id}")
def ztest(year: int, gp: str, driver_id: str,
          alternative: str = Query("two-sided", regex="^(two-sided|less|greater)$"),
          alpha: float = Query(0.05, gt=0.0, lt=1.0)):
    from analytics.stats.inference import z_test_pit_stop_time
    pit_stops = _get_pit_stops(year, gp)
    if not pit_stops or not any("pit_duration" in s and s["pit_duration"] is not None
                                for s in pit_stops):
        return _j({
            "available": False,
            "message": "Pit duration data cannot be estimated for this race.",
        })
    result = z_test_pit_stop_time(
        pit_stops,
        driver_id.upper(),
        alternative=alternative,
        alpha=alpha,
    )
    if not result:
        return _j({"available": False, "message": "Not enough pit stop data."})
    if "error" in result:
        return _j({"available": False, "message": result["error"]})
    result["available"] = True
    return _j(result)


# ── Unit 3 ─────────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/anova/one-way")
def anova_one_way(year: int, gp: str,
                   group: str = Query("team")):
    from analytics.stats.anova import one_way_anova
    laps = _get_laps(year, gp)
    result = one_way_anova(laps, group_col=group)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


@router.get("/{year}/{gp}/anova/two-way")
def anova_two_way(year: int, gp: str):
    from analytics.stats.anova import two_way_anova
    laps = _get_laps(year, gp)
    result = two_way_anova(laps)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


# ── Units 4–5 ──────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/regression/ols")
def regression_ols(year: int, gp: str,
                   target: str = Query("position", regex="^(position|avg_lap_time)$")):
    from analytics.stats.regression import ols_regression
    laps = _get_laps(year, gp)
    pit_stops = _get_pit_stops(year, gp)
    results = _get_results(year, gp)
    result = ols_regression(laps, pit_stops, results, target=target)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


@router.get("/{year}/{gp}/regression/regularised")
def regression_regularised(year: int, gp: str,
                          target: str = Query("position", regex="^(position|avg_lap_time)$")):
    from analytics.stats.regression import lasso_ridge_regression
    laps = _get_laps(year, gp)
    pit_stops = _get_pit_stops(year, gp)
    results = _get_results(year, gp)
    result = lasso_ridge_regression(laps, pit_stops, results, target=target)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


@router.get("/{year}/{gp}/correlation-matrix")
def correlation_matrix(year: int, gp: str):
    from analytics.stats.regression import correlation_matrix as cm
    laps = _get_laps(year, gp)
    pit_stops = _get_pit_stops(year, gp)
    results = _get_results(year, gp)
    result = cm(laps, pit_stops, results)
    if not result or "error" in result:
        raise HTTPException(400, (result or {}).get("error", "Insufficient data."))
    return _j(result)


# ── Unit 6 ─────────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/logistic")
def logistic(year: int, gp: str):
    from analytics.stats.logistic import logistic_regression
    laps = _get_laps(year, gp)
    pit_stops = _get_pit_stops(year, gp)
    results = _get_results(year, gp)
    result = logistic_regression(laps, pit_stops, results)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


@router.get("/{year}/{gp}/model-comparison")
def model_comparison(year: int, gp: str):
    from analytics.stats.logistic import compare_models
    from analytics.predictor import RaceOutcomePredictor
    laps = _get_laps(year, gp)
    pit_stops = _get_pit_stops(year, gp)
    results = _get_results(year, gp)
    xgb_preds = None
    result = compare_models(laps, pit_stops, results, xgb_preds)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


# ── Unit 7 ─────────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/wilcoxon/{driver_id}")
def wilcoxon(year: int, gp: str, driver_id: str):
    from analytics.stats.nonparametric import wilcoxon_before_after_sc
    from analytics.events import detect_safety_car_laps
    import pandas as pd
    laps = _get_laps(year, gp)
    try:
        laps_df = pd.DataFrame(laps)
        sc_df = detect_safety_car_laps(laps_df)
        sc_laps = sc_df[sc_df["sc_candidate"]]["lap_number"].tolist()
    except Exception:
        sc_laps = []
    if not sc_laps:
        return _j({
            "test": "Wilcoxon Signed-Rank",
            "driver_id": driver_id.upper(),
            "available": False,
            "message": "No safety car laps in this race. Try Monaco or Bahrain.",
        })
    result = wilcoxon_before_after_sc(laps, sc_laps, driver_id.upper())
    if "error" in result:
        return _j({
            "test": "Wilcoxon Signed-Rank",
            "driver_id": driver_id.upper(),
            "available": False,
            "message": result["error"],
        })
    result["available"] = True
    return _j(result)


@router.get("/{year}/{gp}/mann-whitney")
def mann_whitney(year: int, gp: str,
                  group_a: str = Query(...),
                  group_b: str = Query(...),
                  group_col: str = Query("team")):
    from analytics.stats.nonparametric import mann_whitney_teams
    laps = _get_laps(year, gp)
    result = mann_whitney_teams(laps, group_a, group_b, group_col)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


@router.get("/{year}/{gp}/friedman")
def friedman(year: int, gp: str,
             drivers: str = Query(...)):
    from analytics.stats.nonparametric import friedman_test
    laps = _get_laps(year, gp)
    drv_list = [d.strip().upper() for d in drivers.split(",")]
    filtered = [l for l in laps if l.get("driver_id") in drv_list]
    if not filtered:
        raise HTTPException(404, "No laps found for specified drivers.")
    result = friedman_test(filtered)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return _j(result)


# ── Summary ────────────────────────────────────────────────────────────────────

@router.get("/{year}/{gp}/summary")
def stats_summary(year: int, gp: str):
    from analytics.stats.dci import compute_dci
    from analytics.stats.anova import one_way_anova
    laps = _get_laps(year, gp)
    if not laps:
        raise HTTPException(404, "No lap data.")
    dci_results = compute_dci(laps)
    anova_result = one_way_anova(laps, group_col="driver_id")
    return _j({
        "dci_top5": dci_results[:5],
        "anova_significant": anova_result.get("significant", False),
        "anova_p_value": anova_result.get("p_value"),
        "n_drivers": len(dci_results),
    })
