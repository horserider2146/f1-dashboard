"""
Ergast API client — historical race results, standings, and qualifying.
Note: Ergast is read-only and rate-limited; responses are cached locally.
"""
import requests
import pandas as pd
import functools
from config.settings import settings

BASE = settings.ergast_base_url


def _get_json(path: str, limit: int = 1000, offset: int = 0) -> dict:
    """Fetch a single Ergast endpoint page and return parsed JSON."""
    url = f"{BASE}/{path}.json"
    params = {"limit": limit, "offset": offset}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["MRData"]


def _paginate(path: str, result_key: str, inner_key: str) -> list[dict]:
    """Fetch all pages for a paginated Ergast endpoint."""
    limit = 100
    offset = 0
    results = []
    while True:
        data = _get_json(path, limit=limit, offset=offset)
        items = data.get(result_key, {}).get(inner_key, [])
        results.extend(items)
        total = int(data.get("total", 0))
        offset += limit
        if offset >= total:
            break
    return results


# ── Season / Race schedule ─────────────────────────────────────────────────────

def get_race_schedule(year: int) -> pd.DataFrame:
    """All races for a season with round, circuit, and date."""
    races = _paginate(f"{year}", "RaceTable", "Races")
    rows = []
    for r in races:
        rows.append({
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "track": r["Circuit"]["circuitName"],
            "location": r["Circuit"]["Location"]["locality"],
            "country": r["Circuit"]["Location"]["country"],
            "date": r.get("date"),
            "season": year,
        })
    return pd.DataFrame(rows)


# ── Race Results ───────────────────────────────────────────────────────────────

def get_race_results(year: int, round_number: int) -> pd.DataFrame:
    """Final classification for a specific race."""
    data = _get_json(f"{year}/{round_number}/results")
    races = data.get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    rows = []
    for result in races[0].get("Results", []):
        driver = result["Driver"]
        rows.append({
            "driver_id": driver["code"],
            "driver_name": f"{driver['givenName']} {driver['familyName']}",
            "team": result["Constructor"]["name"],
            "grid_position": int(result.get("grid", 0)),
            "final_position": int(result["position"]),
            "points": float(result.get("points", 0)),
            "status": result.get("status", ""),
            "fastest_lap": result.get("FastestLap", {}).get("rank") == "1",
        })
    return pd.DataFrame(rows)


def get_all_results_for_season(year: int) -> pd.DataFrame:
    """Race results for every round in a season."""
    schedule = get_race_schedule(year)
    frames = []
    for _, row in schedule.iterrows():
        df = get_race_results(year, int(row["round"]))
        df["round"] = row["round"]
        df["race_name"] = row["race_name"]
        df["season"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Qualifying ─────────────────────────────────────────────────────────────────

def get_qualifying_results(year: int, round_number: int) -> pd.DataFrame:
    data = _get_json(f"{year}/{round_number}/qualifying")
    races = data.get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    rows = []
    for result in races[0].get("QualifyingResults", []):
        driver = result["Driver"]
        rows.append({
            "driver_id": driver["code"],
            "driver_name": f"{driver['givenName']} {driver['familyName']}",
            "team": result["Constructor"]["name"],
            "position": int(result["position"]),
            "q1": result.get("Q1"),
            "q2": result.get("Q2"),
            "q3": result.get("Q3"),
        })
    return pd.DataFrame(rows)


# ── Standings ──────────────────────────────────────────────────────────────────

def get_driver_standings(year: int, round_number: int | None = None) -> pd.DataFrame:
    path = f"{year}/driverStandings" if round_number is None else f"{year}/{round_number}/driverStandings"
    data = _get_json(path)
    standings_list = data.get("StandingsTable", {}).get("StandingsLists", [])
    if not standings_list:
        return pd.DataFrame()

    rows = []
    for entry in standings_list[0].get("DriverStandings", []):
        driver = entry["Driver"]
        rows.append({
            "position": int(entry["position"]),
            "driver_id": driver["code"],
            "driver_name": f"{driver['givenName']} {driver['familyName']}",
            "team": entry["Constructors"][0]["name"] if entry["Constructors"] else "",
            "points": float(entry["points"]),
            "wins": int(entry["wins"]),
        })
    return pd.DataFrame(rows)


def get_constructor_standings(year: int, round_number: int | None = None) -> pd.DataFrame:
    path = (f"{year}/constructorStandings" if round_number is None
            else f"{year}/{round_number}/constructorStandings")
    data = _get_json(path)
    standings_list = data.get("StandingsTable", {}).get("StandingsLists", [])
    if not standings_list:
        return pd.DataFrame()

    rows = []
    for entry in standings_list[0].get("ConstructorStandings", []):
        rows.append({
            "position": int(entry["position"]),
            "team": entry["Constructor"]["name"],
            "nationality": entry["Constructor"].get("nationality", ""),
            "points": float(entry["points"]),
            "wins": int(entry["wins"]),
        })
    return pd.DataFrame(rows)


# ── Pit stops ─────────────────────────────────────────────────────────────────

def get_pit_stops(year: int, round_number: int) -> pd.DataFrame:
    data = _get_json(f"{year}/{round_number}/pitstops")
    races = data.get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    rows = []
    for stop in races[0].get("PitStops", []):
        rows.append({
            "driver_id": stop["driverId"].upper()[:3],
            "stop_number": int(stop["stop"]),
            "lap": int(stop["lap"]),
            "time_of_day": stop.get("time"),
            "duration_s": float(stop["duration"]) if stop.get("duration") else None,
        })
    return pd.DataFrame(rows)


# ── Lap times ─────────────────────────────────────────────────────────────────

def get_lap_times(year: int, round_number: int, driver_id: str | None = None) -> pd.DataFrame:
    """Lap-by-lap times from Ergast (less granular than FastF1 but always available)."""
    path = f"{year}/{round_number}/laps"
    if driver_id:
        path = f"{year}/{round_number}/drivers/{driver_id}/laps"
    laps = _paginate(path, "RaceTable", "Races")
    rows = []
    for race in laps:
        for lap in race.get("Laps", []):
            lap_number = int(lap["number"])
            for timing in lap.get("Timings", []):
                rows.append({
                    "driver_id": timing["driverId"].upper()[:3],
                    "lap_number": lap_number,
                    "position": int(timing["position"]),
                    "lap_time": timing.get("time"),
                })
    return pd.DataFrame(rows)
