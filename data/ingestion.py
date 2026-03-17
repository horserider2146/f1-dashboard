"""
Data ingestion orchestrator.
Loads a race via FastF1, transforms the data, and persists it to PostgreSQL.
"""
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import SessionLocal
from data.fastf1_loader import (
    load_session,
    get_lap_data,
    get_telemetry,
    get_driver_info,
    get_race_metadata,
)


# ── Low-level upsert helpers ───────────────────────────────────────────────────

def _upsert_driver(db: Session, row: dict):
    db.execute(text("""
        INSERT INTO drivers (driver_id, driver_name, team, nationality, team_color)
        VALUES (:driver_id, :driver_name, :team, :nationality, :team_color)
        ON CONFLICT (driver_id) DO UPDATE
            SET driver_name = EXCLUDED.driver_name,
                team        = EXCLUDED.team,
                team_color  = EXCLUDED.team_color
    """), {
        "driver_id": row.get("driver_id", ""),
        "driver_name": row.get("driver_name", ""),
        "team": row.get("team", ""),
        "nationality": row.get("nationality", ""),
        "team_color": row.get("team_color", "#FFFFFF"),
    })


def _insert_race(db: Session, meta: dict) -> int:
    """Insert a race record and return its race_id."""
    result = db.execute(text("""
        INSERT INTO races (race_name, track, season, date, total_laps, round)
        VALUES (:race_name, :track, :season, :date, :total_laps, :round)
        ON CONFLICT DO NOTHING
        RETURNING race_id
    """), meta)
    row = result.fetchone()
    if row:
        return row[0]

    # Already exists — fetch the id
    existing = db.execute(text("""
        SELECT race_id FROM races WHERE season = :season AND round = :round
    """), {"season": meta["season"], "round": meta["round"]}).fetchone()
    return existing[0] if existing else None


def _insert_laps(db: Session, race_id: int, df: pd.DataFrame):
    df = df.copy()
    df["race_id"] = race_id
    # Replace NA with None for SQL compatibility
    df = df.where(pd.notna(df), other=None)

    records = df.to_dict(orient="records")
    if not records:
        return

    db.execute(text("""
        INSERT INTO lap_data
            (race_id, driver_id, lap_number, lap_time_s, sector1_s, sector2_s,
             sector3_s, compound, tyre_age, speed_avg, position, is_pit_lap)
        VALUES
            (:race_id, :driver_id, :lap_number, :lap_time_s, :sector1_s, :sector2_s,
             :sector3_s, :compound, :tyre_age, :speed_avg, :position, :is_pit_lap)
    """), records)


def _insert_telemetry(db: Session, race_id: int, df: pd.DataFrame):
    df = df.copy()
    df["race_id"] = race_id
    df = df.where(pd.notna(df), other=None)

    # Telemetry can be very large — insert in chunks
    chunk_size = 5000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].to_dict(orient="records")
        db.execute(text("""
            INSERT INTO telemetry
                (race_id, driver_id, lap_number, ts, speed, throttle,
                 brake, gear, rpm, x, y, drs)
            VALUES
                (:race_id, :driver_id, :lap_number, :ts, :speed, :throttle,
                 :brake, :gear, :rpm, :x, :y, :drs)
        """), chunk)


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_race(year: int, gp: str, session_type: str = "R",
                include_telemetry: bool = True) -> int:
    """
    Full ingestion pipeline for one race.

    1. Load session via FastF1
    2. Upsert drivers
    3. Insert race metadata → get race_id
    4. Insert lap data
    5. (Optional) Insert telemetry for every driver

    Returns the race_id of the inserted/existing race.
    """
    print(f"[ingest] Loading {year} {gp} ({session_type}) via FastF1…")
    session = load_session(year, gp, session_type)

    meta = get_race_metadata(session)
    driver_df = get_driver_info(session)
    lap_df = get_lap_data(session)

    with SessionLocal() as db:
        # Drivers
        for _, row in driver_df.iterrows():
            _upsert_driver(db, row.to_dict())
        db.flush()

        # Race
        race_id = _insert_race(db, meta)
        if race_id is None:
            print("[ingest] Could not insert/find race record. Aborting.")
            db.rollback()
            return -1

        print(f"[ingest] Race ID = {race_id}")

        # Lap data
        _insert_laps(db, race_id, lap_df)
        print(f"[ingest] Inserted {len(lap_df)} lap rows.")

        # Telemetry
        if include_telemetry:
            drivers = lap_df["driver_id"].dropna().unique().tolist()
            for drv in drivers:
                tel_df = get_telemetry(session, drv)
                if not tel_df.empty:
                    _insert_telemetry(db, race_id, tel_df)
                    print(f"[ingest] Telemetry for {drv}: {len(tel_df)} rows.")

        db.commit()
        print(f"[ingest] Done — race_id={race_id}")
        return race_id


def ingest_season(year: int, include_telemetry: bool = False):
    """Ingest all races for an entire season (telemetry off by default — very large)."""
    from data.fastf1_loader import list_available_races
    schedule = list_available_races(year)
    for _, event in schedule.iterrows():
        gp_name = event["EventName"]
        try:
            ingest_race(year, gp_name, include_telemetry=include_telemetry)
        except Exception as exc:
            print(f"[ingest] Failed for {gp_name}: {exc}")
