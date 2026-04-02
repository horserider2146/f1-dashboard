"""
Pydantic schemas for all API request/response models.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── Common ─────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    message: str = ""


# ── Race / Season ──────────────────────────────────────────────────────────────

class RaceEvent(BaseModel):
    round: int
    event_name: str
    location: str
    date: str


class RaceMetadata(BaseModel):
    race_name: str
    track: str
    season: int
    date: str
    total_laps: Optional[int] = None
    round: int


# ── Driver ─────────────────────────────────────────────────────────────────────

class DriverInfo(BaseModel):
    driver_id: str
    driver_name: str
    team: str
    team_color: Optional[str] = None


# ── Lap data ───────────────────────────────────────────────────────────────────

class LapRecord(BaseModel):
    driver_id: str
    lap_number: int
    lap_time_s: Optional[float] = None
    sector1_s: Optional[float] = None
    sector2_s: Optional[float] = None
    sector3_s: Optional[float] = None
    compound: Optional[str] = None
    tyre_age: Optional[int] = None
    speed_avg: Optional[float] = None
    position: Optional[int] = None
    is_pit_lap: bool = False


class FastestLap(BaseModel):
    driver_id: str
    lap_number: int
    lap_time_s: float
    compound: Optional[str] = None


# ── Telemetry ──────────────────────────────────────────────────────────────────

class TelemetryPoint(BaseModel):
    driver_id: str
    lap_number: int
    ts: float
    speed: Optional[float] = None
    throttle: Optional[float] = None
    brake: Optional[bool] = None
    gear: Optional[int] = None
    rpm: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    drs: Optional[int] = None


# ── Analytics ──────────────────────────────────────────────────────────────────

class TyreStint(BaseModel):
    driver_id: str
    stint_number: int
    compound: str
    start_lap: int
    end_lap: int
    laps_on_tyre: int


class PitStopEvent(BaseModel):
    driver_id: str
    lap: int
    old_compound: str
    new_compound: str
    stop_number: int


class UndercutEvent(BaseModel):
    undercut_driver: str
    target_driver: str
    pit_lap_undercut: int
    pit_lap_target: int
    pit_gap_laps: int
    avg_time_undercut: float
    avg_time_target: float
    time_gain_s: float
    success: bool


class OvertakeEvent(BaseModel):
    lap_number: int
    driver_id: str
    position_before: int
    position_after: int
    positions_gained: int


class SafetyCarLap(BaseModel):
    lap_number: int
    avg_lap_time: float
    median_lap_time: float
    sc_candidate: bool


class TyreDegPrediction(BaseModel):
    driver_id: str
    compound: str
    lap_number: int
    tyre_age: int
    predicted_lap_time: float


class StrategyComparison(BaseModel):
    driver_id: str
    strategy: str
    num_stops: int


class CompoundSummary(BaseModel):
    compound: str
    mean_lap_time_s: float
    deg_rate_s_per_lap: float
    stint_count: int


class RacePrediction(BaseModel):
    driver_id: str
    predicted_position: int
    confidence: Optional[float] = None
