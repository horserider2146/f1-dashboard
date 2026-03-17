-- F1 Analytics Database Schema

CREATE TABLE IF NOT EXISTS drivers (
    driver_id   VARCHAR(10) PRIMARY KEY,
    driver_name VARCHAR(100) NOT NULL,
    team        VARCHAR(100),
    nationality VARCHAR(60),
    team_color  VARCHAR(7) DEFAULT '#FFFFFF'
);

CREATE TABLE IF NOT EXISTS races (
    race_id     SERIAL PRIMARY KEY,
    race_name   VARCHAR(150) NOT NULL,
    track       VARCHAR(150),
    season      INTEGER NOT NULL,
    date        DATE,
    total_laps  INTEGER,
    round       INTEGER
);

CREATE TABLE IF NOT EXISTS lap_data (
    id          BIGSERIAL PRIMARY KEY,
    race_id     INTEGER REFERENCES races(race_id) ON DELETE CASCADE,
    driver_id   VARCHAR(10) REFERENCES drivers(driver_id),
    lap_number  INTEGER,
    lap_time_s  REAL,
    sector1_s   REAL,
    sector2_s   REAL,
    sector3_s   REAL,
    compound    VARCHAR(20),
    tyre_age    INTEGER,
    speed_avg   REAL,
    position    INTEGER,
    is_pit_lap  BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS telemetry (
    id          BIGSERIAL PRIMARY KEY,
    race_id     INTEGER REFERENCES races(race_id) ON DELETE CASCADE,
    driver_id   VARCHAR(10) REFERENCES drivers(driver_id),
    lap_number  INTEGER,
    ts          REAL,
    speed       REAL,
    throttle    REAL,
    brake       BOOLEAN,
    gear        INTEGER,
    rpm         REAL,
    x           REAL,
    y           REAL,
    drs         INTEGER
);

CREATE TABLE IF NOT EXISTS pit_stops (
    id          SERIAL PRIMARY KEY,
    race_id     INTEGER REFERENCES races(race_id) ON DELETE CASCADE,
    driver_id   VARCHAR(10) REFERENCES drivers(driver_id),
    lap         INTEGER,
    stop_number INTEGER,
    duration_s  REAL,
    compound_in VARCHAR(20),
    compound_out VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS race_results (
    id              SERIAL PRIMARY KEY,
    race_id         INTEGER REFERENCES races(race_id) ON DELETE CASCADE,
    driver_id       VARCHAR(10) REFERENCES drivers(driver_id),
    final_position  INTEGER,
    grid_position   INTEGER,
    points          REAL,
    status          VARCHAR(50),
    fastest_lap     BOOLEAN DEFAULT FALSE
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_lap_race_driver ON lap_data(race_id, driver_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_race_driver ON telemetry(race_id, driver_id);
CREATE INDEX IF NOT EXISTS idx_races_season ON races(season);
