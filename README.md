#  F1 Analytics Dashboard

A full-stack Formula 1 data analytics platform built with **FastAPI**, **Streamlit**, and **FastF1**. Provides race telemetry, live timing, strategy analysis, machine learning predictions, and university-level statistical analysis — all in an interactive web dashboard.

---

##  Features at a Glance

| Page | What it does |
|---|---|
|  **Race Data** | Lap times, position history, fastest laps |
|  **Telemetry** | Speed traces, track maps, multi-driver animation |
|  **Live Timing** | Real-time leaderboard, pit stops, track status |
|  **Analytics** | Tyre strategy, degradation model, undercuts, ML predictor |
|  **Statistical Analysis** | MLE, ANOVA, regression, logistic regression, nonparametric tests |

---

##  Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Data:** FastF1 (official F1 timing data, 2018–present)
- **Machine Learning:** XGBoost, scikit-learn (Ridge, Lasso, Logistic Regression)
- **Statistics:** scipy.stats, statsmodels
- **Visualisation:** Plotly

---

##  Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/f1-dashboard.git
cd f1-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

##  Running the Application

You need **two terminal windows** open at the same time.

### Terminal 1 — API Backend
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Wait for: `Application startup complete.`

### Terminal 2 — Dashboard
```bash
streamlit run dashboard/app.py --server.port 8502
```

Then open your browser:
- **Dashboard:** http://localhost:8502
- **API Docs:** http://localhost:8000/docs

### Stopping
Press `Ctrl+C` in each terminal window.

If a port is stuck:
```bash
taskkill /F /IM python.exe   # Windows
kill $(lsof -t -i:8000)      # Mac/Linux
```

---

##  Data & Caching

Race data is loaded via the **FastF1** library from official F1 timing servers.

- **First load** of any race: 30–90 seconds (downloads and caches to disk)
- **Subsequent loads:** Instant (served from local cache at `cache/fastf1/`)
- **Supported years:** 2018 onwards

> The `cache/` folder is excluded from git. Each user builds their own local cache on first use.

---

##  Pages

### 🏁 Race Data
Complete race overview — lap times table, fastest laps per driver, and an interactive position history chart showing every position change lap by lap.

###  Telemetry
- **Speed Trace** — speed (km/h) across every meter of a lap
- **Track Map** — GPS circuit layout colour-coded by speed
- **Multi-Driver Animation** — real-time positional animation for multiple drivers across a lap range

###  Live Timing
Real-time data during active F1 sessions via the OpenF1 API — leaderboard, pit stop feed, and track status (SC, VSC, Red Flag).

###  Analytics
| Feature | Description |
|---|---|
| Tyre Stints | Compound, lap range, and stint length per driver |
| Strategy Comparison | Human-readable strategy string per driver |
| Pit Stops | All stop events with compound changes |
| Undercut Detection | Identifies undercut/overcut attempts and success |
| Tyre Degradation | Degradation model, stint prediction, optimal pit window |
| Overtake Detection | On-track position changes (excludes pit stop changes) |
| Safety Car Laps | Detects SC/VSC laps from lap time spikes |
| Lap Delta | Head-to-head lap time comparison between two drivers |
| Position Changes | Net grid-to-finish position changes per driver |
| Race Predictor | XGBoost model trained on race data, predicts finishing order |

###  Statistical Analysis
Covers 7 statistical units plus a novel Driver Consistency Index (DCI):

| Unit | Feature |
|---|---|
| Novel | Driver Consistency Index (DCI = 1/Var, normalised 0–1) |
| Unit 1 | MLE lap-time distribution + Bayesian win probability |
| Unit 2 | Welch's t-test (head-to-head) + Z-test (pit stops) |
| Unit 3 | One-way ANOVA + Tukey HSD + Two-way ANOVA (team × compound) |
| Units 4–5 | OLS regression + Ridge/Lasso + Pearson correlation matrix |
| Unit 6 | Logistic regression P(podium) + model comparison vs XGBoost |
| Unit 7 | Wilcoxon signed-rank + Mann-Whitney U + Friedman test |

---

##  Race Outcome Predictor

The XGBoost predictor must be trained once per race:

```
POST http://localhost:8000/analytics/{year}/{gp}/train-predictor
```

The model saves to `cache/models/` automatically and reloads on server restart.

Features used: `grid_position`, `pace_gap_to_leader`, `num_pit_stops`, `compound_diversity`

---

##  Project Structure

```
f1_dashboard/
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   ├── races.py         # Race data endpoints
│   │   ├── telemetry.py     # Telemetry endpoints
│   │   ├── live.py          # Live timing endpoints
│   │   ├── analytics.py     # Strategy & ML endpoints
│   │   └── stats.py         # Statistical analysis endpoints
│   └── schemas.py           # Pydantic models
├── analytics/
│   ├── tyre_model.py        # Tyre degradation model
│   ├── strategy.py          # Stint & strategy analysis
│   ├── events.py            # Overtake & SC detection
│   ├── predictor.py         # XGBoost race predictor
│   └── stats/               # Statistical analysis modules
│       ├── dci.py           # Driver Consistency Index
│       ├── inference.py     # MLE, Bayesian, t-test, z-test
│       ├── anova.py         # One-way & two-way ANOVA
│       ├── regression.py    # OLS, Ridge, Lasso, correlation
│       ├── logistic.py      # Logistic regression
│       └── nonparametric.py # Wilcoxon, Mann-Whitney, Friedman
├── dashboard/
│   ├── app.py               # Streamlit app entry point
│   ├── api_client.py        # HTTP client for API
│   └── views/               # One file per dashboard page
├── data/
│   ├── fastf1_loader.py     # FastF1 session loading
│   └── ergast_client.py     # Ergast API client
├── cache/                   # Auto-generated, gitignored
│   ├── fastf1/              # FastF1 race data cache
│   └── models/              # Trained ML models
└── README.md
```

---

##  API Documentation

All endpoints are documented at **http://localhost:8000/docs** (Swagger UI).

Key endpoint groups:
- `GET /races/{year}/{gp}/...` — Race data
- `GET /telemetry/{year}/{gp}/...` — Telemetry
- `GET /live/...` — Live timing
- `GET /analytics/{year}/{gp}/...` — Strategy & predictions
- `POST /analytics/{year}/{gp}/train-predictor` — Train ML model
- `GET /stats/{year}/{gp}/...` — Statistical analysis

---

##  Documentation

A full user guide is included: **`F1_Dashboard_Guide.docx`**

Covers every feature in detail with interpretation guides for all statistical outputs.

---

##  Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 timing data library
- [OpenF1 API](https://openf1.org) — Live timing data
- [Ergast API](http://ergast.com/mrd/) — Historical F1 results
