"""
FastAPI application entry point.
Run with:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import races, telemetry, analytics, stats
from database.connection import check_connection

app = FastAPI(
    title="F1 Analytics API",
    description="Formula 1 race data, telemetry and analytics.",
    version="1.0.0",
)

# Allow Streamlit (and any other frontend) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(races.router)
app.include_router(telemetry.router)
app.include_router(analytics.router)
app.include_router(stats.router)


@app.get("/", tags=["health"])
def root():
    return {"message": "F1 Analytics API is running."}


@app.get("/health", tags=["health"])
def health():
    db_ok = check_connection()
    return {
        "api": "ok",
        "database": "ok" if db_ok else "unreachable",
    }
