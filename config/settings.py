from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "postgresql://f1user:f1pass@localhost:5432/f1_analytics"
    fastf1_cache_dir: str = "./cache/fastf1"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501
    openf1_base_url: str = "https://api.openf1.org/v1"
    ergast_base_url: str = "http://ergast.com/api/f1"
    live_refresh_seconds: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure FastF1 cache directory exists
Path(settings.fastf1_cache_dir).mkdir(parents=True, exist_ok=True)
