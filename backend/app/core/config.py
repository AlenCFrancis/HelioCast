import os
from pathlib import Path
from typing import List, Union
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


def find_project_root() -> Path:
    """Locate the root directory containing configs/ and models/."""
    # Check environment variable first
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root).resolve()

    # Traverse upward from this file
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists() and (parent / "models").exists():
            return parent
        if (parent / "models" / "scaler.joblib").exists():
            return parent

    # Fallback to grandparent of backend (standard layout)
    return Path(__file__).resolve().parent.parent.parent.parent


ROOT_DIR = find_project_root()


class Settings(BaseSettings):
    PROJECT_NAME: str = "Solar Flare Forecasting API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Path settings
    PROJECT_ROOT: Path = ROOT_DIR
    CONFIG_PATH: Path = ROOT_DIR / "configs" / "config.yaml"
    DATA_PATH: Path = ROOT_DIR / "data" / "processed" / "solar_features.csv"
    MODELS_DIR: Path = ROOT_DIR / "models"
    METRICS_DIR: Path = ROOT_DIR / "results" / "metrics"
    
    # ML settings
    HISTORY_HOURS: int = 24
    FORECAST_HOURS: int = 24
    
    # CORS Origins (comma-separated or JSON list via env, or default list)
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"


settings = Settings()
