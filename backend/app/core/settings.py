"""12-factor configuration: everything comes from environment variables with safe local defaults."""
from __future__ import annotations
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'app.db'}"))
    secret_key: str = field(default_factory=lambda: os.environ.get("SECRET_KEY") or secrets.token_urlsafe(32))
    token_minutes: int = field(default_factory=lambda: int(os.environ.get("TOKEN_MINUTES", "720")))
    cors_origins: tuple = field(default_factory=lambda: tuple(o for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",") if o))
    job_workers: int = field(default_factory=lambda: int(os.environ.get("JOB_WORKERS", "2")))
    frontend_dist: Path = field(default_factory=lambda: Path(os.environ.get("FRONTEND_DIST", ROOT / "frontend" / "dist")))
    auth_rate_limit: int = field(default_factory=lambda: int(os.environ.get("AUTH_RATE_LIMIT", "20")))   # attempts / minute / IP
    env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "dev"))


def get_settings() -> Settings:
    return Settings()


SETTINGS = get_settings()
if SETTINGS.env == "prod" and not os.environ.get("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be set when APP_ENV=prod (tokens would be invalid after every restart).")
