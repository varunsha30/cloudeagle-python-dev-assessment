"""
Central settings — all runtime config lives here.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Gemini
    google_api_key: str  = Field(...)
    gemini_model: str    = Field("gemini-2.5-flash")

    # REST countries API
    rest_countries_base_url: str = "https://restcountries.com/v3.1"
    http_timeout_seconds: int    = Field(10)
    max_retries: int             = Field(2)

    # Service
    app_name: str = "Country Information AI Agent"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()