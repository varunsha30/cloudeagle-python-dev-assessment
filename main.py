"""
Application entrypoint.

Run locally:
  uvicorn main:app --reload

Run in production:
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import get_settings
from app.api.routes import router
from app.tools.countries_api import close_client

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown resources."""
    # Startup — nothing to init eagerly (graph is lazy-loaded on first request)
    yield
    # Shutdown — close the shared HTTP client cleanly
    await close_client()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI agent that answers questions about countries using live data.",
    lifespan=lifespan,
)

app.include_router(router)