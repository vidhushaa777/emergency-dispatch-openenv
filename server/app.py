"""
server/app.py — OpenEnv entry point (required by openenv validate).
Re-exports the FastAPI app from app/main.py.
"""
from app.main import app  # noqa: F401

__all__ = ["app"]
