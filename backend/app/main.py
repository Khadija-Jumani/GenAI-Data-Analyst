"""
backend/app/main.py
--------------------
FastAPI application entry point for the GenAI Data Analyst backend.

Run locally with:
    cd backend
    uvicorn app.main:app --reload --port 8000

Interactive API docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI instance with:
    - Metadata (title, version, description)
    - CORS middleware (allow frontend origin)
    - API router mounted at /api/v1
    - Root health-check endpoint
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Backend API for the GenAI Data Analyst platform. "
            "Provides dataset upload, automated EDA, visualisation, "
            "and Generative AI–powered analytics."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.DEBUG,
    )

    # ------------------------------------------------------------------ #
    # CORS Middleware                                                       #
    # ------------------------------------------------------------------ #
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # API Routes                                                           #
    # ------------------------------------------------------------------ #
    application.include_router(api_router, prefix=settings.API_PREFIX)

    # ------------------------------------------------------------------ #
    # Root endpoint — health check                                         #
    # ------------------------------------------------------------------ #
    @application.get("/", tags=["Health"], summary="Health check")
    def health_check() -> dict[str, str]:
        """Confirm the API is running and return basic metadata."""
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }

    return application


app = create_application()

