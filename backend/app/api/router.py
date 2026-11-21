"""
backend/app/api/router.py
--------------------------
Central API router.

All feature routers are registered here and mounted onto the FastAPI app
in main.py. Add new phase routers by importing and including them below.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import dataset as dataset_routes

# Root router — all routes under /api/v1
api_router = APIRouter()

# Phase 2 — Dataset upload & preprocessing
api_router.include_router(dataset_routes.router)

# Phase 3 — Automated EDA
from app.api.routes import eda as eda_routes
api_router.include_router(eda_routes.router)

# Phase 4 — Visualisation
from app.api.routes import visualization as viz_routes
api_router.include_router(viz_routes.router)

# Phase 5 — Generative AI (placeholder)
# from app.api.routes import ai as ai_routes
# api_router.include_router(ai_routes.router)

