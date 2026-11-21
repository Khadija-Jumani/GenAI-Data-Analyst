"""
backend/app/schemas/report.py
-----------------------------
Pydantic schemas for Phase 8: Report Generation and Export.

Defines response models for:
- Analytical Dossier Summary
- Report Export Metadata
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReportSummaryResponse(BaseModel):
    """Metadata overview of the generated analytical report."""

    dataset_id: str
    filename: str
    generated_at: str = Field(..., description="ISO timestamp of report creation.")
    overall_health_score: float = Field(..., description="Data health score (0-100).")
    sections_included: list[str] = Field(..., description="List of report sections generated.")
    download_urls: dict[str, str] = Field(
        ...,
        description="Direct download endpoints for Markdown, HTML, and JSON formats.",
    )
