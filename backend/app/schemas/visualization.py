"""
backend/app/schemas/visualization.py
------------------------------------
Pydantic schemas for Phase 4: Interactive Data Visualization.

Defines the structure for Plotly chart specifications returned to the Next.js frontend.
Plotly figures serialize to standard JSON containing `data`, `layout`, and `config`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlotlyChartSpec(BaseModel):
    """Plotly specification ready for direct consumption by React-Plotly / Plotly.js."""

    data: list[dict[str, Any]] = Field(..., description="Plotly traces data.")
    layout: dict[str, Any] = Field(..., description="Plotly layout configuration.")
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        description="Plotly rendering configuration options.",
    )


class PlotlyChartResponse(BaseModel):
    """Response model for a single generated interactive chart."""

    chart_id: str
    title: str
    chart_type: str = Field(
        ...,
        description="'heatmap', 'histogram', 'box', 'scatter', 'bar', etc.",
    )
    description: str = Field(default="", description="Contextual description of the chart.")
    spec: PlotlyChartSpec


class DashboardVisualizationsResponse(BaseModel):
    """Collection of essential interactive charts generated for the dataset overview."""

    dataset_id: str
    filename: str
    charts: list[PlotlyChartResponse]
