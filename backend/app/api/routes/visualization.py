"""
backend/app/api/routes/visualization.py
---------------------------------------
FastAPI routes for Phase 4: Interactive Data Visualization.

Provides Plotly JSON chart specifications for:
- Correlation Heatmaps
- Feature Distributions (Histogram with marginal box plot / Bar charts)
- Box Plots (Outliers & group comparisons)
- Scatter Plots (Bivariate relationships + colors)
- Bar Charts (Frequencies & aggregated statistics)
- Dashboard Suite (Curated suite of starter charts)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.visualization import (
    DashboardVisualizationsResponse,
    PlotlyChartResponse,
)
from app.services import visualization_service
from app.services.dataset_service import get_dataframe

router = APIRouter(prefix="/visualizations", tags=["Interactive Visualizations"])


@router.get(
    "/{dataset_id}/heatmap",
    response_model=PlotlyChartResponse,
    summary="Interactive correlation heatmap",
    description="Generate an annotated correlation heatmap for numeric features.",
)
def get_correlation_heatmap(
    dataset_id: str,
    method: str = Query(default="pearson", pattern="^(pearson|spearman)$", description="'pearson' or 'spearman'"),
) -> PlotlyChartResponse:
    try:
        df, _ = get_dataframe(dataset_id)
        return visualization_service.create_correlation_heatmap(df, method=method)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/distribution",
    response_model=PlotlyChartResponse,
    summary="Feature distribution chart",
    description="Histogram with marginal box plot for numeric column, or frequency bar chart for categorical column.",
)
def get_distribution_chart(
    dataset_id: str,
    column: str = Query(..., description="Target feature column name to analyze."),
) -> PlotlyChartResponse:
    try:
        df, _ = get_dataframe(dataset_id)
        return visualization_service.create_distribution_plot(df, column=column)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/box",
    response_model=PlotlyChartResponse,
    summary="Box and whisker plot",
    description="Outlier and quartile visualization for a numeric column, optionally grouped by a category.",
)
def get_box_plot(
    dataset_id: str,
    y_column: str = Query(..., description="Numeric column to plot on Y-axis."),
    x_column: str | None = Query(default=None, description="Optional categorical column for grouping."),
) -> PlotlyChartResponse:
    try:
        df, _ = get_dataframe(dataset_id)
        return visualization_service.create_box_plot(df, y_column=y_column, x_column=x_column)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/scatter",
    response_model=PlotlyChartResponse,
    summary="Interactive scatter plot",
    description="Plot relationship between two numeric features with optional grouping colors.",
)
def get_scatter_plot(
    dataset_id: str,
    x_column: str = Query(..., description="Numeric column on X-axis."),
    y_column: str = Query(..., description="Numeric column on Y-axis."),
    color_column: str | None = Query(default=None, description="Optional column to color data points by."),
    add_trendline: bool = Query(default=False, description="Whether to include a linear trendline."),
) -> PlotlyChartResponse:
    try:
        df, _ = get_dataframe(dataset_id)
        return visualization_service.create_scatter_plot(
            df,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            add_trendline=add_trendline,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/bar",
    response_model=PlotlyChartResponse,
    summary="Interactive bar chart",
    description="Bar chart displaying categorical counts or aggregated values (mean/sum).",
)
def get_bar_chart(
    dataset_id: str,
    x_column: str = Query(..., description="Categorical column for categories."),
    y_column: str | None = Query(default=None, description="Optional numeric column to aggregate."),
    aggregation: str = Query(default="count", pattern="^(count|mean|sum)$", description="'count', 'mean', or 'sum'"),
) -> PlotlyChartResponse:
    try:
        df, _ = get_dataframe(dataset_id)
        return visualization_service.create_bar_chart(
            df,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/dashboard",
    response_model=DashboardVisualizationsResponse,
    summary="Curated dashboard charts suite",
    description="Returns a bundle of initial starter charts for the dataset overview.",
)
def get_dashboard_visualizations(dataset_id: str) -> DashboardVisualizationsResponse:
    try:
        return visualization_service.generate_dashboard_charts(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
