"""
backend/app/services/visualization_service.py
---------------------------------------------
Interactive visualization generation service for Phase 4 using Plotly.

Generates:
1. Correlation Heatmaps (annotated Pearson / Spearman)
2. Distribution Plots (Histogram + KDE or frequency bars)
3. Box & Whisker Plots (Outlier inspection + group comparisons)
4. Scatter Plots (Bivariate relationships + group colors + trendlines)
5. Bar Charts (Category frequencies or aggregations)
6. Automated Dashboard Chart Suite (Initial bundle for overview)
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from app.schemas.visualization import (
    DashboardVisualizationsResponse,
    PlotlyChartResponse,
    PlotlyChartSpec,
)
from app.services.dataset_service import get_dataframe

# Modern UI theme defaults
CHART_THEME_LAYOUT = {
    "font": {"family": "Inter, system-ui, -apple-system, sans-serif", "size": 12, "color": "#1f2937"},
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#f9fafb",
    "margin": {"l": 50, "r": 40, "t": 60, "b": 50},
    "hoverlabel": {"bgcolor": "#1f2937", "font_size": 12, "font_color": "#ffffff"},
}


def _fig_to_spec(fig: go.Figure) -> PlotlyChartSpec:
    """Convert a Plotly Figure to JSON-serializable PlotlyChartSpec."""
    # Use PlotlyJSONEncoder to handle numpy types, timestamps, NaNs
    raw_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
    data_dict = json.loads(raw_json)

    data = data_dict.get("data", [])
    layout = data_dict.get("layout", {})

    # Merge modern styling
    for key, val in CHART_THEME_LAYOUT.items():
        if key not in layout or not layout[key]:
            layout[key] = val

    return PlotlyChartSpec(data=data, layout=layout)


def create_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
) -> PlotlyChartResponse:
    """Generate an annotated correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    cols = numeric_df.columns.tolist()

    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation heatmap.",
            showarrow=False,
            font={"size": 14, "color": "#ef4444"},
        )
    else:
        corr_matrix = numeric_df.corr(method=method).round(3)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels={"color": f"{method.capitalize()} Correlation"},
            title=f"Correlation Heatmap ({method.capitalize()})",
        )
        fig.update_layout(title_x=0.5)

    return PlotlyChartResponse(
        chart_id=f"heatmap_{uuid.uuid4().hex[:8]}",
        title=f"Correlation Heatmap ({method.capitalize()})",
        chart_type="heatmap",
        description=f"Pairwise {method} correlation coefficients across numeric features.",
        spec=_fig_to_spec(fig),
    )


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
) -> PlotlyChartResponse:
    """Generate a distribution histogram (numeric) or category count bar chart (categorical)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    series = df[column].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric:
        fig = px.histogram(
            df,
            x=column,
            marginal="box",
            nbins=30,
            opacity=0.85,
            color_discrete_sequence=["#3b82f6"],
            title=f"Distribution of {column}",
        )
        fig.update_layout(
            bargap=0.08,
            title_x=0.5,
            xaxis_title=column,
            yaxis_title="Count",
        )
        desc = f"Histogram with marginal box plot showing distribution and spread of {column}."
    else:
        # Categorical counts
        val_counts = series.value_counts().reset_index()
        val_counts.columns = [column, "count"]
        val_counts = val_counts.head(20)  # Top 20 categories

        fig = px.bar(
            val_counts,
            x=column,
            y="count",
            color="count",
            color_continuous_scale="Blues",
            title=f"Frequency Count of {column}",
        )
        fig.update_layout(
            title_x=0.5,
            xaxis_title=column,
            yaxis_title="Frequency",
        )
        desc = f"Bar chart showing category counts for {column}."

    return PlotlyChartResponse(
        chart_id=f"dist_{uuid.uuid4().hex[:8]}",
        title=f"Distribution of {column}",
        chart_type="histogram" if is_numeric else "bar",
        description=desc,
        spec=_fig_to_spec(fig),
    )


def create_box_plot(
    df: pd.DataFrame,
    y_column: str,
    x_column: str | None = None,
) -> PlotlyChartResponse:
    """Generate interactive box plots for outlier detection and group comparisons."""
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' does not exist in dataset.")
    if x_column and x_column not in df.columns:
        raise ValueError(f"Grouping column '{x_column}' does not exist in dataset.")

    title = f"Box Plot of {y_column}" + (f" by {x_column}" if x_column else "")

    fig = px.box(
        df,
        y=y_column,
        x=x_column,
        color=x_column if x_column else None,
        points="outliers",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title=title,
    )
    fig.update_layout(title_x=0.5)

    return PlotlyChartResponse(
        chart_id=f"box_{uuid.uuid4().hex[:8]}",
        title=title,
        chart_type="box",
        description=f"Box-and-whisker plot highlighting median, IQR, and statistical outliers for {y_column}.",
        spec=_fig_to_spec(fig),
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
    add_trendline: bool = False,
) -> PlotlyChartResponse:
    """Generate an interactive scatter plot between two features with optional group colors."""
    if x_column not in df.columns:
        raise ValueError(f"X-column '{x_column}' not found.")
    if y_column not in df.columns:
        raise ValueError(f"Y-column '{y_column}' not found.")
    if color_column and color_column not in df.columns:
        raise ValueError(f"Color column '{color_column}' not found.")

    trendline_type = "ols" if add_trendline else None

    # Filter out nulls for chosen columns to avoid plotting errors
    cols_to_use = [x_column, y_column] + ([color_column] if color_column else [])
    clean_df = df[cols_to_use].dropna()

    try:
        fig = px.scatter(
            clean_df,
            x=x_column,
            y=y_column,
            color=color_column if color_column else None,
            trendline=trendline_type,
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            title=f"{y_column} vs. {x_column}",
        )
    except Exception:
        # Fallback if OLS fails (e.g. statsmodels not installed or too few points)
        fig = px.scatter(
            clean_df,
            x=x_column,
            y=y_column,
            color=color_column if color_column else None,
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            title=f"{y_column} vs. {x_column}",
        )

    fig.update_layout(title_x=0.5)

    return PlotlyChartResponse(
        chart_id=f"scatter_{uuid.uuid4().hex[:8]}",
        title=f"{y_column} vs. {x_column}",
        chart_type="scatter",
        description=f"Scatter plot analyzing relationship between {x_column} and {y_column}.",
        spec=_fig_to_spec(fig),
    )


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None = None,
    aggregation: str = "count",
) -> PlotlyChartResponse:
    """Generate bar charts for category frequencies or aggregations (mean, sum)."""
    if x_column not in df.columns:
        raise ValueError(f"X-column '{x_column}' not found.")

    if y_column:
        if y_column not in df.columns:
            raise ValueError(f"Y-column '{y_column}' not found.")

        agg_func = "mean" if aggregation == "mean" else "sum"
        agg_df = df.groupby(x_column)[y_column].agg(agg_func).reset_index()
        agg_df[y_column] = agg_df[y_column].round(2)

        fig = px.bar(
            agg_df,
            x=x_column,
            y=y_column,
            color=y_column,
            color_continuous_scale="Teal",
            title=f"{aggregation.capitalize()} of {y_column} by {x_column}",
        )
        title = f"{aggregation.capitalize()} of {y_column} by {x_column}"
        desc = f"Aggregated bar chart displaying {aggregation} of {y_column} grouped by {x_column}."
    else:
        # Frequency count
        val_counts = df[x_column].value_counts().reset_index()
        val_counts.columns = [x_column, "Count"]
        fig = px.bar(
            val_counts,
            x=x_column,
            y="Count",
            color="Count",
            color_continuous_scale="Viridis",
            title=f"Count of Records by {x_column}",
        )
        title = f"Count of Records by {x_column}"
        desc = f"Bar chart of record frequencies across categories of {x_column}."

    fig.update_layout(title_x=0.5)

    return PlotlyChartResponse(
        chart_id=f"bar_{uuid.uuid4().hex[:8]}",
        title=title,
        chart_type="bar",
        description=desc,
        spec=_fig_to_spec(fig),
    )


def generate_dashboard_charts(dataset_id: str) -> DashboardVisualizationsResponse:
    """Generate a curated suite of starter charts for the dataset overview dashboard."""
    df, filename = get_dataframe(dataset_id)
    charts: list[PlotlyChartResponse] = []

    # 1. Correlation Heatmap (if at least 2 numeric features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        try:
            charts.append(create_correlation_heatmap(df))
        except Exception:
            pass

    # 2. Distributions of top 2 numeric features
    for col in numeric_cols[:2]:
        try:
            charts.append(create_distribution_plot(df, col))
        except Exception:
            pass

    # 3. Categorical distribution of primary categorical column
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        try:
            charts.append(create_distribution_plot(df, cat_cols[0]))
        except Exception:
            pass

    # 4. Box plot comparing first numeric column grouped by first categorical column
    if numeric_cols and cat_cols:
        try:
            charts.append(create_box_plot(df, y_column=numeric_cols[0], x_column=cat_cols[0]))
        except Exception:
            pass

    # 5. Scatter plot between first two numeric columns
    if len(numeric_cols) >= 2:
        try:
            color_col = cat_cols[0] if cat_cols else None
            charts.append(
                create_scatter_plot(
                    df,
                    x_column=numeric_cols[0],
                    y_column=numeric_cols[1],
                    color_column=color_col,
                )
            )
        except Exception:
            pass

    return DashboardVisualizationsResponse(
        dataset_id=dataset_id,
        filename=filename,
        charts=charts,
    )
