"""
backend/app/schemas/eda.py
--------------------------
Pydantic schemas for Phase 3: Automated Exploratory Data Analysis (EDA).

Models for:
- Missing Value Detection & Analysis
- Duplicate Record Analysis
- Detailed Numerical Feature Statistics (skew, kurtosis, outliers, quantiles)
- Detailed Categorical Feature Statistics (frequencies, cardinality)
- Correlation Matrices (Pearson & Spearman, top correlated pairs)
- Distribution Profiles (histograms and value counts)
- Comprehensive EDA Report
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ======================================================================== #
# Missing Value Analysis
# ======================================================================== #

class ColumnMissingValue(BaseModel):
    """Missing value breakdown for a single column."""

    column: str
    missing_count: int
    missing_percentage: float
    dtype: str


class MissingValueAnalysis(BaseModel):
    """Overall dataset missing value assessment."""

    total_missing_cells: int
    total_cells: int
    overall_missing_percentage: float
    columns_with_missing_count: int
    columns: list[ColumnMissingValue]


# ======================================================================== #
# Duplicate Record Analysis
# ======================================================================== #

class DuplicateAnalysis(BaseModel):
    """Analysis of duplicate rows in the dataset."""

    total_rows: int
    duplicate_rows_count: int
    duplicate_percentage: float
    has_duplicates: bool
    sample_duplicates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 5 duplicate row samples, if any exist.",
    )


# ======================================================================== #
# Numerical Feature Analysis
# ======================================================================== #

class QuantileStats(BaseModel):
    """Detailed quantile thresholds."""

    p1: float = Field(..., alias="1%")
    p5: float = Field(..., alias="5%")
    q1: float = Field(..., alias="25%")
    median: float = Field(..., alias="50%")
    q3: float = Field(..., alias="75%")
    p95: float = Field(..., alias="95%")
    p99: float = Field(..., alias="99%")

    model_config = {"populate_by_name": True}


class OutlierStats(BaseModel):
    """Outliers detected using Tukey's IQR rule (1.5 * IQR)."""

    iqr: float
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_percentage: float


class DetailedNumericFeature(BaseModel):
    """Comprehensive statistical profile for a numeric column."""

    column: str
    count: int
    missing_count: int
    mean: float
    std: float
    variance: float
    min: float
    max: float
    range: float
    skewness: float
    kurtosis: float
    quantiles: QuantileStats
    outliers: OutlierStats
    distribution_shape: str = Field(
        ...,
        description="'Symmetric', 'Right-Skewed', 'Left-Skewed', or 'Uniform'",
    )


# ======================================================================== #
# Categorical Feature Analysis
# ======================================================================== #

class CategoryFrequency(BaseModel):
    """Frequency count for a single categorical value."""

    value: str
    count: int
    percentage: float


class DetailedCategoricalFeature(BaseModel):
    """Profile for a categorical / text column."""

    column: str
    count: int
    missing_count: int
    unique_count: int
    cardinality: str = Field(
        ...,
        description="'Low' (<=10), 'Medium' (11-50), 'High' (>50)",
    )
    top_categories: list[CategoryFrequency] = Field(
        ...,
        description="Top most frequent categories (up to 10).",
    )
    has_rare_categories: bool = Field(
        ...,
        description="True if categories exist with < 1% share.",
    )


# ======================================================================== #
# Correlation Analysis
# ======================================================================== #

class CorrelatedPair(BaseModel):
    """A pair of features and their correlation score."""

    feature_a: str
    feature_b: str
    correlation: float
    strength: str = Field(
        ...,
        description="'Strong Positive', 'Moderate Positive', 'Weak', 'Moderate Negative', 'Strong Negative'",
    )


class CorrelationAnalysis(BaseModel):
    """Pearson and Spearman correlation matrices and highlights."""

    numeric_columns: list[str]
    pearson_matrix: dict[str, dict[str, float]]
    spearman_matrix: dict[str, dict[str, float]]
    top_positive_correlations: list[CorrelatedPair]
    top_negative_correlations: list[CorrelatedPair]


# ======================================================================== #
# Distribution Analysis (Histograms & Frequencies)
# ======================================================================== #

class NumericHistogram(BaseModel):
    """Binned histogram frequencies for interactive plotting."""

    column: str
    bin_edges: list[float]
    counts: list[int]


class DistributionAnalysis(BaseModel):
    """Distribution representations for frontend charts."""

    numeric_histograms: list[NumericHistogram]
    categorical_frequencies: dict[str, list[CategoryFrequency]]


# ======================================================================== #
# Comprehensive EDA Report
# ======================================================================== #

class ComprehensiveEDAResponse(BaseModel):
    """Consolidated full EDA report."""

    dataset_id: str
    filename: str
    total_rows: int
    total_columns: int
    missing_analysis: MissingValueAnalysis
    duplicate_analysis: DuplicateAnalysis
    numeric_features: list[DetailedNumericFeature]
    categorical_features: list[DetailedCategoricalFeature]
    correlations: CorrelationAnalysis
    distributions: DistributionAnalysis
    key_insights: list[str] = Field(
        ...,
        description="Auto-derived data quality highlights and potential warnings.",
    )
