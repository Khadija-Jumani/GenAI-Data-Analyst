"""
backend/app/schemas/dataset.py
-------------------------------
Pydantic response models for all dataset-related API endpoints.

These models define the exact JSON shape returned to the frontend
for dataset upload, preview, structure inspection, and summary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ======================================================================== #
# Upload
# ======================================================================== #

class DatasetUploadResponse(BaseModel):
    """Returned immediately after a successful file upload."""

    dataset_id: str = Field(..., description="Unique identifier for this dataset session.")
    filename: str = Field(..., description="Original uploaded filename.")
    file_size_bytes: int = Field(..., description="Size of the uploaded file in bytes.")
    rows: int = Field(..., description="Total number of rows in the dataset.")
    columns: int = Field(..., description="Total number of columns in the dataset.")
    extension: str = Field(..., description="File extension: .csv, .xlsx, or .xls.")
    message: str = Field(default="Dataset uploaded successfully.")


# ======================================================================== #
# Preview
# ======================================================================== #

class DatasetPreviewResponse(BaseModel):
    """First N rows of the dataset as a list of row dictionaries."""

    dataset_id: str
    filename: str
    rows_returned: int = Field(..., description="Number of rows in this preview.")
    total_rows: int = Field(..., description="Total rows in the full dataset.")
    total_columns: int
    columns: list[str] = Field(..., description="Ordered list of column names.")
    data: list[dict[str, Any]] = Field(..., description="Preview rows as JSON records.")


# ======================================================================== #
# Structure / Column Info
# ======================================================================== #

class ColumnInfo(BaseModel):
    """Metadata for a single dataset column."""

    name: str
    dtype: str = Field(..., description="Pandas dtype string, e.g. int64, float64, object.")
    dtype_category: str = Field(
        ...,
        description="High-level category: 'numeric', 'categorical', 'datetime', 'boolean', 'other'.",
    )
    null_count: int = Field(..., description="Number of missing / NaN values.")
    null_percentage: float = Field(..., description="Percentage of missing values (0–100).")
    unique_count: int = Field(..., description="Number of distinct values.")
    unique_percentage: float = Field(..., description="Percentage of distinct values (0–100).")
    sample_values: list[Any] = Field(
        ..., description="Up to 5 representative non-null sample values."
    )


class DatasetStructureResponse(BaseModel):
    """Full structural metadata for the dataset."""

    dataset_id: str
    filename: str
    total_rows: int
    total_columns: int
    memory_usage_kb: float = Field(..., description="Approximate in-memory size in kilobytes.")
    columns: list[ColumnInfo]
    numeric_column_count: int
    categorical_column_count: int
    datetime_column_count: int


# ======================================================================== #
# Summary / Basic Statistics
# ======================================================================== #

class NumericStats(BaseModel):
    """Descriptive statistics for a single numeric column."""

    column: str
    count: int
    mean: float
    std: float
    min: float
    q25: float = Field(..., alias="25%")
    median: float = Field(..., alias="50%")
    q75: float = Field(..., alias="75%")
    max: float

    model_config = {"populate_by_name": True}


class CategoricalStats(BaseModel):
    """Frequency statistics for a single categorical column."""

    column: str
    count: int
    unique: int
    top_value: str | None = Field(None, description="Most frequent value.")
    top_frequency: int | None = Field(None, description="Occurrences of the most frequent value.")
    top_percentage: float | None = Field(None, description="Percentage share of the top value.")


class DatasetSummaryResponse(BaseModel):
    """Comprehensive statistical summary of the dataset."""

    dataset_id: str
    filename: str
    total_rows: int
    total_columns: int

    # Data quality
    total_missing_values: int
    missing_value_percentage: float
    total_duplicate_rows: int
    duplicate_row_percentage: float

    # Per-column statistics
    numeric_stats: list[NumericStats]
    categorical_stats: list[CategoricalStats]

