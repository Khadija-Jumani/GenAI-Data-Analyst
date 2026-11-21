"""
backend/app/services/dataset_service.py
----------------------------------------
Core data processing logic for Phase 2: Dataset Upload & Preprocessing.

Responsibilities
----------------
- Validate and store uploaded files.
- Load CSV / Excel files with Pandas.
- Generate dataset previews.
- Extract column-level structural metadata.
- Compute descriptive statistics and data-quality metrics.

Storage strategy (Phase 2)
---------------------------
An in-memory registry maps `dataset_id` → file path.
This is intentionally simple for the prototype phase and will be
replaced with persistent storage (database / object store) in a later phase.
"""

from __future__ import annotations

import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.dataset import (
    CategoricalStats,
    ColumnInfo,
    DatasetPreviewResponse,
    DatasetStructureResponse,
    DatasetSummaryResponse,
    DatasetUploadResponse,
    NumericStats,
)

# --------------------------------------------------------------------------- #
# In-memory dataset registry                                                   #
# dataset_id  →  {"path": str, "filename": str, "size": int}                  #
# --------------------------------------------------------------------------- #
_registry: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _dtype_category(dtype: np.dtype) -> str:
    """Classify a Pandas dtype into a human-readable category."""
    kind = dtype.kind
    if kind in ("i", "u", "f"):
        return "numeric"
    if kind in ("M",):
        return "datetime"
    if kind in ("b",):
        return "boolean"
    if hasattr(dtype, "categories"):          # CategoricalDtype
        return "categorical"
    return "categorical"                       # object / string → categorical


def _safe_float(value: Any) -> float:
    """Convert a value to float, replacing NaN/inf with 0.0."""
    try:
        f = float(value)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


def _load_dataframe(file_path: str, extension: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a Pandas DataFrame."""
    if extension == ".csv":
        return pd.read_csv(file_path, low_memory=False)
    return pd.read_excel(file_path)


# --------------------------------------------------------------------------- #
# Public service functions                                                     #
# --------------------------------------------------------------------------- #

def save_upload(file_bytes: bytes, filename: str) -> DatasetUploadResponse:
    """
    Validate and persist an uploaded file.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content received from the upload endpoint.
    filename : str
        Original filename provided by the client.

    Returns
    -------
    DatasetUploadResponse
        Metadata about the saved dataset.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the file exceeds the size limit.
    """
    # --- Extension check ---
    extension = Path(filename).suffix.lower()
    if extension not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # --- Size check ---
    file_size = len(file_bytes)
    if file_size > settings.max_upload_size_bytes:
        raise ValueError(
            f"File size ({file_size / 1_048_576:.1f} MB) exceeds the "
            f"{settings.MAX_UPLOAD_SIZE_MB} MB upload limit."
        )

    # --- Persist to temp directory ---
    tmp_dir = tempfile.gettempdir()
    dataset_id = uuid.uuid4().hex
    safe_name = f"{dataset_id}{extension}"
    file_path = os.path.join(tmp_dir, safe_name)

    with open(file_path, "wb") as fh:
        fh.write(file_bytes)

    # --- Load to get shape ---
    df = _load_dataframe(file_path, extension)
    rows, columns = df.shape

    # --- Register ---
    _registry[dataset_id] = {
        "path": file_path,
        "filename": filename,
        "size": file_size,
        "extension": extension,
    }

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=filename,
        file_size_bytes=file_size,
        rows=rows,
        columns=columns,
        extension=extension,
    )


def get_preview(dataset_id: str, n_rows: int = 10) -> DatasetPreviewResponse:
    """
    Return the first `n_rows` rows of a dataset as JSON records.

    Parameters
    ----------
    dataset_id : str
        ID returned by `save_upload`.
    n_rows : int
        Number of rows to return (default 10, max 100).

    Returns
    -------
    DatasetPreviewResponse

    Raises
    ------
    KeyError
        If `dataset_id` is not found in the registry.
    """
    entry = _get_entry(dataset_id)
    df = _load_dataframe(entry["path"], entry["extension"])

    n_rows = min(max(1, n_rows), 100)
    preview_df = df.head(n_rows)

    # Replace NaN with None for clean JSON serialization
    records = preview_df.where(pd.notnull(preview_df), other=None).to_dict(orient="records")

    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        filename=entry["filename"],
        rows_returned=len(records),
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=df.columns.tolist(),
        data=records,
    )


def get_structure(dataset_id: str) -> DatasetStructureResponse:
    """
    Return detailed column-level metadata for a dataset.

    Parameters
    ----------
    dataset_id : str

    Returns
    -------
    DatasetStructureResponse
    """
    entry = _get_entry(dataset_id)
    df = _load_dataframe(entry["path"], entry["extension"])

    total_rows, total_columns = df.shape
    memory_kb = df.memory_usage(deep=True).sum() / 1024

    column_infos: list[ColumnInfo] = []
    numeric_count = categorical_count = datetime_count = 0

    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        category = _dtype_category(dtype)

        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        null_pct = round(null_count / total_rows * 100, 2) if total_rows else 0.0
        unique_pct = round(unique_count / total_rows * 100, 2) if total_rows else 0.0

        # Sample up to 5 non-null values
        non_null = series.dropna()
        samples: list[Any] = non_null.head(5).tolist()

        # Convert numpy scalars to Python natives for JSON safety
        samples = [
            v.item() if isinstance(v, np.generic) else v
            for v in samples
        ]

        column_infos.append(
            ColumnInfo(
                name=col,
                dtype=str(dtype),
                dtype_category=category,
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique_count,
                unique_percentage=unique_pct,
                sample_values=samples,
            )
        )

        if category == "numeric":
            numeric_count += 1
        elif category == "datetime":
            datetime_count += 1
        else:
            categorical_count += 1

    return DatasetStructureResponse(
        dataset_id=dataset_id,
        filename=entry["filename"],
        total_rows=total_rows,
        total_columns=total_columns,
        memory_usage_kb=round(memory_kb, 2),
        columns=column_infos,
        numeric_column_count=numeric_count,
        categorical_column_count=categorical_count,
        datetime_column_count=datetime_count,
    )


def get_summary(dataset_id: str) -> DatasetSummaryResponse:
    """
    Compute descriptive statistics and data-quality metrics.

    Parameters
    ----------
    dataset_id : str

    Returns
    -------
    DatasetSummaryResponse
    """
    entry = _get_entry(dataset_id)
    df = _load_dataframe(entry["path"], entry["extension"])

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    # --- Data quality ---
    total_missing = int(df.isna().sum().sum())
    missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_rows / total_rows * 100, 2) if total_rows else 0.0

    # --- Numeric columns ---
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_stats: list[NumericStats] = []

    for col in numeric_df.columns:
        desc = numeric_df[col].describe()
        numeric_stats.append(
            NumericStats(
                column=col,
                count=int(desc.get("count", 0)),
                mean=_safe_float(desc.get("mean")),
                std=_safe_float(desc.get("std")),
                min=_safe_float(desc.get("min")),
                **{"25%": _safe_float(desc.get("25%"))},
                **{"50%": _safe_float(desc.get("50%"))},
                **{"75%": _safe_float(desc.get("75%"))},
                max=_safe_float(desc.get("max")),
            )
        )

    # --- Categorical columns ---
    cat_df = df.select_dtypes(include=["object", "category"])
    categorical_stats: list[CategoricalStats] = []

    for col in cat_df.columns:
        series = df[col].dropna().astype(str)
        count = int(series.count())
        unique = int(series.nunique())
        top_val: str | None = None
        top_freq: int | None = None
        top_pct: float | None = None

        if count > 0:
            vc = series.value_counts()
            top_val = str(vc.index[0])
            top_freq = int(vc.iloc[0])
            top_pct = round(top_freq / count * 100, 2)

        categorical_stats.append(
            CategoricalStats(
                column=col,
                count=count,
                unique=unique,
                top_value=top_val,
                top_frequency=top_freq,
                top_percentage=top_pct,
            )
        )

    return DatasetSummaryResponse(
        dataset_id=dataset_id,
        filename=entry["filename"],
        total_rows=total_rows,
        total_columns=total_columns,
        total_missing_values=total_missing,
        missing_value_percentage=missing_pct,
        total_duplicate_rows=duplicate_rows,
        duplicate_row_percentage=duplicate_pct,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
    )


# --------------------------------------------------------------------------- #
# Internal                                                                     #
# --------------------------------------------------------------------------- #

def _get_entry(dataset_id: str) -> dict[str, Any]:
    """Retrieve a registry entry or raise a descriptive KeyError."""
    entry = _registry.get(dataset_id)
    if entry is None:
        raise KeyError(
            f"Dataset '{dataset_id}' not found. "
            "It may have been cleared from memory or never uploaded."
        )
    return entry

