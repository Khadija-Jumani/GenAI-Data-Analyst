"""
backend/app/api/routes/dataset.py
----------------------------------
FastAPI route handlers for all dataset-related endpoints.

Endpoints
---------
POST   /api/v1/datasets/upload          Upload a CSV or Excel file.
GET    /api/v1/datasets/{id}/preview    Preview first N rows.
GET    /api/v1/datasets/{id}/structure  Column-level structural metadata.
GET    /api/v1/datasets/{id}/summary    Descriptive statistics + data quality.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.schemas.dataset import (
    DatasetPreviewResponse,
    DatasetStructureResponse,
    DatasetSummaryResponse,
    DatasetUploadResponse,
)
from app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["Datasets"])


# --------------------------------------------------------------------------- #
# Upload                                                                       #
# --------------------------------------------------------------------------- #

@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a dataset",
    description=(
        "Upload a CSV or Excel (.xlsx / .xls) file. "
        "Returns a `dataset_id` used for all subsequent endpoints."
    ),
)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV or Excel dataset file."),
) -> DatasetUploadResponse:
    """Accept an uploaded file, validate it, and register it in the session."""
    file_bytes = await file.read()

    try:
        result = dataset_service.save_upload(
            file_bytes=file_bytes,
            filename=file.filename or "upload",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return result


# --------------------------------------------------------------------------- #
# Preview                                                                      #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    summary="Preview dataset rows",
    description="Return the first N rows of an uploaded dataset (max 100).",
)
def preview_dataset(
    dataset_id: str,
    rows: int = Query(default=10, ge=1, le=100, description="Number of rows to return."),
) -> DatasetPreviewResponse:
    """Return the first `rows` records of the dataset."""
    try:
        return dataset_service.get_preview(dataset_id, n_rows=rows)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# --------------------------------------------------------------------------- #
# Structure                                                                    #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}/structure",
    response_model=DatasetStructureResponse,
    summary="Inspect dataset structure",
    description=(
        "Return per-column metadata: data types, null counts, "
        "unique value counts, and sample values."
    ),
)
def dataset_structure(dataset_id: str) -> DatasetStructureResponse:
    """Return detailed column-level metadata for the dataset."""
    try:
        return dataset_service.get_structure(dataset_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}/summary",
    response_model=DatasetSummaryResponse,
    summary="Dataset statistical summary",
    description=(
        "Return descriptive statistics for numeric columns, "
        "frequency statistics for categorical columns, "
        "and data-quality metrics (missing values, duplicates)."
    ),
)
def dataset_summary(dataset_id: str) -> DatasetSummaryResponse:
    """Return full descriptive statistics and data-quality metrics."""
    try:
        return dataset_service.get_summary(dataset_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

