"""
backend/app/api/routes/eda.py
------------------------------
FastAPI routes for Phase 3: Automated Exploratory Data Analysis (EDA).

Endpoints:
- GET /api/v1/eda/{dataset_id}/missing
- GET /api/v1/eda/{dataset_id}/duplicates
- GET /api/v1/eda/{dataset_id}/numeric
- GET /api/v1/eda/{dataset_id}/categorical
- GET /api/v1/eda/{dataset_id}/correlations
- GET /api/v1/eda/{dataset_id}/distributions
- GET /api/v1/eda/{dataset_id}/comprehensive
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.eda import (
    ComprehensiveEDAResponse,
    CorrelationAnalysis,
    DetailedCategoricalFeature,
    DetailedNumericFeature,
    DistributionAnalysis,
    DuplicateAnalysis,
    MissingValueAnalysis,
)
from app.services import eda_service
from app.services.dataset_service import get_dataframe

router = APIRouter(prefix="/eda", tags=["Exploratory Data Analysis"])


@router.get(
    "/{dataset_id}/missing",
    response_model=MissingValueAnalysis,
    summary="Missing value detection & analysis",
    description="Analyze missing and null values per column and across the dataset.",
)
def get_missing_values(dataset_id: str) -> MissingValueAnalysis:
    try:
        df, _ = get_dataframe(dataset_id)
        return eda_service.analyze_missing_values(df)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/duplicates",
    response_model=DuplicateAnalysis,
    summary="Duplicate record detection",
    description="Identify duplicate rows, duplicate frequency, and view sample duplicate rows.",
)
def get_duplicates(dataset_id: str) -> DuplicateAnalysis:
    try:
        df, _ = get_dataframe(dataset_id)
        return eda_service.analyze_duplicates(df)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/numeric",
    response_model=list[DetailedNumericFeature],
    summary="Detailed numerical feature analysis",
    description="Compute mean, std, quantiles, skewness, kurtosis, and Tukey IQR outliers.",
)
def get_numeric_features(dataset_id: str) -> list[DetailedNumericFeature]:
    try:
        df, _ = get_dataframe(dataset_id)
        return eda_service.analyze_numeric_features(df)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/categorical",
    response_model=list[DetailedCategoricalFeature],
    summary="Categorical feature analysis",
    description="Inspect cardinality, frequency distribution of top values, and rare labels.",
)
def get_categorical_features(dataset_id: str) -> list[DetailedCategoricalFeature]:
    try:
        df, _ = get_dataframe(dataset_id)
        return eda_service.analyze_categorical_features(df)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/correlations",
    response_model=CorrelationAnalysis,
    summary="Correlation analysis (Pearson & Spearman)",
    description="Compute correlation matrices and extract top positive and negative correlated pairs.",
)
def get_correlations(dataset_id: str) -> CorrelationAnalysis:
    try:
        df, _ = get_dataframe(dataset_id)
        return eda_service.compute_correlations(df)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/distributions",
    response_model=DistributionAnalysis,
    summary="Distribution profiling for visual charts",
    description="Histogram bins for numeric columns and frequency tables for categorical columns.",
)
def get_distributions(dataset_id: str) -> DistributionAnalysis:
    try:
        df, _ = get_dataframe(dataset_id)
        return eda_service.compute_distributions(df)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/comprehensive",
    response_model=ComprehensiveEDAResponse,
    summary="Comprehensive automated EDA report",
    description="Consolidate all exploratory analyses, statistics, correlations, and insights into one full report.",
)
def get_comprehensive_eda(dataset_id: str) -> ComprehensiveEDAResponse:
    try:
        return eda_service.generate_comprehensive_eda(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
