"""
backend/app/api/routes/ai.py
----------------------------
FastAPI routes for Phase 5: Google Gemini Generative AI Integration.

Endpoints:
- GET  /api/v1/ai/{dataset_id}/insights
- GET  /api/v1/ai/{dataset_id}/explain-feature
- GET  /api/v1/ai/{dataset_id}/explain-correlation
- POST /api/v1/ai/{dataset_id}/ask
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.ai import (
    AskDatasetRequest,
    AskDatasetResponse,
    CorrelationExplanationResponse,
    DatasetInsightsResponse,
    FeatureExplanationResponse,
)
from app.services import ai_service

router = APIRouter(prefix="/ai", tags=["Generative AI Insights"])


@router.get(
    "/{dataset_id}/insights",
    response_model=DatasetInsightsResponse,
    summary="Executive dataset insights & risks",
    description="Generate high-level narrative summary, key statistical findings, anomalies, and recommendations using Gemini.",
)
def get_dataset_insights(dataset_id: str) -> DatasetInsightsResponse:
    try:
        return ai_service.generate_dataset_insights(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/explain-feature",
    response_model=FeatureExplanationResponse,
    summary="AI feature explanation",
    description="Deep domain and statistical interpretation of an individual dataset column.",
)
def get_feature_explanation(
    dataset_id: str,
    column: str = Query(..., description="Column name to explain."),
) -> FeatureExplanationResponse:
    try:
        return ai_service.explain_feature(dataset_id, column=column)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/explain-correlation",
    response_model=CorrelationExplanationResponse,
    summary="AI correlation interpretation",
    description="Explain bivariate statistical relationship, potential confounders, and causation caveats between two features.",
)
def get_correlation_explanation(
    dataset_id: str,
    feature_a: str = Query(..., description="First feature name."),
    feature_b: str = Query(..., description="Second feature name."),
) -> CorrelationExplanationResponse:
    try:
        return ai_service.explain_correlation(dataset_id, feature_a=feature_a, feature_b=feature_b)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post(
    "/{dataset_id}/ask",
    response_model=AskDatasetResponse,
    summary="Ask questions about the dataset",
    description="Natural language Q&A: Ask questions about trends, distributions, or metrics and receive AI-generated answers.",
)
def ask_question(
    dataset_id: str,
    payload: AskDatasetRequest,
) -> AskDatasetResponse:
    try:
        return ai_service.ask_dataset(dataset_id, question=payload.question)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
