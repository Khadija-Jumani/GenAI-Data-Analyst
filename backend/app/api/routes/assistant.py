"""
backend/app/api/routes/assistant.py
----------------------------------
FastAPI routes for Phase 6: Natural-Language Data Analysis Assistant.

Endpoints:
- POST   /api/v1/assistant/{dataset_id}/chat     Send a query (with chart intent auto-detection)
- GET    /api/v1/assistant/{dataset_id}/history  Retrieve chat session history
- DELETE /api/v1/assistant/{dataset_id}/history  Reset/clear session dialogue
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.assistant import (
    AssistantMessageResponse,
    ChatSessionResponse,
    UserQueryRequest,
)
from app.services import assistant_service

router = APIRouter(prefix="/assistant", tags=["Natural-Language Assistant"])


@router.post(
    "/{dataset_id}/chat",
    response_model=AssistantMessageResponse,
    summary="Chat with the data analyst assistant",
    description=(
        "Send natural-language prompts, statistical inquiries, or visualization requests. "
        "The assistant responds with markdown commentary, automated Plotly chart attachments, "
        "and suggested follow-up questions."
    ),
)
def chat_with_assistant(
    dataset_id: str,
    payload: UserQueryRequest,
) -> AssistantMessageResponse:
    try:
        return assistant_service.process_user_query(
            dataset_id=dataset_id,
            message=payload.message,
            session_id=payload.session_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/history",
    response_model=ChatSessionResponse,
    summary="Get conversation history",
    description="Retrieve all turns (user and assistant) from the active chat session.",
)
def get_session_history(
    dataset_id: str,
    session_id: str | None = Query(default=None, description="Optional specific session ID to look up."),
) -> ChatSessionResponse:
    try:
        return assistant_service.get_chat_history(dataset_id=dataset_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete(
    "/{dataset_id}/history",
    summary="Clear conversation history",
    description="Reset the chat memory for a given dataset session.",
)
def clear_session_history(
    dataset_id: str,
    session_id: str | None = Query(default=None, description="Optional session ID to clear."),
) -> dict[str, str]:
    try:
        return assistant_service.clear_chat_history(dataset_id=dataset_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
