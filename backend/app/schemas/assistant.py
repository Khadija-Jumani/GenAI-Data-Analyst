"""
backend/app/schemas/assistant.py
--------------------------------
Pydantic schemas for Phase 6: Natural-Language Data Analysis Assistant.

Models for:
- Chat Messages (with optional chart attachments and follow-up prompt pills)
- Multi-turn Conversational Sessions
- User Prompt Requests & Assistant Responses
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.visualization import PlotlyChartResponse


class ChatMessage(BaseModel):
    """A single turn in the assistant conversation."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text in Markdown format.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of message creation.",
    )
    chart_attachment: PlotlyChartResponse | None = Field(
        default=None,
        description="Optional interactive Plotly chart generated in response to the user's inquiry.",
    )
    referenced_columns: list[str] = Field(
        default_factory=list,
        description="Dataset columns relevant to this message.",
    )


class UserQueryRequest(BaseModel):
    """User prompt to the data assistant."""

    message: str = Field(..., min_length=1, max_length=1000, description="User question or analysis instruction.")
    session_id: str | None = Field(
        default=None,
        description="Optional existing session ID to continue multi-turn dialogue.",
    )


class AssistantMessageResponse(BaseModel):
    """Assistant reply including generated insights, optional visual attachment, and suggested prompts."""

    session_id: str
    dataset_id: str
    reply: ChatMessage
    suggested_followups: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up analytical questions or instructions.",
    )


class ChatSessionResponse(BaseModel):
    """Complete chat history for a session."""

    session_id: str
    dataset_id: str
    filename: str
    message_count: int
    messages: list[ChatMessage]
