"""
backend/app/api/routes/agent.py
-------------------------------
FastAPI routes for Phase 7: Agentic AI and Automated Analytical Workflows.

Endpoints:
- POST /api/v1/agent/{dataset_id}/workflow   Execute an autonomous agent analytical investigation
- GET  /api/v1/agent/{dataset_id}/workflows  Catalog available autonomous workflow types
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.agent import (
    AgentWorkflowRequest,
    AgentWorkflowResponse,
    AvailableWorkflowsResponse,
)
from app.services import agent_service

router = APIRouter(prefix="/agent", tags=["Autonomous Analytics Agent"])


@router.get(
    "/{dataset_id}/workflows",
    response_model=AvailableWorkflowsResponse,
    summary="List available agent workflows",
    description="Returns the catalog of specialized multi-step autonomous workflows.",
)
def get_available_workflows(dataset_id: str) -> AvailableWorkflowsResponse:
    return agent_service.list_available_workflows()


@router.post(
    "/{dataset_id}/workflow",
    response_model=AgentWorkflowResponse,
    summary="Run autonomous agent analytical workflow",
    description=(
        "Executes a multi-step tool-assisted analytical workflow "
        "('comprehensive_audit', 'feature_engineering', or 'ml_readiness'). "
        "Returns execution audit trails, health scores, and an executive action dossier."
    ),
)
def run_agent_workflow(
    dataset_id: str,
    payload: AgentWorkflowRequest,
) -> AgentWorkflowResponse:
    try:
        return agent_service.execute_agent_workflow(
            dataset_id=dataset_id,
            workflow_type=payload.workflow_type,
            target_column=payload.target_column,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
