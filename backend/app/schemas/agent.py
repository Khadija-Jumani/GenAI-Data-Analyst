"""
backend/app/schemas/agent.py
----------------------------
Pydantic schemas for Phase 7: Agentic AI and Automated Analytical Workflows.

Defines schemas for:
- Autonomous Multi-Step Execution Workflows
- Diagnostic Tool Execution Logs
- Data Health Scoring & ML Readiness Assessments
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentExecutionStep(BaseModel):
    """An individual step executed by the autonomous agent."""

    step_number: int
    step_name: str
    tool_invoked: str
    status: str = Field(default="Completed", description="'Pending', 'Running', 'Completed', 'Failed'")
    observations: str = Field(..., description="Key findings discovered during this step.")


class AgentWorkflowRequest(BaseModel):
    """Request to initiate an autonomous analytical workflow."""

    workflow_type: str = Field(
        default="comprehensive_audit",
        pattern="^(comprehensive_audit|feature_engineering|ml_readiness)$",
        description="'comprehensive_audit', 'feature_engineering', or 'ml_readiness'",
    )
    target_column: str | None = Field(
        default=None,
        description="Optional target variable name for ML readiness or supervised feature analysis.",
    )


class AgentWorkflowResponse(BaseModel):
    """Output dossier produced by the autonomous analytical agent."""

    workflow_id: str
    dataset_id: str
    filename: str
    workflow_type: str
    overall_health_score: float = Field(
        ...,
        description="Data quality & integrity index (0.0 to 100.0).",
    )
    execution_steps: list[AgentExecutionStep] = Field(
        ...,
        description="Audit trail of autonomous tool-assisted investigation steps.",
    )
    executive_summary: str = Field(..., description="High-level narrative synthesis.")
    key_findings: list[str] = Field(..., description="Major statistical and structural patterns.")
    critical_risks: list[str] = Field(..., description="Data hygiene issues, severe skewness, outliers, or leaks.")
    prescribed_actions: list[str] = Field(..., description="Concrete next steps for engineering and validation.")
    recommended_models: list[str] = Field(
        default_factory=list,
        description="Candidate machine learning algorithms suited for this dataset.",
    )


class AvailableWorkflowsResponse(BaseModel):
    """Catalog of autonomous agent workflows available."""

    workflows: list[dict[str, str]]
