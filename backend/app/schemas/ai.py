"""
backend/app/schemas/ai.py
-------------------------
Pydantic schemas for Phase 5: Google Gemini API Integration.

Models for:
- Executive Dataset Insights & Risk Assessment
- Specific Feature Explanations
- Correlation Interpretation
- Context-Aware Dataset Q&A
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetInsightsResponse(BaseModel):
    """AI-generated executive summary and strategic findings."""

    dataset_id: str
    filename: str
    executive_summary: str = Field(..., description="High-level narrative overview of the dataset.")
    key_findings: list[str] = Field(..., description="Top statistical and domain patterns identified.")
    anomalies_and_risks: list[str] = Field(..., description="Data quality issues, outliers, and potential biases.")
    actionable_recommendations: list[str] = Field(..., description="Practical recommendations for next steps or modeling.")
    model_used: str = Field(..., description="Gemini model utilized (or mock mode if no API key).")


class FeatureExplanationResponse(BaseModel):
    """Deep AI interpretation of an individual column."""

    dataset_id: str
    column: str
    statistical_summary: str = Field(..., description="Summary of feature metrics (mean, spread, skew).")
    domain_significance: str = Field(..., description="Domain-specific interpretation (e.g. healthcare significance).")
    potential_concerns: list[str] = Field(..., description="Outliers, missingness, or skewness implications.")
    engineering_suggestions: list[str] = Field(..., description="Suggested transformations (e.g. log transform, one-hot).")


class CorrelationExplanationResponse(BaseModel):
    """AI interpretation of relationship between two features."""

    dataset_id: str
    feature_a: str
    feature_b: str
    correlation_score: float
    relationship_narrative: str = Field(..., description="Explanation of what this correlation implies.")
    potential_confounders: list[str] = Field(..., description="Possible lurking or confounding variables.")
    causation_caution: str = Field(..., description="Reminder on correlation vs causation for this pair.")


class AskDatasetRequest(BaseModel):
    """Request model for asking questions about the dataset in natural language."""

    question: str = Field(..., min_length=3, max_length=500, description="Natural language analytical question.")


class AskDatasetResponse(BaseModel):
    """AI answer to natural language dataset query."""

    dataset_id: str
    question: str
    answer: str = Field(..., description="Context-aware analytical response.")
    relevant_columns: list[str] = Field(default_factory=list, description="Columns referenced in the answer.")
    confidence: str = Field(default="High", description="'High', 'Medium', or 'Low'")
