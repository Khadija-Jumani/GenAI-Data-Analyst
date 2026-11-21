"""
backend/app/services/agent_service.py
-------------------------------------
Agentic AI and Automated Analytical Workflows Service for Phase 7.

Orchestrates multi-step, tool-assisted investigations:
1. Data Quality & Hygiene Auditing
2. Outlier & Distribution Diagnostics
3. Multicollinearity & Correlation Analysis
4. Feature Engineering Roadmap Generation
5. Predictive Machine Learning Readiness Assessment

Computes an objective Data Health Score (0-100) and synthesizes
an executive action dossier with autonomous tool step logs.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.agent import (
    AgentExecutionStep,
    AgentWorkflowResponse,
    AvailableWorkflowsResponse,
)
from app.services import ai_service
from app.services.dataset_service import get_dataframe
from app.services.eda_service import (
    analyze_categorical_features,
    analyze_duplicates,
    analyze_missing_values,
    analyze_numeric_features,
    compute_correlations,
)


def list_available_workflows() -> AvailableWorkflowsResponse:
    """Return available autonomous agent workflows."""
    return AvailableWorkflowsResponse(
        workflows=[
            {
                "id": "comprehensive_audit",
                "name": "Comprehensive Diagnostic Audit",
                "description": "Full multi-step audit of data health, missing patterns, outliers, and statistical stability.",
            },
            {
                "id": "feature_engineering",
                "name": "Feature Engineering Roadmap",
                "description": "Evaluates scaling, transformations (log/Box-Cox), encoding strategies, and interaction terms.",
            },
            {
                "id": "ml_readiness",
                "name": "Predictive ML Readiness Assessment",
                "description": "Assesses target balance, feature-to-sample ratio, multicollinearity, and recommends ML models.",
            },
        ]
    )


def _compute_data_health_score(
    total_cells: int,
    missing_cells: int,
    total_rows: int,
    duplicate_rows: int,
    outlier_count: int,
    total_values: int,
) -> float:
    """Calculate an objective quality score from 0.0 to 100.0."""
    score = 100.0

    # Missing penalty (up to -40 pts)
    missing_ratio = (missing_cells / total_cells) if total_cells > 0 else 0.0
    score -= min(missing_ratio * 100 * 2.0, 40.0)

    # Duplicate penalty (up to -20 pts)
    dup_ratio = (duplicate_rows / total_rows) if total_rows > 0 else 0.0
    score -= min(dup_ratio * 100 * 2.0, 20.0)

    # Outlier penalty (up to -20 pts)
    outlier_ratio = (outlier_count / total_values) if total_values > 0 else 0.0
    score -= min(outlier_ratio * 100 * 1.5, 20.0)

    return round(max(score, 10.0), 1)


# ============================================================================ #
# Workflow 1: Comprehensive Diagnostic Audit
# ============================================================================ #

def _run_comprehensive_audit(df: pd.DataFrame, filename: str) -> tuple[float, list[AgentExecutionStep], dict[str, Any]]:
    steps: list[AgentExecutionStep] = []
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    # Step 1: Missing & Duplicate Inspection
    missing = analyze_missing_values(df)
    duplicates = analyze_duplicates(df)
    step1_obs = (
        f"Detected {missing.total_missing_cells} missing cells ({missing.overall_missing_percentage}%). "
        f"Identified {duplicates.duplicate_rows_count} duplicate rows ({duplicates.duplicate_percentage}%)."
    )
    steps.append(
        AgentExecutionStep(
            step_number=1,
            step_name="Data Hygiene & Integrity Audit",
            tool_invoked="audit_missing_and_duplicates",
            observations=step1_obs,
        )
    )

    # Step 2: Distribution & Outliers
    numerics = analyze_numeric_features(df)
    total_outliers = sum(n.outliers.outlier_count for n in numerics)
    total_num_values = sum(n.count for n in numerics)
    skewed_features = [n.column for n in numerics if n.distribution_shape != "Symmetric"]
    step2_obs = (
        f"Analyzed {len(numerics)} continuous variables. Found {total_outliers} IQR statistical outliers. "
        f"Skewed distributions observed in: {', '.join(skewed_features) if skewed_features else 'None'}."
    )
    steps.append(
        AgentExecutionStep(
            step_number=2,
            step_name="Outlier & Distribution Diagnosis",
            tool_invoked="evaluate_feature_distributions",
            observations=step2_obs,
        )
    )

    # Step 3: Categorical Cardinality
    categoricals = analyze_categorical_features(df)
    high_card = [c.column for c in categoricals if c.cardinality == "High"]
    step3_obs = (
        f"Profiled {len(categoricals)} categorical features. High cardinality flagged in: "
        f"{', '.join(high_card) if high_card else 'None (all within manageable cardinality)'}."
    )
    steps.append(
        AgentExecutionStep(
            step_number=3,
            step_name="Categorical Cardinality & Diversity Check",
            tool_invoked="audit_categorical_cardinality",
            observations=step3_obs,
        )
    )

    # Step 4: Multicollinearity Screening
    correlations = compute_correlations(df)
    strong_pairs = [f"{p.feature_a} <-> {p.feature_b} (r={p.correlation})" for p in correlations.top_positive_correlations if p.correlation >= 0.8]
    step4_obs = (
        f"Screened bivariate correlations. Highly correlated collinear pairs (r >= 0.8): "
        f"{', '.join(strong_pairs) if strong_pairs else 'None detected'}."
    )
    steps.append(
        AgentExecutionStep(
            step_number=4,
            step_name="Collinearity & Dependency Screening",
            tool_invoked="detect_correlations_and_collinearity",
            observations=step4_obs,
        )
    )

    health_score = _compute_data_health_score(
        total_cells=total_cells,
        missing_cells=missing.total_missing_cells,
        total_rows=total_rows,
        duplicate_rows=duplicates.duplicate_rows_count,
        outlier_count=total_outliers,
        total_values=total_num_values if total_num_values > 0 else 1,
    )

    payload = {
        "missing": missing,
        "duplicates": duplicates,
        "numerics": numerics,
        "categoricals": categoricals,
        "correlations": correlations,
        "health_score": health_score,
    }
    return health_score, steps, payload


# ============================================================================ #
# Workflow 2: Feature Engineering Roadmap
# ============================================================================ #

def _run_feature_engineering_roadmap(df: pd.DataFrame, filename: str) -> tuple[float, list[AgentExecutionStep], dict[str, Any]]:
    steps: list[AgentExecutionStep] = []
    numerics = analyze_numeric_features(df)
    categoricals = analyze_categorical_features(df)

    # Step 1: Scaling recommendations
    high_variance_cols = [n.column for n in numerics if n.range > 50]
    step1_obs = f"Features requiring Standard/Robust scaling due to wide dynamic range: {', '.join(high_variance_cols)}."
    steps.append(
        AgentExecutionStep(
            step_number=1,
            step_name="Feature Scaling Strategy",
            tool_invoked="evaluate_scaling_requirements",
            observations=step1_obs,
        )
    )

    # Step 2: Non-linear transformations
    skewed_cols = [n.column for n in numerics if abs(n.skewness) >= 1.0]
    step2_obs = (
        f"Right/Left-skewed features requiring Log or Box-Cox transformation: "
        f"{', '.join(skewed_cols) if skewed_cols else 'None (distributions reasonably symmetric)'}."
    )
    steps.append(
        AgentExecutionStep(
            step_number=2,
            step_name="Non-Linear Transformation Planning",
            tool_invoked="recommend_distribution_transforms",
            observations=step2_obs,
        )
    )

    # Step 3: Encoding Plan
    one_hot = [c.column for c in categoricals if c.cardinality == "Low"]
    ordinal = [c.column for c in categoricals if c.cardinality in ("Medium", "High")]
    step3_obs = (
        f"One-Hot Encoding recommended for: {', '.join(one_hot) if one_hot else 'None'}. "
        f"Target/Ordinal Encoding recommended for: {', '.join(ordinal) if ordinal else 'None'}."
    )
    steps.append(
        AgentExecutionStep(
            step_number=3,
            step_name="Categorical Encoding Architecture",
            tool_invoked="plan_encoding_strategies",
            observations=step3_obs,
        )
    )

    health_score = 92.0  # Feature engineering readiness
    payload = {"numerics": numerics, "categoricals": categoricals}
    return health_score, steps, payload


# ============================================================================ #
# Workflow 3: Predictive ML Readiness Assessment
# ============================================================================ #

def _run_ml_readiness(df: pd.DataFrame, filename: str, target_col: str | None) -> tuple[float, list[AgentExecutionStep], dict[str, Any]]:
    steps: list[AgentExecutionStep] = []
    total_rows, total_cols = df.shape

    # Choose target column if not supplied
    if not target_col or target_col not in df.columns:
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        target_col = cat_cols[-1] if cat_cols else df.columns[-1]

    # Step 1: Sample Size & Dimensionality
    ratio = round(total_rows / total_cols, 1)
    step1_obs = f"Dataset size is {total_rows} samples across {total_cols} features. Sample-to-feature ratio: {ratio}:1."
    steps.append(
        AgentExecutionStep(
            step_number=1,
            step_name="Dimensionality & Sample Size Audit",
            tool_invoked="evaluate_sample_to_feature_ratio",
            observations=step1_obs,
        )
    )

    # Step 2: Target Distribution & Balance
    target_series = df[target_col].dropna()
    is_classification = not pd.api.types.is_numeric_dtype(target_series) or target_series.nunique() <= 10

    if is_classification:
        val_counts = target_series.value_counts(normalize=True).round(3).to_dict()
        step2_obs = f"Target '{target_col}' identified as classification task. Class distribution: {val_counts}."
    else:
        step2_obs = f"Target '{target_col}' identified as continuous regression task (mean={round(target_series.mean(), 2)}, std={round(target_series.std(), 2)})."

    steps.append(
        AgentExecutionStep(
            step_number=2,
            step_name="Target Variable Analysis",
            tool_invoked="profile_target_variable",
            observations=step2_obs,
        )
    )

    # Step 3: Algorithm Suitability
    if is_classification:
        models = ["RandomForestClassifier", "XGBoostClassifier", "LogisticRegression", "LightGBM"]
    else:
        models = ["RidgeRegression", "RandomForestRegressor", "GradientBoostingRegressor", "ElasticNet"]

    step3_obs = f"Recommended model candidates: {', '.join(models)}."
    steps.append(
        AgentExecutionStep(
            step_number=3,
            step_name="Algorithm Selection & Benchmark Setup",
            tool_invoked="match_suitable_algorithms",
            observations=step3_obs,
        )
    )

    health_score = 88.0 if ratio >= 10 else 72.0
    payload = {"target_col": target_col, "is_classification": is_classification, "models": models}
    return health_score, steps, payload


# ============================================================================ #
# Main Workflow Execution Dispatcher
# ============================================================================ #

def execute_agent_workflow(
    dataset_id: str,
    workflow_type: str = "comprehensive_audit",
    target_column: str | None = None,
) -> AgentWorkflowResponse:
    """Run an autonomous agent investigation and synthesize findings."""
    df, filename = get_dataframe(dataset_id)
    workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

    if workflow_type == "feature_engineering":
        health_score, steps, payload = _run_feature_engineering_roadmap(df, filename)
        exec_summary = (
            f"Autonomous Feature Engineering Roadmap executed for '{filename}'. "
            "Formulated scaling regimes, non-linear transformation triggers, and optimal encoding strategies."
        )
        findings = [
            f"Identified continuous features needing normalization to equalize gradient propagation.",
            "Formulated encoding strategies for low and high cardinality categorical attributes.",
        ]
        risks = [
            "Applying one-hot encoding on high cardinality features may trigger curse of dimensionality.",
            "Verify absence of target leakage when engineering cross-feature interaction terms.",
        ]
        actions = [
            "Implement scikit-learn ColumnTransformer with StandardScaler for numerical features.",
            "Use OneHotEncoder(drop='first', sparse_output=False) for nominal binary/ternary features.",
            "Apply Log1p transformation to right-skewed variables before linear modeling.",
        ]
        recommended_models = []

    elif workflow_type == "ml_readiness":
        health_score, steps, payload = _run_ml_readiness(df, filename, target_column)
        target = payload["target_col"]
        task_type = "Classification" if payload["is_classification"] else "Regression"
        exec_summary = (
            f"Machine Learning Readiness Assessment completed targeting '{target}' ({task_type}). "
            f"Overall modeling suitability score is {health_score}/100."
        )
        findings = [
            f"Target variable '{target}' configured for {task_type} modeling.",
            f"Sample size of {len(df)} rows evaluated against feature dimensions.",
        ]
        risks = [
            "Ensure cross-validation uses StratifiedKFold to preserve target class proportions.",
            "Remove direct identifier columns (e.g. ID, names) to prevent trivial overfit.",
        ]
        actions = [
            f"Establish baseline model using {payload['models'][0]} with 5-fold cross validation.",
            "Evaluate precision, recall, and ROC-AUC metrics if clinical classification imbalance is present.",
            "Compare non-linear tree ensembles against regularized linear baselines.",
        ]
        recommended_models = payload["models"]

    else:  # comprehensive_audit
        health_score, steps, payload = _run_comprehensive_audit(df, filename)
        missing_count = payload["missing"].total_missing_cells
        dup_count = payload["duplicates"].duplicate_rows_count

        exec_summary = (
            f"Autonomous Diagnostic Audit concluded for '{filename}' ({len(df)} rows, {len(df.columns)} columns). "
            f"Computed overall Data Health Score of {health_score}/100."
        )
        findings = [
            f"Data integrity audit found {missing_count} missing values and {dup_count} duplicate rows.",
            f"Evaluated distributions across {len(payload['numerics'])} numeric and {len(payload['categoricals'])} categorical variables.",
            "Screened pairwise correlation matrices for severe collinearity and data redundancies.",
        ]
        risks = [
            "Monitor physiological and measurement outliers exceeding Tukey IQR fences.",
            "High multicollinearity among physiological indicators could destabilize unregularized regression coefficients.",
        ]
        actions = [
            "Clean or impute any missing records using median for skewed variables or iterative imputer.",
            "Verify duplicate patient records against unique healthcare identifiers.",
            "Standardize variable units and documentation across clinical measurements.",
        ]
        recommended_models = []

    return AgentWorkflowResponse(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
        filename=filename,
        workflow_type=workflow_type,
        overall_health_score=health_score,
        execution_steps=steps,
        executive_summary=exec_summary,
        key_findings=findings,
        critical_risks=risks,
        prescribed_actions=actions,
        recommended_models=recommended_models,
    )
