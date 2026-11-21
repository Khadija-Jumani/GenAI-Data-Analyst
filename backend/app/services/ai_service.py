"""
backend/app/services/ai_service.py
-----------------------------------
Google Gemini Generative AI Service for Phase 5.

Features:
- Configures Google Gemini SDK (`google.generativeai`)
- Builds rich statistical context from EDA service (structure, summary, outliers, correlations)
- Generates:
  1. Executive Dataset Insights & Risk Assessment
  2. Deep Feature Interpretations
  3. Bivariate Correlation Explanations
  4. Natural-Language Dataset Q&A
- Graceful Mock Fallback: When no Gemini API key is configured or offline,
  it generates heuristic-based intelligent analyses so the frontend and API
  function seamlessly without throwing runtime errors.
"""

from __future__ import annotations

import json
from typing import Any

import google.generativeai as genai
import pandas as pd

from app.core.config import settings
from app.schemas.ai import (
    AskDatasetResponse,
    CorrelationExplanationResponse,
    DatasetInsightsResponse,
    FeatureExplanationResponse,
)
from app.services.dataset_service import get_dataframe
from app.services.eda_service import (
    analyze_categorical_features,
    analyze_duplicates,
    analyze_missing_values,
    analyze_numeric_features,
    compute_correlations,
)


def _is_gemini_available() -> bool:
    """Check if a non-placeholder Gemini API key is configured."""
    key = settings.GEMINI_API_KEY.strip()
    return bool(key and key != "your_gemini_api_key_here")


def _init_gemini() -> genai.GenerativeModel | None:
    """Initialize and return a Gemini GenerativeModel instance if configured."""
    if not _is_gemini_available():
        return None
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        return genai.GenerativeModel(settings.GEMINI_MODEL)
    except Exception:
        return None


def _build_context_summary(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Compile structured statistical context for prompt engineering."""
    total_rows, total_cols = df.shape
    missing = analyze_missing_values(df)
    duplicates = analyze_duplicates(df)
    numerics = analyze_numeric_features(df)
    categoricals = analyze_categorical_features(df)
    correlations = compute_correlations(df)

    # Top numeric summaries
    num_summary = [
        {
            "column": n.column,
            "mean": n.mean,
            "std": n.std,
            "min": n.min,
            "max": n.max,
            "shape": n.distribution_shape,
            "outliers_count": n.outliers.outlier_count,
        }
        for n in numerics[:8]
    ]

    # Top categorical summaries
    cat_summary = [
        {
            "column": c.column,
            "unique_count": c.unique_count,
            "cardinality": c.cardinality,
            "top_categories": [cat.value for cat in c.top_categories[:4]],
        }
        for c in categoricals[:6]
    ]

    top_corrs = [
        {"feature_a": p.feature_a, "feature_b": p.feature_b, "correlation": p.correlation, "strength": p.strength}
        for p in correlations.top_positive_correlations[:3] + correlations.top_negative_correlations[:3]
    ]

    return {
        "filename": filename,
        "rows": total_rows,
        "columns": total_cols,
        "total_missing_cells": missing.total_missing_cells,
        "duplicate_rows": duplicates.duplicate_rows_count,
        "numeric_features": num_summary,
        "categorical_features": cat_summary,
        "top_correlations": top_corrs,
        "sample_preview": df.head(3).to_dict(orient="records"),
    }


# ============================================================================ #
# 1. Executive Dataset Insights
# ============================================================================ #

def generate_dataset_insights(dataset_id: str) -> DatasetInsightsResponse:
    """Generate high-level strategic insights and data quality assessments."""
    df, filename = get_dataframe(dataset_id)
    context = _build_context_summary(df, filename)
    model = _init_gemini()

    if model:
        prompt = f"""
You are a Principal Data Scientist and Healthcare Analytics Expert.
Analyze the following dataset context and produce a structured JSON response:

Dataset Context:
{json.dumps(context, indent=2)}

Required JSON Schema:
{{
  "executive_summary": "2-3 sentences providing an executive overview of the dataset structure, domain, and purpose.",
  "key_findings": ["3 to 5 bullet points highlighting the most significant statistical and domain trends."],
  "anomalies_and_risks": ["2 to 4 bullet points highlighting outliers, missing data, skewness, or potential clinical/business risks."],
  "actionable_recommendations": ["3 to 4 actionable recommendations for feature engineering, data cleaning, or machine learning."]
}}
Return ONLY valid JSON with no markdown backticks or commentary.
"""
        try:
            response = model.generate_content(prompt)
            clean_text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean_text)
            return DatasetInsightsResponse(
                dataset_id=dataset_id,
                filename=filename,
                executive_summary=data["executive_summary"],
                key_findings=data["key_findings"],
                anomalies_and_risks=data["anomalies_and_risks"],
                actionable_recommendations=data["actionable_recommendations"],
                model_used=settings.GEMINI_MODEL,
            )
        except Exception:
            pass  # Fallback to intelligent rule-based generation

    # Rule-based fallback
    num_cols = len(context["numeric_features"])
    cat_cols = len(context["categorical_features"])
    corr_highlights = [f"{p['feature_a']} & {p['feature_b']} (r={p['correlation']})" for p in context["top_correlations"][:2]]
    corr_str = ", ".join(corr_highlights) if corr_highlights else "independent features"

    return DatasetInsightsResponse(
        dataset_id=dataset_id,
        filename=filename,
        executive_summary=(
            f"The '{filename}' dataset contains {context['rows']} records and {context['columns']} variables "
            f"spanning {num_cols} numerical features and {cat_cols} categorical indicators. "
            "The data represents clinical patient vitals and health risk factors suited for predictive modeling and risk stratification."
        ),
        key_findings=[
            f"Dataset dimensions: {context['rows']} patients across {context['columns']} diagnostic variables.",
            f"Key statistical correlations detected between: {corr_str}.",
            f"Overall data integrity: {context['total_missing_cells']} missing values and {context['duplicate_rows']} duplicate records.",
        ],
        anomalies_and_risks=[
            "Evaluate clinical outlier boundaries (e.g. elevated blood pressure and glucose spikes) to ensure measurement validity.",
            "Potential class imbalance across target categories could bias classification algorithms if unweighted.",
        ],
        actionable_recommendations=[
            "Apply standard scaling (Z-score normalization) to continuous features before distance-based modeling.",
            "Encode categorical risk factors with ordinal or one-hot encoding depending on target model architecture.",
            "Explore interaction terms between blood pressure, glucose, and cholesterol indicators.",
        ],
        model_used="GenAI-RuleEngine (Configure GEMINI_API_KEY in .env for Live Gemini Flash)",
    )


# ============================================================================ #
# 2. Deep Feature Explanation
# ============================================================================ #

def explain_feature(dataset_id: str, column: str) -> FeatureExplanationResponse:
    """Provide detailed domain and statistical interpretation of a specific feature."""
    df, filename = get_dataframe(dataset_id)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    series = df[column].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    model = _init_gemini()

    feature_meta = {
        "column": column,
        "is_numeric": is_numeric,
        "count": len(series),
        "missing_count": int(df[column].isna().sum()),
        "unique_values": int(series.nunique()),
        "sample_values": series.head(5).tolist(),
    }
    if is_numeric:
        feature_meta.update({
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2) if len(series) > 1 else 0,
            "min": float(series.min()),
            "max": float(series.max()),
            "skewness": round(float(series.skew()), 2) if len(series) > 2 else 0,
        })

    if model:
        prompt = f"""
You are an expert Data Scientist. Explain the statistical profile and domain significance of the feature '{column}'.
Context:
{json.dumps(feature_meta, indent=2)}

Required JSON Schema:
{{
  "statistical_summary": "1-2 sentences summarizing distribution, spread, and values.",
  "domain_significance": "1-2 sentences on what this variable represents in healthcare/business context.",
  "potential_concerns": ["1-3 items on outliers, skewness, or missingness."],
  "engineering_suggestions": ["1-3 suggestions on preprocessing/feature engineering."]
}}
Return ONLY valid JSON.
"""
        try:
            response = model.generate_content(prompt)
            clean_text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean_text)
            return FeatureExplanationResponse(
                dataset_id=dataset_id,
                column=column,
                statistical_summary=data["statistical_summary"],
                domain_significance=data["domain_significance"],
                potential_concerns=data["potential_concerns"],
                engineering_suggestions=data["engineering_suggestions"],
            )
        except Exception:
            pass

    # Rule-based fallback
    if is_numeric:
        stat_sum = (
            f"The feature '{column}' is continuous ranging from {feature_meta['min']} to {feature_meta['max']} "
            f"with a mean of {feature_meta['mean']} (std: {feature_meta['std']}) and skewness of {feature_meta.get('skewness', 0)}."
        )
        concerns = [
            f"Check values near the extremes ({feature_meta['min']} and {feature_meta['max']}) for potential physiological outliers.",
            "Verify measurement scale consistency across patient recording intervals.",
        ]
        eng = [
            "Consider robust scaling (RobustScaler) if extreme outliers represent valid clinical spikes.",
            "Generate binned categorical representations (e.g. Normal, Borderline, Elevated) for clinical interpretability.",
        ]
    else:
        stat_sum = f"'{column}' is categorical containing {feature_meta['unique_values']} distinct categories."
        concerns = ["Ensure consistent casing and whitespace formatting across categorical entries."]
        eng = ["Apply One-Hot Encoding for nominal data or Ordinal Encoding if categories have a natural ranking."]

    return FeatureExplanationResponse(
        dataset_id=dataset_id,
        column=column,
        statistical_summary=stat_sum,
        domain_significance=f"In analytical modeling, '{column}' provides critical discriminant signal regarding patient health profiles.",
        potential_concerns=concerns,
        engineering_suggestions=eng,
    )


# ============================================================================ #
# 3. Correlation Explanation
# ============================================================================ #

def explain_correlation(dataset_id: str, feature_a: str, feature_b: str) -> CorrelationExplanationResponse:
    """Explain the bivariate correlation between two features."""
    df, filename = get_dataframe(dataset_id)
    if feature_a not in df.columns or feature_b not in df.columns:
        raise ValueError(f"Features '{feature_a}' or '{feature_b}' not found in dataset.")

    series_a = df[feature_a].dropna()
    series_b = df[feature_b].dropna()
    corr_score = round(float(series_a.corr(series_b)), 4) if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b) else 0.0

    model = _init_gemini()
    if model:
        prompt = f"""
Analyze the correlation coefficient of r = {corr_score} between '{feature_a}' and '{feature_b}'.
Required JSON Schema:
{{
  "relationship_narrative": "2 sentences explaining the magnitude and clinical/business meaning of this correlation.",
  "potential_confounders": ["2-3 possible confounding or mediating variables."],
  "causation_caution": "1 sentence reminding about non-causality."
}}
Return ONLY valid JSON.
"""
        try:
            response = model.generate_content(prompt)
            clean_text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean_text)
            return CorrelationExplanationResponse(
                dataset_id=dataset_id,
                feature_a=feature_a,
                feature_b=feature_b,
                correlation_score=corr_score,
                relationship_narrative=data["relationship_narrative"],
                potential_confounders=data["potential_confounders"],
                causation_caution=data["causation_caution"],
            )
        except Exception:
            pass

    # Fallback
    abs_c = abs(corr_score)
    strength = "strong" if abs_c >= 0.7 else ("moderate" if abs_c >= 0.3 else "weak")
    direction = "positive" if corr_score > 0 else "negative"

    narrative = (
        f"A {strength} {direction} correlation (r = {corr_score}) exists between '{feature_a}' and '{feature_b}'. "
        f"As {feature_a} increases, {feature_b} tends to systematically {'increase' if corr_score > 0 else 'decrease'}."
    )

    return CorrelationExplanationResponse(
        dataset_id=dataset_id,
        feature_a=feature_a,
        feature_b=feature_b,
        correlation_score=corr_score,
        relationship_narrative=narrative,
        potential_confounders=[
            "Patient age and lifestyle factors acting as common causes.",
            "Underlying metabolic and cardiovascular feedback mechanisms.",
        ],
        causation_caution=f"Note: A correlation of {corr_score} confirms statistical co-movement, not direct causation.",
    )


# ============================================================================ #
# 4. Natural-Language Dataset Q&A
# ============================================================================ #

def ask_dataset(dataset_id: str, question: str) -> AskDatasetResponse:
    """Answer natural language inquiries about the dataset."""
    df, filename = get_dataframe(dataset_id)
    context = _build_context_summary(df, filename)
    model = _init_gemini()

    if model:
        prompt = f"""
You are an intelligent AI Data Analyst. Answer the user's analytical question based strictly on the dataset context provided.

Dataset Context:
{json.dumps(context, indent=2)}

User Question:
"{question}"

Required JSON Schema:
{{
  "answer": "Clear, direct, and insightful answer to the question citing specific metrics or numbers from the context where relevant.",
  "relevant_columns": ["List of column names directly related to the question and answer."],
  "confidence": "'High', 'Medium', or 'Low'"
}}
Return ONLY valid JSON.
"""
        try:
            response = model.generate_content(prompt)
            clean_text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean_text)
            return AskDatasetResponse(
                dataset_id=dataset_id,
                question=question,
                answer=data["answer"],
                relevant_columns=data.get("relevant_columns", []),
                confidence=data.get("confidence", "High"),
            )
        except Exception:
            pass

    # Heuristic fallback matching question keywords
    q_lower = question.lower()
    cols = df.columns.tolist()
    matched_cols = [c for c in cols if c.lower() in q_lower]

    if "missing" in q_lower or "null" in q_lower:
        answer = f"The dataset '{filename}' has {context['total_missing_cells']} missing values across its {context['columns']} columns."
    elif "row" in q_lower or "size" in q_lower or "count" in q_lower or "patient" in q_lower:
        answer = f"The dataset '{filename}' contains {context['rows']} records and {context['columns']} variables."
    elif "correlation" in q_lower or "relationship" in q_lower:
        top_c = context["top_correlations"]
        c_str = ", ".join([f"{c['feature_a']} with {c['feature_b']} (r={c['correlation']})" for c in top_c[:2]])
        answer = f"The strongest observed correlations in the data are: {c_str}."
    else:
        answer = (
            f"Based on the dataset '{filename}' ({context['rows']} rows, {context['columns']} columns), "
            f"your question touches on features: {', '.join(matched_cols) if matched_cols else 'overall dataset metrics'}. "
            "For full generative conversational reasoning, set your GEMINI_API_KEY in the .env file."
        )

    return AskDatasetResponse(
        dataset_id=dataset_id,
        question=question,
        answer=answer,
        relevant_columns=matched_cols,
        confidence="High" if matched_cols else "Medium",
    )
