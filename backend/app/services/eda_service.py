"""
backend/app/services/eda_service.py
------------------------------------
Core statistical analysis engine for Phase 3: Automated Exploratory Data Analysis.

Calculates:
- Missing value metrics per column and overall
- Duplicate row detection and sample extraction
- Advanced numeric metrics: skewness, kurtosis, variance, IQR outliers, quantiles
- Categorical cardinality, frequencies, and rare labels
- Correlation matrices (Pearson and Spearman) & top correlated feature pairs
- Histogram binning for charting
- Heuristic-based data quality insight generation
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.schemas.eda import (
    CategoryFrequency,
    ColumnMissingValue,
    ComprehensiveEDAResponse,
    CorrelationAnalysis,
    CorrelatedPair,
    DetailedCategoricalFeature,
    DetailedNumericFeature,
    DistributionAnalysis,
    DuplicateAnalysis,
    MissingValueAnalysis,
    NumericHistogram,
    OutlierStats,
    QuantileStats,
)
from app.services.dataset_service import get_dataframe


def _safe_float(val: Any) -> float:
    """Sanitize float to handle NaN, Inf, and Numpy types."""
    try:
        f = float(val)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def analyze_missing_values(df: pd.DataFrame) -> MissingValueAnalysis:
    """Analyze missing values per column and across the entire dataset."""
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    columns_res: list[ColumnMissingValue] = []
    total_missing = 0
    cols_with_missing = 0

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        total_missing += null_count
        if null_count > 0:
            cols_with_missing += 1

        columns_res.append(
            ColumnMissingValue(
                column=col,
                missing_count=null_count,
                missing_percentage=null_pct,
                dtype=str(df[col].dtype),
            )
        )

    # Sort columns by missing percentage descending
    columns_res.sort(key=lambda x: x.missing_percentage, reverse=True)

    overall_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0

    return MissingValueAnalysis(
        total_missing_cells=total_missing,
        total_cells=total_cells,
        overall_missing_percentage=overall_pct,
        columns_with_missing_count=cols_with_missing,
        columns=columns_res,
    )


def analyze_duplicates(df: pd.DataFrame) -> DuplicateAnalysis:
    """Detect and analyze duplicate records in the dataset."""
    total_rows = len(df)
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    dup_pct = round((dup_count / total_rows * 100), 2) if total_rows > 0 else 0.0

    sample_dups: list[dict[str, Any]] = []
    if dup_count > 0:
        sample_df = df[dup_mask].head(5).where(pd.notnull(df), None)
        sample_dups = sample_df.to_dict(orient="records")

    return DuplicateAnalysis(
        total_rows=total_rows,
        duplicate_rows_count=dup_count,
        duplicate_percentage=dup_pct,
        has_duplicates=dup_count > 0,
        sample_duplicates=sample_dups,
    )


def analyze_numeric_features(df: pd.DataFrame) -> list[DetailedNumericFeature]:
    """Compute comprehensive statistical metrics for all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    features: list[DetailedNumericFeature] = []

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        count = int(len(series))
        missing_count = int(df[col].isna().sum())

        if count == 0:
            continue

        mean = _safe_float(series.mean())
        std = _safe_float(series.std(ddof=1) if count > 1 else 0.0)
        variance = _safe_float(series.var(ddof=1) if count > 1 else 0.0)
        col_min = _safe_float(series.min())
        col_max = _safe_float(series.max())
        col_range = _safe_float(col_max - col_min)

        # Skewness and Kurtosis
        skewness = _safe_float(series.skew()) if count >= 3 else 0.0
        kurtosis = _safe_float(series.kurtosis()) if count >= 4 else 0.0

        # Distribution shape
        if abs(skewness) < 0.5:
            dist_shape = "Symmetric"
        elif skewness >= 0.5:
            dist_shape = "Right-Skewed"
        else:
            dist_shape = "Left-Skewed"

        # Quantiles
        q_vals = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        quantiles = QuantileStats(
            **{
                "1%": _safe_float(q_vals.loc[0.01]),
                "5%": _safe_float(q_vals.loc[0.05]),
                "25%": _safe_float(q_vals.loc[0.25]),
                "50%": _safe_float(q_vals.loc[0.50]),
                "75%": _safe_float(q_vals.loc[0.75]),
                "95%": _safe_float(q_vals.loc[0.95]),
                "99%": _safe_float(q_vals.loc[0.99]),
            }
        )

        # Tukey's Outlier Method
        q25 = quantiles.q1
        q75 = quantiles.q3
        iqr = _safe_float(q75 - q25)
        lower_bound = _safe_float(q25 - 1.5 * iqr)
        upper_bound = _safe_float(q75 + 1.5 * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / count * 100), 2) if count > 0 else 0.0

        outliers = OutlierStats(
            iqr=iqr,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_count=outlier_count,
            outlier_percentage=outlier_pct,
        )

        features.append(
            DetailedNumericFeature(
                column=col,
                count=count,
                missing_count=missing_count,
                mean=mean,
                std=std,
                variance=variance,
                min=col_min,
                max=col_max,
                range=col_range,
                skewness=skewness,
                kurtosis=kurtosis,
                quantiles=quantiles,
                outliers=outliers,
                distribution_shape=dist_shape,
            )
        )

    return features


def analyze_categorical_features(df: pd.DataFrame) -> list[DetailedCategoricalFeature]:
    """Analyze all categorical / string columns for cardinality and frequencies."""
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    features: list[DetailedCategoricalFeature] = []

    for col in cat_df.columns:
        series = cat_df[col].dropna().astype(str)
        count = int(len(series))
        missing_count = int(df[col].isna().sum())
        unique_count = int(series.nunique())

        if unique_count <= 10:
            cardinality = "Low"
        elif unique_count <= 50:
            cardinality = "Medium"
        else:
            cardinality = "High"

        # Top 10 categories
        top_cats: list[CategoryFrequency] = []
        has_rare = False

        if count > 0:
            vc = series.value_counts()
            for val, freq in vc.head(10).items():
                pct = round((freq / count * 100), 2)
                top_cats.append(
                    CategoryFrequency(
                        value=str(val),
                        count=int(freq),
                        percentage=pct,
                    )
                )

            # Check if any category has < 1% of non-null records
            has_rare = bool((vc / count < 0.01).any())

        features.append(
            DetailedCategoricalFeature(
                column=col,
                count=count,
                missing_count=missing_count,
                unique_count=unique_count,
                cardinality=cardinality,
                top_categories=top_cats,
                has_rare_categories=has_rare,
            )
        )

    return features


def compute_correlations(df: pd.DataFrame) -> CorrelationAnalysis:
    """Compute Pearson and Spearman correlation matrices and identify top pairs."""
    numeric_df = df.select_dtypes(include=[np.number])
    cols = numeric_df.columns.tolist()

    if len(cols) < 2:
        return CorrelationAnalysis(
            numeric_columns=cols,
            pearson_matrix={},
            spearman_matrix={},
            top_positive_correlations=[],
            top_negative_correlations=[],
        )

    # Compute matrices
    pearson_df = numeric_df.corr(method="pearson")
    spearman_df = numeric_df.corr(method="spearman")

    pearson_matrix: dict[str, dict[str, float]] = {}
    spearman_matrix: dict[str, dict[str, float]] = {}

    for c1 in cols:
        pearson_matrix[c1] = {}
        spearman_matrix[c1] = {}
        for c2 in cols:
            pearson_matrix[c1][c2] = _safe_float(pearson_df.loc[c1, c2])
            spearman_matrix[c1][c2] = _safe_float(spearman_df.loc[c1, c2])

    # Find top correlated pairs (excluding self-correlation and duplicates)
    pairs: list[CorrelatedPair] = []
    seen = set()

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            pair_key = (min(c1, c2), max(c1, c2))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            score = _safe_float(pearson_df.loc[c1, c2])

            # Strength label
            abs_score = abs(score)
            if score >= 0.7:
                strength = "Strong Positive"
            elif score >= 0.3:
                strength = "Moderate Positive"
            elif score <= -0.7:
                strength = "Strong Negative"
            elif score <= -0.3:
                strength = "Moderate Negative"
            else:
                strength = "Weak"

            pairs.append(
                CorrelatedPair(
                    feature_a=c1,
                    feature_b=c2,
                    correlation=score,
                    strength=strength,
                )
            )

    # Separate positive and negative
    pos_pairs = sorted([p for p in pairs if p.correlation > 0], key=lambda x: x.correlation, reverse=True)[:5]
    neg_pairs = sorted([p for p in pairs if p.correlation < 0], key=lambda x: x.correlation)[:5]

    return CorrelationAnalysis(
        numeric_columns=cols,
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        top_positive_correlations=pos_pairs,
        top_negative_correlations=neg_pairs,
    )


def compute_distributions(df: pd.DataFrame) -> DistributionAnalysis:
    """Compute histogram data for numeric columns and value counts for categoricals."""
    histograms: list[NumericHistogram] = []
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue
        try:
            counts, bin_edges = np.histogram(series, bins="auto")
            histograms.append(
                NumericHistogram(
                    column=col,
                    bin_edges=[_safe_float(b) for b in bin_edges],
                    counts=[int(c) for c in counts],
                )
            )
        except Exception:
            # Fallback to fixed 10 bins if auto fails
            counts, bin_edges = np.histogram(series, bins=10)
            histograms.append(
                NumericHistogram(
                    column=col,
                    bin_edges=[_safe_float(b) for b in bin_edges],
                    counts=[int(c) for c in counts],
                )
            )

    # Categorical distributions
    cat_freqs: dict[str, list[CategoryFrequency]] = {}
    cat_df = df.select_dtypes(include=["object", "category", "bool"])

    for col in cat_df.columns:
        series = cat_df[col].dropna().astype(str)
        count = len(series)
        col_list: list[CategoryFrequency] = []
        if count > 0:
            for val, freq in series.value_counts().head(15).items():
                col_list.append(
                    CategoryFrequency(
                        value=str(val),
                        count=int(freq),
                        percentage=round((freq / count * 100), 2),
                    )
                )
        cat_freqs[col] = col_list

    return DistributionAnalysis(
        numeric_histograms=histograms,
        categorical_frequencies=cat_freqs,
    )


def generate_insights(
    df: pd.DataFrame,
    missing: MissingValueAnalysis,
    duplicates: DuplicateAnalysis,
    numerics: list[DetailedNumericFeature],
    categoricals: list[DetailedCategoricalFeature],
    correlations: CorrelationAnalysis,
) -> list[str]:
    """Generate rule-based data quality and statistical insights."""
    insights: list[str] = []

    # 1. Missing Values
    if missing.total_missing_cells == 0:
        insights.append("✅ Data Quality: No missing values detected across the dataset.")
    elif missing.overall_missing_percentage > 20:
        insights.append(
            f"⚠️ Data Quality: High proportion of missing data ({missing.overall_missing_percentage}% total). "
            f"{missing.columns_with_missing_count} column(s) contain missing values."
        )
    else:
        insights.append(
            f"ℹ️ Data Quality: {missing.overall_missing_percentage}% missing values detected in {missing.columns_with_missing_count} column(s)."
        )

    # 2. Duplicates
    if duplicates.has_duplicates:
        insights.append(
            f"⚠️ Duplicates: Found {duplicates.duplicate_rows_count} duplicate row(s) ({duplicates.duplicate_percentage}%). Consider deduplicating prior to modeling."
        )
    else:
        insights.append("✅ Duplicates: No duplicate rows detected.")

    # 3. Outliers
    high_outlier_cols = [n.column for n in numerics if n.outliers.outlier_percentage > 5.0]
    if high_outlier_cols:
        insights.append(
            f"⚠️ Outliers: Significant outliers (>5%) detected in numeric feature(s): {', '.join(high_outlier_cols)}."
        )

    # 4. Skewness
    skewed_cols = [f"{n.column} ({n.distribution_shape})" for n in numerics if n.distribution_shape != "Symmetric"]
    if skewed_cols:
        insights.append(
            f"📊 Distributions: Skewed distributions observed in: {', '.join(skewed_cols[:3])}."
        )

    # 5. Correlations
    strong_pos = correlations.top_positive_correlations
    if strong_pos and strong_pos[0].correlation >= 0.7:
        p = strong_pos[0]
        insights.append(
            f"🔗 Correlation: Strong positive relationship detected between '{p.feature_a}' and '{p.feature_b}' (r = {p.correlation})."
        )

    strong_neg = correlations.top_negative_correlations
    if strong_neg and strong_neg[0].correlation <= -0.7:
        p = strong_neg[0]
        insights.append(
            f"🔗 Correlation: Strong negative relationship detected between '{p.feature_a}' and '{p.feature_b}' (r = {p.correlation})."
        )

    # 6. High Cardinality
    high_card_cols = [c.column for c in categoricals if c.cardinality == "High"]
    if high_card_cols:
        insights.append(
            f"ℹ️ Categorical: High cardinality features detected ({', '.join(high_card_cols)}). May require encoding or binning."
        )

    return insights


def generate_comprehensive_eda(dataset_id: str) -> ComprehensiveEDAResponse:
    """Generate the full automated EDA report for a dataset."""
    df, filename = get_dataframe(dataset_id)
    total_rows, total_cols = df.shape

    missing = analyze_missing_values(df)
    duplicates = analyze_duplicates(df)
    numerics = analyze_numeric_features(df)
    categoricals = analyze_categorical_features(df)
    correlations = compute_correlations(df)
    distributions = compute_distributions(df)
    insights = generate_insights(df, missing, duplicates, numerics, categoricals, correlations)

    return ComprehensiveEDAResponse(
        dataset_id=dataset_id,
        filename=filename,
        total_rows=total_rows,
        total_columns=total_cols,
        missing_analysis=missing,
        duplicate_analysis=duplicates,
        numeric_features=numerics,
        categorical_features=categoricals,
        correlations=correlations,
        distributions=distributions,
        key_insights=insights,
    )
