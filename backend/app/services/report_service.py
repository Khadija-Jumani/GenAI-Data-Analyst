"""
backend/app/services/report_service.py
--------------------------------------
Report Generation and Export Service for Phase 8.

Generates comprehensive analytical reports in three formats:
1. Markdown (.md) — Portable, readable documentation with GitHub Flavored Markdown.
2. Standalone HTML (.html) — Polished executive report with modern responsive styling,
   metric cards, stat tables, and print-ready CSS.
3. Machine-readable JSON (.json) — Structured analytical dossier.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.schemas.report import ReportSummaryResponse
from app.services import agent_service, ai_service, eda_service
from app.services.dataset_service import get_dataframe


def get_report_summary(dataset_id: str) -> ReportSummaryResponse:
    """Return high-level metadata and available export links for a dataset report."""
    df, filename = get_dataframe(dataset_id)
    now = datetime.now(timezone.utc).isoformat()
    audit = agent_service.execute_agent_workflow(dataset_id, "comprehensive_audit")

    return ReportSummaryResponse(
        dataset_id=dataset_id,
        filename=filename,
        generated_at=now,
        overall_health_score=audit.overall_health_score,
        sections_included=[
            "Executive Overview",
            "Data Health & Hygiene Audit",
            "Detailed Numerical Statistics",
            "Categorical Feature Analysis",
            "Correlation & Collinearity Matrix",
            "Key Insights & Risk Factors",
            "Actionable Recommendations",
        ],
        download_urls={
            "markdown": f"/api/v1/reports/{dataset_id}/markdown",
            "html": f"/api/v1/reports/{dataset_id}/html",
            "json": f"/api/v1/reports/{dataset_id}/json",
        },
    )


def generate_json_dossier(dataset_id: str) -> dict[str, Any]:
    """Assemble complete analytical dossier as a structured JSON dictionary."""
    df, filename = get_dataframe(dataset_id)
    now = datetime.now(timezone.utc).isoformat()
    eda_comp = eda_service.generate_comprehensive_eda(dataset_id)
    ai_insights = ai_service.generate_dataset_insights(dataset_id)
    audit = agent_service.execute_agent_workflow(dataset_id, "comprehensive_audit")

    return {
        "report_title": f"GenAI Data Analyst — Analytical Dossier for {filename}",
        "dataset_id": dataset_id,
        "filename": filename,
        "generated_at": now,
        "data_health_score": audit.overall_health_score,
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "executive_insights": ai_insights.model_dump(),
        "eda_report": eda_comp.model_dump(),
        "agent_audit": audit.model_dump(),
    }


def generate_markdown_report(dataset_id: str) -> str:
    """Generate a clean, professional Markdown report."""
    df, filename = get_dataframe(dataset_id)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    audit = agent_service.execute_agent_workflow(dataset_id, "comprehensive_audit")
    ai_insights = ai_service.generate_dataset_insights(dataset_id)
    eda_comp = eda_service.generate_comprehensive_eda(dataset_id)

    lines = [
        f"# 📊 GenAI Data Analyst — Analytical Report",
        f"**Dataset:** `{filename}` | **Generated:** `{now}` | **Data Health Score:** `{audit.overall_health_score}/100`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        ai_insights.executive_summary,
        "",
        "### Key Findings",
    ]
    for kf in ai_insights.key_findings:
        lines.append(f"- {kf}")

    lines.extend([
        "",
        "### Data Quality & Hygiene Assessment",
        f"- **Total Records (Rows):** {len(df)}",
        f"- **Total Variables (Columns):** {len(df.columns)}",
        f"- **Missing Cells:** {eda_comp.missing_analysis.total_missing_cells} ({eda_comp.missing_analysis.overall_missing_percentage}%)",
        f"- **Duplicate Records:** {eda_comp.duplicate_analysis.duplicate_rows_count} ({eda_comp.duplicate_analysis.duplicate_percentage}%)",
        "",
        "---",
        "",
        "## 2. Numerical Features Analysis",
        "| Feature | Count | Mean | Std | Min | Median | Max | Skewness | Outliers |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for nf in eda_comp.numeric_features:
        lines.append(
            f"| `{nf.column}` | {nf.count} | {nf.mean} | {nf.std} | {nf.min} | "
            f"{nf.quantiles.median} | {nf.max} | {nf.skewness} ({nf.distribution_shape}) | {nf.outliers.outlier_count} ({nf.outliers.outlier_percentage}%) |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Categorical Features Analysis",
        "| Feature | Count | Unique | Cardinality | Top Categories |",
        "|---|---|---|---|---|",
    ])

    for cf in eda_comp.categorical_features:
        top_str = ", ".join([f"{c.value} ({c.percentage}%)" for c in cf.top_categories[:3]])
        lines.append(f"| `{cf.column}` | {cf.count} | {cf.unique_count} | {cf.cardinality} | {top_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Top Statistical Correlations",
        "| Feature A | Feature B | Pearson r | Strength |",
        "|---|---|---|---|",
    ])

    for pair in eda_comp.correlations.top_positive_correlations[:3]:
        lines.append(f"| `{pair.feature_a}` | `{pair.feature_b}` | +{pair.correlation} | {pair.strength} |")
    for pair in eda_comp.correlations.top_negative_correlations[:3]:
        lines.append(f"| `{pair.feature_a}` | `{pair.feature_b}` | {pair.correlation} | {pair.strength} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Critical Risks & Anomalies",
    ])
    for risk in audit.critical_risks:
        lines.append(f"- ⚠️ {risk}")

    lines.extend([
        "",
        "## 6. Prescribed Actions & Next Steps",
    ])
    for action in audit.prescribed_actions:
        lines.append(f"- 💡 {action}")

    lines.extend([
        "",
        "---",
        "*Report auto-generated by GenAI Data Analyst Platform.*",
    ])

    return "\n".join(lines)


def generate_html_report(dataset_id: str) -> str:
    """Generate a self-contained, beautifully styled HTML report."""
    df, filename = get_dataframe(dataset_id)
    now = datetime.now(timezone.utc).strftime("%B %d, %Y - %H:%M UTC")
    audit = agent_service.execute_agent_workflow(dataset_id, "comprehensive_audit")
    ai_insights = ai_service.generate_dataset_insights(dataset_id)
    eda_comp = eda_service.generate_comprehensive_eda(dataset_id)

    # Health badge color
    score = audit.overall_health_score
    badge_color = "#10b981" if score >= 85 else ("#f59e0b" if score >= 70 else "#ef4444")

    # Build numeric rows HTML
    num_rows_html = ""
    for nf in eda_comp.numeric_features:
        num_rows_html += f"""
        <tr>
            <td class="font-mono">{nf.column}</td>
            <td>{nf.count}</td>
            <td>{nf.mean}</td>
            <td>{nf.std}</td>
            <td>{nf.min}</td>
            <td>{nf.quantiles.median}</td>
            <td>{nf.max}</td>
            <td>{nf.skewness} <span class="tag">{nf.distribution_shape}</span></td>
            <td>{nf.outliers.outlier_count} <span class="tag-alert">({nf.outliers.outlier_percentage}%)</span></td>
        </tr>
        """

    # Build categorical rows HTML
    cat_rows_html = ""
    for cf in eda_comp.categorical_features:
        top_cats = ", ".join([f"<strong>{c.value}</strong> ({c.percentage}%)" for c in cf.top_categories[:3]])
        cat_rows_html += f"""
        <tr>
            <td class="font-mono">{cf.column}</td>
            <td>{cf.count}</td>
            <td>{cf.unique_count}</td>
            <td><span class="tag">{cf.cardinality}</span></td>
            <td>{top_cats}</td>
        </tr>
        """

    # Build findings HTML
    findings_html = "".join([f"<li>{f}</li>" for f in ai_insights.key_findings])
    risks_html = "".join([f"<li>⚠️ {r}</li>" for r in audit.critical_risks])
    actions_html = "".join([f"<li>💡 {a}</li>" for a in audit.prescribed_actions])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI Data Analyst Report — {filename}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --text: #1f2937;
            --bg: #f9fafb;
            --card-bg: #ffffff;
            --border: #e5e7eb;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            border-bottom: 2px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ margin: 0 0 8px 0; color: #111827; font-size: 28px; }}
        .meta {{ color: #6b7280; font-size: 14px; }}
        .score-box {{
            text-align: right;
            background: #f3f4f6;
            padding: 12px 20px;
            border-radius: 8px;
            border-left: 4px solid {badge_color};
        }}
        .score-val {{ font-size: 28px; font-weight: bold; color: {badge_color}; margin: 0; }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: #f9fafb;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            text-align: center;
        }}
        .stat-card .val {{ font-size: 22px; font-weight: bold; color: var(--primary); }}
        .stat-card .lbl {{ font-size: 13px; color: #6b7280; text-transform: uppercase; margin-top: 4px; }}
        h2 {{ color: #1f2937; font-size: 20px; border-left: 4px solid var(--primary); padding-left: 12px; margin-top: 36px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #f3f4f6; font-weight: 600; color: #374151; }}
        tr:hover {{ background-color: #f9fafb; }}
        .font-mono {{ font-family: monospace; font-size: 13px; }}
        .tag {{ background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .tag-alert {{ background: #fee2e2; color: #b91c1c; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 8px; }}
        .footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border); text-align: center; color: #9ca3af; font-size: 13px; }}
        @media print {{
            body {{ padding: 0; background: white; }}
            .container {{ box-shadow: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>GenAI Data Analyst Report</h1>
                <div class="meta">Dataset: <strong>{filename}</strong> | Generated: {now}</div>
            </div>
            <div class="score-box">
                <div class="score-val">{audit.overall_health_score}/100</div>
                <div style="font-size: 12px; color: #6b7280;">Data Health Score</div>
            </div>
        </div>

        <div class="grid-stats">
            <div class="stat-card">
                <div class="val">{len(df)}</div>
                <div class="lbl">Total Records</div>
            </div>
            <div class="stat-card">
                <div class="val">{len(df.columns)}</div>
                <div class="lbl">Variables</div>
            </div>
            <div class="stat-card">
                <div class="val">{eda_comp.missing_analysis.overall_missing_percentage}%</div>
                <div class="lbl">Missing Rate</div>
            </div>
            <div class="stat-card">
                <div class="val">{eda_comp.duplicate_analysis.duplicate_rows_count}</div>
                <div class="lbl">Duplicates</div>
            </div>
        </div>

        <h2>Executive Summary</h2>
        <p>{ai_insights.executive_summary}</p>

        <h2>Key Analytical Findings</h2>
        <ul>{findings_html}</ul>

        <h2>Continuous Features Profile</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th><th>Skewness</th><th>Outliers</th>
                </tr>
            </thead>
            <tbody>{num_rows_html}</tbody>
        </table>

        <h2>Categorical Features Profile</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th><th>Count</th><th>Unique</th><th>Cardinality</th><th>Top Categories</th>
                </tr>
            </thead>
            <tbody>{cat_rows_html}</tbody>
        </table>

        <h2>Identified Risks & Anomalies</h2>
        <ul>{risks_html}</ul>

        <h2>Prescribed Actions & Engineering Roadmap</h2>
        <ul>{actions_html}</ul>

        <div class="footer">
            Generated autonomously by GenAI Data Analyst Platform &bull; MIT License
        </div>
    </div>
</body>
</html>
"""
